"""Performs hyperparameter search for multiple models and evaluates the best
model.

Searches for the best hyperparameters for multiple models using randomized search The search results can be used to compare the performance of different models. The model performance can be compared visually using cv_predict().

Typical usage example:

    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

    data, band_names = load_multi_band_raster("data.tif")
    target, _ = load_multi_band_raster("target.tif")
    data, target = drop_nan_rows(data, target)

    search_space = {
        RandomForestRegressor(): {
            "n_estimators": [100, 200, 300],
        },
        ExtraTreesRegressor(): {
            "max_depth": [5, 10, 15],
        },
    }

    scorers = {
        "mse": make_scorer(mean_squared_error, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    }

    # Perform hyperparameter search
    search_results = hyperparam_search(data, target, search_space, scorers, refit="r2")

    score_df = best_scores(search_results, scorers)

    cv_predict(
        search_results,
        data_path,
        target_path,
        rgb_bands=["B4", "B3", "B2"],
        random_state=42,
    )
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import dill
import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
import rasterio
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typeguard import typechecked

from ltm.data import BROADLEAF_AREA, CONIFER_AREA, list_bands, list_indices
from ltm.features import load_raster, np2pd_like


@typechecked
class EndMemberSplitter(BaseCrossValidator):
    """K-fold splitter that only uses end members for training.

    End members are defined as instances with label 0 or 1. Using this option with area per leaf type as labels is experimental.

    Attributes:
        n_splits:
            Number of splits to use for kfold splitting.
        k_fold:
            KFold object to use for splitting.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initializes the EndMemberSplitter.

        Args:
            n_splits:
                Number of splits to use for kfold splitting.
            shuffle:
                Whether to shuffle the data before splitting into batches. Defaults to False.
            random_state:
                Random state to use for reproducible results.
        """
        self.n_splits = n_splits
        self.k_fold = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def _iter_test_indices(
        self,
        X: npt.ArrayLike | None = None,
        y: npt.ArrayLike | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[np.ndarray]:
        """Generates integer indices corresponding to test sets.

        Args:
            X:
                Data to split.
            y:
                Labels to split.
            groups:
                Group labels to split.

        Yields:
            Integer indices corresponding to test sets.
        """
        fun = self.k_fold._iter_test_indices  # pylint: disable=protected-access
        for test in fun(X, y, groups):
            if y is not None:
                indices = np.where((y[test] == 0) | (y[test] == 1))[0]
                test = test[indices]
            yield test

    def split(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generates indices to split data into training and test set.

        Args:
            data:
                Data to split.
            target:
                Labels to split.
            groups:
                Group labels to split. Not used.

        Yields:
            Tuple of indices for training and test set.
        """
        X = np.array(X)
        y = np.array(y)
        for train, test in self.k_fold.split(X, y):
            indices = np.where((y[train] == 0) | (y[train] == 1))[0]

            end_member_train = train[indices]

            if end_member_train.shape[0] == 0:
                raise ValueError(
                    "No end members in one training set. Maybe you are just unlucky, try another random state."
                )

            yield end_member_train, test

    def get_n_splits(
        self,
        X: npt.ArrayLike | None = None,
        y: npt.ArrayLike | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Args:
            X:
                Data to split. Not used.
            y:
                Target to split. Not used.
            groups:
                Group labels to split. Not used.

        Returns:
            Number of splitting iterations in the cross-validator.
        """
        return self.n_splits


@typechecked
def _target2raster(
    target: pd.Series | pd.DataFrame,
    indices: np.ndarray,
    plot_shape: Tuple[int, int, int],
    area2mixture: bool = False,
) -> np.ndarray:
    # Create target values array of shape (n_samples, n_features)
    target_values = target.values
    if len(target_values.shape) == 1:
        target_values = np.expand_dims(target_values, axis=1)

    # Create raster from target with the help of an indices array
    raster_shape = (plot_shape[1] * plot_shape[2], plot_shape[0])
    raster = np.full(raster_shape, np.nan)
    raster[indices] = target_values
    raster = raster.reshape(
        plot_shape[1], plot_shape[2], plot_shape[0]
    ).transpose(2, 0, 1)

    # Use indices of BROADLEAF_AREA and CONIFER_AREA to compute mixture = broadleaf / (broadleaf + conifer)
    if area2mixture:
        columns = list(target.columns)
        broadleaf_index = columns.index(BROADLEAF_AREA)
        conifer_index = columns.index(CONIFER_AREA)
        broadleaf = raster[broadleaf_index, :, :]
        conifer = raster[conifer_index, :, :]
        raster = broadleaf / (broadleaf + conifer)
        raster = np.expand_dims(raster, axis=0)

    return raster


@typechecked
def _has_nan_error(
    estimator: BaseEstimator,
) -> bool:
    # Checks if estimator raises a ValueError when predicting on NaN
    try:
        estimator.fit([[0]], [0])
        estimator.predict([[np.nan]])
        return False
    except ValueError:
        return True


@typechecked
def _build_pipeline(
    model: BaseEstimator,
    do_standardize: bool,
    do_pca: bool,
    n_components: int | None,
    model_params: Dict[str, Any],
) -> Pipeline:
    # Set params first, as it can change _has_nan_error
    model = model.set_params(**model_params)

    steps = []
    if _has_nan_error(model) or do_pca:
        steps.append(("imputer", KNNImputer()))
    if do_standardize:
        steps.append(("scaler", StandardScaler()))
    if do_pca:
        if n_components is None:
            raise ValueError("n_components must be set if do_pca is True.")
        steps.append(("pca", PCA(n_components=n_components)))

    steps.append(("model", model))

    return Pipeline(steps=steps)


@typechecked
def _study2model(
    study: optuna.study.Study,
    model: BaseEstimator,
    data: npt.ArrayLike,
    target: npt.ArrayLike,
) -> Pipeline:
    # Define preprocessing steps for best model
    model_params = {
        param: value
        for param, value in study.best_params.items()
        if param not in ["do_standardize", "do_pca", "n_components"]
    }

    n_components = None
    if study.best_params["do_pca"]:
        n_components = study.best_params["n_components"]

    best_model = _build_pipeline(
        model,
        study.best_params["do_standardize"],
        study.best_params["do_pca"],
        n_components,
        model_params,
    )

    # Fit best model
    best_model.fit(data, target)

    return best_model


@typechecked
def _create_paths(
    model: BaseEstimator,
    save_folder: str,
) -> Tuple[Path, Path, Path]:
    save_path_obj = Path(save_folder)
    if not save_path_obj.parent.exists():
        raise ValueError(
            f"Directory of save_path does not exist: {save_path_obj.parent}"
        )

    # Check if files already exist
    model_name = model.__class__.__name__
    cache_path = save_path_obj / f"{model_name}_cache.pkl"
    study_path = save_path_obj / f"{model_name}_study.pkl"
    model_path = save_path_obj / f"{model_name}.pkl"

    return study_path, model_path, cache_path


@typechecked
def _check_save_folder(
    model: BaseEstimator,
    data: npt.ArrayLike,
    target: npt.ArrayLike,
    save_folder: str | None,
    use_caching: bool,
) -> Tuple[Pipeline, optuna.study.Study] | None:
    # Check for valid save_path
    if save_folder is not None:
        study_path, model_path, _ = _create_paths(model, save_folder)

        if study_path.exists() and model_path.exists():
            print(
                f"Files already exist, skipping search: {study_path}, {model_path}"
            )

            # Load best model and study
            with open(model_path, "rb") as file:
                best_model = dill.load(file)
            with open(study_path, "rb") as file:
                study = dill.load(file)

            return best_model, study

        if study_path.exists():
            # Inform user
            print("Creating model from study file...")

            # Load the study and create the best model
            with open(study_path, "rb") as file:
                study = dill.load(file)
            best_model = _study2model(study, model, data, target)

            # Save best model
            with open(model_path, "wb") as file:
                dill.dump(best_model, file)

            return best_model, study

        if model_path.exists():
            # Raise error if model file exists but study file is missing
            raise ValueError(
                f"Study file is missing, please delete the model file manually and rerun the script: {model_path}"
            )
    elif use_caching:
        print(
            "Warning: use_caching=True but save_folder=None, caching is disabled."
        )

    return None


@typechecked
def _save_study_model(
    study: optuna.study.Study,
    best_model: Pipeline,
    study_path: Path,
    model_path: Path,
) -> None:
    # Save study
    with open(study_path, "wb") as file:
        dill.dump(study, file)

    # Save best model
    with open(model_path, "wb") as file:
        dill.dump(best_model, file)


@typechecked
def bands_from_importance(
    band_importance_path: str,
    top_n: int = 30,
    level_2a: bool = True,
) -> Tuple[List[str], List[str]]:
    """Extracts the band names of sentinel bands and indices from the band
    importance file.

    The last band with an optimum in one of the scores is interpreted as the last band to be kept. All bands after this band are removed. The bands are then divided into sentinel bands and indices.

    Args:
        band_importance_path:
            Path to the file with band names and their scores.
        top_n:
            Number of top bands to keep. Defaults to 30.
        level_2a:
            Whether the band importance file is from a level 2A dataset. Defaults to True.

    Returns:
        Tuple of lists of sentinel band names and index names.
    """
    # Check path
    if not Path(band_importance_path).exists():
        raise ValueError(f"File does not exist: {band_importance_path}")

    # Read band importance file
    df = pd.read_csv(band_importance_path, index_col=0)
    band_names = df.index
    df = df.reset_index()

    # Divide bands into sentinel bands and indices
    best_bands = list(band_names[:top_n])
    valid_sentinel_bands = list_bands(level_2a)
    valid_index_bands = list_indices()
    sentinel_bands = [
        band for band in valid_sentinel_bands if band in best_bands
    ]
    index_bands = [band for band in valid_index_bands if band in best_bands]

    # Sanity check
    if len(sentinel_bands) + len(index_bands) != len(best_bands):
        raise ValueError(
            "The sum of sentinel bands and index bands does not equal the number of best bands. This should not happen..."
        )

    return sentinel_bands, index_bands


@typechecked
def area2mixture_scorer(scorer: _BaseScorer) -> _BaseScorer:
    """Modifies the score function of a scorer to use the computed leaf type
    mixture from leaf type areas.

    Args:
        scorer:
            A _BaseScorer scorer, e.g. returned by make_scorer().

    Returns:
        Scorer with modified score function.
    """
    score_func = scorer._score_func  # pylint: disable=protected-access

    def mixture_score_func(
        target_true: npt.ArrayLike,
        target_pred: npt.ArrayLike,
        *args,
        **kwargs,
    ) -> Callable:
        # Convert to np.ndarray
        target_true = np.array(target_true)
        target_pred = np.array(target_pred)

        # broadleaf is 0, conifer is 1
        target_true = target_true[:, 0] / (
            target_true[:, 0] + target_true[:, 1]
        )
        target_pred = target_pred[:, 0] / (
            target_pred[:, 0] + target_pred[:, 1]
        )

        return score_func(target_true, target_pred, *args, **kwargs)

    scorer._score_func = mixture_score_func  # pylint: disable=protected-access

    return scorer


@typechecked
def hyperparam_search(  # pylint: disable=too-many-arguments,too-many-locals
    model: BaseEstimator,
    search_space: List[Tuple[str, Tuple, Dict[str, Any]]],
    data: npt.ArrayLike,
    target: npt.ArrayLike,
    scorer: _BaseScorer,
    cv: int | BaseCrossValidator = 5,
    n_trials: int = 100,
    n_jobs: int = 1,
    random_state: int | None = None,
    save_folder: str | None = None,
    use_caching: bool = True,
    always_standardize: bool = False,
) -> Tuple[Pipeline, optuna.study.Study]:
    """Performs hyperparameter search for a model using optuna.

    The search space will be explored using optuna's TPE sampler, together with standardization and PCA for preprocessing. The first trial is the model with its default parameter values. The best pipeline will be returned along with the optuna study. A KNNImputer is used for estimators not supporting NaN values.

    Args:
        model:
            Model to perform hyperparameter search for.
        search_space:
            List of tuples with the name of the method to suggest, the arguments and the keyword arguments. For example [("suggest_float", ("alpha", 1e-10, 1e-1), {"log": True})].
        data:
            Features to use for hyperparameter search.
        target:
            Labels to use for hyperparameter search.
        scorer:
            Scorer to use for hyperparameter search. Please make sure to set greater_is_better=False when using make_scorer if you want to minimize a metric.
        cv:
            Number of folds or BaseCrossValidator instance to use for cross validation. Defaults to 5.
        n_trials:
            Number of trials to perform for hyperparameter search. Defaults to 100.
        n_jobs:
            Number of jobs to use for hyperparameter search. Set it to -1 to maximize parallelization.  Defaults to 1, as otherwise optuna becomes non-deterministic.
        random_state:
            Integer to be used as random state for reproducible results. Defaults to None.
        save_folder:
            Folder to save the study PKL and model PKL to. Uses model.__class__.__name__ to name the files. Skips search if files already exist. Defaults to None.
        use_caching:
            Whether to use caching for the search. Saves a [model]_cache.pkl for each step and resumes if a cache exists. The cache is deleted after the final study is saved. Defaults to True.
        always_standardize:
            Whether to always standardize the data. Recommended for SVM based estimators. Defaults to False.

    Returns:
        Tuple of the best pipeline and the optuna study.
    """
    result = _check_save_folder(
        model,
        data,
        target,
        save_folder,
        use_caching,
    )
    if result is not None:
        return result

    # Create paths
    if save_folder is not None:
        study_path, model_path, cache_path = _create_paths(model, save_folder)

    def callback(study, _):
        # Save intermediate study
        if use_caching and save_folder is not None:
            with open(
                cache_path,  # pylint: disable=possibly-used-before-assignment
                "wb",
            ) as file:
                dill.dump(study, file)

    def objective(trial):
        # Choose whether to standardize and apply PCA
        standardize_options = [True, False]
        if always_standardize:
            standardize_options = [True]
        do_standardize = trial.suggest_categorical(
            "do_standardize", standardize_options
        )
        do_pca = trial.suggest_categorical("do_pca", [True, False])

        # Build pipeline
        n_components = None
        if do_pca:
            n_splits = cv if isinstance(cv, int) else cv.get_n_splits()
            max_components = min(data.shape) - np.ceil(
                min(data.shape) / n_splits
            ).astype(int)
            n_components = trial.suggest_int("n_components", 1, max_components)

        params = {
            args[0]: getattr(trial, name)(*args, **kwargs)
            for name, args, kwargs in search_space
        }

        nonlocal model
        pipe = _build_pipeline(
            model, do_standardize, do_pca, n_components, params
        )

        # Cross validate pipeline
        try:
            cv_results = cross_validate(
                pipe, data, target, cv=cv, scoring=scorer, n_jobs=-1
            )
            score = cv_results["test_score"].mean()

            return score
        # Catch case that all fits fail
        except ValueError:
            print("All fits failed, returning NaN.")
            return np.nan

    if use_caching and save_folder is not None and Path(cache_path).exists():
        # Resume search from cache
        with open(cache_path, "rb") as file:
            study = dill.load(file)
        n_trials -= len(study.trials)

        print(f"Resuming search from cache at trial {len(study.trials)}.")
    else:
        # Start new search
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=random_state),
            direction="maximize",
            study_name=model.__class__.__name__,
        )

    # Optimize study
    study.optimize(
        objective, callbacks=[callback], n_trials=n_trials, n_jobs=n_jobs
    )

    best_model = _study2model(study, model, data, target)

    if save_folder is not None:
        _save_study_model(study, best_model, study_path, model_path)

        # Delete cache
        if use_caching:
            Path(cache_path).unlink()

    return best_model, study


@typechecked
def cv_predict(
    model: BaseEstimator,
    data_path: str,
    target_path: str,
    cv: int | BaseCrossValidator | None = None,
) -> np.ndarray:
    """Predicts on rasters using cross_val_predict.

    Args:
        model:
            Regressor to use for prediction.
        data_path:
            A string with path to the data in GeoTIFF format.
        target_path:
            A string with path to the target data in GeoTIFF format.
        cv:
            An integer for the number of folds or a BaseCrossValidator object for performing cross validation. Will be passed to sklearn.model_selection.cross_val_predict(). Defaults to None.

    Returns:
        A numpy raster of the prediction in the format of read() from rasterio.
    """
    # Load data and plot shape
    data = load_raster(data_path)
    target = load_raster(target_path)

    with rasterio.open(target_path) as src:
        shape = src.read().shape

    # Remove NaNs while keeping the same indices
    indices_array = np.arange(shape[1] * shape[2])
    mask = target.notna()
    data, target, indices_array = data[mask], target[mask], indices_array[mask]

    # Predict using cross_val_predict
    target_pred = cross_val_predict(model, data, target, cv=cv, n_jobs=-1)
    target_pred = np2pd_like(target_pred, target)

    plot = _target2raster(target_pred, indices_array, shape)

    return plot
