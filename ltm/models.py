"""Performs hyperparameter search for multiple models and evaluates the best
model.

Searches for the best hyperparameters for multiple models using randomized search The search results can be used to compare the performance of different models. The model performance can be compared visually using cv_predict().

Typical usage example: TODO

    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

    X, band_names = load_multi_band_raster("X.tif")
    y, _ = load_multi_band_raster("y.tif")
    X, y = drop_nan_rows(X, y)

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
    search_results = hyperparam_search(X, y, search_space, scorers, refit="r2")

    score_df = best_scores(search_results, scorers)

    cv_predict(
        search_results,
        X_path,
        y_path,
        rgb_bands=["B4", "B3", "B2"],
        random_state=42,
    )
"""

import io
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import dill
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import optuna
import pandas as pd
import rasterio
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics._scorer import _BaseScorer, _PredictScorer
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    RandomizedSearchCV,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typeguard import typechecked

from ltm.data import (
    BROADLEAF_AREA,
    CONIFER_AREA,
    CONIFER_PROPORTION,
    list_bands,
    list_indices,
)
from ltm.features import drop_nan_rows, load_raster, np2pd_like


@typechecked
class EndMemberSplitter(BaseCrossValidator):  # pylint: disable=abstract-method
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

    def split(
        self,
        X: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        groups: np.ndarray | None = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generates indices to split data into training and test set.

        Args:
            X:
                Data to split.
            y:
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
        X: np.typing.ArrayLike | None = None,
        y: np.typing.ArrayLike | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Args:
            X:
                Data to split. Not used.
            y:
                Labels to split. Not used.
            groups:
                Group labels to split. Not used.

        Returns:
            Number of splitting iterations in the cross-validator.
        """
        return self.n_splits


@typechecked
def _y2raster(
    y: np.ndarray | pd.Series | pd.DataFrame,
    indices: np.ndarray,
    plot_shape: Tuple[int, int, int],
    area2mixture: bool = False,
) -> np.ndarray:
    if isinstance(y, np.ndarray):
        y_values = y
    else:
        y_values = y.values

    if len(y_values.shape) == 1:
        y_values = np.expand_dims(y_values, axis=1)

    # Create raster from y with the help of an indices array
    raster_shape = (plot_shape[1] * plot_shape[2], plot_shape[0])
    raster = np.full(raster_shape, np.nan)
    raster[indices] = y_values
    raster = raster.reshape(
        plot_shape[1], plot_shape[2], plot_shape[0]
    ).transpose(2, 0, 1)

    # Use indices of BROADLEAF_AREA and CONIFER_AREA to compute mixture = broadleaf / (broadleaf + conifer)
    if area2mixture:
        columns = list(y.columns)
        broadleaf_index = columns.index(BROADLEAF_AREA)
        conifer_index = columns.index(CONIFER_AREA)
        broadleaf = raster[broadleaf_index, :, :]
        conifer = raster[conifer_index, :, :]
        raster = broadleaf / (broadleaf + conifer)
        raster = np.expand_dims(raster, axis=0)

    return raster


@typechecked
def plot_report(
    df: pd.DataFrame,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label_rotation: int = 0,
    replace_labels: dict | None = None,
    categorical_x: bool = True,
    **kwargs: Any,
) -> plt.Axes:
    """Plots a DataFrame with a title, x and y label and legend.

    The index of the DataFrame is used for the x-axis and the columns for the lines in the plot. The legend is placed to the right of the plot.

    Args:
        df:
            DataFrame to plot.
        title:
            Title of the plot. Defaults to None.
        xlabel:
            Label for the x-axis. Defaults to None.
        ylabel:
            Label for the y-axis. Defaults to None.
        label_rotation:
            Rotation of the x-axis labels. Defaults to 0.
        replace_labels:
            Dictionary to replace labels in the legend. Works also for replacing a subset of the labels. Defaults to None.
        categorical_x:
            Whether the x-axis is categorial and thus each index is an xtick. Not recommended for very long DataFrames. Defaults to True.
        **kwargs:
            Additional keyword arguments to pass to df.plot(), e.g. figsize=(10, 5) or marker="o".

    Returns:
        Axes of the plot.
    """
    # Create empty dictionary if replace is None
    if replace_labels is None:
        replace_labels = {}

    # Plot
    ax = df.plot(**kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set xticks
    if categorical_x:
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=label_rotation)
        ax.xaxis.set_tick_params(which="minor", bottom=False, top=False)

    # Replace labels
    labels = ax.get_legend_handles_labels()[1]
    for i, label in enumerate(labels):
        if label in replace_labels.keys():
            labels[i] = replace_labels[label]

    # Set legend
    golden_ratio = (1 + 5**0.5) / 2
    ax.legend(labels, loc="center left", bbox_to_anchor=(1, 1 / golden_ratio))

    return ax


@typechecked
def bands_from_importance(
    band_importance_path: str,
    level_2a: bool = True,
) -> Tuple[List[str], List[str]]:
    """Extracts the band names of sentinel bands and indices from the band importance file.

    The last band with an optimum in one of the scores is interpreted as the last band to be kept. All bands after this band are removed. The bands are then divided into sentinel bands and indices.

    Args:
        band_importance_path:
            Path to the file with band names and their scores.
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

    # Find first band with an optimum in one of the scores
    best_r2_idx = df["R2 Score"].idxmax()
    best_mae_idx = df["Mean Absolute Error"].idxmin()
    best_rmse_idx = df["Root Mean Squared Error"].idxmin()
    best_idx = min(best_r2_idx, best_mae_idx, best_rmse_idx)

    # Divide bands into sentinel bands and indices
    best_bands = list(band_names[: best_idx + 1])
    best_bands = [band.replace(" ", "_") for band in best_bands]
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
def raster2rgb(
    raster: np.ndarray,
    bands: Tuple[str, ...] | None = None,
    rgb_bands: List[str | None] | None = None,
    mask_nan: bool = True,
) -> np.ndarray:
    """Creates an RGB image from a rasterio raster.

    Args:
        raster:
            Rasterio raster to convert to RGB.
        bands:
            Tuple of band names. Must not be None if rgb_bands is not None. Defaults to None.
        rgb_bands:
            A list of strings representing the band names to use for the RGB image. Defaults to the first three bands if None. Except for when there is only one band, then the RGB image will be grayscale. Or for two bands only R and G will be used. You get the idea. I had to do something for default.
        mask_nan:
            A boolean whether to mask NaN values. If any band has a NaN value, the whole pixel will be masked. Defaults to True.

    Returns:
        A NumPy array RGB image.
    """
    # Check whether bands exists if rgb_bands is not None
    if rgb_bands is not None and bands is None:
        raise ValueError("bands must not be None if rgb_bands is not None")

    # Check if bands is equal to the number of bands in raster
    if bands is not None and len(bands) != raster.shape[0]:
        raise ValueError(
            f"bands must have same length as number of bands in raster: {len(bands)} != {raster.shape[0]}"
        )

    # Default bands to tuple of integers if None
    if bands is None:
        bands = tuple(str(number) for number in range(raster.shape[0]))

    # Check if rgb_bands has length of three or is None
    if rgb_bands is not None and len(rgb_bands) != 3:
        raise ValueError("rgb_bands must be a list of length 3 or None")

    # Fill rgb_bands with bands and maybe None if it is None
    if rgb_bands is None:
        if len(bands) == 1:
            rgb_bands = [bands[0]] * 3
        elif len(bands) == 2:
            rgb_bands = [bands[0], bands[1], None]
        else:
            rgb_bands = list(bands[:3])

    # Create RGB image bands
    rgb_plot = []
    for rgb_band in rgb_bands:
        if rgb_band is not None:
            idx = bands.index(rgb_band)
            rgb_plot.append(raster[idx])
        else:
            rgb_plot.append(np.zeros_like(raster[0, :, :]))

    # Stack RGB image bands
    rgb_plot = np.dstack(rgb_plot)

    # Mask NaN values
    if mask_nan:
        nan_mask = np.any(np.isnan(raster), axis=0)
        rgb_plot[nan_mask] = np.nan

    return rgb_plot


@typechecked
def plot2array(fig: plt.Figure | None = None) -> np.ndarray:
    """Converts a matplotlib figure to a numpy array.

    Args:
        fig:
            Matplotlib figure to convert. Defaults to None, which will use the current figure.

    Returns:
        Numpy array of the figure.
    """
    with io.BytesIO() as buff:
        if fig is None:
            fig = plt.gcf()
        fig.savefig(buff, format="png")
        buff.seek(0)
        im = plt.imread(buff)

    return im


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
    score_func = scorer._score_func

    def mixture_score_func(
        y_true: np.typing.ArrayLike,
        y_pred: np.typing.ArrayLike,
        *args,
        **kwargs,
    ) -> Callable:
        # Convert to np.ndarray
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # broadleaf is 0, conifer is 1
        y_true = y_true[:, 0] / (y_true[:, 0] + y_true[:, 1])
        y_pred = y_pred[:, 0] / (y_pred[:, 0] + y_pred[:, 1])

        return score_func(y_true, y_pred, *args, **kwargs)

    scorer._score_func = mixture_score_func

    return scorer


@typechecked
def best_scores(
    search_results: List[RandomizedSearchCV],
    scoring: Dict[str, _PredictScorer],
) -> pd.DataFrame:
    """Returns the best scores for each model.

    Args:
        search_results:
            List of search results from hyperparameter search.
        scoring:
            Dictionary of scorers to use for each model.

    Returns:
        Dictionary of best scores for each model.
    """
    # Check if all models have different names
    model_names = [
        result.best_estimator_.__class__.__name__ for result in search_results
    ]
    if len(set(model_names)) != len(model_names):
        raise ValueError(
            "All models must be of different kind, as they are used as keys in the dictionary."
        )

    # Extract the scores of each metric for each model at best_index
    df_dict = defaultdict(list)
    for metric_name, scorer in scoring.items():
        for result in search_results:
            best_index = np.nonzero(
                result.cv_results_[f"rank_test_{metric_name}"] == 1
            )[0][0]
            df_dict[metric_name].append(
                result.cv_results_["mean_test_" + metric_name][best_index]
                * scorer._sign
            )

    # Create a new dataframe with the scores
    df = pd.DataFrame(df_dict, index=model_names)

    return df


@typechecked
def hyperparam_search(
    model: BaseEstimator,
    search_space: List[Tuple[str, Tuple, Dict[str, Any]]],
    X: np.typing.ArrayLike,
    y: np.typing.ArrayLike,
    scorer: _BaseScorer,
    cv: int = 5,
    n_trials: int = 100,
    n_jobs: int = 1,
    random_state: int | None = None,
    save_folder: str | None = None,
) -> Tuple[Pipeline, optuna.study.Study]:
    """Performs hyperparameter search for a model using optuna.

    The search space will be explored using optuna's TPE sampler, together with standardization and PCA for preprocessing. The first trial is the model with its default parameter values. The best model will be returned along with the optuna study.

    Args:
        model:
            Model to perform hyperparameter search for.
        search_space:
            List of tuples with the name of the method to suggest, the arguments and the keyword arguments. For example [("suggest_float", ("alpha", 1e-10, 1e-1), {"log": True})].
        X:
            Features to use for hyperparameter search.
        y:
            Labels to use for hyperparameter search.
        scorer:
            Scorer to use for hyperparameter search. Please make sure to set greater_is_better=False when using make_scorer if you want to minimize a metric.
        cv:
            Number of folds to use for cross validation. Defaults to 5.
        n_trials:
            Number of trials to perform for hyperparameter search. Defaults to 100.
        n_jobs:
            Number of jobs to use for hyperparameter search. Set it to -1 to maximize parallelization.  Defaults to 1, as otherwise optuna becomes non-deterministic.
        random_state:
            Integer to be used as random state for reproducible results. Defaults to None.
        save_folder:
            Folder to save the study SQLite database and model PKL to. Uses model.__class__.__name__ to name the files. Skips search if files already exist. Defaults to None.
    """
    # Check for valid save_path
    if save_folder is not None:
        save_path_obj = Path(save_folder)
        if not save_path_obj.parent.exists():
            raise ValueError(
                f"Directory of save_path does not exist: {save_path_obj.parent}"
            )

        # Check if files already exist
        model_name = model.__class__.__name__
        study_path = save_path_obj / f"{model_name}_study.pkl"
        model_path = save_path_obj / f"{model_name}.pkl"
        if study_path.exists() and model_path.exists():
            print(
                f"Files already exist, skipping search: {study_path}, {model_path}"
            )
            with open(model_path, "rb") as file:
                best_model = dill.load(file)
            with open(study_path, "rb") as file:
                study = dill.load(file)

            return best_model, study
        if study_path.exists() or model_path.exists():
            raise ValueError(
                f"Only one file exists, please delete it manually: {study_path}, {model_path}"
            )

    def objective(trial):
        # Choose whether to standardize and apply PCA
        do_standardize = trial.suggest_categorical(
            "do_standardize", [True, False]
        )
        do_pca = trial.suggest_categorical("do_pca", [True, False])

        # Define pipeline steps
        steps = []
        if do_standardize:
            steps.append(("scaler", StandardScaler()))
        if do_pca:
            max_components = min(X.shape) - np.ceil(min(X.shape) / cv).astype(
                int
            )
            n_components = trial.suggest_int("n_components", 1, max_components)
            steps.append(("pca", PCA(n_components=n_components)))

        # Define model step
        nonlocal model
        params = {
            args[0]: getattr(trial, name)(*args, **kwargs)
            for name, args, kwargs in search_space
        }
        model = model.set_params(**params)
        steps.append(("model", model))

        # Cross validate pipeline
        pipe = Pipeline(steps=steps)
        cv_results = cross_validate(
            pipe, X, y, cv=cv, scoring=scorer, n_jobs=-1
        )
        score = cv_results["test_score"].mean()

        return score

    # Optimize study with random state
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        sampler=sampler, direction="maximize", study_name=model_name
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Save study results and best model using dill
    if save_folder is not None:
        # Save study
        with open(study_path, "wb") as file:
            dill.dump(study, file)

        # Define pipeline steps for best model
        steps = []
        if study.best_params["do_standardize"]:
            steps.append(("scaler", StandardScaler()))
        if study.best_params["do_pca"]:
            steps.append(
                ("pca", PCA(n_components=study.best_params["n_components"]))
            )
        model_params = {
            args[0]: study.best_params[args[0]] for _, args, _ in search_space
        }
        model = model.set_params(**model_params)

        # Create and save best model
        steps.append(("model", model))
        best_model = Pipeline(steps=steps)
        with open(model_path, "wb") as file:
            dill.dump(best_model, file)

    return best_model, study


# TODO: improve formatting of figure
@typechecked
def cv_predict(
    search_results: List[RandomizedSearchCV],
    X_path: str | List[str],
    y_path: str | List[str],
    rgb_bands: List[str] | None = None,
    cv: int | BaseCrossValidator | None = None,
    area2mixture: bool = False,
    save_path: str | None = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Predicts the labels using cross_val_predict and plots the results for
    each model.

    Args:
        search_results:
            List of search results from hyperparameter search.
        X_path:
            Single string with path to the X data in GeoTIFF format.
        y_path:
            Single string with path to the y data in GeoTIFF format.
        rgb_bands:
            A list of strings representing the band names to use for the RGB image. Defaults to the first three bands if None. Except for when there is only one band, then the RGB image will be grayscale. Or for two bands only R and G will be used. You get the idea. I had to do something for default.
        cv:
            An integer for the number of folds or a BaseCrossValidator object for performing cross validation. Defaults to None.
        random_state:
            Integer to be used as random state for reproducible results.
        area2mixture:
            Whether to show the computed leaf type mixture from leaf type areas as labels. Defaults to False.
        save_path:
            Path to save the plot to with either JPG or PNG suffix. Defaults to None.

    Returns:
        Tuple of ground truth image and list of predicted images.
    """
    # Check for valid save_path and valid suffix
    if save_path is not None:
        save_path_obj = Path(save_path)
        if save_path_obj.suffix not in [".jpg", ".png"]:
            raise ValueError(
                f"save_path must have suffix '.jpg' or '.png' or be None: {save_path}"
            )
        if not save_path_obj.parent.exists():
            raise ValueError(
                f"Directory of save_path does not exist: {save_path_obj.parent}"
            )

    # Load data and plot shape
    X = load_raster(X_path)
    y = load_raster(y_path)

    with rasterio.open(y_path) as src:
        bands = src.descriptions
        shape = src.read().shape

    # Check if y has only one band and area2mixture is True -> raise error
    if area2mixture:
        if shape[0] > 1:
            bands = tuple([CONIFER_PROPORTION])
        else:
            print(
                "'area2mixture=True' is ignored: y has only one band, so computing the mixture from area is not possible."
            )
            area2mixture = False

    # Remove NaNs while keeping the same indices
    indices_array = np.arange(shape[1] * shape[2])
    X, y, indices_array = drop_nan_rows(X, y, indices_array)

    # Predict using cross_val_predict
    plots = []
    rgb_plots = []
    for result in search_results:
        y_pred = cross_val_predict(result.best_estimator_, X, y, cv=cv)
        y_pred = np2pd_like(y_pred, y)

        plot = _y2raster(
            y_pred, indices_array, shape, area2mixture=area2mixture
        )
        plots.append(plot)
        rgb_plot = raster2rgb(plot, bands, rgb_bands)
        rgb_plots.append(rgb_plot)

    # Create ground truth image
    gt_plot = _y2raster(y, indices_array, shape, area2mixture=area2mixture)
    rgb_gt_plot = raster2rgb(gt_plot, bands, rgb_bands)

    # Prepare plots for plotting by normalizing and removing nan
    maximum = np.nanmax(
        [np.nanmax(rgb_plot) for rgb_plot in rgb_plots]
        + [np.nanmax(rgb_gt_plot)]
    )

    rgb_gt_plot = rgb_gt_plot / maximum
    rgb_plots = [rgb_plot / maximum for rgb_plot in rgb_plots]

    rgb_gt_plot[np.isnan(rgb_gt_plot)] = 0
    for rgb_plot in rgb_plots:
        mask = np.logical_or(np.isnan(rgb_plot), rgb_plot < 0)
        rgb_plot[mask] = 0

    # Convert plots to 2D if y has only one band
    if gt_plot.shape[0] == 1:
        rgb_gt_plot = rgb_gt_plot[:, :, 0]
        rgb_plots = [rgb_plot[:, :, 0] for rgb_plot in rgb_plots]

    # Plot original image with title "original"
    ax = plt.subplot()
    ax.imshow(rgb_gt_plot, interpolation="nearest")
    ax.set_title("Ground Truth")
    if save_path is not None:
        gt_image = plot2array()

    # Plot predicted images with title "predicted" and name of regressor as subtitle for each plot
    number_plots = len(rgb_plots)
    ncols = np.ceil(number_plots**0.5).astype(int)
    nrows = np.ceil(number_plots / ncols).astype(int)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle("Predicted")
    axs = np.asarray(axs).flatten()
    for rgb_plot, ax, result in zip(rgb_plots, axs, search_results):
        ax.imshow(rgb_plot, interpolation="nearest")
        ax.set_title(result.best_estimator_.__class__.__name__)

    for ax in axs[len(rgb_plots) :]:
        ax.axis("off")

    if save_path is not None:
        pred_image = plot2array(fig)
        image = np.concatenate((gt_image, pred_image), axis=0)
        plt.imsave(save_path, image)

    plt.show()

    return gt_plot, plots
