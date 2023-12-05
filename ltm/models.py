from collections import defaultdict
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.base import RegressorMixin
from sklearn.metrics._scorer import _BaseScorer, _PredictScorer
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from typeguard import typechecked

from ltm.features import drop_nan, load_multi_band_raster, raster2rgb


@typechecked
class _EndMemberSplitter(BaseCrossValidator):
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
        random_state: Optional[int] = None,
    ) -> None:
        self.n_splits = n_splits
        self.k_fold = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
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
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        return self.n_splits


@typechecked
def _y2raster(
    y: np.ndarray, indices: np.ndarray, plot_shape: Tuple[int, int, int]
) -> np.ndarray:
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)

    # Create raster from y with the help of an indices array
    raster_shape = (plot_shape[1] * plot_shape[2], plot_shape[0])
    raster = np.full(raster_shape, np.nan)
    raster[indices] = y
    raster = raster.reshape(plot_shape[1], plot_shape[2], plot_shape[0]).transpose(
        2, 0, 1
    )

    return raster


@typechecked
def area2mixture_scorer(scorer: _BaseScorer) -> _BaseScorer:
    """Modifies the score function of a scorer to use the computed leaf type mixture from leaf type areas.

    Args:
        scorer:
            A _BaseScorer scorer, e.g. returned by make_scorer().

    Returns:
        Scorer with modified score function.
    """
    score_func = scorer._score_func

    def mixture_score_func(
        y_true: np.ndarray, y_pred: np.ndarray, *args, **kwargs
    ) -> Callable:
        # broadleaf is 0, conifer is 1
        y_true = y_true[:, 0] / (y_true[:, 0] + y_true[:, 1])
        y_pred = y_pred[:, 0] / (y_pred[:, 0] + y_pred[:, 1])

        return score_func(y_true, y_pred, *args, **kwargs)

    scorer._score_func = mixture_score_func

    return scorer


@typechecked
def hyperparam_search(
    X: np.ndarray,
    y: np.ndarray,
    search_space: Dict[RegressorMixin, Dict[str, Any]],
    scoring: Dict[str, _PredictScorer],
    refit: str,
    kfold_from_endmembers: bool = False,
    kfold_n_splits: int = 5,
    kfold_n_iter: int = 10,
    random_state: Optional[int] = None,
) -> List[RandomizedSearchCV]:
    """Performs hyperparameter search for multiple models.

    Args:
        search_space:
            Dictionary of models and their respective hyperparameter search spaces.
        scorers:
            Dictionary of scorers to use for each model.
        refit:
            Name of the scorer to use for refitting the best model.
        kfold_from_endmembers:
            Boolean for whether to use only endmembers for kfold splitting. Endmembers are defined as instances with label 0 or 1. Using this option with area per leaf type as labels is experimental. Defaults to False.
        kfold_n_splits:
            Number of splits to use for kfold splitting.
        kfold_n_iter:
            Number of iterations to use for kfold splitting.
        random_state:
            Random state to use for reproducible results.
    """
    # Use custom kfold splitter if kfold_from_endmembers is True
    if kfold_from_endmembers:
        cv = _EndMemberSplitter(kfold_n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(kfold_n_splits, shuffle=True, random_state=random_state)

    # Perform hyperparameter search for each model
    search_results = []
    for model, param_distributions in search_space.items():
        param_distributions["random_state"] = [random_state]
        # model.random_state = random_state
        search = RandomizedSearchCV(
            model,
            param_distributions,
            scoring=scoring,
            refit=refit,
            cv=cv,
            return_train_score=True,
            n_iter=kfold_n_iter,
            verbose=1,
            random_state=random_state,
        )
        search.fit(X, y)

        search_results.append(search)

    return search_results


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
def cv_predict(
    search_results: List[RandomizedSearchCV],
    X_path: str | List[str],
    y_path: str | List[str],
    rgb_bands: Optional[List[str]] = None,
    kfold_n_splits: int = 5,
    kfold_from_endmembers: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Predicts the labels using cross_val_predict and plots the results for each model.

    Args:
        search_results:
            List of search results from hyperparameter search.
        X_path:
            Single string with path to the X data in GeoTIFF format.
        y_path:
            Single string with path to the y data in GeoTIFF format.
        rgb_bands:
            A list of strings representing the band names to use for the RGB image. Defaults to the first three bands if None. Except for when there is only one band, then the RGB image will be grayscale. Or for two bands only R and G will be used. You get the idea. I had to do something for default.
        kfold_n_splits:
            Integer for number of splits to use for kfold splitting.
        kfold_from_endmembers:
            Boolean for whether to use only endmembers for kfold splitting. Endmembers are defined as instances with label 0 or 1. Using this option with area per leaf type as labels is experimental. Defaults to False.
        random_state:
            Integer to be used as random state for reproducible results.

    Returns:
        Tuple of ground truth image and list of predicted images.
    """
    # Load data and plot shape
    X, _ = load_multi_band_raster(X_path)
    y, _ = load_multi_band_raster(y_path)
    if y.shape[1] == 1:
        y = y.ravel()
    elif kfold_from_endmembers:
        print(
            "Warning: Using kfold_from_endmembers with area per leaf type as labels is experimental."
        )

    with rasterio.open(y_path) as src:
        bands = src.descriptions
        shape = src.read().shape

    # Remove NaNs while keeping the same indices
    indices_array = np.arange(shape[1] * shape[2])
    X, y, indices_array = drop_nan(X, y, indices_array)

    # Use custom kfold splitter if kfold_from_endmembers is True
    if kfold_from_endmembers:
        cv = _EndMemberSplitter(kfold_n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(kfold_n_splits, shuffle=True, random_state=random_state)

    # Predict using cross_val_predict
    plots = []
    rgb_plots = []
    for result in search_results:
        y_pred = cross_val_predict(result.best_estimator_, X, y, cv=cv)

        plot = _y2raster(y_pred, indices_array, shape)
        plots.append(plot)
        rgb_plot = raster2rgb(plot, bands, rgb_bands)
        rgb_plots.append(rgb_plot)

    # Create ground truth image
    gt_plot = _y2raster(y, indices_array, shape)
    rgb_gt_plot = raster2rgb(gt_plot, bands, rgb_bands)

    # Prepare plots for plotting by normalizing and removing nan
    maximum = np.nanmax(
        [np.nanmax(rgb_plot) for rgb_plot in rgb_plots] + [np.nanmax(rgb_gt_plot)]
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
    plt.show()

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

    plt.show()

    return gt_plot, plots
