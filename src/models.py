from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.base import RegressorMixin
from sklearn.metrics._scorer import _PredictScorer
from sklearn.model_selection import (BaseCrossValidator, KFold,
                                     RandomizedSearchCV, cross_val_predict)

from src.features import drop_nan, load_multi_band_raster, raster2rgb


class _EndMemberSplitter(BaseCrossValidator):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle,
                            random_state=random_state)

    def split(self, X, y, groups=None):
        for train, test in self.k_fold.split(X, y):
            indices = np.where((y[train] == 0)
                               | (y[train] == 1))[0]

            end_member_train = train[indices]

            if end_member_train.shape[0] == 0:
                raise ValueError(
                    "No end members in one training set. Maybe you are just unlucky, try another random state.")

            yield end_member_train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def _y2raster(y, indices, plot_shape):
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)

    raster_shape = (plot_shape[1] * plot_shape[2], plot_shape[0])
    raster = np.full(raster_shape, np.nan)
    raster[indices] = y
    raster = raster.reshape(
        plot_shape[1], plot_shape[2], plot_shape[0]).transpose(2, 0, 1)

    return raster


def hyperparam_search(
        X: np.ndarray,
        y: np.ndarray,
        search_space: Dict[RegressorMixin, Dict[str, Any]],
        scoring: Dict[str, _PredictScorer],
        refit: str,
        kfold_from_endmembers: bool = False,
        kfold_n_splits: int = 5,
        kfold_n_iter: int = 10,
        random_state: int = None,
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
    # Type check
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if not isinstance(search_space, dict):
        raise TypeError("search_space must be a dictionary")
    if not isinstance(scoring, dict):
        raise TypeError("scoring must be a dictionary")
    if not isinstance(refit, str):
        raise TypeError("refit must be a string")
    if not isinstance(kfold_from_endmembers, bool):
        raise TypeError("kfold_from_endmembers must be a boolean")
    if not isinstance(kfold_n_splits, int):
        raise TypeError("kfold_n_splits must be an integer")
    if not isinstance(kfold_n_iter, int):
        raise TypeError("kfold_n_iter must be an integer")
    if not isinstance(random_state, (int, type(None))):
        raise TypeError("random_state must be an integer")

    # Use custom kfold splitter if kfold_from_endmembers is True
    if kfold_from_endmembers:
        cv = _EndMemberSplitter(
            kfold_n_splits, shuffle=True, random_state=random_state)
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


def best_scores(
        search_results: List[RandomizedSearchCV],
        scoring: Dict[str, _PredictScorer],
) -> Dict[str, float]:
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
        result.best_estimator_.__class__.__name__ for result in search_results]
    if len(set(model_names)) != len(model_names):
        raise ValueError(
            "All models must be of different kind, as they are used as keys in the dictionary.")

    # Extract the scores of each metric for each model at best_index
    df_dict = defaultdict(list)
    for metric_name, scorer in scoring.items():
        for result in search_results:
            best_index = np.nonzero(
                result.cv_results_[f"rank_test_{metric_name}"] == 1)[0][0]
            df_dict[metric_name].append(
                result.cv_results_["mean_test_" + metric_name][best_index] * scorer._sign)

    # Create a new dataframe with the scores
    df = pd.DataFrame(df_dict,
                      index=model_names)

    return df


def cv_predict(
        search_results: List[RandomizedSearchCV],
        X_path: str | List[str],
        y_path: str | List[str],
        rgb_bands: List[str] = None,
        kfold_n_splits: int = 5,
        kfold_from_endmembers: bool = False,
        random_state: int = None,
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
    # Type check
    if not isinstance(search_results, list):
        raise TypeError("search_results must be a list")
    if any(not isinstance(result, RandomizedSearchCV) for result in search_results):
        raise TypeError(
            "search_results must be a list of RandomizedSearchCV objects")
    if not isinstance(X_path, str):
        raise TypeError("X_path must be a string")
    if not isinstance(y_path, str):
        raise TypeError("y_path must be a string")
    # rgb_bands is checked in raster2rgb()
    if not isinstance(kfold_n_splits, int):
        raise TypeError("kfold_n_splits must be an integer")
    if not isinstance(kfold_from_endmembers, bool):
        raise TypeError("kfold_from_endmembers must be a boolean")

    # Load data and plot shape
    X, _ = load_multi_band_raster(X_path)
    y, _ = load_multi_band_raster(y_path)
    if y.shape[1] == 1:
        y = y.ravel()
    elif kfold_from_endmembers:
        print("Warning: Using kfold_from_endmembers with area per leaf type as labels is experimental.")

    with rasterio.open(y_path) as src:
        bands = src.descriptions
        shape = src.read().shape

    # Remove NaNs while keeping the same indices
    indices_array = np.arange(shape[1] * shape[2])
    X, y, indices_array = drop_nan(X, y, indices_array)

    # Use custom kfold splitter if kfold_from_endmembers is True
    if kfold_from_endmembers:
        cv = _EndMemberSplitter(
            kfold_n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(kfold_n_splits, shuffle=True, random_state=random_state)

    # Predict using cross_val_predict
    plots = []
    rgb_plots = []
    for result in search_results:
        y_pred = cross_val_predict(
            result.best_estimator_, X, y, cv=cv)

        plot = _y2raster(y_pred, indices_array, shape)
        plots.append(plot)
        rgb_plot = raster2rgb(plot, bands, rgb_bands)
        rgb_plots.append(rgb_plot)

    # Create ground truth image
    gt_plot = _y2raster(y, indices_array, shape)
    rgb_gt_plot = raster2rgb(gt_plot, bands, rgb_bands)

    # Prepare plots for plotting by normalizing and removing nan
    maximum = np.nanmax([np.nanmax(rgb_plot)
                        for rgb_plot in rgb_plots] + [np.nanmax(rgb_gt_plot)])

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

    for ax in axs[len(rgb_plots):]:
        ax.axis("off")

    plt.show()

    return gt_plot, plots
