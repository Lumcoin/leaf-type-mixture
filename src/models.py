from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.base import RegressorMixin
from sklearn.metrics._scorer import _PredictScorer
from sklearn.model_selection import (BaseCrossValidator, KFold,
                                     RandomizedSearchCV, cross_val_predict)

from src.features import drop_nan, load_X_and_band_names, load_y


class _EndMemberSplitter(BaseCrossValidator):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle,
                            random_state=random_state)

    def split(self, X, y, groups=None):
        for train, test in self.k_fold.split(X, y):
            end_member_train = np.where((y[train] == 0)
                                        | (y[train] == 1))[0]

            if end_member_train.shape[0] == 0:
                raise ValueError(
                    "No end members in one training set. Maybe you are just unlucky, try another random state.")

            yield end_member_train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def hyperparam_search(
        X: np.ndarray,
        y: np.ndarray,
        search_space: Dict[RegressorMixin, Dict[str, Any]],
        scoring: Dict[str, _PredictScorer],
        refit: str,
        kfold_from_endmembers: bool = True,
        kfold_n_splits: int = 5,
        kfold_n_iter: int = 10,
        random_state: int = None,
) -> List[RandomizedSearchCV]:
    """
    Performs hyperparameter search for multiple models.

    Args:
        search_space:
            Dictionary of models and their respective hyperparameter search spaces.
        scorers:
            Dictionary of scorers to use for each model.
        refit:
            Name of the scorer to use for refitting the best model.
        kfold_from_endmembers:
            Whether to use only endmembers for kfold splitting. Endmembers are defined as instances with label 0 or 1.
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
    """
    Returns the best scores for each model.

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
        kfold_n_splits: int = 5,
        kfold_from_endmembers: bool = True,
        random_state: int = None,
) -> None:
    """
    Predicts the labels using cross_val_predict and plots the results for each model.

    Args:
        search_results:
            List of search results from hyperparameter search.
        X_path:
            String or list of strings with path to the X data in GeoTIFF format.
        y_path:
            String or list of strings with path to the y data in GeoTIFF format.
        kfold_n_splits:
            Integer for number of splits to use for kfold splitting.
        kfold_from_endmembers:
            Boolean for whether to use only endmembers for kfold splitting. Endmembers are defined as instances with label 0 or 1.
        random_state:
            Integer to be used as random state for reproducible results.
    """
    # Type check
    if not isinstance(search_results, list):
        raise TypeError("search_results must be a list")
    if any(not isinstance(result, RandomizedSearchCV) for result in search_results):
        raise TypeError(
            "search_results must be a list of RandomizedSearchCV objects")
    if not isinstance(X_path, (str, list)):
        raise TypeError("X_path must be a string or a list of strings")
    if not isinstance(y_path, (str, list)):
        raise TypeError("y_path must be a string or a list of strings")
    if not isinstance(kfold_n_splits, int):
        raise TypeError("kfold_n_splits must be an integer")
    if not isinstance(kfold_from_endmembers, bool):
        raise TypeError("kfold_from_endmembers must be a boolean")

    # Check if y_path and X_path are of same length
    if isinstance(y_path, str):
        y_path = [y_path]
    if isinstance(X_path, str):
        X_path = [X_path]
    if len(y_path) != len(X_path):
        raise ValueError(
            "y_path and X_path must have the same number of paths")

    # Load data and plot shape
    X, _ = load_X_and_band_names(X_path)
    y = load_y(y_path)
    with rasterio.open(y_path) as src:
        plot_shape = src.read(1).shape

    # Remove NaNs while keeping the same indices
    indices_array = np.arange(len(X))
    X, y, indices_array = drop_nan(X, y, indices_array)

    # Use custom kfold splitter if kfold_from_endmembers is True
    if kfold_from_endmembers:
        cv = _EndMemberSplitter(
            kfold_n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(kfold_n_splits, shuffle=True, random_state=random_state)

    # Predict using cross_val_predict
    plots = []
    for result in search_results:
        y_pred = cross_val_predict(
            result.best_estimator_, X, y, cv=cv)
        plot_pred = np.full(np.prod(plot_shape), np.nan)
        plot_pred[indices_array] = y_pred
        plots.append(plot_pred.reshape(plot_shape))

    # Create original image
    original_plot = np.full(np.prod(plot_shape), np.nan)
    original_plot[indices_array] = y
    original_plot = original_plot.reshape(plot_shape)

    # Plot original image with title "original"
    ax = plt.subplot()
    ax.imshow(original_plot, interpolation="nearest")
    ax.set_title("Ground Truth")
    plt.show()

    # Plot predicted images with title "predicted" and name of regressor as subtitle for each plot
    number_plots = len(plots)
    ncols = np.ceil(number_plots**0.5).astype(int)
    nrows = np.ceil(number_plots / ncols).astype(int)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle("Predicted")
    axs = np.asarray(axs).flatten()
    for plot, ax, result in zip(plots, axs, search_results):
        ax.imshow(plot, interpolation="nearest")
        ax.set_title(result.best_estimator_.__class__.__name__)

    for ax in axs[len(plots):]:
        ax.axis("off")

    plt.show()

    return original_plot, plots
