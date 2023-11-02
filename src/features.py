import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm


def load_X_and_band_names(
        X_path: str | List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Loads the dataset image and returns the data and band names.

    Args:
        X_path:
            A string or list of strings representing the file paths to the dataset images.

    Returns:
        A tuple of a numpy array containing the data and a list of strings representing the band names.
    """
    # Type check
    if not isinstance(X_path, (str, list)):
        raise TypeError(
            f"Expected string or list, got {type(X_path)} instead.")

    # Ensure X_path is a list
    if isinstance(X_path, str):
        X_path = [X_path]

    # Check if all paths are valid
    for path in X_path:
        if not isinstance(path, str):
            raise TypeError(
                f"Expected string, got {type(path)} instead.")
        if not path.endswith(".tif"):
            raise ValueError(
                f"Expected path to .tif file, got '{path}' instead.")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find file '{path}'.")

    X = None
    band_names = []
    for path in X_path:
        # Load dataset image
        with rasterio.open(path) as src:
            raster = src.read()
            band_count = src.count
            curr_band_names = list(src.descriptions)
            curr_X = raster.transpose(1, 2, 0).reshape(-1, band_count)

        # First image
        if X is None:
            # Check if band names are unique
            if len(set(band_names)) != len(band_names):
                raise ValueError(
                    "All band names must be unique.")

            X = curr_X
            band_names = curr_band_names
        else:
            # Check if band names are the same
            if band_names != curr_band_names:
                raise ValueError(
                    "All band names must be the same.")

            X = np.vstack((X, curr_X))

    return X, band_names


def load_y(
        y_path: str | List[str],
) -> np.ndarray:
    """
    Loads the label image and returns the data.

    Args:
        y_path:
            A string or a list of strings representing the file path(s) to the label image(s).

    Returns:
        A numpy array containing the data.
    """
    # Type check
    if not isinstance(y_path, (str, list)):
        raise TypeError(
            f"Expected string or list, got {type(y_path)} instead.")

    # Ensure y_path is a list
    if isinstance(y_path, str):
        y_path = [y_path]

    # Check if all paths are valid
    for path in y_path:
        if not isinstance(path, str):
            raise TypeError(
                f"Expected string, got {type(path)} instead.")
        if not path.endswith(".tif"):
            raise ValueError(
                f"Expected path to .tif file, got '{path}' instead.")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find file '{path}'.")

    # Load label image(s)
    y_list = []
    for path in y_path:
        with rasterio.open(path) as src:
            y = src.read(1).flatten()
            y_list.append(y)

    # Concatenate label images
    y = np.concatenate(y_list)

    return y


def drop_nan(
        *arrays: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drops rows with NaN values.

    Args:
        arrays:
            A tuple of numpy arrays of either one or two dimensions.

    Returns:
        A tuple of numpy arrays containing the dataset image and label image without NaN values.
    """
    # Type check
    for array in arrays:
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"Expected numpy array, got {type(array)} instead.")

    # Check if all arrays have the same number of rows and at max two dimensions
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError(
            "All arrays must have the same number of rows.")
    if any(len(array.shape) > 2 for array in arrays):
        raise ValueError(
            "All arrays must have at max two dimensions.")

    # Drop rows with NaN values
    mask = np.full(arrays[0].shape[0], fill_value=True)
    for array in arrays:
        new_mask = ~np.isnan(array)
        if len(new_mask.shape) == 2:
            new_mask = new_mask.all(axis=1)
        mask = np.logical_and(mask, new_mask)

    return tuple(array[mask] for array in arrays)


def get_similarity_matrix(
        X: np.array,
        band_names: List[str] = None,
        method: str = "pearson",
) -> pd.DataFrame:
    """
    Calculates the similarity matrix for the data.

    Args:
        X:
            A numpy array containing the data.
        band_names:
            A list of strings representing the band names.
        method:
            A string representing the method to use for calculating the similarity matrix. Must be either 'pearson', 'spearman', or 'mutual_info'. Defaults to 'pearson'.

    Returns:
        A numpy array containing the similarity matrix. It is symmetrical, has a diagonal of ones and values from 0 to 1.
    """
    # Type check
    if not isinstance(X, np.ndarray):
        raise TypeError(
            f"Expected numpy array, got {type(X)} instead.")
    if band_names is not None and not isinstance(band_names, list):
        raise TypeError(
            f"Expected list, got {type(band_names)} instead.")
    illegal_band_names = [
        band_name for band_name in band_names if not isinstance(band_name, str)]
    if any(illegal_band_names):
        raise TypeError(
            f"Expected string for every band name, got {', '.join(str(type(band_name)) for band_name in illegal_band_names)} instead.")
    if not isinstance(method, str):
        raise TypeError(
            f"Expected string, got {type(method)} instead.")

    # Check if X has two dimensions
    if len(X.shape) != 2:
        raise ValueError(
            "X must have two dimensions.")

    # Check if all band names are unique
    if len(set(band_names)) != len(band_names):
        raise ValueError(
            "All band names must be unique.")

    # Check if method is valid
    valid_methods = ["pearson", "spearman", "mutual_info"]
    if method not in valid_methods:
        raise ValueError(
            f"Method must be one of {', '.join(valid_methods)}. Got '{method}' instead.")

    # Calculate similarity matrix
    if method == "pearson":
        similarity_matrix = np.corrcoef(X, rowvar=False)
    elif method == "spearman":
        similarity_matrix = spearmanr(X).correlation
    elif method == "mutual_info":
        # EXPERIMENTAL, most likely scientifically wrong
        n_neighbors = min(3, X.shape[0]-1)
        similarity_matrix = np.full((X.shape[1], X.shape[1]), np.nan)
        for i, band_1 in tqdm(enumerate(X.T)):
            for j, band_2 in enumerate(X.T):
                similarity_matrix[i, j] = mutual_info_regression(band_1.reshape(-1, 1),
                                                                 band_2,
                                                                 n_neighbors=n_neighbors)[0]
        # Esoteric way to achieve a diagonal of 1s
        entropy = np.zeros_like(similarity_matrix)
        for i in range(similarity_matrix.shape[0]):
            component = similarity_matrix[i, i]
            entropy[:, i] += component
            entropy[i, :] += component
            entropy[i, i] = component * 2

        similarity_matrix = similarity_matrix / (entropy - similarity_matrix)

    # Raise error if similarity matrix is NaN
    if similarity_matrix is np.nan:
        raise ValueError(
            f"Could not compute similarity matrix... This commonly occurs if a band has deviation of zero")

    # Ensure the similarity matrix is normalized, symmetric, with diagonal of ones
    similarity_matrix = abs(similarity_matrix)
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    similarity_matrix /= np.nanmax(similarity_matrix)
    similarity_matrix[np.isnan(similarity_matrix)] = 1  # in case xD
    np.fill_diagonal(similarity_matrix, 1)

    # Convert to pandas DataFrame
    similarity_matrix = pd.DataFrame(
        similarity_matrix, columns=band_names, index=band_names)

    return similarity_matrix


def show_similarity_matrix(
        similarity_matrix: pd.DataFrame,
) -> plt.Axes:
    """
    Displays the similarity matrix.

    Args:
        similarity_matrix:
            A pandas DataFrame containing the similarity matrix.
    """
    # Type check
    if not isinstance(similarity_matrix, pd.DataFrame):
        raise TypeError(
            f"Expected pandas DataFrame, got {type(similarity_matrix)} instead.")

    # Check if similarity matrix is square
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(
            "Similarity matrix must be square.")

    # Show similarity matrix
    fig, ax = plt.subplots(figsize=(
        similarity_matrix.columns.shape[0]*0.3,
        similarity_matrix.columns.shape[0]*0.3,
    ))
    image = ax.imshow(similarity_matrix,
                      interpolation="nearest", vmin=0, vmax=1)
    fig.colorbar(image)
    ax.set_xticks(range(similarity_matrix.columns.shape[0]))
    ax.set_yticks(range(similarity_matrix.columns.shape[0]))
    ax.set_xticklabels(similarity_matrix.columns, rotation="vertical")
    ax.set_yticklabels(similarity_matrix.columns)

    return ax


def show_dendrogram(
        similarity_matrix: pd.DataFrame,
) -> plt.Axes:
    """
    Displays a dendrogram according to the similarity matrix.

    Args:
        similarity_matrix:
            A pandas DataFrame containing the similarity matrix.

    Returns:
        A matplotlib Axes object.
    """
    # Type check
    if not isinstance(similarity_matrix, pd.DataFrame):
        raise TypeError(
            f"Expected pandas DataFrame, got {type(similarity_matrix)} instead.")

    # Check if similarity matrix is square
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(
            "Similarity matrix must be square.")

    # Show dendrogram
    fig, ax = plt.subplots(figsize=(16, 16))
    distance_matrix = 1 - similarity_matrix
    dist_linkage = ward(squareform(distance_matrix))
    dendro = dendrogram(
        dist_linkage,
        labels=similarity_matrix.columns,
        ax=ax,
        leaf_rotation=90
    )

    return ax
