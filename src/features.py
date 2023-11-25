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


def load_multi_band_raster(
        raster_path: str | List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Loads one or many rasters with multiple bands and returns the data ready to use with sklearn and band names.

    Args:
        raster_path:
            A string or list of strings representing the file paths to the images.

    Returns:
        A tuple of a numpy array containing the data and a list of strings representing the band names.
    """
    # Type check
    if not isinstance(raster_path, (str, list)):
        raise TypeError(
            f"Expected string or list, got {type(raster_path)} instead.")

    # Ensure X_path is a list
    if isinstance(raster_path, str):
        raster_path = [raster_path]

    # Check if all paths are valid
    for path in raster_path:
        if not isinstance(path, str):
            raise TypeError(
                f"Expected string, got {type(path)} instead.")
        if not path.endswith(".tif"):
            raise ValueError(
                f"Expected path to .tif file, got '{path}' instead.")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find file '{path}'.")

    data = None
    band_names = []
    for path in raster_path:
        # Load dataset image
        with rasterio.open(path) as src:
            raster = src.read()
            band_count = src.count
            curr_band_names = list(src.descriptions)
            curr_data = raster.transpose(1, 2, 0).reshape(-1, band_count)

        # First image
        if data is None:
            # Check if band names are unique
            if len(set(band_names)) != len(band_names):
                raise ValueError(
                    "All band names must be unique.")

            data = curr_data
            band_names = curr_band_names
        else:
            # Check if band names are the same
            if band_names != curr_band_names:
                raise ValueError(
                    "All band names must be the same.")

            data = np.vstack((data, curr_data))

    return data, band_names


def interpolate_X_and_bands(
        X: np.ndarray,
        band_names: List[str],
        cyclic: bool = True,
        method: str = "linear",
        order: int = None,
) -> Tuple[np.ndarray, List[str]]:
    """Interpolate missing time series values in X using the given method.

    Args:
        X:
            A numpy array containing the data.
        band_names:
            A list of strings representing the band names. Necessary for deducing the number of composites.
        cyclic:
            A boolean representing whether the data is cyclic. If so, the interpolation of values at the start will use values from the end of the time series and vice versa. Defaults to True.
        method:
            A string representing the method to use for interpolation. Methods 'polynomial' and 'spline' require an integer 'order' as additional argument. Defaults to 'linear'. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
        order:
            An integer representing the order of the polynomial or spline interpolation. Defaults to None.

    Returns:
        A tuple of a numpy array containing the interpolated data and a list of strings representing the band names. The band names are unchanged.
    """
    # Type check
    if not isinstance(X, np.ndarray):
        raise TypeError(
            f"Expected numpy array, got {type(X)} instead.")
    if not isinstance(band_names, list):
        raise TypeError(
            f"Expected list, got {type(band_names)} instead.")
    illegal_band_names = [
        band_name for band_name in band_names if not isinstance(band_name, str)]
    if any(illegal_band_names):
        raise TypeError(
            f"Expected string for every band name, got {', '.join(str(type(band_name)) for band_name in illegal_band_names)} instead.")
    if not isinstance(cyclic, bool):
        raise TypeError(
            f"Expected boolean, got {type(cyclic)} instead.")
    if not isinstance(method, str):
        raise TypeError(
            f"Expected string, got {type(method)} instead.")
    if order is not None and not isinstance(order, int):
        raise TypeError(
            f"Expected integer, got {type(order)} instead.")

    # Get the number of composites and number of bands
    num_composites = int(band_names[-1].split()[0])
    num_bands = len(band_names) // num_composites

    # Reshape into DataFrame with one row per composite
    reshaped_X = X.reshape(-1, num_composites, num_bands)
    reshaped_X = reshaped_X.transpose(0, 2, 1)
    reshaped_X = reshaped_X.reshape(-1, num_composites).T
    df = pd.DataFrame(reshaped_X)

    # Interpolate
    if cyclic:
        df = pd.concat([df] * 3, ignore_index=True)
    df.interpolate(method=method, inplace=True, order=order)
    if cyclic:
        start = len(df) // 3
        end = 2 * len(df) // 3
        df = df.iloc[start:end].reset_index(drop=True)

    # Reshape back into original shape
    interpolated_X = df.values.T.reshape(-1, num_bands, num_composites)
    interpolated_X = interpolated_X.transpose(0, 2, 1)
    interpolated_X = interpolated_X.reshape(-1,
                                            num_bands * num_composites)

    return interpolated_X, band_names


def save_raster(
        X: np.ndarray,
        band_names: List[str],
        source_path: str,
        destination_path: str,
) -> str:
    """Saves the data as a raster image.

    Args:
        X:
            A numpy array containing the data.
        band_names:
            A list of strings representing the band names.
        source_path:
            A string representing the file path to the source image. Used for copying the raster profile.
        destination_path:
            A string representing the file path to the destination image.

    Returns:
        A string representing the file path to the destination image.
    """
    # Type check
    if not isinstance(X, np.ndarray):
        raise TypeError(
            f"Expected numpy array, got {type(X)} instead.")
    if not isinstance(band_names, list):
        raise TypeError(
            f"Expected list, got {type(band_names)} instead.")
    illegal_band_names = [
        band_name for band_name in band_names if not isinstance(band_name, str)]
    if any(illegal_band_names):
        raise TypeError(
            f"Expected string for every band name, got {', '.join(str(type(band_name)) for band_name in illegal_band_names)} instead.")
    if not isinstance(source_path, str):
        raise TypeError(
            f"Expected string, got {type(source_path)} instead.")
    if not isinstance(destination_path, str):
        raise TypeError(
            f"Expected string, got {type(destination_path)} instead.")

    # Copy X
    X = X.copy()

    # Read raster profile and shape
    with rasterio.open(source_path) as raster:
        profile = raster.profile
        shape = raster.read().shape

    # Reshape X
    X = X.reshape(shape[1], shape[2], shape[0]).transpose(2, 0, 1)

    # Write raster
    with rasterio.open(destination_path, "w", **profile) as dst:
        dst.write(X)
        dst.descriptions = band_names

    return destination_path


def raster2rgb(
        raster: np.ndarray,
        bands: Tuple[str],
        rgb_bands: List[str] = None,
) -> np.ndarray:
    """Creates an RGB image from the raster ready for plt.plot().

    Args:
        raster:
            A numpy array containing the data.
        bands:
            A tuple of strings representing the band names.
        rgb_bands:
            A list of strings representing the band names to use for the RGB image. Defaults to the first three bands if None. Except for when there is only one band, then the RGB image will be grayscale. Or for two bands only R and G will be used. You get the idea. I had to do something for default.

    Returns:
        A numpy array containing the RGB image ready for plt.plot().
    """
    # Type check
    if not isinstance(raster, np.ndarray):
        raise TypeError(
            f"Expected numpy array, got {type(raster)} instead.")
    if not isinstance(bands, tuple):
        raise TypeError(
            f"Expected tuple, got {type(bands)} instead.")
    illegal_bands = [
        band for band in bands if not isinstance(band, str)]
    if any(illegal_bands):
        raise TypeError(
            f"Expected string for every band, got {', '.join(str(type(band)) for band in illegal_bands)} instead.")
    if rgb_bands is not None:
        if not isinstance(rgb_bands, list):
            raise TypeError(
                f"Expected list, got {type(rgb_bands)} instead.")

        illegal_rgb_bands = [
            band for band in rgb_bands if not isinstance(band, str)]
        if any(illegal_rgb_bands):
            raise TypeError(
                f"Expected string for every band, got {', '.join(str(type(band)) for band in illegal_rgb_bands)} instead.")

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
            rgb_bands = bands[:3]

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

    return rgb_plot


def drop_nan(
        *arrays: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drops rows with NaN values.

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

    # Raise an error if all rows are dropped
    if not mask.any():
        raise ValueError(
            "All rows contain NaN values. Have you tried interpolating using interpolate_X_and_bands?")

    return tuple(array[mask] for array in arrays)


def get_similarity_matrix(
        X: np.ndarray,
        band_names: List[str] = None,
        method: str = "pearson",
) -> pd.DataFrame:
    """Calculates the similarity matrix for the data.

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
    """Displays the similarity matrix.

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
    """Displays a dendrogram according to the similarity matrix.

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
