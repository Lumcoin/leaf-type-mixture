"""Loads, manipulates, and saves multi-band raster data.

Typical usage example:

    from ltm.features import load_raster, interpolate_data, save_raster

    data_path = "data.tif"
    target_path = "target.tif"

    data = load_raster(data_path)
    target = load_raster(target_path)

    data = interpolate_data(data)

    save_raster(data, data_path, "data_interpolated.tif")
"""

import os
from collections import defaultdict
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from tqdm.notebook import tqdm
from typeguard import typechecked

from slc.data import split_band_name


@typechecked
def genera2target(
    genera_areas: pd.DataFrame,
    *,
    regression: bool = False,
    evergreen_larix: bool = False,
) -> pd.Series | pd.DataFrame:
    """Converts a DataFrame containing base area per genera to a Series or DataFrame containing the target.

    Args:
        genera_areas:
            A pandas DataFrame containing the base area per genera. The column names are used as genera names.
        regression:
            A boolean representing whether to use regression instead of classification. Defaults to False.
        evergreen_larix:
            A boolean representing whether to classify Larix as evergreen. Only relevant for comparing targets with Dominant Leaf Type 2018. Defaults to False.

    Returns:
        A pandas Series or DataFrame containing the target. If regression is True, the target is a DataFrame with columns 'evergreen' and 'deciduous' containing the respective base areas. If regression is False, the target is a Series with values 'evergreen' and 'deciduous'.

    """
    evergreen = {
        "Abies": True,
        "Acer": False,
        "Aesculus": False,
        "Alnus": False,
        "Betula": False,
        "Carpinus": False,
        "Crataegus": False,
        "Cornus": False,
        "Corylus": False,
        "Euonymus": False,
        "Fagus": False,
        "Fraxinus": False,
        "Juglans": False,
        "Larix": evergreen_larix,  # Larch, deciduous conifer
        "Picea": True,
        "Pinus": True,
        "Populus": False,
        "Prunus": False,
        "Pseudotsuga": True,
        "Pyrus": False,
        "Rhamnus": False,
        "Quercus": False,
        "Sambucus": False,
        "Sorbus": False,
        "Taxus": True,
        "Tilia": False,
        "Thuja": True,
        "Ulmus": False,
        "Unidentified broadleaf": False,  # Assumed for all unknown broadleafs
        "Unidentified conifer": True,  # Assumed for all unknown conifers
    }

    target = genera_areas.copy()
    do_mask = target.isna().all(axis=1)
    target.fillna(0, inplace=True)

    evergreen_columns = [col for col in target.columns if evergreen[col]]
    evergreen_area = target[evergreen_columns].sum(axis=1)

    deciduous_columns = [col for col in target.columns if not evergreen[col]]
    deciduous_area = target[deciduous_columns].sum(axis=1)

    if regression:
        target["evergreen"] = evergreen_area
        target["deciduous"] = deciduous_area
        target[do_mask] = np.nan

        return target[["evergreen", "deciduous"]]

    target = (evergreen_area > deciduous_area).map(
        {True: "evergreen", False: "deciduous"}
    )
    target[do_mask] = np.nan

    return target


@typechecked
def load_dataset(
    data_path: str | PathLike, target_path: str | PathLike
) -> tuple[pd.DataFrame, pd.Series]:
    """Loads the dataset from the given paths.

    Args:
        data_path:
            A string or PathLike representing the file path to the data. The suffix '.tif' (GeoTIFF) is expected. If a directory is given, all .tif files in the directory are loaded.
        target_path:
            A string or PathLike representing the file path to the target. The suffix '.tif' (GeoTIFF) is expected. If a directory is given, all .tif files in the directory are loaded.

    Returns:
        A tuple of a pandas DataFrame containing the data and a pandas Series containing the target. The target is 1, representing evergreen, or 0, representing deciduous.

    """
    data_path = Path(data_path)
    target_path = Path(target_path)

    if data_path.is_dir() and target_path.is_dir():
        data_names = set([f.name for f in data_path.glob("*.tif")])
        target_names = set([f.name for f in target_path.glob("*.tif")])
        common_names = data_names.intersection(target_names)

        if not common_names:
            raise FileNotFoundError(
                "No common files found in data and target directories."
            )

        lonely_files = list(data_names.union(target_names).difference(common_names))
        if lonely_files:
            print(
                "Warning: All files in data and target directories without a corresponding file are discarded:",
                lonely_files,
            )

        data = []
        target = []
        for name in sorted(common_names):  # reproducibility
            data.append(load_raster(data_path / name))
            target.append(load_raster(target_path / name, monochrome_as_dataframe=True))

        data = pd.concat(data)
        target = pd.concat(target)

        data.reset_index(drop=True, inplace=True)
        target.reset_index(drop=True, inplace=True)
    elif (not data_path.is_dir()) and (not target_path.is_dir()):
        data = load_raster(data_path)
        target = load_raster(target_path, monochrome_as_dataframe=True)
    else:
        raise ValueError(
            "Expected data_path and target_path to be either directories or files, got a mix instead."
        )

    target = genera2target(target)

    mask = target.notna()
    data, target = data[mask], target[mask]
    data.reset_index(drop=True, inplace=True)
    target.reset_index(drop=True, inplace=True)

    target = target.map({"evergreen": 1, "deciduous": 0})

    return data, target


@typechecked
def np2pd_like(
    np_obj: np.ndarray,
    like_pd_obj: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Converts a numpy array to a pandas Series or DataFrame with the same column names or series name as the given pandas object.

    Args:
        np_obj:
            A numpy array of shape=(rows,) for a Series or shape=(rows, columns) for a DataFrame.
        like_pd_obj:
            A pandas Series or DataFrame.

    Returns:
        A pandas Series or DataFrame with the same column names or series name as the given pandas object.

    """
    if isinstance(like_pd_obj, pd.Series):
        if np_obj.ndim != 1:
            raise ValueError(
                f"Expected 1-dimensional numpy array, got {np_obj.ndim}-dimensional array instead."
            )
        return pd.Series(np_obj, name=like_pd_obj.name)

    if isinstance(like_pd_obj, pd.DataFrame):
        if np_obj.ndim != 2:
            raise ValueError(
                f"Expected 2-dimensional numpy array, got {np_obj.ndim}-dimensional array instead."
            )
        if np_obj.shape[1] != len(like_pd_obj.columns):
            raise ValueError(
                f"Expected numpy array with {len(like_pd_obj.columns)} columns, got {np_obj.shape[1]} columns instead."
            )

    return pd.DataFrame(np_obj, columns=like_pd_obj.columns)


@typechecked
def to_float32(data: pd.DataFrame) -> pd.DataFrame:
    """Converts the data to float32 and replaces infinities with the maximum and minimum float32 values.

    Args:
        data:
            A DataFrame containing the data.

    Returns:
        A DataFrame containing the data as float32.

    """
    # Check if dtype is numeric
    dtypes = data.dtypes
    illegal_dtypes = [
        str(dtype) for dtype in dtypes if not np.issubdtype(dtype, np.number)
    ]
    if illegal_dtypes:
        raise ValueError(
            f"Expected numeric data, found {', '.join(illegal_dtypes)} instead."
        )

    data = data.copy()

    float32_max = np.finfo(np.float32).max
    float32_min = np.finfo(np.float32).min

    data[data > float32_max] = float32_max
    data[data < float32_min] = float32_min
    data = data.astype(np.float32)

    return data


@typechecked
def interpolate_data(
    data: pd.DataFrame,
    method: str = "linear",
    order: int | None = None,
    *,
    cyclic: bool = True,
) -> pd.DataFrame:
    """Interpolate missing time series values in data using the given method.

    Args:
        data:
            A pd.DataFrame containing the data as values and band names as column names.
        method:
            A string representing the method to use for interpolation. Methods 'polynomial' and 'spline' require an integer 'order' as additional argument. Defaults to 'linear'. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
        order:
            An integer representing the order of the polynomial or spline interpolation. Defaults to None.
        cyclic:
            A boolean representing whether the data is cyclic. If so, the interpolation of values at the start will use values from the end of the time series and vice versa. Defaults to True.

    Returns:
        A dataframe containing the interpolated data.

    """
    # Return if no NaN values found
    if not data.isna().values.any():
        return data

    # Separate band names and values
    band_names = list(data.columns)
    values = data.values

    # Get the number of composites and number of bands
    num_composites = split_band_name(band_names[-1])[0]
    num_bands = len(band_names) // num_composites

    # Reshape into DataFrame with one row per composite
    reshaped = values.reshape(-1, num_composites, num_bands)
    reshaped = reshaped.transpose(0, 2, 1)
    reshaped = reshaped.reshape(-1, num_composites).T
    df = pd.DataFrame(reshaped)

    # Interpolate
    if cyclic:
        df = pd.concat([df] * 3, ignore_index=True)
    df.interpolate(method=method, inplace=True, order=order)
    if cyclic:
        start = len(df) // 3
        end = 2 * len(df) // 3
        df = df.iloc[start:end].reset_index(drop=True)

    # Reshape back into original shape
    interpolated_data = df.values.T.reshape(-1, num_bands, num_composites)
    interpolated_data = interpolated_data.transpose(0, 2, 1)
    interpolated_data = interpolated_data.reshape(-1, num_bands * num_composites)

    interpolated_data = pd.DataFrame(interpolated_data, columns=band_names)

    return interpolated_data


@typechecked
def load_raster(
    raster_path: str | PathLike,
    *,
    monochrome_as_dataframe: bool = False,
) -> pd.DataFrame | pd.Series:
    """Loads a raster and returns the data ready to use with sklearn.

    Args:
        raster_path:
            A string or PathLike representing the file path to the raster. The suffix '.tif' (GeoTIFF) is expected.
        monochrome_as_dataframe:
            A boolean representing whether to return a monochrome raster as a pandas DataFrame instead of a Series. Defaults to False.

    Returns:
        A DataFrame containing the data with band names as column names. If the raster is monochrome and 'monochrome_as_dataframe' is False, a Series is returned instead. This is expected by sklearn.

    """
    raster_path = str(raster_path)
    # Check if all paths are valid
    if not raster_path.endswith(".tif"):
        raise ValueError(f"Expected path to .tif file, got '{raster_path}' instead.")
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Could not find file '{raster_path}'.")

    with rasterio.open(raster_path) as src:
        raster = src.read()
        band_names = [
            band_name if isinstance(band_name, str) else str(i)
            for i, band_name in enumerate(src.descriptions)
        ]
        band_count = src.count
        values = raster.transpose(1, 2, 0).reshape(-1, band_count)

    if monochrome_as_dataframe or band_count > 1:
        return pd.DataFrame(values, columns=band_names)

    return pd.Series(values[:, 0], name=band_names[0])


@typechecked
def save_raster(
    data: pd.DataFrame,
    source_path: str | PathLike,
    destination_path: str | PathLike,
) -> None:
    """Saves the data as a raster image.

    Args:
        data:
            A pandas DataFrame containing the data. The column names are used as band names.
        source_path:
            A string or PathLike of the file path to the source image. Used for copying the raster profile.
        destination_path:
            A string or PathLike of the file path to the destination image.

    """
    # Copy data
    data_values = data.values.copy()

    # Read raster profile and shape
    with rasterio.open(source_path) as raster:
        profile = dict(raster.profile)
        profile["count"] = len(data.columns)
        shape = raster.read().shape

    # Reshape data
    data_values = data_values.reshape(shape[1], shape[2], len(data.columns)).transpose(
        2, 0, 1
    )

    # Write raster
    with rasterio.open(destination_path, "w", **profile) as dst:
        dst.write(data_values)
        dst.descriptions = list(data)


@typechecked
def get_similarity_matrix(
    data: pd.DataFrame,
    method: str = "pearson",
    seed: int | None = None,
) -> pd.DataFrame:
    """Calculates the similarity matrix for the data.

    Args:
        data:
            A pandas DataFrame containing the data. The column names are used as band names.
        method:
            A string representing the method to use for calculating the similarity matrix. Must be either 'pearson', 'spearman', or 'mutual_info'. Defaults to 'pearson'.
        seed:
            An optional integer representing the seed for mutual information. Defaults to None.

    Returns:
        A numpy array containing the similarity matrix. It is symmetrical, has a diagonal of ones and values from 0 to 1.

    """
    # Check if all column names are unique
    if len(set(data.columns)) != len(data.columns):
        raise ValueError("All column names must be unique.")

    # Check if method is valid
    valid_methods = ["pearson", "spearman", "mutual_info"]
    if method not in valid_methods:
        raise ValueError(
            f"Method must be one of {', '.join(valid_methods)}. Got '{method}' instead."
        )

    # Calculate similarity matrix
    data_values = data.values
    if method == "pearson":
        similarity_matrix = np.corrcoef(data_values, rowvar=False)
    elif method == "spearman":
        similarity_matrix = spearmanr(data_values).correlation
    elif method == "mutual_info":
        # EXPERIMENTAL, most likely scientifically wrong
        n_neighbors = min(3, data_values.shape[0] - 1)
        similarity_matrix = np.full(
            (data_values.shape[1], data_values.shape[1]), np.nan
        )
        for i, band_1 in tqdm(enumerate(data_values.T)):
            for j, band_2 in enumerate(data_values.T):
                similarity_matrix[i, j] = mutual_info_regression(
                    band_1.reshape(-1, 1),
                    band_2,
                    n_neighbors=n_neighbors,
                    random_state=seed,
                )[0]
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
            "Could not compute similarity matrix... This commonly occurs if a band has deviation of zero"
        )

    # Ensure the similarity matrix is normalized, symmetric, with diagonal of ones
    similarity_matrix = abs(similarity_matrix)
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    similarity_matrix /= np.nanmax(similarity_matrix)
    similarity_matrix[np.isnan(similarity_matrix)] = 1  # in case xD
    np.fill_diagonal(similarity_matrix, 1)

    # Convert to pandas DataFrame
    similarity_matrix = pd.DataFrame(
        similarity_matrix, columns=data.columns, index=data.columns
    )

    return similarity_matrix


@typechecked
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
            f"Expected pandas DataFrame, got {type(similarity_matrix)} instead."
        )

    # Check if similarity matrix is square
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Similarity matrix must be square.")

    # Show similarity matrix
    fig, ax = plt.subplots(
        figsize=(
            similarity_matrix.columns.shape[0] * 0.3,
            similarity_matrix.columns.shape[0] * 0.3,
        )
    )
    image = ax.imshow(similarity_matrix, interpolation="nearest", vmin=0, vmax=1)
    fig.colorbar(image)
    ax.set_xticks(range(similarity_matrix.columns.shape[0]))
    ax.set_yticks(range(similarity_matrix.columns.shape[0]))
    ax.set_xticklabels(similarity_matrix.columns, rotation="vertical")
    ax.set_yticklabels(similarity_matrix.columns)

    return ax


@typechecked
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
            f"Expected pandas DataFrame, got {type(similarity_matrix)} instead."
        )

    # Check if similarity matrix is square
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Similarity matrix must be square.")

    # Show dendrogram
    _, ax = plt.subplots(figsize=(16, 16))
    distance_matrix = 1 - similarity_matrix
    dist_linkage = ward(squareform(distance_matrix))
    _ = dendrogram(
        dist_linkage, labels=similarity_matrix.columns, ax=ax, leaf_rotation=90
    )

    return ax


@typechecked
def dendrogram_dim_red(
    data: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Performs dimensionality reduction using the threshold of the dendrogram.

    Implements approach described in https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#handling-multicollinear-features

    Args:
        data:
            A pandas DataFrame containing the data. The column names are used as band names.
        similarity_matrix:
            A pandas DataFrame containing the similarity matrix.
        threshold:
            A float representing the threshold for the dendrogram. Choose the value by inspecting the dendrogram plotted with show_dendrogram().

    Returns:
        A tuple of a numpy array containing the data and a list of strings representing the band names.

    """
    # Check if similarity matrix is square
    index = list(similarity_matrix.index)
    columns = list(similarity_matrix.columns)
    band_names = list(data.columns)
    if index != band_names or columns != band_names:
        raise ValueError(
            "Similarity matrix must be square with column names of data as index and columns."
        )

    # Compute linkage
    distance_matrix = 1 - similarity_matrix
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    # Compute clusters
    cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
    cluster_id_to_band_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_band_ids[cluster_id].append(idx)
    selected_bands = [v[0] for v in cluster_id_to_band_ids.values()]
    selected_band_names = list(np.array(band_names)[selected_bands])

    selected_data = data[selected_band_names]

    return selected_data
