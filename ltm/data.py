"""Functions for processing satellite data for leaf type mixture analysis.

Google Earth Engine is used to retrieve Sentinel 2 satellite images and compute
composites. The composites are saved as GeoTIFFs. The composites can be used as
input data for machine learning models. The labels are computed from plot data
on the individual tree level and saved as GeoTIFFs. The labels can be used as
target data for machine learning models.

Typical usage example:

    from ltm.data import compute_label, sentinel_composite
    import pandas as pd

    plot = pd.read_csv("plot.csv")

    y_path = "y.tif"
    X_path = "X.tif"

    compute_label(
        y_path=y_path,
        plot=plot,
    )

    sentinel_composite(
        y_path_from=y_path,
        X_path_to=X_path,
        time_window=("2020-01-01", "2020-12-31"),
    )
"""

import datetime
import json
import math
from itertools import product
from pathlib import Path
from typing import List, Set, Tuple

import ee
import eemont
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import utm
from pyproj import CRS
from rasterio.io import MemoryFile
from shapely import MultiPolygon, to_geojson
from tqdm import tqdm
from typeguard import typechecked

BROADLEAF_AREA = "Broadleaf Area"
CONIFER_AREA = "Conifer Area"
CONIFER_PROPORTION = "Conifer Proportion"

level_2a_bands = None
level_1c_bands = None
index_bands = None


@typechecked
def _initialize_ee() -> None:
    try:
        getattr(ee.Reducer, "mean")
    except AttributeError:
        print("Initializing Earth Engine API...")
        ee.Initialize()


@typechecked
def _sentinel_crs(
    latitude: float,
    longitude: float,
) -> str:
    zone_number = utm.latlon_to_zone_number(latitude, longitude)
    is_south = utm.latitude_to_zone_letter(latitude) < "N"

    crs = CRS.from_dict({"proj": "utm", "zone": zone_number, "south": is_south})
    crs = f"EPSG:{crs.to_epsg()}"

    return crs


@typechecked
def _split_time_window(
    time_window: Tuple[datetime.date, datetime.date],
    num_splits: int,
) -> List[Tuple[datetime.date, datetime.date]]:
    start = time_window[0]
    end = time_window[1]
    delta = (end - start + datetime.timedelta(days=1)) / num_splits

    if delta.days < 1:
        raise ValueError(
            f"Time window {time_window} is too small to split into {num_splits} sub windows"
        )

    sub_windows = []
    for i in range(num_splits):
        sub_window = (
            start + i * delta,
            start + (i + 1) * delta - datetime.timedelta(days=1),
        )
        sub_windows.append(sub_window)

    return sub_windows


@typechecked
def _get_roi_scale_crs(
    y_path: str,
) -> Tuple[ee.Geometry, float, str]:
    # Get region of interest (ROI), scale, and coordinate reference system (CRS)
    with rasterio.open(y_path) as src:
        crs = src.crs.to_string()
        res = src.res
        if res[0] != res[1]:
            raise ValueError("resolution is not square!")
        scale = res[0]
        bounds = src.bounds

    fc = ee.FeatureCollection(
        [
            ee.Geometry.Point([x, y], proj=crs)
            for x, y in product(bounds[::2], bounds[1::2])
        ]
    )
    roi = fc.geometry().bounds(0.01, proj=crs)

    return roi, scale, crs


@typechecked
def _split_reducer_band_name(
    band_name: str,
    reverse_and_space: bool = False,
) -> Tuple[str, str]:
    valid_bands = set(list_bands(level_2a=True))
    valid_bands = valid_bands.union(set(list_bands(level_2a=False)))
    valid_bands = valid_bands.union(set(list_indices()))

    if reverse_and_space:
        parts = band_name.split(" ")
        partial_band = parts[-1]
        band = partial_band if partial_band in valid_bands else ""
        for part in parts[-2::-1]:
            partial_band = f"{part}_{partial_band}"
            if partial_band in valid_bands:
                band = partial_band

        band = band.replace("_", " ")
        reducer = band_name[: -len(band) - 1]

        return band, reducer

    parts = band_name.split("_")
    partial_band = parts[0]
    band = partial_band if partial_band in valid_bands else ""
    for part in parts[1:]:
        partial_band += f"_{part}"
        if partial_band in valid_bands:
            band = partial_band

    reducer = band_name[len(band) + 1 :]

    return band, reducer


@typechecked
def _download_image(
    image: ee.Image,
) -> requests.Response:
    download_params = {
        "scale": image.projection().nominalScale().getInfo(),
        "crs": image.projection().getInfo()["crs"],
        "format": "GEO_TIFF",
    }
    url = image.getDownloadURL(download_params)

    return requests.get(url)


@typechecked
def _image_response2file(
    image: bytes,
    file_path: str,
    mask: bytes,
    bands: List[str],
) -> None:
    with MemoryFile(image) as memfile, MemoryFile(mask) as mask_memfile:
        with memfile.open() as dataset, mask_memfile.open() as mask_dataset:
            profile = dataset.profile
            profile["nodata"] = np.nan
            with rasterio.open(file_path, "w", **profile) as dst:
                raster = dataset.read()
                mask_raster = mask_dataset.read()
                raster[mask_raster == 0] = np.nan
                dst.write(raster)
                dst.descriptions = tuple(bands)


@typechecked
def _save_image(
    image: ee.Image,
    file_path: str,
) -> None:
    try:
        image = image.toDouble()
        image_response = _download_image(image)
        mask_response = _download_image(image.mask())
    except ee.ee_exception.EEException as exc:
        raise ValueError(
            "Failed to compute image. Please check the timewindow first, maybe it is too small. This error can occur if there is no data for the given timewindow."
        ) from exc

    if image_response.status_code == mask_response.status_code == 200:
        _image_response2file(
            image_response.content,
            file_path,
            mask=mask_response.content,
            bands=image.bandNames().getInfo(),
        )
        print(f"GeoTIFF saved as {file_path}")
    else:
        print(
            f"FAILED to either compute or download the image for {file_path} most likely due to computational limits of Google Earth Engine."
        )


@typechecked
def _mask_s2_clouds(
    image: ee.Image,
) -> ee.Image:
    """Masks clouds in a Sentinel-2 image using the QA band.

    From https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#colab-python

    Args:
        image (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: A cloud-masked Sentinel-2 image.
    """
    qa = image.select("QA60")

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    return image.updateMask(mask).divide(10000)


@typechecked
def _prettify_band_name(band_name: str) -> str:
    composite_idx, _, band_reducer = band_name.split("_", 2)
    band_label, reducer_label = _split_reducer_band_name(band_reducer)
    band_label = band_label.replace("_", " ")
    reducer_label = reducer_label.replace("_", " ")
    composite_idx = int(composite_idx) + 1
    reducer_label = "".join(
        [
            a if a.isupper() else b
            for a, b in zip(reducer_label, reducer_label.title())
        ]
    )
    pretty_name = f"{composite_idx} {reducer_label} {band_label}"

    return pretty_name


@typechecked
def _compute_area(
    fc: ee.FeatureCollection,
    scale: float,
    fine_scale: float,
    crs: str,
) -> ee.Image:
    # Compute area by masking a constant image and multiplying by scale**2
    fc_area = ee.Image.constant(1).clip(fc).mask()
    fc_area = ee.Image.constant(scale**2).multiply(fc_area)

    # Reduce to coarse resolution
    fc_area = fc_area.reproject(scale=fine_scale, crs=crs)
    fc_area = fc_area.reduceResolution(ee.Reducer.mean(), maxPixels=10_000)

    return fc_area


@typechecked
def _path_check(
    *paths: str,
    suffix: str,
    check_parent: bool = True,
    check_self: bool = False,
) -> None:
    for path in paths:
        pathlib = Path(path)
        if pathlib.suffix != suffix:
            raise ValueError(f"{path} must end with {suffix}")
        if check_parent and not pathlib.parent.exists():
            raise ValueError(f"{path} parent directory does not exist")
        if check_self and not pathlib.exists():
            raise ValueError(f"{path} does not exist")


@typechecked
def list_reducers(use_buffered_reducers: bool = True) -> List[str]:
    """Lists all valid reducers in the Earth Engine API.

    Args:
        use_buffered_reducers:
            A boolean indicating whether to use the buffered reducers. If False the Google Earth Engine API is used to retrieve all current reducers. Defaults to True.

    Returns:
        A list of strings representing the valid reducers.
    """
    if use_buffered_reducers:
        return [
            "And",
            "Or",
            "allNonZero",
            "anyNonZero",
            "circularMean",
            "circularStddev",
            "circularVariance",
            "count",
            "countDistinct",
            "countDistinctNonNull",
            "countRuns",
            "first",
            "firstNonNull",
            "kendallsCorrelation",
            "kurtosis",
            "last",
            "lastNonNull",
            "max",
            "mean",
            "median",
            "min",
            "minMax",
            "mode",
            "product",
            "sampleStdDev",
            "sampleVariance",
            "skew",
            "stdDev",
            "sum",
            "variance",
        ]

    # Check for cached reducers
    global reducers
    if reducers is not None:
        return reducers

    # Initialize Earth Engine API
    _initialize_ee()
    print("Checking for valid reducers...")

    # Get all valid reducers
    ic = ee.ImageCollection([ee.Image.constant(1).toDouble()])
    point = ee.Geometry.Point(0, 0)
    valid_reducers = []
    for attr in tqdm(dir(ee.Reducer)):
        try:
            reducer = getattr(ee.Reducer, attr)()
            if isinstance(reducer, ee.Reducer):
                image = ic.reduce(reducer)
                result = image.reduceRegion(ee.Reducer.first(), geometry=point)
                for value in result.getInfo().values():
                    if value is not None:
                        float(value)

                valid_reducers.append(attr)
        except (TypeError, ee.ee_exception.EEException):
            continue

    # Cache reducers
    reducers = valid_reducers

    return valid_reducers


@typechecked
def list_bands(level_2a: bool = True) -> List[str]:
    """Lists all valid bands in the Earth Engine API.

    Args:
        level_2a:
            A boolean indicating whether to list bands from Level-2A or Level-1C Sentinel 2 data. Level-2A if True. Defaults to True.

    Returns:
        A list of strings representing the valid bands.
    """
    # Initialize Earth Engine API
    _initialize_ee()

    # Check for cached bands
    global level_2a_bands, level_1c_bands
    if level_2a and level_2a_bands is not None:
        return level_2a_bands
    if not level_2a and level_1c_bands is not None:
        return level_1c_bands

    # Get all valid bands
    if level_2a:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    else:
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    bands = s2.first().bandNames().getInfo()

    # Cache bands
    if level_2a:
        level_2a_bands = bands
    else:
        level_1c_bands = bands

    return bands


@typechecked
def list_indices() -> List[str]:
    """Lists all valid indices for Sentinel-2.

    Returns:
        A list of strings representing the valid indices.
    """
    # Initialize Earth Engine API
    _initialize_ee()

    # Check for cached indices
    global index_bands
    if index_bands is not None:
        return index_bands

    # Get all valid indices
    indices = eemont.common.indices()
    s2_indices = [
        index
        for index in indices
        if "Sentinel-2" in eemont.common.indices()[index]["platforms"]
    ]

    # Remove NIRvP (does need bands not available in Sentinel-2)
    s2_indices.remove("NIRvP")

    # Cache indices
    index_bands = s2_indices

    return s2_indices


@typechecked
def combine_band_name(
    composite_idx: int,
    reducer: str,
    band_label: str,
) -> str:
    """Combines a composite index, reducer, and band label into a band name.

    Args:
        composite_idx:
            An integer for the composite index starting at 1.
        reducer:
            A string for the reducer, possibly containing space characters.
        band_label:
            A string for the band label.

    Returns:
        A string for the band name.
    """
    return f"{composite_idx} {reducer} {band_label}"


@typechecked
def split_band_name(
    band_name: str,
) -> Tuple[int, str, str]:
    """Splits a band name into its composite index, reducer, and band label.

    Args:
        A string for the band name.

    Returns:
        A tuple of the composite index starting at 1, reducer (possibly containing space characters), and band label.
    """
    composite_idx, band_name = band_name.split(maxsplit=1)

    band_label, reducer = _split_reducer_band_name(
        band_name, reverse_and_space=True
    )

    return int(composite_idx), reducer, band_label


@typechecked
def show_timeseries(
    raster_path: str,
    reducer: str,
    rgb_bands: List[str] | None = None,
) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    # Read raster and get band names
    with rasterio.open(raster_path) as src:
        raster = src.read()
        bands = src.descriptions
        bands_lower = [band.lower() for band in bands]

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

        for i, rgb_band in enumerate(rgb_bands):
            if rgb_band is not None:
                _, _, rgb_bands[i] = split_band_name(rgb_bands[i])

    # Get the rasters for the rgb bands
    num_composites, reducer_title, _ = split_band_name(bands[-1])
    rgb_rasters = []
    for i in range(1, num_composites + 1):
        rgb_raster = []
        for b in rgb_bands:
            if b is not None:
                raster_band = combine_band_name(i, reducer, b).lower()
                band_raster = raster[bands_lower.index(raster_band)]
            else:
                band_raster = np.zeros_like(raster[0])

            rgb_raster.append(band_raster)

        rgb_raster = np.stack(rgb_raster)
        rgb_raster = rgb_raster.transpose(1, 2, 0)
        rgb_rasters.append(rgb_raster)

    # Get min and max values for normalization
    min_value = min(np.nanmin(rgb_raster) for rgb_raster in rgb_rasters)
    max_value = max(np.nanmax(rgb_raster) for rgb_raster in rgb_rasters)

    # Plot the rasters below each other
    fig, axs = plt.subplots(nrows=num_composites, figsize=(10, 10))
    if num_composites == 1:
        axs = np.array([axs])
    fig.tight_layout()
    for i, (ax, rgb_raster) in enumerate(zip(axs, rgb_rasters), start=1):
        # normalize values and apply gamma correction
        rgb_raster -= min_value
        rgb_raster /= max_value - min_value
        rgb_raster **= 1 / 2.2
        rgb_raster[np.isnan(rgb_raster)] = 0
        ax.imshow(rgb_raster)
        ax.set_title(f"{i} {reducer_title}")
        ax.axis("off")

    # Show the plot
    plt.show()

    return fig, axs


@typechecked
def shape2mask(
    y_path: str,
    shape: MultiPolygon,
    crs: str,
) -> str:
    """Creates a raster mask from a shape.

    Args:
        y_path:
            A string representing the file path to save the raster mask. Cells inside the shape will be 1 and cells outside the shape will be 0. Border cells are proportional to the area inside the shape between 0 and 1.
        shape:
            A shapely MultiPolygon representing the shape to create the raster mask from.
        crs:
            A string representing the coordinate reference system (CRS) of the shape.

    Returns:
        The file path to the raster mask as a string.
    """
    # Initialize Earth Engine API
    _initialize_ee()
    print("Preparing mask...")

    # Ensure proper path format
    _path_check(y_path, check_parent=True, check_self=False, suffix=".tif")

    # Convert shape to bounds in output crs
    geo_series = gpd.GeoSeries(shape, crs=crs)
    geo_series = geo_series.to_crs("EPSG:4326")
    geojson = to_geojson(geo_series[0])
    coordinates = json.loads(geojson)["coordinates"]
    multipolygon = ee.Geometry.MultiPolygon(coordinates)

    # Get CRS in epsg format for center of the shape
    longitude, latitude = multipolygon.centroid(0.01).coordinates().getInfo()
    out_crs = _sentinel_crs(latitude, longitude)

    # Check if rectangle has reasonable size
    shape_area = multipolygon.area(0.01).getInfo()
    if shape_area == 0:
        raise ValueError(
            "Shape bounding box has area 0. Check if shape coordinates are valid."
        )
    if shape_area > 1e7:
        raise ValueError(
            "Shape bounding box has area > 1e7. Check if shape coordinates are valid."
        )

    # Define scale at 10 m/pixel, same as max Sentinel 2 resolution
    scale = 10

    # Clip to shape and reproject
    image = ee.Image.constant(1)
    image = image.clip(multipolygon).mask()
    image = image.reproject(scale=scale, crs=out_crs)

    # Save shape
    print("Computing mask...")
    _save_image(image, y_path)

    return y_path


@typechecked
def compute_label(
    y_path: str,
    plot: pd.DataFrame,
    area_as_y: bool = False,
) -> str:
    """Computes the label for a plot.

    Args:
        y_path:
            A string representing the file path to save the raster with values between 0 and 1 for the conifer proportion.
        plot:
            A pandas DataFrame containing data on a per tree basis with two columns for longitude and latitude, one column for DBH, and one for whether or not the tree is a broadleaf (1 is broadleaf, 0 is conifer). The column names must be 'longitude', 'latitude', 'dbh', and 'broadleaf' respectively. This function is case insensitive regarding column names.
        area_as_y:
            A boolean indicating whether to compute the area per leaf type instead of the leaf type mixture as labels. Results in a y with two bands, one for each leaf type. Defaults to False.

    Returns:
        A string representing the file path to the raster with values between 0 and 1 for the conifer proportion. 1 being fully conifer and 0 being fully broadleaf for a given raster cell.
    """
    # Initialize Earth Engine API
    _initialize_ee()
    print("Preparing labels...")

    # Ensure proper plot DataFrame format
    expected_dtypes = {
        "broadleaf": np.int8,
        "dbh": np.float64,
        "latitude": np.float64,
        "longitude": np.float64,
    }
    expected_columns = set(expected_dtypes.keys())
    plot = plot.rename(columns=str.lower)
    columns = set(plot.columns)
    if expected_columns != columns.intersection(expected_columns):
        raise ValueError("Columns do not match expected columns")
    plot = plot.astype(expected_dtypes)

    # Ensure proper path format
    y_pathlib = Path(y_path)
    if y_pathlib.suffix != ".tif":
        raise ValueError("y_path must be a string ending in .tif")
    if not y_pathlib.parent.exists():
        raise ValueError(
            f"y_path parent directory does not exist: {y_pathlib.parent}"
        )

    # Upload plot
    broadleafs = []
    conifers = []
    for _, row in plot.iterrows():
        circle = ee.Geometry.Point(
            [
                row["longitude"],
                row["latitude"],
            ]
        ).buffer(row["dbh"] / 2)

        if row["broadleaf"]:
            broadleafs.append(circle)
        else:
            conifers.append(circle)
    broadleafs = ee.FeatureCollection(broadleafs)
    conifers = ee.FeatureCollection(conifers)

    # Get region of interest (ROI)
    roi = broadleafs.merge(conifers).geometry()

    # Get CRS in epsg format for center of the roi
    longitude, latitude = roi.centroid(1).getInfo()["coordinates"]
    crs = _sentinel_crs(latitude, longitude)

    # Convert ROI to bounds in output crs
    roi = roi.bounds(0.01, crs)

    # Check if rectangle has reasonable size
    roi_area = roi.area(0.01).getInfo()
    if roi_area == 0:
        raise ValueError(
            "Plot bounding box has area 0. Check if plot coordinates are valid."
        )
    if roi_area > 1e7:
        raise ValueError(
            "Plot bounding box has area > 1e7. Check if plot coordinates are valid."
        )

    # Define scale at 10 m/pixel, same as max Sentinel 2 resolution
    scale = 10

    # Render plot as fine resolution image, then reduce to coarse resolution
    fine_scale = min(plot["dbh"].min() * 5, scale)
    if plot["dbh"].min() < 0.05:
        print(
            "Info: DBH < 0.05 m found. Google Earth Engine might ignore small trees."
        )
        fine_scale = 0.25

    # Compute broadleaf and conifer area
    broadleaf_area = _compute_area(broadleafs, scale, fine_scale, crs)
    conifer_area = _compute_area(conifers, scale, fine_scale, crs)

    # Compute y (conifer proportion) from broadleaf_area and conifer_area
    total_area = broadleaf_area.add(conifer_area)
    if area_as_y:
        y = broadleaf_area.addBands(conifer_area)
        y = y.updateMask(total_area.gt(0))  # Remove pixels with no trees
        y = y.rename([BROADLEAF_AREA, CONIFER_AREA])
    else:
        y = conifer_area.divide(total_area)
        y = y.updateMask(total_area.gt(0))  # Remove pixels with no trees
        y = y.rename(CONIFER_PROPORTION)

    # Clip to roi and reproject
    y = y.clip(roi)
    y = y.reproject(scale=scale, crs=crs)

    # Save y
    print("Computing labels...")
    _save_image(y, y_path)

    return y_path


@typechecked
def sentinel_composite(
    y_path_from: str,
    X_path_to: str,
    time_window: Tuple[datetime.date, datetime.date],
    num_composites: int = 1,
    temporal_reducers: List[str] | None = None,
    indices: List[str] | None = None,
    level_2a: bool = True,
    sentinel_bands: List[str] | None = None,
    remove_clouds: bool = True,
) -> str:
    """Creates a composite from many Sentinel 2 satellite images for a given
    label image.

    The raster will be saved to 'X_path_to' after processing. The processing itself can take several minutes, depending on Google Earth Engine and the size of your region of interest. If you hit some limit of Google Earth Engine, the function will raise an error.

    Args:
        y_path_from:
            A string representing the file path to the label raster. This is used to derive the bounds, coordinate reference system and resolution/pixel size of the image.
        X_path_to:
            A string representing the output file path to save the composite raster.
        time_window:
            A tuple of two dates representing the start and end of the time window in which the satellite images are retrieved. The dates are converted to milliseconds. The end date is technically exclusive, but only by one millisecond.
        num_composites:
            An integer representing the number of composites to create. Defaults to 1.
        temporal_reducers:
            A list of strings representing the temporal reducers to use when creating the composite. Run data.list_reducers() or see https://developers.google.com/earth-engine/guides/reducers_intro for more information. Defaults to ['mean'] if None.
        indices:
            A list of strings representing the spectral indices to add to the composite as additional bands. Run data.list_indices() or see https://eemont.readthedocs.io/en/latest/guide/spectralIndices.html for more information.
        level_2a:
            A boolean indicating whether to use Level-2A or Level-1C Sentinel 2 data. Defaults to True.
        sentinel_bands:
            A list of strings representing the bands to use from the Sentinel 2 data. For available bands run data.list_bands() or see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED#bands (Level-1C) and https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands (Level-2A). All bands are used if sentinel_bands is None. Defaults to None.
        remove_clouds:
            A boolean indicating whether to remove clouds from the satellite images based on the QA60 band. Defaults to True.

    Returns:
        A string representing the file path to the composite raster.
    """
    # Initialize Earth Engine API
    _initialize_ee()
    print("Preparing Sentinel-2 data...")

    # Ensure proper time windows
    if round(time_window[0].timestamp() * 1000) >= round(
        time_window[1].timestamp() * 1000
    ):
        raise ValueError(
            f"start ({time_window[0]}) must be before the end ({time_window[1]}) of timewindow"
        )
    if level_2a and time_window[0] < datetime.datetime(2017, 3, 28):
        if time_window[0] >= datetime.datetime(2015, 6, 27):
            raise ValueError(
                "Level-2A data is not available before 2017-03-28. Use Level-1C data instead."
            )
        raise ValueError("Level-2A data is not available before 2017-03-28.")
    if not level_2a and time_window[0] < datetime.datetime(2015, 6, 27):
        raise ValueError("Level-1C data is not available before 2015-06-27.")

    # Ensure proper path format
    _path_check(y_path_from, X_path_to, suffix=".tif")

    # Check if indices are valid eemont indices
    if indices is not None:
        valid_indices = list_indices()
        invalid_indices = [
            index for index in indices if index not in valid_indices
        ]
        if invalid_indices:
            raise ValueError(
                f"Invalid indices not in eemont package: {', '.join(invalid_indices)}"
            )

    # Split time window into sub windows and convert to strings
    time_windows = _split_time_window(time_window, num_composites)
    time_windows = [
        (round(start.timestamp() * 1000), round(end.timestamp() * 1000))
        for start, end in time_windows
    ]

    # Use ee.Reducer.mean() if temporal_reducers is None
    if temporal_reducers is None:
        temporal_reducers = ["mean"]
    if len(set(temporal_reducers)) < len(temporal_reducers):
        raise ValueError(
            "temporal_reducers must not contain duplicate reducers"
        )

    # Check if all reducers are valid
    valid_reducers = list_reducers()
    invalid_reducers = [
        reducer
        for reducer in temporal_reducers
        if reducer not in valid_reducers
    ]
    if invalid_reducers:
        raise ValueError(
            f"Invalid reducers not in ee.Reducer: {', '.join(invalid_reducers)}"
        )

    # Check if all bands are valid. Use all bands if sentinel_bands is None
    valid_bands = list_bands(level_2a)
    if sentinel_bands is None:
        sentinel_bands = valid_bands
    illegal_bands = [b for b in sentinel_bands if b not in valid_bands]
    if any(illegal_bands):
        raise ValueError(f"{illegal_bands} not available in: {valid_bands}")

    # Combine sentinel_bands and indices
    bands = sentinel_bands.copy()
    if indices is not None:
        bands += indices

    # Check if the limit of 5000 bands is exceeded
    if len(bands) * len(temporal_reducers) > 5000:
        raise ValueError(
            f"You exceed the 5000 bands max limit of GEE: {len(bands) * len(temporal_reducers)} bands"
        )

    # Get region of interest (ROI), scale, and coordinate reference system (CRS)
    roi, scale, crs = _get_roi_scale_crs(y_path_from)

    # Get Sentinel 2 image collection filtered by bounds
    if level_2a:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    else:
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    s2 = s2.filterBounds(roi)

    # Loop through time windows
    data = []
    for start, end in time_windows:
        # Filter by roi and timewindow
        s2_window = s2.filterDate(start, end)

        # Remove clouds
        if remove_clouds:
            s2_window = s2_window.map(_mask_s2_clouds)

        # Add indices before possibly removing bands necessary for computing indices
        if indices:
            s2_window = s2_window.spectralIndices(indices)

        # Select bands
        s2_window = s2_window.select(bands)

        # Reduce by temporal_reducers
        reduced_images = []
        for temporal_reducer in temporal_reducers:
            reducer = getattr(ee.Reducer, temporal_reducer)()
            reduced_image = s2_window.reduce(reducer)

            band_names = reduced_image.bandNames().getInfo()
            if len(band_names) == len(bands):
                band_names = [
                    f"{_split_reducer_band_name(band_name)[0]}_{temporal_reducer}"
                    for band_name in reduced_image.bandNames().getInfo()
                ]
            else:
                # Rename bands and remove temporal_reducer from band name
                new_band_names = []
                for band_name in band_names:
                    band, reducer_label = _split_reducer_band_name(band_name)
                    new_band_name = f"{band}_{temporal_reducer}_{reducer_label}"
                    new_band_names.append(new_band_name)
                band_names = new_band_names

            reduced_image = reduced_image.rename(band_names)

            reduced_images.append(reduced_image)

        # Combine reduced_images into one image
        datum = ee.ImageCollection(reduced_images).toBands()

        # Add to data
        data.append(datum)

    # Stack images
    data = ee.ImageCollection(data).toBands()
    data = data.clip(roi)
    data = data.reproject(scale=scale, crs=crs)

    # Prettify band names
    pretty_names = []
    for band_name in data.bandNames().getInfo():
        pretty_name = _prettify_band_name(band_name)
        pretty_names.append(pretty_name)
    data = data.rename(pretty_names)

    # Save data (X)
    print("Computing data...")
    _save_image(data, X_path_to)

    return X_path_to


@typechecked
def palsar_raster(
    y_path_from: str,
    X_path_to: str,
    timestamp: datetime.date,
    sin_cos_angle: bool = True,
    exclude_future: bool = False,
) -> str:
    """Creates a raster containing PALSAR data for a given label image.

    The raster will be saved to 'X_path_to' after processing. The processing itself can take several minutes, depending on Google Earth Engine and the size of your region of interest. The "Global PALSAR-2/PALSAR Yearly Mosaic, version 2" dataset is used for the raster. It is available at https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR_YEARLY_SAR_EPOCH and covers the years 2015 to 2022.

    Args:
        y_path_from:
            A string representing the file path to the label raster. This is used to derive the bounds and coordinate reference system.
        X_path_to:
            A string representing the output file path to save the raster.
        timestamp:
            A date representing the timestamp of the raster. As the images are sparse, the closest image to the timestamp is used.
        sin_cos_angle:
            A boolean indicating whether to include sine and cosine of the angle as bands in the raster instead of the angle. This prevents discontinuity between 359° and 0°. Defaults to True.
        exclude_future:
            A boolean indicating whether to exclude future images past the 'timestamp'. This would make sense for forests that are drastically changed by an event like a storm. Defaults to False.

    Returns:
        A string representing the file path to the raster.
    """
    # Initialize Earth Engine API
    _initialize_ee()
    print("Preparing PALSAR data...")

    # Ensure proper path format
    _path_check(y_path_from, X_path_to, suffix=".tif")

    # Convert timestamp to milliseconds
    date_format = "%Y-%m-%d"
    date = timestamp.strftime(date_format)
    milliseconds = ee.Date(date).millis().getInfo()

    # Get region of interest (ROI), scale, and coordinate reference system (CRS)
    roi, scale, crs = _get_roi_scale_crs(y_path_from)

    # Get PALSAR image collection
    palsar = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
    if exclude_future:
        palsar = palsar.filterDate(0, milliseconds + 1)

    # Compute time difference between timestamp and image timestamps
    def add_time_delta(image):
        # Compute the time difference in milliseconds
        image_timestamp = ee.Number(image.get("system:time_start"))
        target_timestamp = ee.Number(milliseconds)

        time_delta = image_timestamp.subtract(target_timestamp).abs()

        # Add the time delta as a new property to the image
        return image.set("time_delta", time_delta)

    # Sort the image collection by time delta and create a single image
    palsar = palsar.map(add_time_delta)
    palsar = palsar.sort(
        "time_delta", False
    )  # False is necessary, as mosaic prioritizes last to first
    palsar = palsar.mosaic()

    # Compute sine and cosine of the angle
    sin_band = (
        palsar.select("angle").multiply(math.pi / 180).sin().rename("sin_angle")
    )
    cos_band = (
        palsar.select("angle").multiply(math.pi / 180).cos().rename("cos_angle")
    )
    palsar = palsar.addBands([sin_band, cos_band])

    remaining_bands = ["HH", "HV"]
    if sin_cos_angle:
        remaining_bands += ["sin_angle", "cos_angle"]
    else:
        remaining_bands += ["angle"]
    palsar = palsar.select(remaining_bands)

    # Resample
    palsar = palsar.clip(roi)
    palsar = palsar.reproject(scale=scale, crs=crs)

    # Save data (X)
    print("Computing data...")
    _save_image(palsar, X_path_to)

    return X_path_to
