"""Functions for processing satellite data for leaf type mixture analysis.

Google Earth Engine is used to retrieve Sentinel 2 satellite images and compute
composites. The composites are saved as GeoTIFFs. The composites can be used as
input data for machine learning models. The labels are computed from plot data
on the individual tree level and saved as GeoTIFFs. The labels can be used as
target data for machine learning models.

Typical usage example:

    from ltm.data import sentinel_composite
    import pandas as pd

    plot = pd.read_csv("plot.csv")

    X_path, y_path = sentinel_composite(
        plot=plot,
        time_window=("2020-01-01", "2020-12-31"),
    )
"""

import datetime
from itertools import product
from pathlib import Path
from typing import List, Tuple

import ee
import eemont
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import utm
from pyproj import CRS
from rasterio.io import MemoryFile
from typeguard import typechecked


@typechecked
def _split_time_window(
    time_window: Tuple[str, str],
    num_splits: int,
) -> List[Tuple[str, str]]:
    start = datetime.datetime.strptime(time_window[0], "%Y-%m-%d")
    end = datetime.datetime.strptime(time_window[1], "%Y-%m-%d")
    delta = (end - start + datetime.timedelta(days=1)) / num_splits

    if delta.days < 1:
        raise ValueError(
            f"Time window {time_window} is too small to split into {num_splits} sub windows"
        )

    sub_windows = []
    for i in range(num_splits):
        sub_window = (
            (start + i * delta).strftime("%Y-%m-%d"),
            (start + (i + 1) * delta - datetime.timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
        )
        sub_windows.append(sub_window)

    return sub_windows


@typechecked
def _download_image(
    image: ee.Image,
    scale: float,
    crs: str,
) -> requests.Response:
    download_params = {
        "scale": scale,
        "crs": crs,
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
    scale: float,
    crs: str,
) -> None:
    try:
        image_response = _download_image(image, scale, crs)
        mask_response = _download_image(image.mask(), scale, crs)
    except ee.ee_exception.EEException as exc:
        raise ValueError(
            "FAILED to compute image. Most likely because the timewindow is too small or outside the availability, see 'Dataset Availability' in the Earth Engine Data Catalog."
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
            f"FAILED to either compute or download the image for {file_path} most likely due to limits of Google Earth Engine."
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
def _prettify_band_name(
    band_name: str,
) -> str:
    composite_idx, _, *band_label, reducer_label = band_name.split("_")
    composite_idx = int(composite_idx) + 1
    pretty_name = (
        f"{composite_idx} {reducer_label.title()} {'_'.join(band_label)}"
    )

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
            A string for the reducer.
        band_label:
            A string for the band label.

    Returns:
        A string for the band name.
    """
    return f"{composite_idx} {reducer.title()} {band_label}"


@typechecked
def split_band_name(
    band_name: str,
) -> Tuple[int, str, str]:
    """Splits a band name into its composite index, reducer, and band label.

    Args:
        A string for the band name.

    Returns:
        A tuple of the composite index starting at 1, reducer, and band label.
    """
    composite_idx, reducer, band_label = band_name.split(maxsplit=2)

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
    num_composites, _, _ = split_band_name(bands[-1])
    rgb_rasters = []
    for i in range(1, num_composites + 1):
        rgb_raster = []
        for b in rgb_bands:
            if b is not None:
                raster_band = combine_band_name(i, reducer, b)
                band_raster = raster[bands.index(raster_band)]
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
    fig, axs = plt.subplots(nrows=6, figsize=(10, 10))
    fig.tight_layout()
    for i, (ax, rgb_raster) in enumerate(zip(axs, rgb_rasters), start=1):
        # normalize values and apply gamma correction
        rgb_raster -= min_value
        rgb_raster /= max_value - min_value
        rgb_raster **= 1 / 2.2
        rgb_raster[np.isnan(rgb_raster)] = 0
        ax.imshow(rgb_raster)
        ax.set_title(f"{i} {reducer}")
        ax.axis("off")

    # Show the plot
    plt.show()

    return fig, axs


@typechecked
def compute_label(
    plot: pd.DataFrame,
    y_path: str = "../data/processed/y.tif",
    areas_as_y: bool = False,
) -> str:
    """Computes the label for a plot.

    Args:
        plot:
            A pandas DataFrame containing data on a per tree basis with two columns for longitude and latitude, one column for DBH, and one for whether or not the tree is a broadleaf (1 is broadleaf, 0 is conifer). The column names must be 'longitude', 'latitude', 'dbh', and 'broadleaf' respectively. This function is case insensitive regarding column names.
        y_path:
            A string representing the file path to save the raster with values between 0 and 1 for the leaf type mixture. Will not be saved if None. Resulting in a speedup.
        areas_as_y:
            A boolean indicating whether to compute the area per leaf type instead of the leaf type mixture as labels. Results in a y with two bands, one for each leaf type. Defaults to False.

    Returns:
        A string representing the file path to the raster with values between 0 and 1 for the leaf type mixture. 1 being broadleaf and 0 being conifer.
    """
    # Initialize Earth Engine API
    if not ee.data._credentials:
        print("Initializing Earth Engine API...")
        ee.Initialize()
    print("Preparing labels...")

    # Ensure proper plot DataFrame format
    expected_dtypes = {
        "longitude": np.float64,
        "latitude": np.float64,
        "dbh": np.float64,
        "broadleaf": np.int8,
    }
    plot = plot.rename(columns=str.lower)
    if set(expected_dtypes.keys()) != set(plot.columns):
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
    zone_number = utm.latlon_to_zone_number(latitude, longitude)
    is_south = utm.latitude_to_zone_letter(latitude) < "N"

    crs = CRS.from_dict({"proj": "utm", "zone": zone_number, "south": is_south})
    crs = crs.to_string()

    # Convert ROI to bounds in output crs
    roi = roi.bounds(0.01, crs)

    # Check if rectangle has reasonable size
    if roi.area(0.01).getInfo() == 0:
        raise ValueError(
            "Plot bounding box has area 0. Check if plot coordinates are valid."
        )
    if roi.area(0.01).getInfo() > 1e7:
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

    # Compute y (leaf type mixture) from broadleaf_area and conifer_area
    total_area = broadleaf_area.add(conifer_area)
    if areas_as_y:
        y = broadleaf_area.addBands(conifer_area)
        y = y.updateMask(total_area.gt(0))  # Remove pixels with no trees
        y = y.rename(["Broadleaf Area", "Conifer Area"])
    else:
        y = broadleaf_area.divide(total_area)
        y = y.updateMask(total_area.gt(0))  # Remove pixels with no trees
        y = y.rename("Leaf Type Mixture")

    # Clip to roi and reproject
    y = y.clip(roi)
    y = y.reproject(scale=scale, crs=crs)

    # Save y
    print("Computing labels...")
    _save_image(y, y_path, scale=scale, crs=crs)

    return y_path


@typechecked
def compute_data(
    time_window: Tuple[datetime.date, datetime.date] | Tuple[str, str],
    y_path,
    X_path: str = "../data/processed/X.tif",
    num_composites: int = 1,
    temporal_reducers: List[str] | None = None,
    indices: List[str] | None = None,
    level_2a: bool = False,
    sentinel_bands: List[str] | None = None,
    remove_clouds: bool = True,
    remove_qa: bool = True,
) -> str:
    """Creates a composite from many Sentinel 2 satellite images for a given label image.

    Args:
        time_window:
            A tuple of two dates or strings representing the start and end of the time window to retrieve the satellite images.
        y_path:
            A string representing the file path to the label raster. This is used to derive the bounds and coordinate reference system.
        X_path:
            A string representing the output file path to save the composite raster.
        num_composites:
            An integer representing the number of composites to create. Defaults to 1.
        temporal_reducers:
            A list of strings representing the temporal reducers to use when creating the composite. See https://developers.google.com/earth-engine/guides/reducers_intro for more information. Defaults to ['mean'] if None.
        indices:
            A list of strings representing the spectral indices to add to the composite as additional bands. See https://eemont.readthedocs.io/en/latest/guide/spectralIndices.html for more information.
        level_2a:
            A boolean indicating whether to use Level-2A or Level-1C Sentinel 2 data.
        sentinel_bands:
            A list of strings representing the bands to use from the Sentinel 2 data EXCLUDING all QA bands. To include QA bands set remove_qa to False. For available bands, see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED#bands (Level-1C) and https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands (Level-2A). All non-QA bands are used if sentinel_bands is None. Defaults to None.
        remove_clouds:
            A boolean indicating whether to remove clouds from the satellite images based on the QA60 band.
        remove_qa:
            A boolean indicating whether to remove the QA bands from the satellite images.

    Returns:
        A string representing the file path to the composite raster.
    """
    # Initialize Earth Engine API
    if not ee.data._credentials:
        print("Initializing Earth Engine API...")
        ee.Initialize()
    print("Preparing data...")

    # Ensure proper time window format and convert to strings
    date_format = "%Y-%m-%d"
    time_window = tuple(
        datetime.datetime.strptime(date, date_format)
        if isinstance(date, str)
        else date
        for date in time_window
    )
    time_window = tuple(date.strftime(date_format) for date in time_window)
    start, end = time_window
    if start > end:
        raise ValueError(
            f"start ({start}) must be before end ({end}) of timewindow"
        )

    # Ensure proper path format
    X_pathlib = Path(X_path)
    y_pathlib = Path(y_path)
    if X_pathlib.suffix != ".tif" or y_pathlib.suffix != ".tif":
        raise ValueError("X_path and y_path must be strings ending in .tif")
    if not X_pathlib.parent.exists():
        raise ValueError(
            f"X_path parent directory does not exist: {X_pathlib.parent}"
        )
    if not y_pathlib.parent.exists():
        raise ValueError(
            f"y_path parent directory does not exist: {y_pathlib.parent}"
        )

    # Check if indices are valid eemont indices
    if indices is not None:
        invalid_indices = [
            index
            for index in indices
            if index not in eemont.common.indices()
            or "Sentinel-2" not in eemont.common.indices()[index]["platforms"]
        ]
        if invalid_indices:
            raise ValueError(
                f"Invalid indices not in eemont package: {', '.join(invalid_indices)}"
            )

    # Split time window into sub windows
    time_windows = _split_time_window(time_window, num_composites)

    # Use ee.Reducer.mean() if temporal_reducers is None
    if temporal_reducers is None:
        temporal_reducers = ["mean"]
    if len(set(temporal_reducers)) < len(temporal_reducers):
        raise ValueError(
            "temporal_reducers must not contain duplicate reducers"
        )

    # Check if all reducers are valid
    valid_reducers = set()
    for attr in dir(ee.Reducer):
        try:
            if isinstance(getattr(ee.Reducer, attr)(), ee.Reducer):
                valid_reducers.add(attr)
        except (TypeError, ee.ee_exception.EEException):
            continue
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
    available_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    ]
    if level_2a:
        available_bands += [
            "AOT",
            "WVP",
            "SCL",
            "TCI_R",
            "TCI_G",
            "TCI_B",
            "MSK_CLDPRB",
            "MSK_SNWPRB",
        ]
        available_bands.pop(available_bands.index("B10"))

    if sentinel_bands is None:
        sentinel_bands = available_bands
    else:
        illegal_bands = [b for b in sentinel_bands if b not in available_bands]
        if any(illegal_bands):
            raise ValueError(
                f"{illegal_bands} not available in: {available_bands}"
            )

    # Get region of interest (ROI), scale, and coordinate reference system (CRS)
    with rasterio.open("../data/processed/separate_y.tif") as src:
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

    # Loop through time windows
    data = []
    for start, end in time_windows:
        # Get Sentinel 2 image collection
        if level_2a:
            s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        else:
            s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")

        # Filter by roi and timewindow
        s2 = s2.filterBounds(roi).filterDate(start, end)

        # Remove clouds
        if remove_clouds:
            s2 = s2.map(_mask_s2_clouds)

        # Remove QA bands
        remaining_bands = sentinel_bands
        if not remove_qa:
            remaining_bands += ["QA20", "QA40", "QA60"]
        s2 = s2.select(remaining_bands)

        # Add indices
        if indices:
            s2 = s2.spectralIndices(indices)

        # Reduce by temporal_reducers
        reduced_images = [
            s2.reduce(getattr(ee.Reducer, temporal_reducer)())
            for temporal_reducer in temporal_reducers
        ]

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
    _save_image(data, X_path, scale=scale, crs=crs)

    return X_path
