
"""
This module contains functions for processing satellite data for leaf type mixture analysis.

Functions:
- sentinel_composite: Creates a composite from many Sentinel 2 satellite images for a given plot.
"""

import datetime
from pathlib import Path
from typing import List, Tuple

import ee
import eemont
import numpy as np
import pandas as pd
import rasterio
import requests
import utm
from pyproj import CRS
from rasterio.io import MemoryFile


def _download_image(image, scale, crs):
    download_params = {
        "scale": scale,
        "crs": crs,
        "format": "GEO_TIFF",
    }
    url = image.getDownloadURL(download_params)

    return requests.get(url)


def _image_response2file(image, file_path, mask, bands):
    with MemoryFile(image) as memfile, MemoryFile(mask) as mask_memfile:
        with memfile.open() as dataset, mask_memfile.open() as mask_dataset:
            profile = dataset.profile
            with rasterio.open(file_path, "w", **profile) as dst:
                raster = dataset.read()
                mask = mask_dataset.read()
                raster[mask == 0] = np.nan
                dst.write(raster)
                dst.descriptions = tuple(bands)


def _save_image(image, file_path, scale, crs):
    image_response = _download_image(image, scale, crs)
    mask_response = _download_image(image.mask(), scale, crs)

    if image_response.status_code == mask_response.status_code == 200:
        _image_response2file(image_response.content,
                             file_path,
                             mask=mask_response.content,
                             bands=image.bandNames().getInfo())
        print(f"GeoTIFF saved as {file_path}")
    else:
        print(f"Failed to download the image for {file_path}")


def _mask_s2_clouds(image):
    """From https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#colab-python

    Masks clouds in a Sentinel-2 image using the QA band.

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


def sentinel_composite(
        timewindow: Tuple[datetime.date, datetime.date] | Tuple[str, str],
        plot: pd.DataFrame,
        X_path: str = "../data/processed/X.tif",
        y_path: str = "../data/processed/y.tif",
        temporal_reducers: List[str] = None,
        indices: List[str] = None,
        level_2a: bool = False,
        remove_clouds: bool = True,
        remove_qa: bool = True,
) -> Tuple[str, str]:
    """
    Creates a composite from many Sentinel 2 satellite images for a given plot.

    Args:
        timewindow:
            A tuple of two dates or strings representing the start and end of the time window to retrieve the satellite images.
        plot:
            A pandas DataFrame containing data on a per tree basis with two columns for longitude and latitude, one column for DBH, and one for whether or not the tree is a broadleaf (1 is broadleaf, 0 is conifer). The column names must be 'longitude', 'latitude', 'dbh', and 'broadleaf' respectively. This function is case insensitive regarding column names.
        X_path:
            A string representing the file path to save the composite raster.
        y_path: 
            A string representing the file path to save the raster with values between 0 and 1 for the leaf type mixture.
        temporal_reducers:
            A list of strings representing the temporal reducers to use when creating the composite. See https://developers.google.com/earth-engine/guides/reducers_intro for more information. Defaults to ['mean'] if None.
        indices:
            A list of strings representing the spectral indices to add to the composite as additional bands. See https://eemont.readthedocs.io/en/latest/guide/spectralIndices.html for more information.
        level_2a:
            A boolean indicating whether to use level 2A or level 1C Sentinel 2 data.
        remove_clouds: 
            A boolean indicating whether to remove clouds from the satellite images.
        remove_qa:
            A boolean indicating whether to remove the QA bands from the satellite images.

    Returns:
        A tuple of two strings representing the file paths to the composite raster and the raster with values between 0 and 1 for the leaf type mixture. 1 being broadleaf and 0 being conifer.
    """
    # Initialize Earth Engine API
    if not ee.data._credentials:
        print("Initializing Earth Engine API...")
        ee.Initialize()
    print("Preparing data...")

    # Type checks
    if not isinstance(timewindow, tuple) or len(timewindow) != 2:
        raise TypeError("timewindow must be a tuple of two dates or strings")
    if not isinstance(timewindow[0], (datetime.date, str)) or not isinstance(timewindow[1], (datetime.date, str)):
        raise TypeError(
            "timewindow elements must be either datetime.date or str")
    if not isinstance(plot, pd.DataFrame):
        raise TypeError("plot must be a pandas DataFrame")
    if not isinstance(X_path, str):
        raise TypeError("X_path must be a string")
    if not isinstance(y_path, str):
        raise TypeError("y_path must be a string")
    if indices is not None and (not isinstance(indices, list) or not all(isinstance(i, str) for i in indices)):
        raise TypeError("indices must be a list of strings")
    if temporal_reducers is not None and (not isinstance(temporal_reducers, list) or not all(isinstance(i, str) for i in temporal_reducers)):
        raise TypeError(
            "temporal_reducers must be a list of strings")
    if not isinstance(level_2a, bool):
        raise TypeError("level_2a must be a boolean")
    if not isinstance(remove_clouds, bool):
        raise TypeError("remove_clouds must be a boolean")

    # Convert timewindow to strings
    date_format = "%Y-%m-%d"
    timewindow = tuple(datetime.datetime.strptime(date, date_format) if isinstance(
        date, str) else date
        for date in timewindow)
    start, end = tuple(date.strftime(date_format)
                       for date in timewindow)
    if start > end:
        raise ValueError(
            f"start ({start}) must be before end ({end}) of timewindow")

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
    X_path = Path(X_path)
    y_path = Path(y_path)
    if X_path.suffix != ".tif" or y_path.suffix != ".tif":
        raise ValueError(
            "X_path and y_path must be strings ending in .tif")
    if not X_path.parent.exists():
        raise ValueError(
            f"X_path parent directory does not exist: {X_path.parent}")
    if not y_path.parent.exists():
        raise ValueError(
            f"y_path parent directory does not exist: {y_path.parent}")

    # Check if indices are valid eemont indices
    if indices is not None:
        invalid_indices = [index for index in indices
                           if index not in eemont.common.indices()
                           or "Sentinel-2" not in eemont.common.indices()[index]["platforms"]]
        if invalid_indices:
            raise ValueError(
                f"Invalid indices not in eemont package: {', '.join(invalid_indices)}")

    # Use ee.Reducer.mean() if temporal_reducers is None
    if temporal_reducers is None:
        temporal_reducers = ["mean"]
    if len(set(temporal_reducers)) < len(temporal_reducers):
        raise ValueError(
            "temporal_reducers must not contain duplicate reducers")

    # Check if all reducers are valid
    valid_reducers = set()
    for attr in dir(ee.Reducer):
        try:
            if isinstance(getattr(ee.Reducer, attr)(), ee.Reducer):
                valid_reducers.add(attr)
        except (TypeError, ee.ee_exception.EEException):
            continue
    invalid_reducers = [reducer for reducer in temporal_reducers
                        if reducer not in valid_reducers]
    if invalid_reducers:
        raise ValueError(
            f"Invalid reducers not in ee.Reducer: {', '.join(invalid_reducers)}")

    # Get roi
    roi = ee.Geometry.Rectangle([
        plot["longitude"].min(),
        plot["latitude"].min(),
        plot["longitude"].max(),
        plot["latitude"].max(),
    ]).buffer(plot["dbh"].max() / 2)

    # Get CRS in epsg format
    longitude, latitude = roi.centroid(1).getInfo()["coordinates"]
    zone_number = utm.latlon_to_zone_number(latitude, longitude)
    is_south = utm.latitude_to_zone_letter(latitude) < "N"

    crs = CRS.from_dict(
        {"proj": "utm", "zone": zone_number, "south": is_south})
    crs = ":".join(crs.to_authority())

    # Define scale at 10 m/pixel, same as max Sentinel 2 resolution
    scale = 10

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
    if remove_qa:
        remaining_bands = s2.first().bandNames().getInfo()
        remaining_bands = [
            band for band in remaining_bands if not band.startswith("QA")]
        # remaining_bands = ["B1", "B2", "B3", "B4", "B5",
        #                    "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
        s2 = s2.select(remaining_bands)

    # Add indices
    if indices:
        s2 = s2.spectralIndices(indices)

    # Reduce by temporal_reducers
    reduced_images = [s2.reduce(getattr(ee.Reducer, temporal_reducer)())
                      for temporal_reducer in temporal_reducers]

    # Stack images
    X = ee.ImageCollection(reduced_images).toBands()
    X = X.reproject(scale=scale, crs=crs)
    X = X.clip(roi)

    # Prettify band names
    pretty_names = []
    for band_name in X.bandNames().getInfo():
        _, *band_label, reducer_label = band_name.split("_")
        pretty_name = f"{reducer_label.title()} {''.join(band_label)}"
        pretty_names.append(pretty_name)
    X = X.rename(pretty_names)

    # Save X
    print("Computing data...")
    _save_image(X, X_path, scale=scale, crs=crs)
    print("Preparing labels...")

    # Upload plot
    plot_fc = ee.FeatureCollection([ee.Feature(
        ee.Geometry.Point([
            row["longitude"],
            row["latitude"],
        ]).buffer(row["dbh"] / 2),
        {"broadleaf": row["broadleaf"]})
        for index, row in plot.iterrows()])

    # Reduce plot to image
    y = plot_fc.reduceToImage(["broadleaf"], ee.Reducer.mean())
    y = y.reproject(scale=scale, crs=crs)
    y = y.clip(roi)

    # Save y
    print("Computing labels...")
    _save_image(y, y_path, scale=scale, crs=crs)

    return X_path, y_path
