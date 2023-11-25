
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


def _split_time_window(time_window, num_splits):
    start = datetime.datetime.strptime(time_window[0], '%Y-%m-%d')
    end = datetime.datetime.strptime(time_window[1], '%Y-%m-%d')
    delta = (end - start + datetime.timedelta(days=1)) / num_splits

    if delta.days < 1:
        raise ValueError(
            f"Time window {time_window} is too small to split into {num_splits} sub windows")

    sub_windows = []
    for i in range(num_splits):
        sub_window = ((start + i * delta).strftime('%Y-%m-%d'),
                      (start + (i + 1) * delta - datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
        sub_windows.append(sub_window)

    return sub_windows


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
            profile["nodata"] = np.nan
            with rasterio.open(file_path, "w", **profile) as dst:
                raster = dataset.read()
                mask = mask_dataset.read()
                raster[mask == 0] = np.nan
                dst.write(raster)
                dst.descriptions = tuple(bands)


def _save_image(image, file_path, scale, crs):
    try:
        image_response = _download_image(image, scale, crs)
        mask_response = _download_image(image.mask(), scale, crs)
    except ee.ee_exception.EEException:
        raise ValueError(
            "FAILED to compute image. Most likely because the timewindow is too small or outside the availability, see 'Dataset Availability' in the Earth Engine Data Catalog.")

    if image_response.status_code == mask_response.status_code == 200:
        _image_response2file(image_response.content,
                             file_path,
                             mask=mask_response.content,
                             bands=image.bandNames().getInfo())
        print(f"GeoTIFF saved as {file_path}")
    else:
        print(
            f"FAILED to either compute or download the image for {file_path} most likely due to limits of Google Earth Engine.")


def _mask_s2_clouds(image):
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


def sentinel_composite(
        plot: pd.DataFrame,
        time_window: Tuple[datetime.date, datetime.date] | Tuple[str, str],
        X_path: str = "../data/processed/X.tif",
        y_path: str = "../data/processed/y.tif",
        num_composites: int = 1,
        temporal_reducers: List[str] = None,
        indices: List[str] = None,
        level_2a: bool = False,
        sentinel_bands: List[str] = None,
        remove_clouds: bool = True,
        remove_qa: bool = True,
        areas_as_y: bool = False,
) -> Tuple[str, str]:
    """Creates a composite from many Sentinel 2 satellite images for a given plot.

    Args:
        plot:
            A pandas DataFrame containing data on a per tree basis with two columns for longitude and latitude, one column for DBH, and one for whether or not the tree is a broadleaf (1 is broadleaf, 0 is conifer). The column names must be 'longitude', 'latitude', 'dbh', and 'broadleaf' respectively. This function is case insensitive regarding column names.
        time_window:
            A tuple of two dates or strings representing the start and end of the time window to retrieve the satellite images.
        X_path:
            A string representing the file path to save the composite raster.
        y_path:
            A string representing the file path to save the raster with values between 0 and 1 for the leaf type mixture. Will not be saved if None. Resulting in a speedup.
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
        areas_as_y:
            A boolean indicating whether to compute the area per leaf type instead of the leaf type mixture as labels. Results in a y with two bands, one for each leaf type. Defaults to False.

    Returns:
        A tuple of two strings representing the file paths to the composite raster and the raster with values between 0 and 1 for the leaf type mixture. 1 being broadleaf and 0 being conifer.
    """
    # Initialize Earth Engine API
    if not ee.data._credentials:
        print("Initializing Earth Engine API...")
        ee.Initialize()
    print("Preparing data...")

    # Type checks
    if not isinstance(plot, pd.DataFrame):
        raise TypeError("plot must be a pandas DataFrame")
    if not isinstance(time_window, tuple) or len(time_window) != 2:
        raise TypeError("timewindow must be a tuple of two dates or strings")
    if not isinstance(time_window[0], (datetime.date, str)) or not isinstance(time_window[1], (datetime.date, str)):
        raise TypeError(
            "timewindow elements must be either datetime.date or str")
    if not isinstance(X_path, str):
        raise TypeError("X_path must be a string")
    if not isinstance(y_path, str):
        raise TypeError("y_path must be a string")
    if indices is not None and (not isinstance(indices, list) or not all(isinstance(i, str) for i in indices)):
        raise TypeError("indices must be a list of strings")
    if not isinstance(num_composites, int) or num_composites < 1:
        raise TypeError("num_composites must be a positive integer")
    if temporal_reducers is not None and (not isinstance(temporal_reducers, list) or not all(isinstance(i, str) for i in temporal_reducers)):
        raise TypeError(
            "temporal_reducers must be a list of strings")
    if not isinstance(level_2a, bool):
        raise TypeError("level_2a must be a boolean")
    if not isinstance(sentinel_bands, list) and sentinel_bands is not None:
        raise TypeError("sentinel_bands must be a list of strings")
    if not isinstance(remove_clouds, bool):
        raise TypeError("remove_clouds must be a boolean")
    if not isinstance(areas_as_y, bool):
        raise TypeError("area_as_y must be a boolean")

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

    # Ensure proper time window format and convert to strings
    date_format = "%Y-%m-%d"
    time_window = tuple(datetime.datetime.strptime(
        date, date_format) if isinstance(date, str) else date for date in time_window)
    time_window = tuple(date.strftime(date_format)
                        for date in time_window)
    start, end = time_window
    if start > end:
        raise ValueError(
            f"start ({start}) must be before end ({end}) of timewindow")

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
    X_path = str(X_path)
    y_path = str(y_path)

    # Check if indices are valid eemont indices
    if indices is not None:
        invalid_indices = [index for index in indices
                           if index not in eemont.common.indices()
                           or "Sentinel-2" not in eemont.common.indices()[index]["platforms"]]
        if invalid_indices:
            raise ValueError(
                f"Invalid indices not in eemont package: {', '.join(invalid_indices)}")

    # Split time window into sub windows
    time_windows = _split_time_window(time_window, num_composites)

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

    # Check if all bands are valid. Use all bands if sentinel_bands is None
    if level_2a:
        available_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11",
                           "B12", "AOT", "WVP", "SCL", "TCI_R", "TCI_G", "TCI_B", "MSK_CLDPRB", "MSK_SNWPRB"]
    else:
        available_bands = ["B1", "B2", "B3", "B4", "B5",
                           "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

    if sentinel_bands is None:
        sentinel_bands = available_bands
    else:
        illegal_bands = [b for b in sentinel_bands if b not in available_bands]
        if any(illegal_bands):
            raise ValueError(
                f"{illegal_bands} not available in: {available_bands}")

    # Get region of interest (ROI)
    roi = ee.Geometry.Rectangle([
        plot["longitude"].min(),
        plot["latitude"].min(),
        plot["longitude"].max(),
        plot["latitude"].max(),
    ]).buffer(plot["dbh"].max() / 2)

    # Check if rectangle has reasonable size
    if roi.area().getInfo() == 0:
        raise ValueError(
            "Plot bounding box has area 0. Check if plot coordinates are valid.")
    if roi.area().getInfo() > 1e7:
        raise ValueError(
            "Plot bounding box has area > 1e7. Check if plot coordinates are valid.")

    # Get CRS in epsg format for center of the roi
    longitude, latitude = roi.centroid(1).getInfo()["coordinates"]
    zone_number = utm.latlon_to_zone_number(latitude, longitude)
    is_south = utm.latitude_to_zone_letter(latitude) < "N"

    crs = CRS.from_dict(
        {"proj": "utm", "zone": zone_number, "south": is_south})
    crs = ":".join(crs.to_authority())

    # Define scale at 10 m/pixel, same as max Sentinel 2 resolution
    scale = 10

    # Convert ROI to bounding box in output crs
    roi = roi.bounds(0.01, crs)

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
        reduced_images = [s2.reduce(getattr(ee.Reducer, temporal_reducer)())
                          for temporal_reducer in temporal_reducers]

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
        composite_idx, _, *band_label, reducer_label = band_name.split("_")
        composite_idx = int(composite_idx) + 1
        pretty_name = f"{composite_idx} {reducer_label.title()} {'_'.join(band_label)}"
        pretty_names.append(pretty_name)
    data = data.rename(pretty_names)

    # Save data (X)
    print("Computing data...")
    _save_image(data, X_path, scale=scale, crs=crs)
    print("Preparing labels...")

    # End here if y_path is None (speedup)
    if y_path is None:
        return X_path, y_path

    # Upload plot
    broadleafs = []
    conifers = []
    for _, row in plot.iterrows():
        circle = ee.Geometry.Point([
            row["longitude"],
            row["latitude"],
        ]).buffer(row["dbh"] / 2)

        if row["broadleaf"]:
            broadleafs.append(circle)
        else:
            conifers.append(circle)
    broadleafs = ee.FeatureCollection(broadleafs)
    conifers = ee.FeatureCollection(conifers)

    # Render plot as fine resolution image, then reduce to coarse resolution
    fine_scale = plot["dbh"].min() * 5
    # not more coarse than Sentinel 2 resolution
    fine_scale = min(fine_scale, 10)
    if plot["dbh"].min() < 0.05:
        print(
            "Info: DBH < 0.05 m found. Small trees might be ignored by Google Earth Engine.")
        fine_scale = 0.25

    # Compute broadleaf area
    broadleaf_area = ee.Image.constant(1).clip(broadleafs).mask()
    broadleaf_area = ee.Image.constant(scale**2).multiply(broadleaf_area)
    broadleaf_area = broadleaf_area.reproject(scale=fine_scale, crs=crs)
    broadleaf_area = broadleaf_area.reduceResolution(
        ee.Reducer.mean(), maxPixels=10_000)

    # Compute conifer area
    conifer_area = ee.Image.constant(1).clip(conifers).mask()
    conifer_area = ee.Image.constant(scale**2).multiply(conifer_area)
    conifer_area = conifer_area.reproject(scale=fine_scale, crs=crs)
    conifer_area = conifer_area.reduceResolution(
        ee.Reducer.mean(), maxPixels=10_000)

    # Compute y (leaf type mixture) from broadleaf_area and conifer_area
    if areas_as_y:
        y = broadleaf_area.addBands(conifer_area)
        # Remove pixels with no trees
        y = y.updateMask(broadleaf_area.add(conifer_area).gt(0))
        y = y.rename(["Broadleaf Area", "Conifer Area"])
    else:
        denominator = broadleaf_area.add(conifer_area)
        y = broadleaf_area.divide(denominator)
        y = y.updateMask(denominator.gt(0))  # Remove pixels with no trees
        y = y.rename("Leaf Type Mixture")

    # Clip to roi
    y = y.clip(roi)
    y = y.reproject(scale=scale, crs=crs)

    # Save y
    print("Computing labels...")
    _save_image(y, y_path, scale=scale, crs=crs)

    return X_path, y_path
