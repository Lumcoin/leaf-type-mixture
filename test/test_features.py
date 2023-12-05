# pylint: disable=missing-module-docstring
import numpy as np
import pytest
import rasterio

from ltm.features import drop_nan, load_multi_band_raster


@pytest.fixture(name="X_path")
def fixture_X_path(tmp_path):  # pylint: disable=invalid-name
    X = np.random.rand(10, 20, 30)  # pylint: disable=invalid-name
    X_file = tmp_path / "X.tif"  # so pylint: disable=invalid-name
    with rasterio.open(
        X_file,
        "w",
        driver="GTiff",
        height=X.shape[1],
        width=X.shape[2],
        count=X.shape[0],
        dtype=X.dtype,
    ) as dst:
        dst.write(X)
        dst.descriptions = tuple(f"Mean B{i+1}" for i in range(X.shape[0]))
    return str(X_file)


@pytest.fixture(name="y_path")
def fixture_y_path(tmp_path):
    y = np.random.rand(10, 20)
    y_file = tmp_path / "y.tif"
    with rasterio.open(
        y_file,
        "w",
        driver="GTiff",
        height=y.shape[0],
        width=y.shape[1],
        count=1,
        dtype=y.dtype,
    ) as dst:
        dst.write(y, 1)

    return str(y_file)


def test_load_X_and_band_names(X_path):  # pylint: disable=invalid-name
    X, band_names = load_multi_band_raster(X_path)  # pylint: disable=invalid-name
    assert isinstance(X, np.ndarray)
    assert isinstance(band_names, list)
    assert X.shape == (600, 10)
    assert len(band_names) == 10
    assert band_names[0] == "Mean B1"


def test_load_y(y_path):
    y, _ = load_multi_band_raster(y_path)
    assert isinstance(y, np.ndarray)
    assert y.shape == (200, 1)


def test_drop_nan():
    X = np.array(  # pylint: disable=invalid-name
        [[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]]
    )
    y = np.array([0, 1, 0])
    X_clean, y_clean = drop_nan(X, y)  # pylint: disable=invalid-name
    assert isinstance(X_clean, np.ndarray)
    assert isinstance(y_clean, np.ndarray)
    assert X_clean.shape == (1, 3)
    assert y_clean.shape == (1,)
    assert X_clean[0, 0] == 7
    assert y_clean[0] == 0
