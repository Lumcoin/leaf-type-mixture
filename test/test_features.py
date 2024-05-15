import numpy as np
import pandas as pd
import pytest
import rasterio

from ltm.features import load_raster


@pytest.fixture(name="data_path")
def fixture_data_path(tmp_path):
    data = np.random.rand(10, 20, 30)
    data_file = tmp_path / "data.tif"
    with rasterio.open(
        data_file,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
    ) as dst:
        dst.write(data)
        dst.descriptions = tuple(f"Mean B{i+1}" for i in range(data.shape[0]))
    return str(data_file)


@pytest.fixture(name="target_path")
def fixture_target_path(tmp_path):
    target = np.random.rand(10, 20)
    target_file = tmp_path / "target.tif"
    with rasterio.open(
        target_file,
        "w",
        driver="GTiff",
        height=target.shape[0],
        width=target.shape[1],
        count=1,
        dtype=target.dtype,
    ) as dst:
        dst.write(target, 1)

    return str(target_file)


def test_load_data_and_band_names(data_path):
    data = load_raster(data_path)
    band_names = list(data.columns)
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (600, 10)
    assert len(band_names) == 10
    assert band_names[0] == "Mean B1"


def test_load_raster(target_path):
    target = load_raster(target_path)
    assert isinstance(target, pd.Series)
    assert len(target) == 200
