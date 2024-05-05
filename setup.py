# pylint: disable=missing-module-docstring
# https://packaging.python.org/en/latest/guides/tool-recommendations/
from setuptools import find_packages, setup

setup(
    name="ltm",
    packages=find_packages(),
    version="0.1.0",
    description="Leaf type mixture prediction with machine learning from satellite data.",
    author="Peter Hofinger",
    license="MIT",
    install_requires=[
        "aiohttp",
        "dask[dataframe,distributed]",  # for scikit-elm
        "dill",  # for pickling objects with lambda functions
        "ee",
        "eemont",
        "geopandas",
        "ipywidgets",  # for tqdm
        "rasterio",
        "matplotlib",
        "numpy",
        "optuna",
        "pandas",
        "pyproj",
        "requests",
        "scikit-elm",
        "scikit-learn",
        "SciencePlots",
        "seaborn",
        "tqdm",
        "typeguard",
        "utm",
        "xgboost",
    ],
    python_requires=">=3.8",
    extras_require={
        "linting": [
            "pylint",
        ],
        "testing": [
            "pytest",
        ],
    },
)
