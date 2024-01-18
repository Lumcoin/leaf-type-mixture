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
        "ee",
        "eemont",
        "geopandas",
        "rasterio",
        "matplotlib",
        "numpy",
        "pandas",
        "pyproj",
        "requests",
        "scikit-learn",
        "SciencePlots",
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
            "dask",
            "distributed",
            "pytest",
            "scikit-elm",
        ],
    },
)
