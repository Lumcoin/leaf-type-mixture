Leaf Type Mixture
==============================

> **Disclaimer:** This repository is still under development. The code is not yet ready for production use. The README is only a placeholder and will be updated in the future.

This repository belongs to the paper ---. Thus most functions in this repository are tailored to the use case of leaf type mixture prediction. However some functions might be useful for general use cases. For example you can create a Sentinel 2 composite for a given label raster in GeoTIFF format. There are functions for reading raster data as pd.DataFrames, etc. General use cases are demonstrated in general_use_cases.ipynb.

All results should be reproducible, if not please create an issue.

Requires Python>=3.10 (!)

If you are new to python, we recommend to install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) and create a environment called "ltm" using `conda create -n ltm python=3.10 ipykernel`. Now navigate to the folder where this repository is cloned to using the `cd [path/to/leaf-type-mixture]` (replace path within square brackets `[]`) command. You can then activate the environment using `conda activate ltm` and install the "ltm" package and its dependencies of this repository using:
```bash
pip install -e .
```

The "`.`" (dot) in above code cell is no typo, it is the path to the current directory. Then call `import ee; ee.Authenticate()` in a code cell to authenticate with Google Earth Engine.

The code in this repository is formatted using following commands:
```bash
pip install isort docformatter black[jupyter]
isort --profile black .
docformatter --in-place --recursive .
black --line-length 80 .
```

This repository uses the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) as a style guide for the code. If you dicover any violations, please open an issue.

Check code quality using pylint:
```bash
pip install pylint
pylint --disable=line-too-long,too-many-lines,no-member ltm
pylint --disable=line-too-long,too-many-lines,no-member,missing-module-docstring,missing-class-docstring,missing-function-docstring test
```

Run tests using pytest:
```bash
pip install pytest
pytest
```

Check test coverage using coverage.py:
```bash
pip install coverage
coverage run -m pytest
coverage report -m
```

Reinstall conda environment:
```bash
%USERPROFILE%/mambaforge/Scripts/activate.bat
mamba activate base
mamba env remove -n ltm
mamba clean -a -y
mamba env create -f environment.yml
```

Note: Random state will be set by sklearn, if scipy >= 0.16 is available.

Fix Pylance error "Import could not be resolved":
- Press Ctrl+Shift+P
- Select "Python: Clear Cache and Reload Window"

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data.py        <- Script to download or generate data
        │
        ├── features.py    <- Script to turn raw data into features for modeling
        │
        └── models.py      <- Scripts to train models and then use trained models to make
                              predictions


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
