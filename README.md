Leaf Type Mixture
==============================

Leaf type mixture prediction with machine learning from satellite data.

Requires Python>=3.10

If you are new to python, we recommend to install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) and create a environment called "ltm" using `conda create -n ltm python=3.10 ipykernel`. You can then activate the environment using `conda activate ltm` and install the "ltm" package of this repository using:
```bash
pip install -e .
```

The "`.`" (dot) in above code cell is no typo, it is the path to the current directory.

Install the ltm package to your active environment:
```bash
pip install -e .
```

The code in this repository is formatted using following commands:
```bash
pip install isort docformatter black
isort --profile black .
docformatter --in-place --recursive .
black --line-length 80 .
```

This repository uses the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) as a style guide for the code. If you dicover any violations, please open an issue.

Check code quality using pylint:
```bash
pip install pylint
pylint --disable=line-too-long **/*.py
```

Check test coverage using coverage.py:
```bash
pip install coverage
coverage run -m pytest
coverage report -m
```

Reinstall conda environment:
```bash
~/mambaforge/Scripts/activate.bat
mamba activate base
mamba env remove -n ltm
mamba clean -a -y
mamba env create -f environment.yml
```

Note: Random state will be set by sklearn, if scipy >= 0.16 is available.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
