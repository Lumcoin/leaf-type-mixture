Leaf Type Mixture
==============================

> **Disclaimer:** This repository is still under development.

This repository supports the paper "Predicting Leaf Type Mixture Using Tailored Sentinel-2 Composites". We aim for all results to be reproducible. If you encounter any issues, please open an issue in this repository.

In this study, we explore the optimization of spectral band combinations and compositing as well as the tuning of hyperparameters for multiple common regression models to tailor Sentinel-2 Level-2A data for leaf type mixture predictions. We first identify the most important spectral bands and indices through a systematic evaluation. This involves creating composite rasters over a one-year period and assessing the importance of bands using recursive feature elimination (RFE) in conjunction with a random forest model. Subsequently, we optimize the compositing process by selecting the best temporal windows and compositing methods. The optimized dataset is used to fine-tune hyperparameters of multiple regression models to further improve performance. Our results highlight the significance of tailoring the data and model to the specific use case. At last, we evaluate the model's generalization capability across different unseen temporal windows and experimental sites. We find that the model's performance is highly dependent on the temporal window and spatial domain used for training and testing. Our results demonstrate the potential of using tailored Sentinel-2 composites for leaf type mixture predictions and provide insights into the optimization of spectral band combinations and compositing methods for this task.

# Installation and Setup

This repository requires Python 3.10 or later. To install the required packages, run the following command in the root directory of this repository:

```bash
pip install -e .
```

To use the Earth Engine API you need to authenticate. Execute the Python code below. Follow the instructions on the website and paste the authentication key into the input window. Repeat this process whenever your token expires.

```python
import ee

ee.Authenticate()
```

# Usage

All experiments of our paper are implemented in the notebooks in the `notebooks` directory. After following the instructions for installation and setup, you can run the notebooks in your IDE of choice.

# CI/CD

This GitHub repository uses [GitHub Actions](https://github.com/features/actions) for a continuous integration (CI) workflow. The CI pipeline is defined in the `.github/workflows` directory. The pipeline checks the code quality of the repository with [pylint](https://pylint.readthedocs.io/) and runs tests using [pytest](https://docs.pytest.org/). Functions using GEE are not included in the tests due to concerns regarding the authentication key.

If you want to run the CI workflow locally, you can use the following commands from the root directory of this repository:
```bash
pip install -e .[linting,testing]
pylint --disable=line-too-long,too-many-lines,no-member ltm
pylint --disable=line-too-long,too-many-lines,no-member,missing-module-docstring,missing-class-docstring,missing-function-docstring test
pytest
```

The code was formatted using following commands:
```bash
pip install isort black[jupyter] 
isort --profile black .
black .
```