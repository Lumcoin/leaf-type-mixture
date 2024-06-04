Leaf Type Mixture
==============================

> **Disclaimer:** This repository is still under development.

Welcome to the repository for the paper "Predicting Leaf Type Mixture Using Tailored Sentinel-2 Composites." Our goal is to ensure all results are reproducible. If you encounter any issues, please open an issue in this repository.

# Overview

In this study, we focus on optimizing spectral band combinations, compositing methods, and hyperparameters for various regression models to enhance Sentinel-2 Level-2A data for predicting leaf type mixtures. The experiments are broken down into:

1. **Band Importance**: We conduct a systematic evaluation using recursive feature elimination (RFE) with a random forest model to identify significant bands and indices.
2. **Compositing**: The best temporal windows and compositing methods are selected to refine the dataset.
3. **Hyperparameter Tuning**: Hyperparameters for multiple regression models are optimized to improve performance.
4. **Generalization**: We evaluate the best model's generalization capability across unseen temporal windows and experimental sites.

Our results highlight the potential of tailored Sentinel-2 composites for leaf type mixture predictions and provide insights into optimizing spectral band combinations and compositing methods.

# Installation and Setup

This repository requires Python 3.10 or later. To install the required packages, run the following command in the root directory of this repository:

```bash
pip install -e .
```

To use the Earth Engine API, authenticate with the following Python code. Follow the website instructions and paste the authentication key into the input window. Repeat this process whenever your token expires.

```python
import ee

ee.Authenticate()
```

# Usage

All experiments described in our paper are implemented in the notebooks within the `notebooks` directory. After completing the installation and setup, you can run these notebooks in your preferred IDE.

# CI/CD

This GitHub repository uses [GitHub Actions](https://github.com/features/actions) for continuous integration (CI). The CI pipeline is defined in the `.github/workflows` directory. It includes:
- **Code Quality Checks** using [pylint](https://pylint.readthedocs.io/)
- **Testing** using [pytest](https://docs.pytest.org/).

> Note: Functions using the GEE API are excluded from the tests due to authentication concerns.

To run the CI workflow locally, execute the following commands from the repository directory:
```bash
pip install -e .[linting,testing]
pylint --disable=line-too-long,too-many-lines,no-member ltm
pylint --disable=line-too-long,too-many-lines,no-member,missing-module-docstring,missing-class-docstring,missing-function-docstring test
pytest
```

The code was formatted using the following commands:
```bash
pip install isort black[jupyter]
isort --profile black .
black .
```
