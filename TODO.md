# TODO List for the Next 2 Weeks

## 1. Rewriting Notebook Markdown Cells
Each notebook has legacy markdown that needs to be rewritten to match the current repository version. Below is the split-up task for each notebook.

### 1.1 `ground truth` Notebook
- [x] Make the code ruff compliant.
- [x] Review and rewrite the introduction section.
- [x] Update any data loading and preprocessing markdown.
- [x] Rewrite explanations for analysis steps.
- [x] Update conclusions and summary markdown.
- [x] Enable figure saving for all plots. (single function call -> skip if exists, mkdir, save)

### 1.2 `band importance` Notebook
- [x] Make the code ruff compliant.
- [x] Review and rewrite the introduction section.
- [x] Update markdown for feature importance analysis.
- [x] Rewrite explanations for plots and figures.
- [x] Update conclusions and summary markdown.
- [x] Enable figure saving for all plots.

### 1.3 `compositing` Notebook
- [x] Make the code ruff compliant.
- [x] Review and rewrite the introduction section.
- [x] Update markdown for compositing methods.
- [x] Rewrite explanations for any results or plots.
- [x] Update conclusions and summary markdown.
- [x] Enable figure saving for all plots.

### 1.4 `hyperparameter tuning` Notebook
- [x] Make the code ruff compliant.
- [ ] Review and rewrite the introduction section.
- [ ] Update markdown for the hyperparameter tuning process.
- [ ] Rewrite explanations for any optimization results.
- [ ] Update conclusions and summary markdown.
- [ ] Enable figure saving for all plots.

### 1.5 `Generalization` Notebook
- [x] Make the code ruff compliant.
- [ ] Compute all scores on the test set, then separately compare the best one to DLT 2018 product
- [ ] Review and rewrite the introduction section.
- [ ] Update markdown for the generalization techniques.
- [ ] Rewrite explanations for results and discussions.
- [ ] Update conclusions and summary markdown.
- [ ] Enable figure saving for all plots.

## 2. Linting with Ruff
The `slc` package needs to be checked against any Ruff linting warnings.

### 2.1 `__init__.py`
- [x] Run Ruff linting on `slc/__init__.py`.
- [x] Resolve any warnings or errors.

### 2.2 `data` Module
- [x] Run Ruff linting on `slc/data.py`.
- [x] Resolve any warnings or errors.

### 2.3 `features` Module
- [x] Run Ruff linting on `slc/features.py`.
- [x] Resolve any warnings or errors.

### 2.4 `models` Module
- [x] Run Ruff linting on `slc/models.py`.
- [x] Resolve any warnings or errors.

### 2.5 `visualize` Module
- [x] Run Ruff linting on `slc/visualize.py`.
- [x] Resolve any warnings or errors.

## 3. Additional Tasks
Ensure other repository files are up-to-date and consistent with the current version.

### 3.1 `pyproject.toml`
- [x] Review `pyproject.toml` for any outdated configurations.
- [x] Update dependencies and settings if necessary.

### 3.2 `README.md`
- [ ] Review and update the `README.md`.
- [ ] Ensure that the README reflects the current state and purpose of the repository.

### 3.3 `reports/` Directory
- [ ] Check the `reports/` directory for any outdated reports.
- [ ] Update or remove old reports if necessary.

### 3.4 `models/` Directory
- [ ] Review the `models/` directory for any outdated models.
- [ ] Update model documentation and any related files.

### 3.5 `data/` Directory
- [ ] Review the `data/` directory for any unnecessary or outdated files.
- [ ] Ensure that the data files are correctly documented and relevant.

### 3.6 Repository Cleanup
- [ ] Review the entire repository for any other outdated files or code.
- [ ] Remove or archive unnecessary files.

## 4. Updating Tests
The tests need to be partially rewritten since they assume an older version of the repository.

### 4.1 `__init__.py` in `test`
- [x] Review and update `test/__init__.py` for current functionality.

### 4.2 `test_data.py`
- [x] Review and update tests in `test/test_data.py`.

### 4.3 `test_features.py`
- [x] Review and update tests in `test/test_features.py`.

### 4.4 `test_models.py`
- [x] Review and update tests in `test/test_models.py`.

### 4.5 `test_visualize.py`
- [x] Review and update tests in `test/test_visualize.py`.

## Final steps
- [x] Compare all models on the test set with the DLT 2018 product (ignoring Larix pixels for the latter)
- [ ] info for the paper: standardize SVM input
- [ ] Geolocation Accuracy of Traunstein dataset
- [ ] Parallelize the data downloading in band-importance and compositing
- [ ] Remove all leftover `TODO` comments
- [ ] Check if the company allows the publishing of all CSV dataset files -> ground_truth notebook Markdown assumes it for NWR Rep. areas
- [ ] Rerun all notebooks and hyperparam tune ALL models with 100 iterations
- [x] Logging with loguru
- [x] Reproducibility of Hyperparameter tuning, especially training vs manual testing