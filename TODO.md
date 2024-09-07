# TODO List for the Next 2 Weeks

## 1. Rewriting Notebook Markdown Cells
Each notebook has legacy markdown that needs to be rewritten to match the current repository version. Below is the split-up task for each notebook.

### 1.1 `ground truth` Notebook
- [x] Make the code ruff compliant.
- [ ] Review and rewrite the introduction section.
- [ ] Update any data loading and preprocessing markdown.
- [ ] Rewrite explanations for analysis steps.
- [ ] Update conclusions and summary markdown.
- [ ] Enable figure saving for all plots. (single function call -> skip if exists, mkdir, save)

### 1.2 `band importance` Notebook
- [x] Make the code ruff compliant.
- [ ] Review and rewrite the introduction section.
- [ ] Update markdown for feature importance analysis.
- [ ] Rewrite explanations for plots and figures.
- [ ] Update conclusions and summary markdown.
- [ ] Enable figure saving for all plots.

### 1.3 `compositing` Notebook
- [x] Make the code ruff compliant.
- [ ] Review and rewrite the introduction section.
- [ ] Update markdown for compositing methods.
- [ ] Rewrite explanations for any results or plots.
- [ ] Update conclusions and summary markdown.
- [ ] Enable figure saving for all plots.

### 1.4 `hyperparameter tuning` Notebook
- [x] Make the code ruff compliant.
- [ ] Review and rewrite the introduction section.
- [ ] Update markdown for the hyperparameter tuning process.
- [ ] Rewrite explanations for any optimization results.
- [ ] Update conclusions and summary markdown.
- [ ] Enable figure saving for all plots.

### 1.5 `Generalization` Notebook
- [x] Make the code ruff compliant.
- [ ] Review and rewrite the introduction section.
- [ ] Update markdown for the generalization techniques.
- [ ] Rewrite explanations for results and discussions.
- [ ] Update conclusions and summary markdown.
- [ ] Enable figure saving for all plots.

## 2. Linting with Ruff
The `slc` package needs to be checked against any Ruff linting warnings.

### 2.1 `__init__.py`
- [x] Run Ruff linting on `slc/__init__.py`.
- [ ] Resolve any warnings or errors.

### 2.2 `data` Module
- [x] Run Ruff linting on `slc/data.py`.
- [ ] Resolve any warnings or errors.

### 2.3 `features` Module
- [x] Run Ruff linting on `slc/features.py`.
- [ ] Resolve any warnings or errors.

### 2.4 `models` Module
- [x] Run Ruff linting on `slc/models.py`.
- [ ] Resolve any warnings or errors.

### 2.5 `visualize` Module
- [x] Run Ruff linting on `slc/visualize.py`.
- [ ] Resolve any warnings or errors.

## 3. Updating Tests
The tests need to be partially rewritten since they assume an older version of the repository.

### 3.1 `__init__.py` in `test`
- [ ] Review and update `test/__init__.py` for current functionality.
- [ ] Ensure the test setup matches the new structure.

### 3.2 `test_data.py`
- [ ] Review and update tests in `test/test_data.py`.
- [ ] Adjust tests to reflect current data processing logic.

### 3.3 `test_features.py`
- [ ] Review and update tests in `test/test_features.py`.
- [ ] Adjust tests to reflect current feature extraction logic.

### 3.4 `test_models.py`
- [ ] Review and update tests in `test/test_models.py`.
- [ ] Adjust tests to reflect current model structure and predictions.

### 3.5 `test_visualize.py`
- [ ] Review and update tests in `test/test_visualize.py`.
- [ ] Adjust tests to reflect current visualization methods.

## 4. Additional Tasks
Ensure other repository files are up-to-date and consistent with the current version.

### 4.1 `pyproject.toml`
- [ ] Review `pyproject.toml` for any outdated configurations.
- [ ] Update dependencies and settings if necessary.

### 4.2 `README.md`
- [ ] Review and update the `README.md`.
- [ ] Ensure that the README reflects the current state and purpose of the repository.

### 4.3 `reports/` Directory
- [ ] Check the `reports/` directory for any outdated reports.
- [ ] Update or remove old reports if necessary.

### 4.4 `models/` Directory
- [ ] Review the `models/` directory for any outdated models.
- [ ] Update model documentation and any related files.

### 4.5 `data/` Directory
- [ ] Review the `data/` directory for any unnecessary or outdated files.
- [ ] Ensure that the data files are correctly documented and relevant.

### 4.6 Repository Cleanup
- [ ] Review the entire repository for any other outdated files or code.
- [ ] Remove or archive unnecessary files.

## Final steps
- [ ] Check if the company allows the publishing of all CSV dataset files
- [ ] Rerun all notebooks and hyperparam tune ALL models with 1000 iterations
- [ ] Remove similarity matrix