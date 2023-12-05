# pylint: disable=missing-module-docstring
import unittest

import pandas as pd
from scipy.stats import loguniform, uniform
from skelm import ELMRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    median_absolute_error,
)
from sklearn.model_selection import RandomizedSearchCV

from ltm.models import best_scores, hyperparam_search


class TestModels(unittest.TestCase):  # pylint: disable=missing-class-docstring
    def setUp(self):
        (  # pylint: disable=unbalanced-tuple-unpacking
            self.X,  # pylint: disable=invalid-name
            self.y,
        ) = make_regression(n_samples=100, n_features=10, random_state=42)
        self.search_space = {
            ELMRegressor(): {
                "alpha": loguniform(1e-8, 1e5),
                "include_original_features": [True, False],
                "n_neurons": loguniform(1, 100 - 1),
                "ufunc": ["tanh", "sigm", "relu", "lin"],
                "density": uniform(0.01, 0.99),
            },
        }
        self.scoring = {
            "median_absolute_error": make_scorer(
                median_absolute_error, greater_is_better=False
            ),
            "mean_squared_error": make_scorer(
                mean_squared_error, greater_is_better=False
            ),
        }
        self.refit = "mean_squared_error"

    def test_hyperparam_search(self):
        results = hyperparam_search(
            self.X,
            self.y,
            self.search_space,
            self.scoring,
            self.refit,
            kfold_from_endmembers=False,
            random_state=42,
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RandomizedSearchCV)
        self.assertIsInstance(results[0].best_estimator_, ELMRegressor)

    def test_reproducible_seed(self):
        results1 = hyperparam_search(
            self.X,
            self.y,
            self.search_space,
            self.scoring,
            self.refit,
            kfold_from_endmembers=False,
            random_state=42,
        )
        results2 = hyperparam_search(
            self.X,
            self.y,
            self.search_space,
            self.scoring,
            self.refit,
            kfold_from_endmembers=False,
            random_state=42,
        )
        self.assertEqual(
            results1[0].best_estimator_.get_params(),
            results2[0].best_estimator_.get_params(),
        )

    def test_best_scores(self):
        results = hyperparam_search(
            self.X,
            self.y,
            self.search_space,
            self.scoring,
            self.refit,
            kfold_from_endmembers=False,
            random_state=42,
        )
        scores = best_scores(results, self.scoring)
        self.assertIsInstance(scores, pd.DataFrame)
        self.assertEqual(len(scores.columns), 2)
        self.assertIn("mean_squared_error", scores)


if __name__ == "__main__":
    unittest.main()
