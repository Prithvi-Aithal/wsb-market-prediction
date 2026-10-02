import unittest

import numpy as np

from src.models import NewtonLogisticRegression


class NewtonLogisticRegressionTests(unittest.TestCase):
    def test_converges_and_predicts_linearly_separable_data(self):
        X = np.array([
            [-2.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -2.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        model = NewtonLogisticRegression().fit(X, y)

        self.assertTrue(model.converged_)
        self.assertLess(model.deltas_[-1], model.tol)
        np.testing.assert_array_equal(model.predict(X), y)


if __name__ == "__main__":
    unittest.main()
