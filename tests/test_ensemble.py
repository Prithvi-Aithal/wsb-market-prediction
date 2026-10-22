import unittest

import numpy as np
import pandas as pd

from src.models import ShrinkageEnsemble, tune_ensemble
from src.wsb_pipeline import COUNTS, relative_features, relative_rank


def toy_data(n=400, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(int) if signal else rng.integers(0, 2, n)
    return np.hstack([X, np.ones((n, 1))]), y   # last column = bias, as in the pipeline


class ShrinkageEnsembleTests(unittest.TestCase):
    def test_lam_zero_predicts_the_training_base_rate(self):
        X, y = toy_data()
        model = ShrinkageEnsemble(lam=0.0).fit(X, y)
        np.testing.assert_allclose(model.predict_proba(X), y.mean())

    def test_lam_one_learns_a_real_signal(self):
        X, y = toy_data()
        model = ShrinkageEnsemble(lam=1.0).fit(X[:300], y[:300])
        self.assertGreater((model.predict(X[300:]) == y[300:]).mean(), 0.9)

    def test_tuning_trusts_features_only_when_they_predict_forward(self):
        X, y = toy_data(signal=True)
        self.assertGreater(tune_ensemble(X, y, n_train=320, n_configs=2)["lam"], 0.5)
        X, y = toy_data(signal=False)
        self.assertLessEqual(tune_ensemble(X, y, n_train=320, n_configs=2)["lam"], 0.25)


class RelativeFeatureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.posts = pd.DataFrame({
            "timestamp": pd.date_range("2021-01-01", periods=30, freq="h"),
            **{c: rng.integers(0, 1000, 30) for c in COUNTS},
        })

    def test_ranks_use_only_earlier_posts(self):
        early = relative_features(self.posts.iloc[:20], window=10)
        full = relative_features(self.posts, window=10)
        pd.testing.assert_frame_equal(early, full.iloc[:20])

    def test_single_post_rank_matches_rolling_rank(self):
        full = relative_features(self.posts, window=10)
        hist = self.posts["post_score"].values[:-1]
        self.assertAlmostEqual(relative_rank(hist, self.posts["post_score"].iloc[-1], window=10),
                               full["post_score"].iloc[-1])


if __name__ == "__main__":
    unittest.main()
