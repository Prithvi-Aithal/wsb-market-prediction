import unittest

import numpy as np
import pandas as pd

from src.wsb_pipeline import FEATURES, chrono_split, log_scale


class ChronologicalSplitTests(unittest.TestCase):
    def test_sorts_by_timestamp_and_keeps_future_rows_in_test(self):
        timestamps = pd.date_range("2020-01-01", periods=10)
        data = pd.DataFrame({"timestamp": timestamps[::-1]})

        train, validation, test = chrono_split(data, fracs=(0.6, 0.2, 0.2))

        self.assertEqual((len(train), len(validation), len(test)), (6, 2, 2))
        self.assertLess(train["timestamp"].max(), validation["timestamp"].min())
        self.assertLess(validation["timestamp"].max(), test["timestamp"].min())
        self.assertEqual(
            list(test["timestamp"]),
            list(pd.date_range("2020-01-09", periods=2)),
        )


class FeatureScalingTests(unittest.TestCase):
    def test_log_transforms_clipped_counts_and_preserves_other_features(self):
        row = pd.DataFrame([{
            "post_score": -1,
            "total_comms_num": 3,
            "len_of_post": 7,
            "positive_sentiment": 0.2,
            "neutral_sentiment": 0.5,
            "negative_sentiment": 0.3,
            "bias": 1.0,
        }])

        scaled = log_scale(row)

        np.testing.assert_allclose(
            scaled[0],
            [0.0, np.log1p(3), np.log1p(7), 0.2, 0.5, 0.3, 1.0],
        )
        self.assertEqual(scaled.shape[1], len(FEATURES))


if __name__ == "__main__":
    unittest.main()
