import unittest

import pandas as pd

from src.wsb_pipeline import add_labels


class LabelBoundaryTests(unittest.TestCase):
    def test_drops_posts_without_a_following_trading_day(self):
        posts = pd.DataFrame({
            "timestamp": pd.to_datetime(["2020-09-02 10:00"]),
            "text": ["final-day post"],
        })
        prices = pd.DataFrame(
            {"Open": [100.0, 105.0]},
            index=pd.to_datetime(["2020-09-01", "2020-09-02"]),
        )

        labelled = add_labels(posts, prices)

        self.assertTrue(labelled.empty)
        self.assertIn("ret", labelled.columns)
        self.assertIn("up", labelled.columns)


if __name__ == "__main__":
    unittest.main()
