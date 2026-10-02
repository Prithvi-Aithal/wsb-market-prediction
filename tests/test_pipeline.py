import unittest

import pandas as pd

from src.wsb_pipeline import add_labels, clean_text


class TextCleaningTests(unittest.TestCase):
    def test_removes_handles_urls_and_single_character_tokens(self):
        self.assertEqual(
            clean_text("GME!!! @Sagar https://example.com A to the moon!"),
            "gme to the moon",
        )


class LabelConstructionTests(unittest.TestCase):
    def setUp(self):
        self.posts = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2020-09-01 10:00",
                "2020-09-02 14:00",
                "2020-09-05 10:00",
            ]),
            "text": ["early post", "after cutoff", "weekend post"],
        })
        self.prices = pd.DataFrame(
            {"Open": [100.0, 110.0, 99.0, 98.0, 96.0]},
            index=pd.to_datetime([
                "2020-09-01",
                "2020-09-02",
                "2020-09-03",
                "2020-09-07",
                "2020-09-08",
            ]),
        )

    def test_paper_labels_use_post_date_and_next_trading_open(self):
        labelled = add_labels(self.posts, self.prices, mode="paper")

        self.assertEqual(
            list(labelled["day"]),
            list(pd.to_datetime(["2020-09-01", "2020-09-02", "2020-09-07"])),
        )
        self.assertAlmostEqual(labelled.iloc[0]["ret"], 0.1)
        self.assertEqual(list(labelled["up"]), [1, 0, 0])

    def test_tradable_labels_apply_cutoff_and_skip_weekends(self):
        labelled = add_labels(self.posts, self.prices, mode="tradable")

        self.assertEqual(
            list(labelled["day"]),
            list(pd.to_datetime(["2020-09-01", "2020-09-03", "2020-09-07"])),
        )
        self.assertAlmostEqual(labelled.iloc[1]["ret"], 98.0 / 99.0 - 1.0)


if __name__ == "__main__":
    unittest.main()
