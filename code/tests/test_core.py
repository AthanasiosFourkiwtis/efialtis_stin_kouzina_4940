from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.io_utils import write_submission
from src.preprocess import build_text, clean_text
from src.scoring import score_st1


# simple unit tests for the core helpers, not for the heavy ML experiments.
class CoreTests(unittest.TestCase):
    def test_normalize_text_collapses_whitespace_and_lowercases(self):
        # checks that preprocessing lowercases and cleans up whitespace.
        self.assertEqual(clean_text("  A\nB\tC  "), "a b c")

    def test_build_text_adds_metadata_tokens(self):
        # checks that country/year/month make it into the text representation.
        df = pd.DataFrame(
            {
                "country": ["us"],
                "year": [2026],
                "month": [5],
                "title": ["Recall"],
                "text": ["Listeria in ham"],
            }
        )
        text = build_text(df).iloc[0]
        self.assertIn("country_us", text)
        self.assertIn("year_2026", text)
        self.assertIn("recall listeria", text)

    def test_st1_score_product_is_conditioned_on_correct_hazard(self):
        # product F1 must be measured only where the hazard is correct.
        parts = score_st1(
            ["a", "b", "b"],
            ["a", "a", "b"],
            ["x", "y", "z"],
            ["x", "wrong", "wrong"],
            return_components=True,
        )
        self.assertGreater(parts["f1_hazard"], 0)
        self.assertEqual(parts["n_hazard_correct"], 2)
        self.assertLess(parts["f1_product_cond"], 1)

    def test_write_submission_validates_shape_and_columns(self):
        # checks that the submission has the right columns and the right size.
        test = pd.DataFrame({"id": [10, 11]})
        with tempfile.TemporaryDirectory() as tmp:
            path = write_submission(test, ["h1", "h2"], ["p1", "p2"], Path(tmp) / "sub.csv")
            sub = pd.read_csv(path)
            self.assertEqual(sub.columns.tolist(), ["id", "hazard-category", "product-category"])
            self.assertEqual(len(sub), 2)


if __name__ == "__main__":
    unittest.main()
