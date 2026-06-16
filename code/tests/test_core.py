from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.io_utils import grapse_submission
from src.preprocess import ftiakse_keimeno, katharise_keimeno
from src.scoring import metrhse_st1


# apla unit tests gia ta core helpers, oxi gia ta varia ML experiments.
class CoreTests(unittest.TestCase):
    def test_normalize_text_collapses_whitespace_and_lowercases(self):
        # elegxei oti to preprocessing kanei lowercase kai katharizei whitespace.
        self.assertEqual(katharise_keimeno("  A\nB\tC  "), "a b c")

    def test_build_text_adds_metadata_tokens(self):
        # elegxei oti country/year/month mpainei sto text representation.
        df = pd.DataFrame(
            {
                "country": ["us"],
                "year": [2026],
                "month": [5],
                "title": ["Recall"],
                "text": ["Listeria in ham"],
            }
        )
        text = ftiakse_keimeno(df).iloc[0]
        self.assertIn("country_us", text)
        self.assertIn("year_2026", text)
        self.assertIn("recall listeria", text)

    def test_st1_score_product_is_conditioned_on_correct_hazard(self):
        # product F1 prepei na metrietai mono ekei pou to hazard einai swsto.
        parts = metrhse_st1(
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
        # elegxei oti to submission exei ta swsta columns kai swsto megethos.
        test = pd.DataFrame({"id": [10, 11]})
        with tempfile.TemporaryDirectory() as tmp:
            path = grapse_submission(test, ["h1", "h2"], ["p1", "p2"], Path(tmp) / "sub.csv")
            sub = pd.read_csv(path)
            self.assertEqual(sub.columns.tolist(), ["id", "hazard-category", "product-category"])
            self.assertEqual(len(sub), 2)


if __name__ == "__main__":
    unittest.main()
