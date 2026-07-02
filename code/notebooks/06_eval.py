# eval/validation helper for sanity-checking the experiments. The script reads
# eval_log.csv, shows the best validation runs, and verifies that the submission
# CSVs have the right columns, the right row count, and no NaNs. Run it after
# experiments so a broken file never gets uploaded to Kaggle.

import sys
from pathlib import Path

import pandas as pd

# add the project root to sys.path so src/ resolves
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io_utils import LOG_PATH, PRED_DIR, load_split


def main():
    # show the best validation runs from the csv log
    # the log is an append-only CSV where every notebook adds the results
    # of each of its experiments. Every row holds ts/notebook/model/features/scores.
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        # st1 may come out of the CSV as a string. Convert to numeric for sorting.
        # errors="coerce": anything malformed becomes NaN instead of crashing.
        log["st1"] = pd.to_numeric(log["st1"], errors="coerce")
        print("=== best validation runs ===")
        # top 15, sorted by ST1 descending
        print(log.sort_values("st1", ascending=False).head(15).to_string(index=False))
    else:
        # if no experiments have run yet, the log won't exist
        print(f"missing log: {LOG_PATH}")

    # check every submission csv for rows/columns/NaN
    # the test set has a fixed row count (997), which is what the Kaggle
    # upload expects. Anything that doesn't match -> reject.
    test = load_split("test")
    # the exact columns Kaggle expects
    expected_cols = ["id", "hazard-category", "product-category"]
    print("\n=== submissions ===")
    # iterate in sorted order for consistent output
    for path in sorted(PRED_DIR.glob("submission*.csv")):
        sub = pd.read_csv(path)
        status = "ok"
        # check the columns are exactly what Kaggle expects
        if sub.columns.tolist() != expected_cols:
            status = f"bad columns {sub.columns.tolist()}"
        # check the row count matches the test set
        elif len(sub) != len(test):
            status = f"bad rows {len(sub)} expected {len(test)}"
        # check there are no NaNs (the Kaggle upload would fail)
        elif sub.isna().any().any():
            status = "contains missing values"
        # padded output for easy alignment
        print(f"{path.name:45s} rows={len(sub):4d} {status}")


if __name__ == "__main__":
    main()
