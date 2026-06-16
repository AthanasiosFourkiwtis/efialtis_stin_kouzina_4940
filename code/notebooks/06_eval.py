# eval/validation helper gia ton elegxo twn peiramatwn. To script diavazei to
# eval_log.csv, deixnei ta kalytera validation runs kai elegxei oti ta submission
# cSV exoun swstes sthles, swsto plithos grammwn kai kanena NaN. To trexw meta
# apo peiramata gia na min anevei lathos arxeio sto Kaggle.

import sys
from pathlib import Path

import pandas as pd

# prosthetw to project root sto sys.path gia na vrei to src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io_utils import LOG_PATH, PRED_DIR, fortwse_split


def main():
    # deixnw ta kalitera validation runs apo to csv log
    # log einai append-only CSV pou kathe notebook prosthetei ta apotelesmata
    # tou kathe peiramatos tou. Kathe row exei ts/notebook/model/features/scores.
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        # st1 mporei na einai string apo to CSV. kanw to numeric gia sort.
        # errors="coerce": an iparxei lathos, ginetai NaN anti gia crash.
        log["st1"] = pd.to_numeric(log["st1"], errors="coerce")
        print("=== best validation runs ===")
        # top-15 alphavithia kat' ST1 fthinontas
        print(log.sort_values("st1", ascending=False).head(15).to_string(index=False))
    else:
        # an akoma den exoun trexw peiramata, to log den iparxei
        print(f"missing log: {LOG_PATH}")

    # elegxw ola ta submission csv gia rows/columns/NaN
    # test set exei sigkekrimeno arithmo rows (997) kai aut o pou ekana
    # upload sto Kaggle anamenei. An kati den tairiazei -> reject.
    test = fortwse_split("test")
    # orizw exact columns pou perimeneti to Kaggle
    expected_cols = ["id", "hazard-category", "product-category"]
    print("\n=== submissions ===")
    # kanw iterate sigkrithika gia consistent ordering sto output
    for path in sorted(PRED_DIR.glob("submission*.csv")):
        sub = pd.read_csv(path)
        status = "ok"
        # elegxw oti ta columns einai akrivws ayta pou perimeneti to Kaggle
        if sub.columns.tolist() != expected_cols:
            status = f"bad columns {sub.columns.tolist()}"
        # elegxw oti idio plhthos rows me to test
        elif len(sub) != len(test):
            status = f"bad rows {len(sub)} expected {len(test)}"
        # elegxw oti kanena NaN (would fail to upload to Kaggle)
        elif sub.isna().any().any():
            status = "contains missing values"
        # kanw padded output gia eukola alignment
        print(f"{path.name:45s} rows={len(sub):4d} {status}")


if __name__ == "__main__":
    main()
