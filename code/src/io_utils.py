# I/O helpers for data, submissions, paths.
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# fixed paths, so no script has to hardcode folders on its own.
RAW_DIR = ROOT.parent / "data" / "raw"
if not RAW_DIR.exists():
    RAW_DIR = ROOT / "data" / "raw"
PRED_DIR = ROOT / "results" / "predictions"
FIG_DIR = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "models"
LOG_PATH = ROOT / "results" / "eval_log.csv"


def load_split(name):
    # loads one of the train/valid/test csv files.
    # train and valid carry an extra index column from the original dataset.
    path = RAW_DIR / f"{name}.csv"
    # train/valid have an extra index column, test does not
    if name in ("train", "valid"):
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path)


def load_data():
    # central helper for all experiments: always returns train, valid, test.
    return load_split("train"), load_split("valid"), load_split("test")


def write_submission(test, hazard_pred, product_pred, path):
    # builds a submission csv with the exact columns Kaggle expects.
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sub = pd.DataFrame({
        "id": test["id"].to_numpy(),
        "hazard-category": hazard_pred,
        "product-category": product_pred,
    })

    # basic sanity checks before the file is written, so a bad csv never gets uploaded.
    if len(sub) != len(test):
        raise ValueError(f"submission has {len(sub)} rows, expected {len(test)}")
    if sub.isna().any().any():
        raise ValueError("submission contains missing predictions")

    sub.to_csv(out, index=False)
    return out
