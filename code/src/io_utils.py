# I/O helpers gia data, submissions, paths.
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# stathera paths gia na min grafw hardcoded folders mesa se kathe script.
RAW_DIR = ROOT.parent / "data" / "raw"
if not RAW_DIR.exists():
    RAW_DIR = ROOT / "data" / "raw"
PRED_DIR = ROOT / "results" / "predictions"
FIG_DIR = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "models"
LOG_PATH = ROOT / "results" / "eval_log.csv"


def fortwse_split(name):
    # fortwnei ena apo ta train/valid/test csv.
    # train kai valid exoun extra index column apo to arxiko dataset.
    path = RAW_DIR / f"{name}.csv"
    # train/valid exoun extra index column, test oxi
    if name in ("train", "valid"):
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path)


def fortwse_dedomena():
    # kentriko helper gia ola ta experiments: gyrnaei panta train, valid, test.
    return fortwse_split("train"), fortwse_split("valid"), fortwse_split("test")


def grapse_submission(test, hazard_pred, product_pred, path):
    # ftiaxnei submission csv me ta akrivi columns pou perimenei to Kaggle.
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sub = pd.DataFrame({
        "id": test["id"].to_numpy(),
        "hazard-category": hazard_pred,
        "product-category": product_pred,
    })

    # kanw basikous elegxoi prin grafthei to arxeio, gia na min anevei lathos csv.
    if len(sub) != len(test):
        raise ValueError(f"submission has {len(sub)} rows, expected {len(test)}")
    if sub.isna().any().any():
        raise ValueError("submission contains missing predictions")

    sub.to_csv(out, index=False)
    return out
