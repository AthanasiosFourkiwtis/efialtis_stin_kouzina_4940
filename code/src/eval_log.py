# append-only logging of ST1 scores into a CSV file.
import csv
from datetime import datetime
from pathlib import Path

from .io_utils import LOG_PATH


HEADER = ["timestamp", "notebook", "model", "features", "st1", "f1_haz", "f1_prod_cond"]


def log_eval(notebook, model, features, parts, log_path=LOG_PATH):
    # opens/creates eval_log.csv and appends one row of results.
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # the header is written only the first time the csv is created.
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(HEADER)
        # All 3 numbers are kept so they can go into the report later.
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            notebook,
            model,
            features,
            f"{parts['st1']:.4f}",
            f"{parts['f1_hazard']:.4f}",
            f"{parts['f1_product_cond']:.4f}",
        ])
