# kanw append-only logging gia ST1 scores se CSV file.
import csv
from datetime import datetime
from pathlib import Path

from .io_utils import LOG_PATH


HEADER = ["timestamp", "notebook", "model", "features", "st1", "f1_haz", "f1_prod_cond"]


def grapse_eval(notebook, model, features, parts, log_path=LOG_PATH):
    # anoigei/dimiourgei to eval_log.csv kai prosthetei mia grammh apotelesmatwn.
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # grafw header grafetai mono tin prwti fora pou dimiourgeitai to csv.
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(HEADER)
        # Krataw kai ta 3 noumera gia na mporw meta na ta balw sto report.
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            notebook,
            model,
            features,
            f"{parts['st1']:.4f}",
            f"{parts['f1_hazard']:.4f}",
            f"{parts['f1_product_cond']:.4f}",
        ])
