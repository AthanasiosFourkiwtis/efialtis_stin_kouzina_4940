# 10_cv_eval.py
# auto to script kanei pio aksiopisti axiologisi me k-fold cross-validation sto
# pooled train kai valid set. To single validation split borei na parasyrei tin
# epilogi hyperparameters, opote edw metraw mean kai standard deviation panw se
# folds. Se kathe fold trexei olo to Efialtis Stin Kouzina pipeline xwris leakage,
# me TF-IDF fit mono sto fold-train kai OOF hazard feature gia to product model.
import sys
import argparse
from pathlib import Path
from datetime import datetime
import csv

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.scoring import metrhse_st1
from src.io_utils import fortwse_dedomena
from src.preprocess import ftiakse_keimeno
from src.models import ekpaideuse_tfidf, efarmose_tfidf

# baseline config: idio me to 08_efialtis_kouzina
N_SPLITS = 5
RANDOM_STATE = 42
C_HAZARD = 1.0
C_PRODUCT = 2.0
HAZ_W = 1.5            # baros tou hazard one-hot feature
INNER_OOF_SPLITS = 5  # gia tin OOF hazard mesa se kathe fold


def one_hot_hazard(pred_labels, all_labels, weight):
    # ftiaxnw sparse one-hot tou predicted hazard (idio me 08_efialtis_kouzina).
    idx = {h: i for i, h in enumerate(all_labels)}
    rows = np.arange(len(pred_labels))
    cols = np.array([idx[h] for h in pred_labels])
    data = np.full(len(pred_labels), weight, dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(pred_labels), len(all_labels)))


def oof_hazard_predictions(X, y, c_hazard, n_splits=INNER_OOF_SPLITS, seed=RANDOM_STATE):
    # kanw inner k-fold OOF hazard predictions panw sto fold-train.
    #
    # idio skeptiko me to 08: to product prepei na dei realistiko (~94%) hazard
    # signal, oxi to 99% pou tha edine ena full-fit hazard model. Etsi to
    # product mathenei na antexei noisy hazard, opws tha symvei sto test.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=y.dtype)
    for tr_idx, va_idx in skf.split(X, y):
        clf = LinearSVC(C=c_hazard, class_weight="balanced", max_iter=5000)
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(X[va_idx])
    return oof


def trekse_fold(df_tr, df_va, *, c_hazard, c_product, haz_w, include_metadata):
    # trexw OLO to Efialtis Stin Kouzina pipeline se ena fold. Kanena leakage:
    # TF-IDF fittarei MONO sto df_tr.
    x_tr_text = ftiakse_keimeno(df_tr, include_metadata=include_metadata)
    x_va_text = ftiakse_keimeno(df_va, include_metadata=include_metadata)

    y_haz_tr = df_tr["hazard-category"].to_numpy()
    y_prod_tr = df_tr["product-category"].to_numpy()
    y_haz_va = df_va["hazard-category"].to_numpy()
    y_prod_va = df_va["product-category"].to_numpy()

    # xrhsimopoiw TF-IDF: fit mono sto fold-train
    word, char, x_tr = ekpaideuse_tfidf(x_tr_text)
    x_va = efarmose_tfidf(word, char, x_va_text)

    # ftiaxnw hazard model + predictions sto fold-valid
    haz_clf = LinearSVC(C=c_hazard, class_weight="balanced", max_iter=5000)
    haz_clf.fit(x_tr, y_haz_tr)
    pred_haz_va = haz_clf.predict(x_va)
    haz_labels = sorted(np.unique(y_haz_tr).tolist())

    # OOF hazard panw sto fold-train -> one-hot feature
    pred_haz_tr_oof = oof_hazard_predictions(x_tr, y_haz_tr, c_hazard)
    H_tr = one_hot_hazard(pred_haz_tr_oof, haz_labels, haz_w)
    H_va = one_hot_hazard(pred_haz_va, haz_labels, haz_w)
    x_tr_p = hstack([x_tr, H_tr]).tocsr()
    x_va_p = hstack([x_va, H_va]).tocsr()

    # ftiaxnw product model panw sto augmented feature space
    prod_clf = LinearSVC(C=c_product, class_weight="balanced", max_iter=5000)
    prod_clf.fit(x_tr_p, y_prod_tr)
    pred_prod_va = prod_clf.predict(x_va_p)

    return metrhse_st1(y_haz_va, pred_haz_va, y_prod_va, pred_prod_va, return_components=True)


def cv_efialtis_kouzina(df, *, n_splits, c_hazard, c_product, haz_w, include_metadata, seed=RANDOM_STATE):
    # k-fold CV tou Efialtis Stin Kouzina. Gyrnaei list apo per-fold component dicts.
    #
    # kanw stratify panw sto hazard-category (idio me to OOF tou 08) wste kathe fold
    # na exei antiproswpeutiki katanomi twn hazard classes.
    y_strat = df["hazard-category"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_parts = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df)), y_strat)):
        df_tr = df.iloc[tr_idx]
        df_va = df.iloc[va_idx]
        parts = trekse_fold(
            df_tr, df_va,
            c_hazard=c_hazard, c_product=c_product,
            haz_w=haz_w, include_metadata=include_metadata,
        )
        fold_parts.append(parts)
        print(
            f"  fold {fold + 1}/{n_splits}: st1={parts['st1']:.4f} "
            f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f}"
        )
    return fold_parts


def synopsi(fold_parts):
    # ypologizw mean +/- std panw sta folds gia kathe metric.
    keys = ["st1", "f1_hazard", "f1_product_cond"]
    out = {}
    for k in keys:
        vals = np.array([p[k] for p in fold_parts])
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out


def grapse_log(label, summary, n_splits, config):
    # grafw to CV apotelesma sto results/cv_eval_log.csv.
    log_path = ROOT / "results" / "cv_eval_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "label", "n_splits", "config",
                "st1_mean", "st1_std",
                "f1_haz_mean", "f1_haz_std",
                "f1_prod_mean", "f1_prod_std",
            ])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            label, n_splits, config,
            f"{summary['st1_mean']:.4f}", f"{summary['st1_std']:.4f}",
            f"{summary['f1_hazard_mean']:.4f}", f"{summary['f1_hazard_std']:.4f}",
            f"{summary['f1_product_cond_mean']:.4f}", f"{summary['f1_product_cond_std']:.4f}",
        ])


def dokimase_config(df, label, n_splits, *, c_hazard, c_product, haz_w, include_metadata):
    # trexw + typwnei + logarei mia config.
    cfg = f"meta={include_metadata} c_haz={c_hazard} c_prod={c_product} haz_w={haz_w}"
    print(f"\n=== {label} ({cfg}) ===")
    fold_parts = cv_efialtis_kouzina(
        df, n_splits=n_splits,
        c_hazard=c_hazard, c_product=c_product,
        haz_w=haz_w, include_metadata=include_metadata,
    )
    summary = synopsi(fold_parts)
    print(
        f"  --> ST1 = {summary['st1_mean']:.4f} +/- {summary['st1_std']:.4f}  "
        f"(haz {summary['f1_hazard_mean']:.4f}, prod {summary['f1_product_cond_mean']:.4f})"
    )
    grapse_log(label, summary, n_splits, cfg)
    return summary


def main():
    parser = argparse.ArgumentParser(description="k-fold CV gia to Efialtis Stin Kouzina pipeline")
    parser.add_argument("--splits", type=int, default=N_SPLITS, help="arithmos folds (default 5)")
    parser.add_argument("--grid", action="store_true", help="sygkrine pollaples configs")
    args = parser.parse_args()

    print("=== CV evaluation: Efialtis Stin Kouzina ===")
    train, valid, test = fortwse_dedomena()
    # pool train+valid: einai akrivws ta data pou exei to final model sto submission,
    # ~5.6k rows dinoun poly pio statheri ektimisi apo ta 565 tou valid mono.
    import pandas as pd
    df = pd.concat([train, valid], axis=0, ignore_index=True)
    print(f"pooled train+valid: {df.shape[0]} rows, {args.splits}-fold CV")

    if not args.grid:
        # metraw mono to baseline (idio me to 08_efialtis_kouzina pou edwse to kalo Kaggle score)
        dokimase_config(
            df, "baseline", args.splits,
            c_hazard=C_HAZARD, c_product=C_PRODUCT, haz_w=HAZ_W, include_metadata=True,
        )
        print("\nTip: --grid gia na sygkrineis configs. To CV mean einai to noumero")
        print("pou prepei na empisteuesai, OXI to single-split valid score.")
        return

    # sto grid mode sygkrinoume merika configs gia na doume ti pragmatika voithaei.
    results = []
    results.append(("baseline", dokimase_config(
        df, "baseline", args.splits,
        c_hazard=C_HAZARD, c_product=C_PRODUCT, haz_w=HAZ_W, include_metadata=True)))
    results.append(("no_metadata", dokimase_config(
        df, "no_metadata", args.splits,
        c_hazard=C_HAZARD, c_product=C_PRODUCT, haz_w=HAZ_W, include_metadata=False)))
    results.append(("c_prod_1", dokimase_config(
        df, "c_prod_1", args.splits,
        c_hazard=C_HAZARD, c_product=1.0, haz_w=HAZ_W, include_metadata=True)))
    results.append(("c_prod_4", dokimase_config(
        df, "c_prod_4", args.splits,
        c_hazard=C_HAZARD, c_product=4.0, haz_w=HAZ_W, include_metadata=True)))
    results.append(("haz_w_3", dokimase_config(
        df, "haz_w_3", args.splits,
        c_hazard=C_HAZARD, c_product=C_PRODUCT, haz_w=3.0, include_metadata=True)))

    print("\n=== SYGKRISI (taxinomimeno me ST1 mean) ===")
    for label, s in sorted(results, key=lambda r: -r[1]["st1_mean"]):
        print(f"  {label:15s} ST1 = {s['st1_mean']:.4f} +/- {s['st1_std']:.4f}")
    print("\nMia config einai pragmatika kaliteri MONO an to mean tis ksexorizei")
    print("perissotero apo ~1 std apo to baseline. Alliws einai thoryvos.")


if __name__ == "__main__":
    main()
