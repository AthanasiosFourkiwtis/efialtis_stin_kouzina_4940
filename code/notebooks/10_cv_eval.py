# 10_cv_eval.py
# This script does a more trustworthy evaluation via k-fold cross-validation on
# the pooled train and valid set. A single validation split can mislead the
# choice of hyperparameters, so here the mean and standard deviation are measured
# across folds. Each fold runs the whole Efialtis Stin Kouzina pipeline leak-free,
# with TF-IDF fit only on the fold-train and the OOF hazard feature for the product model.
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
from src.scoring import score_st1
from src.io_utils import load_data
from src.preprocess import build_text
from src.models import fit_tfidf, apply_tfidf

# baseline config: same as 08_efialtis_kouzina
N_SPLITS = 5
RANDOM_STATE = 42
C_HAZARD = 1.0
C_PRODUCT = 2.0
HAZ_W = 1.5            # weight of the hazard one-hot feature
INNER_OOF_SPLITS = 5  # for the OOF hazard inside each fold


def one_hot_hazard(pred_labels, all_labels, weight):
    # sparse one-hot of the predicted hazard (same as 08_efialtis_kouzina).
    idx = {h: i for i, h in enumerate(all_labels)}
    rows = np.arange(len(pred_labels))
    cols = np.array([idx[h] for h in pred_labels])
    data = np.full(len(pred_labels), weight, dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(pred_labels), len(all_labels)))


def oof_hazard_predictions(X, y, c_hazard, n_splits=INNER_OOF_SPLITS, seed=RANDOM_STATE):
    # inner k-fold OOF hazard predictions over the fold-train.
    #
    # same rationale as 08: the product must see a realistic (~94%) hazard
    # signal, not the 99% a full-fit hazard model would give. That way the
    # product learns to cope with a noisy hazard, as it will have to at test time.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=y.dtype)
    for tr_idx, va_idx in skf.split(X, y):
        clf = LinearSVC(C=c_hazard, class_weight="balanced", max_iter=5000)
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(X[va_idx])
    return oof


def run_fold(df_tr, df_va, *, c_hazard, c_product, haz_w, include_metadata):
    # run the WHOLE Efialtis Stin Kouzina pipeline on one fold. Zero leakage:
    # TF-IDF fits ONLY on df_tr.
    x_tr_text = build_text(df_tr, include_metadata=include_metadata)
    x_va_text = build_text(df_va, include_metadata=include_metadata)

    y_haz_tr = df_tr["hazard-category"].to_numpy()
    y_prod_tr = df_tr["product-category"].to_numpy()
    y_haz_va = df_va["hazard-category"].to_numpy()
    y_prod_va = df_va["product-category"].to_numpy()

    # TF-IDF: fit on the fold-train only
    word, char, x_tr = fit_tfidf(x_tr_text)
    x_va = apply_tfidf(word, char, x_va_text)

    # hazard model + predictions on the fold-valid
    haz_clf = LinearSVC(C=c_hazard, class_weight="balanced", max_iter=5000)
    haz_clf.fit(x_tr, y_haz_tr)
    pred_haz_va = haz_clf.predict(x_va)
    haz_labels = sorted(np.unique(y_haz_tr).tolist())

    # OOF hazard over the fold-train -> one-hot feature
    pred_haz_tr_oof = oof_hazard_predictions(x_tr, y_haz_tr, c_hazard)
    H_tr = one_hot_hazard(pred_haz_tr_oof, haz_labels, haz_w)
    H_va = one_hot_hazard(pred_haz_va, haz_labels, haz_w)
    x_tr_p = hstack([x_tr, H_tr]).tocsr()
    x_va_p = hstack([x_va, H_va]).tocsr()

    # product model over the augmented feature space
    prod_clf = LinearSVC(C=c_product, class_weight="balanced", max_iter=5000)
    prod_clf.fit(x_tr_p, y_prod_tr)
    pred_prod_va = prod_clf.predict(x_va_p)

    return score_st1(y_haz_va, pred_haz_va, y_prod_va, pred_prod_va, return_components=True)


def cv_efialtis_kouzina(df, *, n_splits, c_hazard, c_product, haz_w, include_metadata, seed=RANDOM_STATE):
    # k-fold CV of Efialtis Stin Kouzina. Returns a list of per-fold component dicts.
    #
    # stratified on hazard-category (like the OOF in 08) so every fold carries
    # a representative distribution of the hazard classes.
    y_strat = df["hazard-category"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_parts = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df)), y_strat)):
        df_tr = df.iloc[tr_idx]
        df_va = df.iloc[va_idx]
        parts = run_fold(
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


def summarize(fold_parts):
    # mean +/- std across the folds for each metric.
    keys = ["st1", "f1_hazard", "f1_product_cond"]
    out = {}
    for k in keys:
        vals = np.array([p[k] for p in fold_parts])
        out[k + "_mean"] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out


def write_log(label, summary, n_splits, config):
    # write the CV result to results/cv_eval_log.csv.
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


def try_config(df, label, n_splits, *, c_hazard, c_product, haz_w, include_metadata):
    # runs + prints + logs one config.
    cfg = f"meta={include_metadata} c_haz={c_hazard} c_prod={c_product} haz_w={haz_w}"
    print(f"\n=== {label} ({cfg}) ===")
    fold_parts = cv_efialtis_kouzina(
        df, n_splits=n_splits,
        c_hazard=c_hazard, c_product=c_product,
        haz_w=haz_w, include_metadata=include_metadata,
    )
    summary = summarize(fold_parts)
    print(
        f"  --> ST1 = {summary['st1_mean']:.4f} +/- {summary['st1_std']:.4f}  "
        f"(haz {summary['f1_hazard_mean']:.4f}, prod {summary['f1_product_cond_mean']:.4f})"
    )
    write_log(label, summary, n_splits, cfg)
    return summary


def main():
    parser = argparse.ArgumentParser(description="k-fold CV for the Efialtis Stin Kouzina pipeline")
    parser.add_argument("--splits", type=int, default=N_SPLITS, help="number of folds (default 5)")
    parser.add_argument("--grid", action="store_true", help="compare multiple configs")
    args = parser.parse_args()

    print("=== CV evaluation: Efialtis Stin Kouzina ===")
    train, valid, test = load_data()
    # pool train+valid: exactly the data the final model trains on for the submission,
    # and ~5.6k rows give a far more stable estimate than the 565 of valid alone.
    import pandas as pd
    df = pd.concat([train, valid], axis=0, ignore_index=True)
    print(f"pooled train+valid: {df.shape[0]} rows, {args.splits}-fold CV")

    if not args.grid:
        # measure only the baseline (same as 08_efialtis_kouzina, which gave the good Kaggle score)
        try_config(
            df, "baseline", args.splits,
            c_hazard=C_HAZARD, c_product=C_PRODUCT, haz_w=HAZ_W, include_metadata=True,
        )
        print("\nTip: --grid compares configs. The CV mean is the number")
        print("to trust, NOT the single-split valid score.")
        return

    # grid mode compares a handful of configs to see what actually helps.
    results = []
    results.append(("baseline", try_config(
        df, "baseline", args.splits,
        c_hazard=C_HAZARD, c_product=C_PRODUCT, haz_w=HAZ_W, include_metadata=True)))
    results.append(("no_metadata", try_config(
        df, "no_metadata", args.splits,
        c_hazard=C_HAZARD, c_product=C_PRODUCT, haz_w=HAZ_W, include_metadata=False)))
    results.append(("c_prod_1", try_config(
        df, "c_prod_1", args.splits,
        c_hazard=C_HAZARD, c_product=1.0, haz_w=HAZ_W, include_metadata=True)))
    results.append(("c_prod_4", try_config(
        df, "c_prod_4", args.splits,
        c_hazard=C_HAZARD, c_product=4.0, haz_w=HAZ_W, include_metadata=True)))
    results.append(("haz_w_3", try_config(
        df, "haz_w_3", args.splits,
        c_hazard=C_HAZARD, c_product=C_PRODUCT, haz_w=3.0, include_metadata=True)))

    print("\n=== COMPARISON (sorted by ST1 mean) ===")
    for label, s in sorted(results, key=lambda r: -r[1]["st1_mean"]):
        print(f"  {label:15s} ST1 = {s['st1_mean']:.4f} +/- {s['st1_std']:.4f}")
    print("\nA config is genuinely better ONLY if its mean stands out from the")
    print("baseline by more than ~1 std. Anything less is noise.")


if __name__ == "__main__":
    main()
