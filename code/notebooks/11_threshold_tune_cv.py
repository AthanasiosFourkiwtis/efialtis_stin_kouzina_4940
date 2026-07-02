# 11_threshold_tune_cv.py
# Tries per-class threshold tuning for the hazard model. The error analysis
# shows the smaller hazard classes suffer from low recall, so offsets are
# added to the LinearSVC's decision_function before the argmax, giving the
# rare classes more of a chance. The offsets are selected on a held-out tune
# slice inside each fold and evaluated on the fold-valid, so the experiment
# stays leak-free and compares cleanly against the baseline argmax.
import sys
import argparse
from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.scoring import score_st1
from src.io_utils import load_data
from src.preprocess import build_text
from src.models import fit_tfidf, apply_tfidf

N_SPLITS = 5
RANDOM_STATE = 42
C_HAZARD = 1.0
C_PRODUCT = 2.0
HAZ_W = 1.5
INNER_OOF_SPLITS = 5
TUNE_FRACTION = 0.2          # how much of the fold-train is held out for offset tuning
OFFSET_GRID = np.linspace(-1.5, 1.5, 31)   # the candidate offsets per class
TUNE_PASSES = 3              # coordinate-ascent passes over the classes


def one_hot_hazard(pred_labels, all_labels, weight):
    # sparse one-hot of the predicted hazard (same as 08_efialtis_kouzina).
    idx = {h: i for i, h in enumerate(all_labels)}
    rows = np.arange(len(pred_labels))
    cols = np.array([idx[h] for h in pred_labels])
    data = np.full(len(pred_labels), weight, dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(pred_labels), len(all_labels)))


def oof_hazard_predictions(X, y, c_hazard, n_splits=INNER_OOF_SPLITS, seed=RANDOM_STATE):
    # inner k-fold OOF hazard predictions (same rationale as 08_efialtis_kouzina).
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=object)
    for tr_idx, va_idx in skf.split(X, y):
        clf = LinearSVC(C=c_hazard, class_weight="balanced", max_iter=5000)
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(X[va_idx])
    return oof


def tune_offsets(decision, y_idx, n_classes, grid=OFFSET_GRID, passes=TUNE_PASSES):
    # coordinate ascent: finds the per-class offsets that maximize
    # macro-F1 over the held-out TUNE set.
    #
    # decision : (n_samples, n_classes) from LinearSVC.decision_function
    # y_idx    : (n_samples,) true labels as indices into [0, n_classes)
    # returns  : (n_classes,) offsets - added to the decision before the argmax.
    offsets = np.zeros(n_classes)

    def macro(off):
        pred = (decision + off).argmax(axis=1)
        return f1_score(y_idx, pred, average="macro", zero_division=0)

    best = macro(offsets)
    for _ in range(passes):
        for c in range(n_classes):
            trial = offsets.copy()
            best_o = offsets[c]
            for g in grid:
                trial[c] = g
                s = macro(trial)
                if s > best:
                    best = s
                    best_o = g
            offsets[c] = best_o
    return offsets


def run_fold(df_tr, df_va, *, c_hazard, c_product, haz_w, seed):
    # runs Efialtis Stin Kouzina on one fold, TWICE (baseline vs tuned hazard).
    # returns (parts_baseline, parts_tuned).
    # split the fold-train into FIT + TUNE (stratified on hazard)
    y_haz_all = df_tr["hazard-category"].to_numpy()
    fit_idx, tune_idx = train_test_split(
        np.arange(len(df_tr)), test_size=TUNE_FRACTION,
        stratify=y_haz_all, random_state=seed,
    )
    df_fit = df_tr.iloc[fit_idx]
    df_tune = df_tr.iloc[tune_idx]

    x_fit_text = build_text(df_fit, include_metadata=True)
    x_tune_text = build_text(df_tune, include_metadata=True)
    x_va_text = build_text(df_va, include_metadata=True)

    y_haz_fit = df_fit["hazard-category"].to_numpy()
    y_prod_fit = df_fit["product-category"].to_numpy()
    y_haz_tune = df_tune["hazard-category"].to_numpy()
    y_haz_va = df_va["hazard-category"].to_numpy()
    y_prod_va = df_va["product-category"].to_numpy()

    # TF-IDF + hazard model (fit ONLY on the FIT subset)
    word, char, x_fit = fit_tfidf(x_fit_text)
    x_tune = apply_tfidf(word, char, x_tune_text)
    x_va = apply_tfidf(word, char, x_va_text)

    haz_clf = LinearSVC(C=c_hazard, class_weight="balanced", max_iter=5000)
    haz_clf.fit(x_fit, y_haz_fit)
    classes = haz_clf.classes_                       # the order of the decision_function columns
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    # tune the offsets over the held-out TUNE set
    dec_tune = haz_clf.decision_function(x_tune)
    # keep only the tune rows whose label exists in classes (edge case:
    # a rare class that never landed in the FIT subset)
    keep = np.array([y in cls_to_idx for y in y_haz_tune])
    y_tune_idx = np.array([cls_to_idx[y] for y in y_haz_tune[keep]])
    offsets = tune_offsets(dec_tune[keep], y_tune_idx, len(classes))

    # hazard predictions on the fold-VALID: baseline vs tuned
    dec_va = haz_clf.decision_function(x_va)
    pred_haz_va_base = classes[dec_va.argmax(axis=1)]
    pred_haz_va_tuned = classes[(dec_va + offsets).argmax(axis=1)]

    # product pipeline: IDENTICAL in both cases
    haz_labels = sorted(np.unique(y_haz_fit).tolist())
    oof = oof_hazard_predictions(x_fit, y_haz_fit, c_hazard)
    H_fit = one_hot_hazard(oof, haz_labels, haz_w)
    x_fit_p = hstack([x_fit, H_fit]).tocsr()

    prod_clf = LinearSVC(C=c_product, class_weight="balanced", max_iter=5000)
    prod_clf.fit(x_fit_p, y_prod_fit)

    def product_st1(pred_haz_va):
        # the product's validation side uses whichever hazard prediction applies
        H_va = one_hot_hazard(pred_haz_va, haz_labels, haz_w)
        x_va_p = hstack([x_va, H_va]).tocsr()
        pred_prod_va = prod_clf.predict(x_va_p)
        return score_st1(y_haz_va, pred_haz_va, y_prod_va, pred_prod_va, return_components=True)

    parts_base = product_st1(pred_haz_va_base)
    parts_tuned = product_st1(pred_haz_va_tuned)
    return parts_base, parts_tuned, offsets, classes


def summarize(fold_parts):
    keys = ["st1", "f1_hazard", "f1_product_cond"]
    return {k: (float(np.mean([p[k] for p in fold_parts])),
                float(np.std([p[k] for p in fold_parts]))) for k in keys}


def write_log(summary_base, summary_tuned, n_splits):
    log_path = ROOT / "results" / "cv_eval_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "label", "n_splits", "config",
                        "st1_mean", "st1_std", "f1_haz_mean", "f1_haz_std",
                        "f1_prod_mean", "f1_prod_std"])
        ts = datetime.now().isoformat(timespec="seconds")
        for label, s in [("thresh_baseline", summary_base), ("thresh_tuned", summary_tuned)]:
            w.writerow([ts, label, n_splits, "hazard offset tuning",
                        f"{s['st1'][0]:.4f}", f"{s['st1'][1]:.4f}",
                        f"{s['f1_hazard'][0]:.4f}", f"{s['f1_hazard'][1]:.4f}",
                        f"{s['f1_product_cond'][0]:.4f}", f"{s['f1_product_cond'][1]:.4f}"])


def main():
    parser = argparse.ArgumentParser(description="CV per-class threshold tuning for the hazard")
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    print("=== CV: per-class threshold tuning on the HAZARD ===")
    train, valid, test = load_data()
    df = pd.concat([train, valid], axis=0, ignore_index=True)
    print(f"pooled train+valid: {df.shape[0]} rows, {args.splits}-fold CV\n")

    y_strat = df["hazard-category"].to_numpy()
    skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=RANDOM_STATE)

    base_parts, tuned_parts = [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df)), y_strat)):
        pb, pt, offsets, classes = run_fold(
            df.iloc[tr_idx], df.iloc[va_idx],
            c_hazard=C_HAZARD, c_product=C_PRODUCT, haz_w=HAZ_W,
            seed=RANDOM_STATE + fold,
        )
        base_parts.append(pb)
        tuned_parts.append(pt)
        delta = pt["st1"] - pb["st1"]
        print(f"  fold {fold + 1}/{args.splits}: "
              f"baseline={pb['st1']:.4f}  tuned={pt['st1']:.4f}  delta={delta:+.4f}")

    sb = summarize(base_parts)
    st = summarize(tuned_parts)
    print("\n=== RESULT ===")
    print(f"  baseline (argmax):        ST1 = {sb['st1'][0]:.4f} +/- {sb['st1'][1]:.4f}  "
          f"(haz {sb['f1_hazard'][0]:.4f})")
    print(f"  tuned (per-class offset): ST1 = {st['st1'][0]:.4f} +/- {st['st1'][1]:.4f}  "
          f"(haz {st['f1_hazard'][0]:.4f})")
    gain = st["st1"][0] - sb["st1"][0]
    pooled_std = (sb["st1"][1] + st["st1"][1]) / 2
    print(f"\n  mean gain = {gain:+.4f}   (pooled std ~ {pooled_std:.4f})")
    if gain > pooled_std:
        print("  -> solid gain (over 1 std). Worth folding into 08_efialtis_kouzina.")
    elif gain > 0:
        print("  -> Small gain but within the noise. Borderline - try --splits 10.")
    else:
        print("  -> No help. Threshold tuning is not the way.")

    write_log(sb, st, args.splits)
    print("\nLogged to results/cv_eval_log.csv")


if __name__ == "__main__":
    main()
