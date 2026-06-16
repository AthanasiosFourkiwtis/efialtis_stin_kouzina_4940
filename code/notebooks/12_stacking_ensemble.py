# Final stacking / soft-voting pipeline. It combines the TF-IDF Efialtis
# model with a MiniLM embedding model; the two make different mistakes.
# With `--submit`, it refits on the pooled labelled set and writes
# results/predictions/submission_stacking.csv.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.scoring import metrhse_st1
from src.io_utils import fortwse_dedomena, grapse_submission, PRED_DIR, MODEL_DIR
from src.preprocess import ftiakse_keimeno
from src.models import ekpaideuse_tfidf, efarmose_tfidf

N_SPLITS = 5
RANDOM_STATE = 42
C_HAZARD = 1.0
C_PRODUCT = 2.0
HAZ_W = 1.5
INNER_OOF_SPLITS = 5
TUNE_FRACTION = 0.2
WEIGHT_GRID = np.linspace(0.0, 1.0, 11)   # 0.0 = mono MiniLM ... 1.0 = mono TF-IDF
CACHE = ROOT / "results" / "cache"


def get_embeddings():
    emb_tr = np.load(CACHE / "emb_minilm_train.npy")
    emb_va = np.load(CACHE / "emb_minilm_valid.npy")
    emb_te = np.load(CACHE / "emb_minilm_test.npy")
    return np.vstack([emb_tr, emb_va]), emb_te


def one_hot_hazard(pred_labels, all_labels, weight=HAZ_W):
    idx = {h: i for i, h in enumerate(all_labels)}
    rows = np.arange(len(pred_labels))
    cols = np.array([idx[h] for h in pred_labels])
    data = np.full(len(pred_labels), weight, dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(pred_labels), len(all_labels)))


def oof_hazard_predictions(X, y, c_hazard, n_splits=INNER_OOF_SPLITS, seed=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=object)
    for tr_idx, va_idx in skf.split(X, y):
        clf = LinearSVC(C=c_hazard, class_weight="balanced", max_iter=5000)
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(X[va_idx])
    return oof


def ekpaideuse_svc(X, y, c):
    # class_weight='balanced' matters because the metric is macro-F1.
    # Calibration is avoided because rare classes can have only 2-3 rows.
    clf = LinearSVC(C=c, class_weight="balanced", max_iter=5000)
    clf.fit(X, y)
    return clf


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def full_proba(model, X, master_classes):
    # Softmax over decision_function gives pseudo-probabilities for voting.
    # Missing fold classes get probability 0 to avoid class-order misalignment.
    dec = model.decision_function(X)
    if dec.ndim == 1:  # binary case: 1-D -> [-dec, dec] gia classes_[0], classes_[1]
        dec = np.vstack([-dec, dec]).T
    p = _softmax(dec)
    out = np.zeros((X.shape[0], len(master_classes)), dtype=np.float64)
    pos = {c: j for j, c in enumerate(master_classes)}
    for j, c in enumerate(model.classes_):
        out[:, pos[c]] = p[:, j]
    return out


def trekse_fold(df_tr, df_va, emb_tr, emb_va, haz_classes, prod_classes, seed):
    # Inner tune split chooses ensemble weights without touching fold validation.
    y_haz_all = df_tr["hazard-category"].to_numpy()
    fit_idx, tune_idx = train_test_split(
        np.arange(len(df_tr)), test_size=TUNE_FRACTION,
        stratify=y_haz_all, random_state=seed,
    )
    df_fit, df_tune = df_tr.iloc[fit_idx], df_tr.iloc[tune_idx]
    emb_fit, emb_tune = emb_tr[fit_idx], emb_tr[tune_idx]

    y_haz_fit = df_fit["hazard-category"].to_numpy()
    y_prod_fit = df_fit["product-category"].to_numpy()
    y_haz_tune = df_tune["hazard-category"].to_numpy()
    y_prod_tune = df_tune["product-category"].to_numpy()
    y_haz_va = df_va["hazard-category"].to_numpy()
    y_prod_va = df_va["product-category"].to_numpy()

    # ===== BASE A: Efialtis Stin Kouzina TF-IDF =====
    word, char, x_fit = ekpaideuse_tfidf(ftiakse_keimeno(df_fit, include_metadata=True))
    x_tune = efarmose_tfidf(word, char, ftiakse_keimeno(df_tune, include_metadata=True))
    x_va = efarmose_tfidf(word, char, ftiakse_keimeno(df_va, include_metadata=True))

    hazA = ekpaideuse_svc(x_fit, y_haz_fit, C_HAZARD)
    pHazA_tune = full_proba(hazA, x_tune, haz_classes)
    pHazA_va = full_proba(hazA, x_va, haz_classes)

    # OOF hazard feature avoids training the product model on perfect hazard labels.
    fit_haz_labels = sorted(np.unique(y_haz_fit).tolist())
    oof = oof_hazard_predictions(x_fit, y_haz_fit, C_HAZARD)
    x_fit_p = hstack([x_fit, one_hot_hazard(oof, fit_haz_labels)]).tocsr()
    prodA = ekpaideuse_svc(x_fit_p, y_prod_fit, C_PRODUCT)

    def productA_proba(x, pHazA):
        haz_pred = np.array(haz_classes)[pHazA.argmax(1)]
        x_p = hstack([x, one_hot_hazard(haz_pred, fit_haz_labels)]).tocsr()
        return full_proba(prodA, x_p, prod_classes)

    pProdA_tune = productA_proba(x_tune, pHazA_tune)
    pProdA_va = productA_proba(x_va, pHazA_va)

    # ===== BASE B: MiniLM embeddings =====
    hazB = ekpaideuse_svc(emb_fit, y_haz_fit, C_HAZARD)
    prodB = ekpaideuse_svc(emb_fit, y_prod_fit, C_PRODUCT)
    pHazB_tune = full_proba(hazB, emb_tune, haz_classes)
    pHazB_va = full_proba(hazB, emb_va, haz_classes)
    pProdB_tune = full_proba(prodB, emb_tune, prod_classes)
    pProdB_va = full_proba(prodB, emb_va, prod_classes)

    # Tune weights on the inner split, optimizing the official ST1.
    haz_arr = np.array(haz_classes)
    prod_arr = np.array(prod_classes)
    best = (-1.0, 1.0, 1.0)
    for wh in WEIGHT_GRID:
        haz_pred = haz_arr[(wh * pHazA_tune + (1 - wh) * pHazB_tune).argmax(1)]
        for wp in WEIGHT_GRID:
            prod_pred = prod_arr[(wp * pProdA_tune + (1 - wp) * pProdB_tune).argmax(1)]
            s = metrhse_st1(y_haz_tune, haz_pred, y_prod_tune, prod_pred)
            if s > best[0]:
                best = (s, wh, wp)
    _, wh, wp = best

    # Baseline A: plain TF-IDF Efialtis.
    parts_A = metrhse_st1(
        y_haz_va, haz_arr[pHazA_va.argmax(1)],
        y_prod_va, prod_arr[pProdA_va.argmax(1)], return_components=True,
    )
    # Ensemble: tuned weighted average.
    haz_pred_ens = haz_arr[(wh * pHazA_va + (1 - wh) * pHazB_va).argmax(1)]
    prod_pred_ens = prod_arr[(wp * pProdA_va + (1 - wp) * pProdB_va).argmax(1)]
    parts_ens = metrhse_st1(y_haz_va, haz_pred_ens, y_prod_va, prod_pred_ens,
                          return_components=True)
    return parts_A, parts_ens, (wh, wp)


def synopsi(parts_list):
    keys = ["st1", "f1_hazard", "f1_product_cond"]
    return {k: (float(np.mean([p[k] for p in parts_list])),
                float(np.std([p[k] for p in parts_list]))) for k in keys}


def grapse_log(s_A, s_ens, n_splits):
    log_path = ROOT / "results" / "cv_eval_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "label", "n_splits", "config", "st1_mean",
                        "st1_std", "f1_haz_mean", "f1_haz_std", "f1_prod_mean", "f1_prod_std"])
        ts = datetime.now().isoformat(timespec="seconds")
        for label, s in [("stacking_base_A", s_A), ("stacking_ensemble", s_ens)]:
            w.writerow([ts, label, n_splits, "tfidf_efialtisKouzina + minilm",
                        f"{s['st1'][0]:.4f}", f"{s['st1'][1]:.4f}",
                        f"{s['f1_hazard'][0]:.4f}", f"{s['f1_hazard'][1]:.4f}",
                        f"{s['f1_product_cond'][0]:.4f}", f"{s['f1_product_cond'][1]:.4f}"])


def submission_path(df, test, emb_pooled, emb_test, haz_classes, prod_classes):
    # Tune weights on 20% of pooled data, then refit bases on 100% before test.
    print("\n[submit] base training + weight tuning...")
    y_haz_all = df["hazard-category"].to_numpy()
    fit_idx, tune_idx = train_test_split(
        np.arange(len(df)), test_size=TUNE_FRACTION,
        stratify=y_haz_all, random_state=RANDOM_STATE)

    haz_arr, prod_arr = np.array(haz_classes), np.array(prod_classes)

    def build_bases(rows_df, rows_emb):
        y_h = rows_df["hazard-category"].to_numpy()
        y_p = rows_df["product-category"].to_numpy()
        word, char, x = ekpaideuse_tfidf(ftiakse_keimeno(rows_df, include_metadata=True))
        hazA = ekpaideuse_svc(x, y_h, C_HAZARD)
        fit_haz_labels = sorted(np.unique(y_h).tolist())
        oof = oof_hazard_predictions(x, y_h, C_HAZARD)
        x_p = hstack([x, one_hot_hazard(oof, fit_haz_labels)]).tocsr()
        prodA = ekpaideuse_svc(x_p, y_p, C_PRODUCT)
        hazB = ekpaideuse_svc(rows_emb, y_h, C_HAZARD)
        prodB = ekpaideuse_svc(rows_emb, y_p, C_PRODUCT)
        return word, char, hazA, prodA, fit_haz_labels, hazB, prodB

    def predict_probs(bases, rows_df, rows_emb):
        word, char, hazA, prodA, fit_haz_labels, hazB, prodB = bases
        x = efarmose_tfidf(word, char, ftiakse_keimeno(rows_df, include_metadata=True))
        pHazA = full_proba(hazA, x, haz_classes)
        haz_pred = haz_arr[pHazA.argmax(1)]
        x_p = hstack([x, one_hot_hazard(haz_pred, fit_haz_labels)]).tocsr()
        pProdA = full_proba(prodA, x_p, prod_classes)
        pHazB = full_proba(hazB, rows_emb, haz_classes)
        pProdB = full_proba(prodB, rows_emb, prod_classes)
        return pHazA, pProdA, pHazB, pProdB

    bases_fit = build_bases(df.iloc[fit_idx], emb_pooled[fit_idx])
    pHazA_t, pProdA_t, pHazB_t, pProdB_t = predict_probs(bases_fit, df.iloc[tune_idx], emb_pooled[tune_idx])
    y_haz_t = df.iloc[tune_idx]["hazard-category"].to_numpy()
    y_prod_t = df.iloc[tune_idx]["product-category"].to_numpy()
    best = (-1.0, 1.0, 1.0)
    for wh in WEIGHT_GRID:
        haz_pred = haz_arr[(wh * pHazA_t + (1 - wh) * pHazB_t).argmax(1)]
        for wp in WEIGHT_GRID:
            prod_pred = prod_arr[(wp * pProdA_t + (1 - wp) * pProdB_t).argmax(1)]
            s = metrhse_st1(y_haz_t, haz_pred, y_prod_t, prod_pred)
            if s > best[0]:
                best = (s, wh, wp)
    _, wh, wp = best
    print(f"[submit] tuned weights: wh={wh:.2f} (TF-IDF share hazard), wp={wp:.2f} (product)")

    bases_full = build_bases(df, emb_pooled)
    pHazA, pProdA, pHazB, pProdB = predict_probs(bases_full, test, emb_test)
    haz_pred = haz_arr[(wh * pHazA + (1 - wh) * pHazB).argmax(1)]
    prod_pred = prod_arr[(wp * pProdA + (1 - wp) * pProdB).argmax(1)]
    out = grapse_submission(test, haz_pred, prod_pred, PRED_DIR / "submission_stacking.csv")
    print(f"[submit] submission -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Stacking ensemble: TF-IDF + MiniLM, CV-validated")
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--submit", action="store_true", help="grapse kai submission_stacking.csv")
    args = parser.parse_args()

    print("=== Stacking ensemble: Efialtis Stin Kouzina (TF-IDF) + MiniLM embeddings ===")
    train, valid, test = fortwse_dedomena()
    df = pd.concat([train, valid], axis=0, ignore_index=True)
    print(f"pooled train+valid: {df.shape[0]} rows | test: {test.shape[0]}")

    emb_pooled, emb_test = get_embeddings()
    print(f"  MiniLM embeddings: pooled {emb_pooled.shape}, test {emb_test.shape}")
    assert len(emb_pooled) == len(df), "embeddings/pooled mismatch!"
    assert len(emb_test) == len(test), "embeddings/test mismatch!"

    haz_classes = sorted(df["hazard-category"].unique().tolist())
    prod_classes = sorted(df["product-category"].unique().tolist())

    print(f"\n[cv] {args.splits}-fold evaluation (A mono vs ensemble)...")
    y_strat = df["hazard-category"].to_numpy()
    skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=RANDOM_STATE)
    A_parts, ens_parts = [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df)), y_strat)):
        pA, pE, (wh, wp) = trekse_fold(
            df.iloc[tr_idx], df.iloc[va_idx], emb_pooled[tr_idx], emb_pooled[va_idx],
            haz_classes, prod_classes, seed=RANDOM_STATE + fold)
        A_parts.append(pA)
        ens_parts.append(pE)
        print(f"  fold {fold + 1}/{args.splits}: A={pA['st1']:.4f}  "
              f"ensemble={pE['st1']:.4f}  delta={pE['st1'] - pA['st1']:+.4f}  "
              f"(wh={wh:.2f} wp={wp:.2f})")

    sA, sE = synopsi(A_parts), synopsi(ens_parts)
    print("\n=== APOTELESMA ===")
    print(f"  base A mono (Efialtis Stin Kouzina): ST1 = {sA['st1'][0]:.4f} +/- {sA['st1'][1]:.4f}")
    print(f"  ensemble (A + MiniLM):       ST1 = {sE['st1'][0]:.4f} +/- {sE['st1'][1]:.4f}")
    gain = sE["st1"][0] - sA["st1"][0]
    pooled_std = (sA["st1"][1] + sE["st1"][1]) / 2
    print(f"\n  mean gain = {gain:+.4f}   (pooled std ~ {pooled_std:.4f})")
    if gain > pooled_std:
        print("  -> sigouro kerdos (panw apo 1 std). Aksizei submission.")
    elif gain > 0:
        print("  -> Mikro kerdos mesa ston thoryvo. Oriako - dokimase --splits 10.")
    else:
        print("  -> Den voithaei to ensemble. Krata to sketo Efialtis Stin Kouzina.")
    grapse_log(sA, sE, args.splits)
    print("  (logged sto results/cv_eval_log.csv)")

    if args.submit:
        submission_path(df, test, emb_pooled, emb_test, haz_classes, prod_classes)


if __name__ == "__main__":
    main()
