# 08_efialtis_kouzina.py
# The project's first conditional model. A hazard LinearSVC is trained first
# on TF-IDF features, and the predicted hazard then enters the product model
# as a one-hot feature. So the product never learns to trust an unrealistically
# good hazard signal, the train set uses out-of-fold predictions.
# This matches the official ST1, which scores the product only when the hazard
# was identified correctly.
import sys
from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.scoring import score_st1
from src.io_utils import MODEL_DIR, PRED_DIR, load_data, write_submission
from src.preprocess import build_text

HAZ_W = 1.5         # weight of the hazard one-hot, comparable in magnitude to a TF-IDF feature
N_SPLITS = 5
RANDOM_STATE = 42
C_HAZARD = 1.0
C_PRODUCT = 2.0


def fit_tfidf(train_text):
    # fits and returns the TF-IDF (word + char_wb) bundle + train features.
    # word vectorizer: captures vocabulary (1- and 2-grams)
    word = TfidfVectorizer(
        ngram_range=(1, 2),     # word 1-grams and 2-grams
        min_df=3,                # enters the vocab only if it appears in >=3 docs
        max_df=0.95,             # dropped if it appears in 95%+ of docs (stopwords)
        sublinear_tf=True,       # 1+log(tf) instead of raw counts - smoother
        max_features=250_000,    # ceiling on the vocab size
    )
    # char vectorizer: captures subword patterns - helps with typos/brands
    char = TfidfVectorizer(
        analyzer="char_wb",      # within-word boundary characters
        ngram_range=(3, 5),      # character 3- to 5-grams
        min_df=3,
        sublinear_tf=True,
        max_features=250_000,
    )
    # fit_transform on train ONLY (so nothing leaks in from valid/test)
    Xw = word.fit_transform(train_text)
    Xc = char.fit_transform(train_text)
    # hstack the 2 sparse arrays side by side into a single CSR matrix
    return word, char, hstack([Xw, Xc]).tocsr()


def apply_tfidf(word, char, text):
    # same as fit_tfidf but transform only (no fit) - for the valid/test sets.
    return hstack([word.transform(text), char.transform(text)]).tocsr()


def one_hot_hazard(pred_labels, all_labels, weight=HAZ_W):
    # sparse one-hot encoding of the predicted hazard: an (N x K) sparse matrix
    # where N = row count and K = number of hazard classes. Every row has ONE
    # nonzero value, in the predicted class's column, equal to weight. weight=1.5
    # came out well in the sweep -- big enough to matter, not so big that it
    # drowns out the rest of the feature space.
    # dict: label string -> column index
    idx = {h: i for i, h in enumerate(all_labels)}
    # row indices: 0, 1, 2, ..., N-1
    rows = np.arange(len(pred_labels))
    # column indices: which class was predicted for each row
    cols = np.array([idx[h] for h in pred_labels])
    # values: all identical, set to weight
    data = np.full(len(pred_labels), weight, dtype=np.float32)
    # csr_matrix via the triplet syntax (data, (rows, cols))
    return csr_matrix((data, (rows, cols)), shape=(len(pred_labels), len(all_labels)))


def oof_hazard_predictions(X, y, n_splits=N_SPLITS):
    # 5-fold Out-Of-Fold hazard predictions over the train set.
    #
    # why not the predictions of the full-train hazard model? That one has seen ALL
    # the train rows, so on train it scores ~99% (overfit). Training the
    # product on such a perfect hazard teaches it to trust it blindly -- but on
    # test the hazard drops to ~93% and the product gets thrown off.
    #
    # The OOF fix: every train row belongs to one fold and gets its prediction from
    # a model trained on the other 4 folds (which never saw that row). This way all
    # the train rows receive a realistic ~94% accurate signal, just like at test time.
    # StratifiedKFold: keeps the class distribution proportional in every fold.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.empty(len(y), dtype=y.dtype)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        # train on 80% (4 folds)
        clf = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
        clf.fit(X[tr_idx], y[tr_idx])
        # predict on the 20% (the 1 held-out fold)
        oof[va_idx] = clf.predict(X[va_idx])
        print(f"  fold {fold + 1}/{n_splits} done")
    return oof


def write_log(label, parts):
    log_path = ROOT / "results/eval_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "notebook", "model", "features", "st1", "f1_haz", "f1_prod_cond"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            "08_efialtis_kouzina", label, "tfidf+oof_hazard_feat",
            f"{parts['st1']:.4f}",
            f"{parts['f1_hazard']:.4f}",
            f"{parts['f1_product_cond']:.4f}",
        ])


def main():
    print("=== Efialtis Stin Kouzina: conditional product head ===")
    train, valid, test = load_data()
    print(f"shapes: train={train.shape} valid={valid.shape} test={test.shape}")

    x_tr_text = build_text(train)
    x_va_text = build_text(valid)
    x_te_text = build_text(test)

    y_haz_tr = train["hazard-category"].to_numpy()
    y_prod_tr = train["product-category"].to_numpy()
    y_haz_va = valid["hazard-category"].to_numpy()
    y_prod_va = valid["product-category"].to_numpy()

    # TF-IDF + the hazard model
    print("\n[1] TF-IDF + hazard LinearSVC")
    word, char, x_tr = fit_tfidf(x_tr_text)
    x_va = apply_tfidf(word, char, x_va_text)
    print(f"  tfidf features: {x_tr.shape[1]}")

    haz_clf = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
    haz_clf.fit(x_tr, y_haz_tr)
    pred_haz_va = haz_clf.predict(x_va)
    haz_labels = sorted(np.unique(y_haz_tr).tolist())
    print(f"  hazard labels: {len(haz_labels)}")

    # OOF hazard over the train set
    print("\n[2] OOF hazard predictions over train (5-fold)")
    pred_haz_tr_oof = oof_hazard_predictions(x_tr, y_haz_tr)
    print(f"  OOF accuracy: {(pred_haz_tr_oof == y_haz_tr).mean():.4f}")

    # build the product model with the hazard feature
    print("\n[3] product LinearSVC with one-hot hazard feature")
    H_tr = one_hot_hazard(pred_haz_tr_oof, haz_labels)
    H_va = one_hot_hazard(pred_haz_va, haz_labels)
    x_tr_p = hstack([x_tr, H_tr]).tocsr()
    x_va_p = hstack([x_va, H_va]).tocsr()
    print(f"  augmented shape: {x_tr_p.shape}")

    prod_clf = LinearSVC(C=C_PRODUCT, class_weight="balanced", max_iter=5000)
    prod_clf.fit(x_tr_p, y_prod_tr)
    pred_prod_va = prod_clf.predict(x_va_p)

    parts = score_st1(y_haz_va, pred_haz_va, y_prod_va, pred_prod_va, return_components=True)
    print(
        f"\nEfialtis Stin Kouzina valid: st1={parts['st1']:.4f} "
        f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f} "
        f"(haz_correct {parts['n_hazard_correct']}/{parts['n_total']})"
    )
    write_log("efialtis_kouzina", parts)

    # final refit on train+valid
    print("\n[4] refit on train+valid and predict on test")
    full = pd.concat([train, valid], axis=0, ignore_index=True)
    x_full_text = build_text(full)
    y_haz_full = full["hazard-category"].to_numpy()
    y_prod_full = full["product-category"].to_numpy()

    word2, char2, x_full = fit_tfidf(x_full_text)
    x_te = apply_tfidf(word2, char2, x_te_text)

    haz_final = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
    haz_final.fit(x_full, y_haz_full)
    pred_haz_te = haz_final.predict(x_te)

    print("  OOF hazard over train+valid")
    pred_haz_full_oof = oof_hazard_predictions(x_full, y_haz_full)

    H_full = one_hot_hazard(pred_haz_full_oof, haz_labels)
    H_te = one_hot_hazard(pred_haz_te, haz_labels)
    x_full_p = hstack([x_full, H_full]).tocsr()
    x_te_p = hstack([x_te, H_te]).tocsr()

    prod_final = LinearSVC(C=C_PRODUCT, class_weight="balanced", max_iter=5000)
    prod_final.fit(x_full_p, y_prod_full)
    pred_prod_te = prod_final.predict(x_te_p)

    sub_path = write_submission(
        test, pred_haz_te, pred_prod_te,
        PRED_DIR / "submission_efialtis_kouzina.csv",
    )
    print(f"  submission -> {sub_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "final_efialtis_kouzina.joblib"
    dump(
        {
            "tfidf_word": word2,
            "tfidf_char": char2,
            "hazard_model": haz_final,
            "product_model": prod_final,
            "hazard_labels": haz_labels,
            "hazard_feature_weight": HAZ_W,
            "text_builder": "src.preprocess.build_text(include_metadata=True)",
        },
        model_path,
    )
    print(f"  model -> {model_path}")


if __name__ == "__main__":
    main()
