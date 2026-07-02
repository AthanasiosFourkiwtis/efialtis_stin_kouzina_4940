# 09_embeddings.py
# Extends the Efialtis Stin Kouzina pipeline with MiniLM sentence embeddings.
# TF-IDF is good at capturing words and n-grams, but it has no idea that two words
# with similar meaning can be different tokens. The embeddings supply that
# semantic signal and get joined onto the sparse TF-IDF features. The hazard
# and product pipelines stay the same as before, with the OOF hazard feature
# guarding against leakage.
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

# put the project root on sys.path so the src/ module resolves
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.scoring import score_st1
from src.io_utils import MODEL_DIR, PRED_DIR, load_data, write_submission
from src.preprocess import build_text

# configuration
# MiniLM-L6-v2: 384-dimensional embeddings, 22M params, runs comfortably on CPU
# Alternative: all-mpnet-base-v2 is bigger (768) and more reliable,
# but takes 3-5x as long to run.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH = 64                          # 64 samples/batch during encode
CACHE_DIR = ROOT / "results" / "cache"    # where the .npy files are kept

# embedding scale: how much weight the embeddings carry in the stacked feature space.
# scale zero means no embeddings (same as 08_efialtis_kouzina)
# scale one means same magnitude as TF-IDF (drowns out the sparse signal)
# scale 0.7 came out optimal in the sweep - a "U-shape" with the worst at 0 and at 1
EMB_SCALES_TO_TRY = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
HAZ_W = 1.5                                # weight of the one-hot hazard feature
C_HAZARD = 1.0                             # LinearSVC C for the hazard model
C_PRODUCT = 2.0                            # LinearSVC C for product (from the 05_sota tuning)
N_SPLITS = 5                               # 5-fold OOF
RANDOM_STATE = 42                          # for reproducibility


def fit_tfidf(train_text):
    # fits TF-IDF (word + char_wb) over a list of train texts.
    # word vectorizer: captures words and bigrams (e.g. "undeclared milk")
    word = TfidfVectorizer(
        ngram_range=(1, 2),         # 1-grams and 2-grams
        min_df=3,                   # a token enters the vocabulary only if >=3 docs contain it
        max_df=0.95,                # washes out tokens that show up too often
        sublinear_tf=True,          # 1+log(tf) instead of raw - smoother
        max_features=250_000,       # ceiling on the feature count
    )
    # char vectorizer: captures subword/character ngrams for robustness
    # char_wb = "within word boundary" - the ngram never crosses words
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),         # character 3- to 5-grams
        min_df=3,
        sublinear_tf=True,
        max_features=250_000,
    )
    # fit_transform on train ONLY (we don't want leakage from valid/test)
    Xw = word.fit_transform(train_text)
    Xc = char.fit_transform(train_text)
    # hstack: joins the 2 sparse arrays side by side (word features + char features)
    # tocsr(): the CSR format LinearSVC wants
    return word, char, hstack([Xw, Xc]).tocsr()


def apply_tfidf(word, char, text):
    # same as fit_tfidf but transform only (no fit) - for valid/test.
    return hstack([word.transform(text), char.transform(text)]).tocsr()


def cached_embeddings(name, texts):
    # embed once with MiniLM, save the .npy, reuse it next time.
    #
    # embedding the 5082 train docs takes ~25 minutes on CPU. Re-running it
    # every time would be prohibitive, so the .npy is kept
    # in the cache.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"emb_minilm_{name}.npy"

    # on a cache hit, load and return (~1 sec)
    if cache_path.exists():
        print(f"  [cache hit] {name}: {cache_path.name}")
        return np.load(cache_path)

    # otherwise compute it. The import lives here, inside the function, so
    # the script doesn't blow up for someone without the package installed
    from sentence_transformers import SentenceTransformer
    print(f"  [computing] {name}: {len(texts)} texts (this can take a while)")
    model = SentenceTransformer(EMBED_MODEL)
    # normalize_embeddings=True: L2 normalization so every vector
    # has norm = 1. That way dot product = cosine similarity.
    arr = model.encode(
        list(texts), batch_size=EMBED_BATCH,
        show_progress_bar=False, normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)  # float32 instead of float64 - half the space, same accuracy
    np.save(cache_path, arr)
    print(f"  [saved] {cache_path.name} shape={arr.shape}")
    return arr


def stack_features(tfidf_sparse, emb_dense, emb_scale=1.0):
    # joins sparse TF-IDF + dense embeddings into a single sparse matrix.
    #
    # the embeddings are dense (virtually every value non-zero). Entering at a large
    # magnitude (scale=1.0) they dominate the sparse TF-IDF -- which has small
    # non-zero values at only a few indices -- and drown it out. Hence the scale.
    #
    # scale=0   -> TF-IDF only (baseline comparison)
    # scale=0.7 -> came out best in the sweep
    # scale=1.0 -> the embeddings take over and the score drops
    if emb_scale == 0.0:
        return tfidf_sparse
    # embeddings * scale, then sparse-ize (csr_matrix wraps a dense array)
    # float32 for consistency
    emb_sparse = csr_matrix((emb_dense * emb_scale).astype(np.float32))
    return hstack([tfidf_sparse, emb_sparse]).tocsr()


def one_hot_hazard(pred_labels, all_labels, weight=HAZ_W):
    # sparse one-hot encoding of the predicted hazard: an (N x K) sparse matrix
    # where K = number of hazard classes. Every row has one nonzero value in
    # its prediction's column. The weight controls how much pull the hazard signal
    # has (HAZ_W=1.5 looked good in the sweep).
    # dict: label -> column index
    idx = {h: i for i, h in enumerate(all_labels)}
    # row indices (0, 1, ..., N-1)
    rows = np.arange(len(pred_labels))
    # column indices (which class was predicted for each row)
    cols = np.array([idx[h] for h in pred_labels])
    # values - all identical, set to weight
    data = np.full(len(pred_labels), weight, dtype=np.float32)
    # csr_matrix via the (data, (rows, cols)) syntax: sparse construction
    return csr_matrix((data, (rows, cols)), shape=(len(pred_labels), len(all_labels)))


def oof_hazard_predictions(X, y, n_splits=N_SPLITS):
    # 5-fold Out-Of-Fold hazard predictions over the train set.
    #
    # same rationale as in 08: the product should take the hazard as a feature,
    # but if the hazard scores ~100% on train (having seen it) the product
    # learns to trust it blindly. With OOF every train row gets its prediction
    # from a model that never saw it (the other 4 folds) - a realistic ~94%
    # signal, just like at test time.
    # StratifiedKFold: preserves the class distribution in every fold.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.empty(len(y), dtype=y.dtype)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        # train on 80% and predict the 20%
        clf = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(X[va_idx])
        print(f"  fold {fold + 1}/{n_splits} done")
    return oof


def write_log(label, parts):
    # append one line to results/eval_log.csv with the experiment's scores.
    log_path = ROOT / "results/eval_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "notebook", "model", "features", "st1", "f1_haz", "f1_prod_cond"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            "09_embeddings", label, "tfidf+minilm+oof_hazard",
            f"{parts['st1']:.4f}",
            f"{parts['f1_hazard']:.4f}",
            f"{parts['f1_product_cond']:.4f}",
        ])


def try_scale(scale, x_tr_tf, x_va_tf, emb_tr, emb_va, y_haz_tr, y_prod_tr, y_haz_va, y_prod_va, haz_labels):
    # full hazard+product evaluation for this particular scale value.
    #
    # returns a parts dict with ST1, hazard F1, conditional product F1 + counts.
    # also returns the OOF hazard preds for possible reuse.
    # stack TF-IDF + scaled embeddings
    x_tr = stack_features(x_tr_tf, emb_tr, emb_scale=scale)
    x_va = stack_features(x_va_tf, emb_va, emb_scale=scale)

    # hazard model: TF-IDF + embeddings -> hazard label
    haz_clf = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
    haz_clf.fit(x_tr, y_haz_tr)
    pred_haz_va = haz_clf.predict(x_va)

    # OOF predictions on train, to build a realistic hazard feature
    pred_haz_tr_oof = oof_hazard_predictions(x_tr, y_haz_tr)

    # product features = TF-IDF + embeddings + one-hot hazard
    # for train, reuse the OOF predictions (realistic noise)
    # for valid, reuse the predictions of the full hazard model
    H_tr = one_hot_hazard(pred_haz_tr_oof, haz_labels)
    H_va = one_hot_hazard(pred_haz_va, haz_labels)
    x_tr_p = hstack([x_tr, H_tr]).tocsr()
    x_va_p = hstack([x_va, H_va]).tocsr()

    # product model over the augmented feature space
    prod_clf = LinearSVC(C=C_PRODUCT, class_weight="balanced", max_iter=5000)
    prod_clf.fit(x_tr_p, y_prod_tr)
    pred_prod_va = prod_clf.predict(x_va_p)

    # official ST1 score
    parts = score_st1(y_haz_va, pred_haz_va, y_prod_va, pred_prod_va, return_components=True)
    return parts, pred_haz_tr_oof


def main():
    print("=== Efialtis Stin Kouzina v2: TF-IDF + MiniLM embeddings (scale sweep) ===")
    # load all the splits
    train, valid, test = load_data()
    print(f"shapes: train={train.shape} valid={valid.shape} test={test.shape}")

    # the text the model will see - title + text + metadata tokens
    x_tr_text = build_text(train).tolist()
    x_va_text = build_text(valid).tolist()
    x_te_text = build_text(test).tolist()

    # labels for the two classification tasks
    y_haz_tr = train["hazard-category"].to_numpy()
    y_prod_tr = train["product-category"].to_numpy()
    y_haz_va = valid["hazard-category"].to_numpy()
    y_prod_va = valid["product-category"].to_numpy()
    # hazard labels in a fixed order for consistent one-hot encoding
    haz_labels = sorted(np.unique(y_haz_tr).tolist())

    # embeddings (cached; no recompute when the .npy files exist)
    print(f"\n[1] sentence embeddings ({EMBED_MODEL})")
    emb_tr = cached_embeddings("train", x_tr_text)
    emb_va = cached_embeddings("valid", x_va_text)
    emb_te = cached_embeddings("test", x_te_text)
    print(f"  embedding dim: {emb_tr.shape[1]}")

    # TF-IDF on train only
    print("\n[2] TF-IDF (fit on train)")
    word, char, x_tr_tf = fit_tfidf(x_tr_text)
    x_va_tf = apply_tfidf(word, char, x_va_text)
    print(f"  tfidf features: {x_tr_tf.shape[1]}")

    # sweep over the embedding scale to find the optimum
    print("\n[3] sweep emb_scale (scale=0 means no embeddings)")
    best = None
    for scale in EMB_SCALES_TO_TRY:
        print(f"\n  -> emb_scale={scale}")
        parts, _ = try_scale(
            scale, x_tr_tf, x_va_tf, emb_tr, emb_va,
            y_haz_tr, y_prod_tr, y_haz_va, y_prod_va, haz_labels,
        )
        print(
            f"     st1={parts['st1']:.4f} "
            f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f}"
        )
        write_log(f"efialtis_kouzina_v2_s{scale:g}", parts)
        # keep the best scale for the final refit
        if best is None or parts["st1"] > best["parts"]["st1"]:
            best = {"scale": scale, "parts": parts}

    print(f"\nbest scale = {best['scale']} with st1={best['parts']['st1']:.4f}")

    # refit on train+valid and predict on test. The best scale is now known
    # from the sweep, so the TF-IDF is rebuilt (its vocabulary now also covering
    # the valid data) and the product trains on a larger data set before
    # being applied to the test.
    print("\n[4] refit on train+valid and predict on test")
    best_scale = best["scale"]
    # concat: stack train and valid on top of each other into one bigger DataFrame
    full = pd.concat([train, valid], axis=0, ignore_index=True)
    x_full_text = build_text(full).tolist()
    y_haz_full = full["hazard-category"].to_numpy()
    y_prod_full = full["product-category"].to_numpy()

    # TF-IDF refit over all the labelled data (train+valid)
    word2, char2, x_full_tf = fit_tfidf(x_full_text)
    x_te_tf = apply_tfidf(word2, char2, x_te_text)

    # no recomputing of embeddings - just concatenate train+valid
    emb_full = np.concatenate([emb_tr, emb_va], axis=0)
    x_full = stack_features(x_full_tf, emb_full, emb_scale=best_scale)
    x_te = stack_features(x_te_tf, emb_te, emb_scale=best_scale)

    # final hazard model over train+valid
    haz_final = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
    haz_final.fit(x_full, y_haz_full)
    pred_haz_te = haz_final.predict(x_te)

    # OOF hazard over the full set, same logic as train-only
    print("  OOF hazard over train+valid")
    pred_haz_full_oof = oof_hazard_predictions(x_full, y_haz_full)

    # product features with OOF hazard over the full set + predicted hazard on test
    H_full = one_hot_hazard(pred_haz_full_oof, haz_labels)
    H_te = one_hot_hazard(pred_haz_te, haz_labels)
    x_full_p = hstack([x_full, H_full]).tocsr()
    x_te_p = hstack([x_te, H_te]).tocsr()

    # final product model
    prod_final = LinearSVC(C=C_PRODUCT, class_weight="balanced", max_iter=5000)
    prod_final.fit(x_full_p, y_prod_full)
    pred_prod_te = prod_final.predict(x_te_p)

    # write the Kaggle CSV
    sub_path = write_submission(
        test, pred_haz_te, pred_prod_te,
        PRED_DIR / "submission_efialtis_kouzina_v2.csv",
    )
    print(f"  submission -> {sub_path}")

    # also save the model as an artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "final_efialtis_kouzina_v2.joblib"
    dump(
        {
            "tfidf_word": word2,
            "tfidf_char": char2,
            "hazard_model": haz_final,
            "product_model": prod_final,
            "hazard_labels": haz_labels,
            "embedding_model_name": EMBED_MODEL,
            "embedding_scale": best_scale,
            "hazard_feature_weight": HAZ_W,
        },
        model_path,
    )
    print(f"  model -> {model_path}")


if __name__ == "__main__":
    main()
