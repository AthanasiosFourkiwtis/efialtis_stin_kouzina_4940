# 09_embeddings.py
# Epektasi tou Efialtis Stin Kouzina pipeline me MiniLM sentence embeddings.
# To TF-IDF krataei kala lexeis kai n-grams, alla den katalavainei oti dyo lexeis
# me paromoio noima mporei na einai diaforetika tokens. Ta embeddings prosthetoun
# auto to semantic signal kai enwnontai me ta sparse TF-IDF features. To hazard
# kai to product pipeline menoun idia me prin, me OOF hazard feature wste na
# apofeugetai to leakage.
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

# bazoume to project root sto sys.path gia na vroume to src/ module
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.scoring import metrhse_st1
from src.io_utils import MODEL_DIR, PRED_DIR, fortwse_dedomena, grapse_submission
from src.preprocess import ftiakse_keimeno

# orizw configuration
# miniLM-L6-v2: 384-dimensional embeddings, 22M params, trexw aneta se CPU
# Alternative: all-mpnet-base-v2 einai pio megalo (768) kai pio aksiopisto
# alla pernw 3-5x ton xrono na trekei.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH = 64                          # 64 sample/batch sto encode
CACHE_DIR = ROOT / "results" / "cache"    # pou sozoume ta .npy

# embedding scale: poso varos na exoun ta embeddings sto stacked feature space.
# scale zero means xwris embeddings (san to 08_efialtis_kouzina)
# scale one means idia magnitude me to TF-IDF (pnigei to sparse signal)
# scale 0.7 means vrethke optimal sto sweep, "U-shape" me xeirotero sto 0 kai sto 1
EMB_SCALES_TO_TRY = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
HAZ_W = 1.5                                # varos tou one-hot hazard feature
C_HAZARD = 1.0                             # LinearSVC C gia hazard model
C_PRODUCT = 2.0                            # LinearSVC C gia product (apo 05_sota tuning)
N_SPLITS = 5                               # 5-fold OOF
RANDOM_STATE = 42                          # gia reproducibility


def ekpaideuse_tfidf(train_text):
    # kanw fit TF-IDF (word + char_wb) panw se ena list apo train texts.
    # xrhsimopoiw word vectorizer: pianei lexeis kai bigrams (px "undeclared milk")
    word = TfidfVectorizer(
        ngram_range=(1, 2),         # 1-grams kai 2-grams
        min_df=3,                   # token mphenei sto lexilogio mono an >=3 docs to exoun
        max_df=0.95,                # ksevgazei tokens pou emfanizontai poly syxna
        sublinear_tf=True,          # 1+log(tf) anti gia raw - smoother
        max_features=250_000,       # ceiling stous max features
    )
    # xrhsimopoiw char vectorizer: pianw subword/character ngrams gia robustness
    # char_wb = "within word boundary" - to ngram den diaperna lexeis
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),         # 3- ews 5-grammata haraktron
        min_df=3,
        sublinear_tf=True,
        max_features=250_000,
    )
    # kanw fit_transform MONO sto train (giati den theloume leakage apo valid/test)
    Xw = word.fit_transform(train_text)
    Xc = char.fit_transform(train_text)
    # xrhsimopoiw hstack: enwnei orizontia ta 2 sparse arrays (word features + char features)
    # tocsr(): to CSR format pou theloun ta LinearSVC
    return word, char, hstack([Xw, Xc]).tocsr()


def efarmose_tfidf(word, char, text):
    # idio me ekpaideuse_tfidf alla mono transform (oxi fit) - gia valid/test.
    return hstack([word.transform(text), char.transform(text)]).tocsr()


def cached_embeddings(name, texts):
    # embed mia fora me to MiniLM, sose to .npy, kane reuse to next time.
    #
    # to embedding panw se 5082 train docs thelei ~25 lepta se CPU. To na to
    # ksanatrexoume kathe fora einai apagoreutiko, opote krataume to .npy
    # sto cache.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"emb_minilm_{name}.npy"

    # an iparxei to cache, fortwsoume kai gyrname (load ~1 sec)
    if cache_path.exists():
        print(f"  [cache hit] {name}: {cache_path.name}")
        return np.load(cache_path)

    # alliws ftiaxnw. To import to kanw edw, mesa sthn function, gia
    # na min skasei to script se kapoion pou den exei to package egkatestimeno
    from sentence_transformers import SentenceTransformer
    print(f"  [computing] {name}: {len(texts)} texts (mporei na parei wra)")
    model = SentenceTransformer(EMBED_MODEL)
    # normalize_embeddings=True: kanw L2 normalization wste kathe vector
    # na exei norm = 1. Etsi to dot product = cosine similarity.
    arr = model.encode(
        list(texts), batch_size=EMBED_BATCH,
        show_progress_bar=False, normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)  # float32 anti float64 - misos chwros, idio accuracy
    np.save(cache_path, arr)
    print(f"  [saved] {cache_path.name} shape={arr.shape}")
    return arr


def stack_features(tfidf_sparse, emb_dense, emb_scale=1.0):
    # enwnei TF-IDF sparse + embeddings dense se ena eniaio sparse matrix.
    #
    # ta embeddings einai dense (kathe value typika non-zero). An mpoun me megalo
    # magnitude (scale=1.0) kyriarxoun panw sto sparse TF-IDF -- pou exei mikra
    # non-zero values mono se ligous indices -- kai to pnigoun. Gi auto to scale.
    #
    # scale=0   -> mono TF-IDF (baseline comparison)
    # scale=0.7 -> vrethke kalytero apo to sweep
    # scale=1.0 -> ta embeddings kyriarxoun kai to score pefti
    if emb_scale == 0.0:
        return tfidf_sparse
    # embeddings * scale, kai meta sparse-ize (csr_matrix wraps a dense array)
    # float32 gia consistency
    emb_sparse = csr_matrix((emb_dense * emb_scale).astype(np.float32))
    return hstack([tfidf_sparse, emb_sparse]).tocsr()


def one_hot_hazard(pred_labels, all_labels, weight=HAZ_W):
    # sparse one-hot encoding tou predicted hazard: ena (N x K) sparse matrix
    # opou K = arithmos hazard classes. Kathe row exei ena mh-mhdeniko value sto
    # column tou prediction. To weight elegxei poso varos exei to hazard signal
    # (HAZ_W=1.5 fanike kalo apo to sweep).
    # dict: label -> column index
    idx = {h: i for i, h in enumerate(all_labels)}
    # row indices (0, 1, ..., N-1)
    rows = np.arange(len(pred_labels))
    # column indices (poio class proeblepe to kathe row)
    cols = np.array([idx[h] for h in pred_labels])
    # values - olla idia, weight
    data = np.full(len(pred_labels), weight, dtype=np.float32)
    # xrhsimopoiw csr_matrix me (data, (rows, cols)) syntax: sparse construct
    return csr_matrix((data, (rows, cols)), shape=(len(pred_labels), len(all_labels)))


def oof_hazard_predictions(X, y, n_splits=N_SPLITS):
    # 5-fold Out-Of-Fold hazard predictions panw sto train.
    #
    # idio skeptiko me to 08: theloume to product na pairnei to hazard san feature,
    # alla an o hazard exei ~100% accuracy sto train (giati to eide) to product
    # mathenei na to empisteuetai panta. Me OOF kathe train row pairnei prediction
    # apo montelo pou den to eide (ta alla 4 folds), ara ena realistiko ~94%
    # signal opws kai sto test.
    # StratifiedKFold: krataei tin katanomi twn classes se kathe fold.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.empty(len(y), dtype=y.dtype)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        # ekpaideuoume sto 80% kai protoume to 20%
        clf = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict(X[va_idx])
        print(f"  fold {fold + 1}/{n_splits} done")
    return oof


def grapse_log(label, parts):
    # grafw mia grammh sto results/eval_log.csv me ta scores tou peiramatos.
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


def dokimase_scale(scale, x_tr_tf, x_va_tf, emb_tr, emb_va, y_haz_tr, y_prod_tr, y_haz_va, y_prod_va, haz_labels):
    # pliro hazard+product evaluation gia auto to sigkekrimeno scale value.
    #
    # epistrefw parts dict me ST1, hazard F1, product F1 cond + counts.
    # gyrnaw kai ta OOF hazard preds gia possible reuse.
    # stack TF-IDF + scaled embeddings
    x_tr = stack_features(x_tr_tf, emb_tr, emb_scale=scale)
    x_va = stack_features(x_va_tf, emb_va, emb_scale=scale)

    # ftiaxnw hazard model: TF-IDF + embeddings -> hazard label
    haz_clf = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
    haz_clf.fit(x_tr, y_haz_tr)
    pred_haz_va = haz_clf.predict(x_va)

    # eksigw oti OOF predictions sto train, gia na ftiaxoume realistiko hazard feature
    pred_haz_tr_oof = oof_hazard_predictions(x_tr, y_haz_tr)

    # ftiaxnw product features = TF-IDF + embeddings + one-hot hazard
    # gia train ksanachrhsimopoioume ta OOF predictions (realistic noise)
    # gia valid ksanachrhsimopoioume to pred apo to full hazard model
    H_tr = one_hot_hazard(pred_haz_tr_oof, haz_labels)
    H_va = one_hot_hazard(pred_haz_va, haz_labels)
    x_tr_p = hstack([x_tr, H_tr]).tocsr()
    x_va_p = hstack([x_va, H_va]).tocsr()

    # ftiaxnw product model panw sto augmented feature space
    prod_clf = LinearSVC(C=C_PRODUCT, class_weight="balanced", max_iter=5000)
    prod_clf.fit(x_tr_p, y_prod_tr)
    pred_prod_va = prod_clf.predict(x_va_p)

    # official ST1 score
    parts = metrhse_st1(y_haz_va, pred_haz_va, y_prod_va, pred_prod_va, return_components=True)
    return parts, pred_haz_tr_oof


def main():
    print("=== Efialtis Stin Kouzina v2: TF-IDF + MiniLM embeddings (scale sweep) ===")
    # fortwnw ola ta splits
    train, valid, test = fortwse_dedomena()
    print(f"shapes: train={train.shape} valid={valid.shape} test={test.shape}")

    # text pou tha dei to montelo - title + text + metadata tokens
    x_tr_text = ftiakse_keimeno(train).tolist()
    x_va_text = ftiakse_keimeno(valid).tolist()
    x_te_text = ftiakse_keimeno(test).tolist()

    # labels gia ta dyo classification tasks
    y_haz_tr = train["hazard-category"].to_numpy()
    y_prod_tr = train["product-category"].to_numpy()
    y_haz_va = valid["hazard-category"].to_numpy()
    y_prod_va = valid["product-category"].to_numpy()
    # mia tetagmenh seira twn hazard labels gia consistent one-hot encoding
    haz_labels = sorted(np.unique(y_haz_tr).tolist())

    # embeddings (cached, kanena re-compute an iparxoun ta .npy)
    print(f"\n[1] sentence embeddings ({EMBED_MODEL})")
    emb_tr = cached_embeddings("train", x_tr_text)
    emb_va = cached_embeddings("valid", x_va_text)
    emb_te = cached_embeddings("test", x_te_text)
    print(f"  embedding dim: {emb_tr.shape[1]}")

    # xrhsimopoiw TF-IDF mono sto train
    print("\n[2] TF-IDF (fit panw sto train)")
    word, char, x_tr_tf = ekpaideuse_tfidf(x_tr_text)
    x_va_tf = efarmose_tfidf(word, char, x_va_text)
    print(f"  tfidf features: {x_tr_tf.shape[1]}")

    # sweep panw sto embedding scale gia na vroume to optimal
    print("\n[3] sweep emb_scale (scale=0 simainei xwris embeddings)")
    best = None
    for scale in EMB_SCALES_TO_TRY:
        print(f"\n  -> emb_scale={scale}")
        parts, _ = dokimase_scale(
            scale, x_tr_tf, x_va_tf, emb_tr, emb_va,
            y_haz_tr, y_prod_tr, y_haz_va, y_prod_va, haz_labels,
        )
        print(
            f"     st1={parts['st1']:.4f} "
            f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f}"
        )
        grapse_log(f"efialtis_kouzina_v2_s{scale:g}", parts)
        # krataw to kalitero scale gia to final refit
        if best is None or parts["st1"] > best["parts"]["st1"]:
            best = {"scale": scale, "parts": parts}

    print(f"\nbest scale = {best['scale']} me st1={best['parts']['st1']:.4f}")

    # refit panw se train+valid kai prediction sto test. Pleon kserw to best
    # scale apo to sweep, opote ksanaftiaxnw to TF-IDF (twra kai me ta valid data
    # sto lexilogio) kai to product ekpaideuetai se megalytero data set prin
    # efarmostei sto test.
    print("\n[4] refit train+valid kai prediction sto test")
    best_scale = best["scale"]
    # concat: panw-katw ta train kai valid se ena pio megalo DataFrame
    full = pd.concat([train, valid], axis=0, ignore_index=True)
    x_full_text = ftiakse_keimeno(full).tolist()
    y_haz_full = full["hazard-category"].to_numpy()
    y_prod_full = full["product-category"].to_numpy()

    # xrhsimopoiw TF-IDF refit panw se ola ta labelled data (train+valid)
    word2, char2, x_full_tf = ekpaideuse_tfidf(x_full_text)
    x_te_tf = efarmose_tfidf(word2, char2, x_te_text)

    # embeddings den ksanaftianoume - apla enwnoume train+valid mazi
    emb_full = np.concatenate([emb_tr, emb_va], axis=0)
    x_full = stack_features(x_full_tf, emb_full, emb_scale=best_scale)
    x_te = stack_features(x_te_tf, emb_te, emb_scale=best_scale)

    # final hazard model panw se train+valid
    haz_final = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
    haz_final.fit(x_full, y_haz_full)
    pred_haz_te = haz_final.predict(x_te)

    # OOF hazard panw sto full set, idia logiki me to train mono
    print("  OOF hazard panw sto train+valid")
    pred_haz_full_oof = oof_hazard_predictions(x_full, y_haz_full)

    # ftiaxnw product features me OOF hazard panw se full set + predicted hazard sto test
    H_full = one_hot_hazard(pred_haz_full_oof, haz_labels)
    H_te = one_hot_hazard(pred_haz_te, haz_labels)
    x_full_p = hstack([x_full, H_full]).tocsr()
    x_te_p = hstack([x_te, H_te]).tocsr()

    # final product model
    prod_final = LinearSVC(C=C_PRODUCT, class_weight="balanced", max_iter=5000)
    prod_final.fit(x_full_p, y_prod_full)
    pred_prod_te = prod_final.predict(x_te_p)

    # grafw to Kaggle CSV
    sub_path = grapse_submission(
        test, pred_haz_te, pred_prod_te,
        PRED_DIR / "submission_efialtis_kouzina_v2.csv",
    )
    print(f"  submission -> {sub_path}")

    # sozw kai to montelo san artifact
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
