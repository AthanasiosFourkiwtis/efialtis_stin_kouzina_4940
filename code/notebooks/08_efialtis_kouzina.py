# 08_efialtis_kouzina.py
# To prwto conditional model tou project. Prwta ekpaideuw ena hazard LinearSVC
# panw se TF-IDF features, kai meta to predicted hazard mpainei san one-hot
# feature sto product model. Gia na min mathei to product na empisteuetai ena
# mh-realistiko hazard signal, sto train xrhsimopoiw out-of-fold predictions.
# Auto tairiazei me to official ST1, pou metraei to product mono otan to hazard
# exei vrethei swsta.
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
from src.scoring import metrhse_st1
from src.io_utils import MODEL_DIR, PRED_DIR, fortwse_dedomena, grapse_submission
from src.preprocess import ftiakse_keimeno

HAZ_W = 1.5         # baros tou hazard one-hot, similar megethos me TF-IDF feature
N_SPLITS = 5
RANDOM_STATE = 42
C_HAZARD = 1.0
C_PRODUCT = 2.0


def ekpaideuse_tfidf(train_text):
    # kanw fit kai gyrnaei to TF-IDF (word + char_wb) bundle + train features.
    # xrhsimopoiw word vectorizer: pianw lexilogio (1- kai 2-grams)
    word = TfidfVectorizer(
        ngram_range=(1, 2),     # 1-grams kai 2-grams leksaewn
        min_df=3,                # mphenei sto vocab mono an emfanizetai >=3 docs
        max_df=0.95,             # to vgazei an emfanizetai sto 95%+ docs (stopwords)
        sublinear_tf=True,       # 1+log(tf) anti gia raw counts - smoother
        max_features=250_000,    # ceiling sto megethos tou vocab
    )
    # xrhsimopoiw char vectorizer: pianw subword patterns - voithaei se typos/brands
    char = TfidfVectorizer(
        analyzer="char_wb",      # within-word boundary characters
        ngram_range=(3, 5),      # 3- ews 5-grammata haraktron
        min_df=3,
        sublinear_tf=True,
        max_features=250_000,
    )
    # kanw fit_transform MONO sto train (gia na min iparxei leakage apo valid/test)
    Xw = word.fit_transform(train_text)
    Xc = char.fit_transform(train_text)
    # xrhsimopoiw hstack orizontia ta 2 sparse arrays se ena eniaio CSR matrix
    return word, char, hstack([Xw, Xc]).tocsr()


def efarmose_tfidf(word, char, text):
    # idio me ekpaideuse_tfidf alla mono transform (oxi fit) - gia valid/test sets.
    return hstack([word.transform(text), char.transform(text)]).tocsr()


def one_hot_hazard(pred_labels, all_labels, weight=HAZ_W):
    # sparse one-hot encoding tou predicted hazard: ena (N x K) sparse matrix
    # opou N = arithmos rows kai K = arithmos hazard classes. Kathe row exei ENA
    # mh-mhdeniko value sto column tou prediction, iso me weight. To weight=1.5
    # vrethke kalo sto sweep -- arketa megalo wste na metraei, oxi toso pou na
    # pnigei to ypoloipo feature space.
    # dict: label string -> column index
    idx = {h: i for i, h in enumerate(all_labels)}
    # row indices: 0, 1, 2, ..., N-1
    rows = np.arange(len(pred_labels))
    # column indices: poio class proeblepe to kathe row
    cols = np.array([idx[h] for h in pred_labels])
    # values: olla idia, sto weight
    data = np.full(len(pred_labels), weight, dtype=np.float32)
    # xrhsimopoiw csr_matrix me triplet syntax (data, (rows, cols))
    return csr_matrix((data, (rows, cols)), shape=(len(pred_labels), len(all_labels)))


def oof_hazard_predictions(X, y, n_splits=N_SPLITS):
    # 5-fold Out-Of-Fold hazard predictions panw sto train.
    #
    # giati oxi to pred apo to full-train hazard model? Ekeino exei dei OLA ta
    # train rows, opote ston train petyxainei ~99% (overfit). An ekpaideusoume to
    # product me toso teleio hazard, mathenei na to empisteuetai panta -- alla sto
    # test to hazard pefti sto ~93% kai tote to product mperdevetai.
    #
    # OOF lysi: kathe train row anikei se ena fold kai pairnei prediction apo
    # montelo pou ekpaideutike sta alla 4 folds (den eide auto to row). Etsi ola
    # ta train rows pairnoun ena realistiko ~94% accurate signal, opws kai sto test.
    # StratifiedKFold: krataei tin katanomi twn classes analogi se kathe fold.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.empty(len(y), dtype=y.dtype)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        # kanw train sto 80% (4 folds)
        clf = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
        clf.fit(X[tr_idx], y[tr_idx])
        # kanw predict sto 20% (1 fold pou meine ekso)
        oof[va_idx] = clf.predict(X[va_idx])
        print(f"  fold {fold + 1}/{n_splits} done")
    return oof


def grapse_log(label, parts):
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
    train, valid, test = fortwse_dedomena()
    print(f"shapes: train={train.shape} valid={valid.shape} test={test.shape}")

    x_tr_text = ftiakse_keimeno(train)
    x_va_text = ftiakse_keimeno(valid)
    x_te_text = ftiakse_keimeno(test)

    y_haz_tr = train["hazard-category"].to_numpy()
    y_prod_tr = train["product-category"].to_numpy()
    y_haz_va = valid["hazard-category"].to_numpy()
    y_prod_va = valid["product-category"].to_numpy()

    # xrhsimopoiw TF-IDF + hazard montelo
    print("\n[1] TF-IDF + hazard LinearSVC")
    word, char, x_tr = ekpaideuse_tfidf(x_tr_text)
    x_va = efarmose_tfidf(word, char, x_va_text)
    print(f"  tfidf features: {x_tr.shape[1]}")

    haz_clf = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
    haz_clf.fit(x_tr, y_haz_tr)
    pred_haz_va = haz_clf.predict(x_va)
    haz_labels = sorted(np.unique(y_haz_tr).tolist())
    print(f"  hazard labels: {len(haz_labels)}")

    # eksigw oti OOF hazard panw sto train
    print("\n[2] OOF hazard predictions panw sto train (5-fold)")
    pred_haz_tr_oof = oof_hazard_predictions(x_tr, y_haz_tr)
    print(f"  OOF accuracy: {(pred_haz_tr_oof == y_haz_tr).mean():.4f}")

    # ftiaxnw product montelo me hazard feature
    print("\n[3] product LinearSVC me one-hot hazard feature")
    H_tr = one_hot_hazard(pred_haz_tr_oof, haz_labels)
    H_va = one_hot_hazard(pred_haz_va, haz_labels)
    x_tr_p = hstack([x_tr, H_tr]).tocsr()
    x_va_p = hstack([x_va, H_va]).tocsr()
    print(f"  augmented shape: {x_tr_p.shape}")

    prod_clf = LinearSVC(C=C_PRODUCT, class_weight="balanced", max_iter=5000)
    prod_clf.fit(x_tr_p, y_prod_tr)
    pred_prod_va = prod_clf.predict(x_va_p)

    parts = metrhse_st1(y_haz_va, pred_haz_va, y_prod_va, pred_prod_va, return_components=True)
    print(
        f"\nEfialtis Stin Kouzina valid: st1={parts['st1']:.4f} "
        f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f} "
        f"(haz_correct {parts['n_hazard_correct']}/{parts['n_total']})"
    )
    grapse_log("efialtis_kouzina", parts)

    # final refit panw se train+valid
    print("\n[4] refit panw se train+valid kai prediction sto test")
    full = pd.concat([train, valid], axis=0, ignore_index=True)
    x_full_text = ftiakse_keimeno(full)
    y_haz_full = full["hazard-category"].to_numpy()
    y_prod_full = full["product-category"].to_numpy()

    word2, char2, x_full = ekpaideuse_tfidf(x_full_text)
    x_te = efarmose_tfidf(word2, char2, x_te_text)

    haz_final = LinearSVC(C=C_HAZARD, class_weight="balanced", max_iter=5000)
    haz_final.fit(x_full, y_haz_full)
    pred_haz_te = haz_final.predict(x_te)

    print("  OOF hazard panw sto train+valid")
    pred_haz_full_oof = oof_hazard_predictions(x_full, y_haz_full)

    H_full = one_hot_hazard(pred_haz_full_oof, haz_labels)
    H_te = one_hot_hazard(pred_haz_te, haz_labels)
    x_full_p = hstack([x_full, H_full]).tocsr()
    x_te_p = hstack([x_te, H_te]).tocsr()

    prod_final = LinearSVC(C=C_PRODUCT, class_weight="balanced", max_iter=5000)
    prod_final.fit(x_full_p, y_prod_full)
    pred_prod_te = prod_final.predict(x_te_p)

    sub_path = grapse_submission(
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
            "text_builder": "src.preprocess.ftiakse_keimeno(include_metadata=True)",
        },
        model_path,
    )
    print(f"  model -> {model_path}")


if __name__ == "__main__":
    main()
