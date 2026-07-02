# 04_neural: simple neural baseline with TF-IDF, SVD, and MLPClassifier.
#
# Covers the rubric's neural-baseline requirement, so a sklearn MLP is used here. Since
# an MLP needs dense input, the sparse TF-IDF is first reduced to 300 SVD
# dimensions and then L2-normalized. The score stays well below the
# classical models, which shows that the dimensionality reduction loses important
# information and that the data is too little for an MLP to work well.

import sys
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

# add the project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.eval_log import log_eval
from src.io_utils import PRED_DIR, load_data, write_submission
from src.models import fit_tfidf, apply_tfidf
from src.preprocess import build_text
from src.scoring import score_st1


def main():
    # load the 3 splits (train + valid with labels, test without)
    train, valid, test = load_data()
    # text = title + text + metadata tokens, lowercase normalized
    x_tr_text = build_text(train)
    x_va_text = build_text(valid)
    x_te_text = build_text(test)

    # fit TF-IDF, but with fewer max features so it runs more comfortably
    # (120k instead of 250k - the MLP is more sensitive to size here)
    word, char, x_tr = fit_tfidf(x_tr_text, word_max_features=120_000, char_max_features=120_000)
    x_va = apply_tfidf(word, char, x_va_text)
    x_te = apply_tfidf(word, char, x_te_text)

    # the MLP needs dense/small vectors, so run SVD first to reduce
    # dimensionality. 300 components = LSA standard.
    # Normalizer L2: gives every row norm=1 (standard after LSA)
    # make_pipeline: chains the two into one pipeline that fits/transforms together
    reducer = make_pipeline(TruncatedSVD(n_components=300, random_state=42), Normalizer(copy=False))
    z_tr = reducer.fit_transform(x_tr)   # (5082, 300) dense float
    z_va = reducer.transform(x_va)        # (565, 300)
    z_te = reducer.transform(x_te)        # (997, 300)

    # same settings for both the hazard and the product MLP
    # hidden_layer_sizes=(256,) = 1 hidden layer with 256 neurons
    # alpha=1e-4 = L2 regularization weight
    # max_iter=80 = up to 80 epochs (sklearn MLP)
    # early_stopping=True = stops if validation doesn't improve within patience
    common = dict(
        hidden_layer_sizes=(256,),
        alpha=1e-4,
        max_iter=80,
        early_stopping=True,
        random_state=42,
    )
    # LabelEncoder: turns string labels into ints 0..N-1 (required by sklearn)
    haz_encoder = LabelEncoder()
    prod_encoder = LabelEncoder()
    y_haz = haz_encoder.fit_transform(train["hazard-category"])
    y_prod = prod_encoder.fit_transform(train["product-category"])

    # train one MLP for hazard and one for product (independently)
    haz = MLPClassifier(**common)
    prod = MLPClassifier(**common)
    haz.fit(z_tr, y_haz)
    prod.fit(z_tr, y_prod)

    # map the numeric labels back to the original string labels for score_st1
    pred_h = haz_encoder.inverse_transform(haz.predict(z_va))
    pred_p = prod_encoder.inverse_transform(prod.predict(z_va))
    # official ST1 with return_components for logging
    parts = score_st1(
        valid["hazard-category"], pred_h, valid["product-category"], pred_p, return_components=True
    )
    print(
        f"mlp_svd300 st1={parts['st1']:.4f} "
        f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f}"
    )
    log_eval("04_neural", "mlp_svd300", "tfidf_word_char+svd300", parts)

    # also write a submission so a baseline csv exists (for consistency with the rest)
    sub_path = write_submission(
        test,
        haz_encoder.inverse_transform(haz.predict(z_te)),
        prod_encoder.inverse_transform(prod.predict(z_te)),
        PRED_DIR / "submission_mlp_svd300.csv",
    )
    print(f"submission -> {sub_path}")


if __name__ == "__main__":
    main()
