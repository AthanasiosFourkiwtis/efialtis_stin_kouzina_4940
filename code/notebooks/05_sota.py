# 05_sota: LinearSVC model with TF-IDF, metadata, and C tuning.
#
# A strong classical baseline ahead of the OOF hazard and embeddings
# experiments. Word and character TF-IDF are combined with metadata tokens, and
# separate LinearSVCs are trained for hazard and product. The grid search shows
# that c_haz=1 and c_prod=2 is a good choice. At the end the model is refit
# on train and valid together, so the test prediction benefits from every
# available label.
import sys
import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.svm import LinearSVC

# add the project root to sys.path for the src/ imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.eval_log import log_eval
from src.io_utils import MODEL_DIR, PRED_DIR, load_data, write_submission
from src.models import fit_tfidf, apply_tfidf
from src.preprocess import build_text
from src.scoring import score_st1


def parse_args() -> argparse.Namespace:
    # CLI args allow skipping the grid search when the best C values are known.
    #
    # `python 05_sota.py` with no args runs the full 3x3 grid search.
    # `python 05_sota.py --best-c-hazard 1 --best-c-product 2`
    # skips the grid and runs only those values - much faster.
    parser = argparse.ArgumentParser(description="Tune/export the final LinearSVC model.")
    parser.add_argument("--best-c-hazard", type=float, default=None,
                        help="Skip grid: use this C for hazard model")
    parser.add_argument("--best-c-product", type=float, default=None,
                        help="Skip grid: use this C for product model")
    return parser.parse_args()


def main():
    args = parse_args()
    # train/valid carry labels; test is for the Kaggle leaderboard (no labels)
    train, valid, test = load_data()

    # build the text the model will see: title + text + metadata tokens
    # (e.g. country_us, year_2020, month_3 as "words" inside the text)
    x_tr_text = build_text(train)
    x_va_text = build_text(valid)
    x_te_text = build_text(test)

    # fit TF-IDF on train only (the vocabulary comes solely from there),
    # then transform valid with that same vocabulary (no leakage)
    print("fitting TF-IDF on train")
    word, char, x_tr = fit_tfidf(x_tr_text)
    x_va = apply_tfidf(word, char, x_va_text)
    print(f"feature shape: {x_tr.shape}")  # typically ~(5082, ~160000)

    if args.best_c_hazard is not None or args.best_c_product is not None:
        # quick run with the C values already known to be best
        if args.best_c_hazard is None or args.best_c_product is None:
            raise ValueError("provide both --best-c-hazard and --best-c-product")
        c_haz = args.best_c_hazard
        c_prod = args.best_c_product
        haz_clf = LinearSVC(C=c_haz, class_weight="balanced", max_iter=5000)
        haz_clf.fit(x_tr, train["hazard-category"])
        prod_clf = LinearSVC(C=c_prod, class_weight="balanced", max_iter=5000)
        prod_clf.fit(x_tr, train["product-category"])
        parts = score_st1(
            valid["hazard-category"], haz_clf.predict(x_va),
            valid["product-category"], prod_clf.predict(x_va),
            return_components=True,
        )
        label = f"linsvc_ch{c_haz:g}_cp{c_prod:g}"
        print(
            f"{label:20s} st1={parts['st1']:.4f} "
            f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f}"
        )
        log_eval("05_sota", label, "tfidf_word12+char35+metadata", parts)
        best = {"label": label, "c_haz": c_haz, "c_prod": c_prod, "parts": parts}
    else:
        # small grid search. Nothing heavy, just a few C values.
        # 3x3 = 9 combinations. The best on the validation set is kept.
        # The runs found C=1 (hazard) and C=2 (product) to be optimal,
        # at ST1=0.7612 on validation.
        best = None
        for c_haz in [0.5, 1.0, 2.0]:
            for c_prod in [0.5, 1.0, 2.0]:
                haz_clf = LinearSVC(C=c_haz, class_weight="balanced", max_iter=5000)
                haz_clf.fit(x_tr, train["hazard-category"])
                prod_clf = LinearSVC(C=c_prod, class_weight="balanced", max_iter=5000)
                prod_clf.fit(x_tr, train["product-category"])
                parts = score_st1(
                    valid["hazard-category"], haz_clf.predict(x_va),
                    valid["product-category"], prod_clf.predict(x_va),
                    return_components=True,
                )
                label = f"linsvc_ch{c_haz:g}_cp{c_prod:g}"
                print(
                    f"{label:20s} st1={parts['st1']:.4f} "
                    f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f}"
                )
                log_eval("05_sota", label, "tfidf_word12+char35+metadata", parts)
                if best is None or parts["st1"] > best["parts"]["st1"]:
                    best = {"label": label, "c_haz": c_haz, "c_prod": c_prod, "parts": parts}

    assert best is not None
    print(f"best on valid: {best['label']} st1={best['parts']['st1']:.4f}")

    # for the final submission, retrain on train+valid (the full labeled set)
    # rationale: after tuning, keep the best C values and train on a LARGER
    # dataset for better generalization on test.
    full = pd.concat([train, valid], axis=0, ignore_index=True)
    x_full_text = build_text(full)
    print("refitting best model on train+valid")
    # a FRESH TF-IDF (not the train-only one) - the vocabulary now
    # includes the valid words too, which the test set may need
    word_f, char_f, x_full = fit_tfidf(x_full_text)
    x_te = apply_tfidf(word_f, char_f, x_te_text)

    # the model is saved too, so the deliverable ships an artifact (reproducibility)
    haz = LinearSVC(C=best["c_haz"], class_weight="balanced", max_iter=5000)
    haz.fit(x_full, full["hazard-category"])
    prod = LinearSVC(C=best["c_prod"], class_weight="balanced", max_iter=5000)
    prod.fit(x_full, full["product-category"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"final_sota_{best['label']}.joblib"
    dump(
        {
            "tfidf_word": word_f,
            "tfidf_char": char_f,
            "hazard_model": haz,
            "product_model": prod,
            "best": best,
            "text_builder": "src.preprocess.build_text(include_metadata=True)",
        },
        model_path,
    )
    # the csv that goes up to Kaggle
    sub_path = write_submission(
        test,
        haz.predict(x_te),
        prod.predict(x_te),
        PRED_DIR / f"submission_sota_{best['label']}.csv",
    )
    print(f"model -> {model_path}")
    print(f"submission -> {sub_path}")


if __name__ == "__main__":
    main()
