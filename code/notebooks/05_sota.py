# kanw 05_sota: LinearSVC montelo me TF-IDF, metadata kai tuning sto C.
#
# dynato classical baseline prin apo ta OOF hazard kai embeddings
# experiments. Xrhsimopoiw word kai character TF-IDF mazi me metadata tokens kai
# ekpaideuw ksexwrista LinearSVC gia hazard kai product. To grid search deixnei
# oti c_haz=1 kai c_prod=2 einai kali epilogh. Sto telos to montelo kanei refit
# se train kai valid mazi, wste to test prediction na xrhsimopoiei ola ta
# diathesima labels.
import sys
import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.svm import LinearSVC

# prosthetw to project root sto sys.path gia src/ imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.eval_log import grapse_eval
from src.io_utils import MODEL_DIR, PRED_DIR, fortwse_dedomena, grapse_submission
from src.models import ekpaideuse_tfidf, efarmose_tfidf
from src.preprocess import ftiakse_keimeno
from src.scoring import metrhse_st1


def parse_args() -> argparse.Namespace:
    # vazw CLI args gia na skipparoume to grid search an kseroume ta best C.
    #
    # otan kaloume `python 05_sota.py` xwris args, kanw olo to 3x3 grid search.
    # otan kaloume `python 05_sota.py --best-c-hazard 1 --best-c-product 2`,
    # apefygei to grid kai trexw mono autes tis times - polu pio grhgoro.
    parser = argparse.ArgumentParser(description="Tune/export the final LinearSVC model.")
    parser.add_argument("--best-c-hazard", type=float, default=None,
                        help="Skip grid: use this C for hazard model")
    parser.add_argument("--best-c-product", type=float, default=None,
                        help="Skip grid: use this C for product model")
    return parser.parse_args()


def main():
    args = parse_args()
    # train/valid exoun labels, test einai gia to Kaggle leaderboard (no labels)
    train, valid, test = fortwse_dedomena()

    # ftiaxnw to text pou tha dei to montelo: title + text + metadata tokens
    # (px country_us, year_2020, month_3 san "lekseis" sto text)
    x_tr_text = ftiakse_keimeno(train)
    x_va_text = ftiakse_keimeno(valid)
    x_te_text = ftiakse_keimeno(test)

    # kanw fit TF-IDF mono sto train (axiopoieiseis to lexilogio mono apo ekei),
    # meta transform sto valid me to idio lexilogio (no leakage)
    print("fitting TF-IDF on train")
    word, char, x_tr = ekpaideuse_tfidf(x_tr_text)
    x_va = efarmose_tfidf(word, char, x_va_text)
    print(f"feature shape: {x_tr.shape}")  # typika ~(5082, ~160000)

    if args.best_c_hazard is not None or args.best_c_product is not None:
        # grigoro run me ta C pou kserw oti einai kalitera
        if args.best_c_hazard is None or args.best_c_product is None:
            raise ValueError("provide both --best-c-hazard and --best-c-product")
        c_haz = args.best_c_hazard
        c_prod = args.best_c_product
        haz_clf = LinearSVC(C=c_haz, class_weight="balanced", max_iter=5000)
        haz_clf.fit(x_tr, train["hazard-category"])
        prod_clf = LinearSVC(C=c_prod, class_weight="balanced", max_iter=5000)
        prod_clf.fit(x_tr, train["product-category"])
        parts = metrhse_st1(
            valid["hazard-category"], haz_clf.predict(x_va),
            valid["product-category"], prod_clf.predict(x_va),
            return_components=True,
        )
        label = f"linsvc_ch{c_haz:g}_cp{c_prod:g}"
        print(
            f"{label:20s} st1={parts['st1']:.4f} "
            f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f}"
        )
        grapse_eval("05_sota", label, "tfidf_word12+char35+metadata", parts)
        best = {"label": label, "c_haz": c_haz, "c_prod": c_prod, "parts": parts}
    else:
        # mikro grid search. Den einai terastio tuning, apla dokimazw liga C.
        # 3x3 = 9 combinations. kratw to kalitero apo to validation set.
        # apo ta runs vrethkan ta C=1 (hazard) kai C=2 (product) san optimal,
        # me ST1=0.7612 sto validation.
        best = None
        for c_haz in [0.5, 1.0, 2.0]:
            for c_prod in [0.5, 1.0, 2.0]:
                haz_clf = LinearSVC(C=c_haz, class_weight="balanced", max_iter=5000)
                haz_clf.fit(x_tr, train["hazard-category"])
                prod_clf = LinearSVC(C=c_prod, class_weight="balanced", max_iter=5000)
                prod_clf.fit(x_tr, train["product-category"])
                parts = metrhse_st1(
                    valid["hazard-category"], haz_clf.predict(x_va),
                    valid["product-category"], prod_clf.predict(x_va),
                    return_components=True,
                )
                label = f"linsvc_ch{c_haz:g}_cp{c_prod:g}"
                print(
                    f"{label:20s} st1={parts['st1']:.4f} "
                    f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f}"
                )
                grapse_eval("05_sota", label, "tfidf_word12+char35+metadata", parts)
                if best is None or parts["st1"] > best["parts"]["st1"]:
                    best = {"label": label, "c_haz": c_haz, "c_prod": c_prod, "parts": parts}

    assert best is not None
    print(f"best on valid: {best['label']} st1={best['parts']['st1']:.4f}")

    # gia final submission ekpaideuw ksana se train+valid (full labeled set)
    # giati: meta to tuning, kratw ta best C kai eknaideuoume me MEGALITERO
    # dataset gia kalitero generalization sto test.
    full = pd.concat([train, valid], axis=0, ignore_index=True)
    x_full_text = ftiakse_keimeno(full)
    print("refitting best model on train+valid")
    # kAINOURGIO TF-IDF (oxi to idio me to train mono) - to lexilogio twra
    # exei kai ta valid words pou mporei na xreiazontai sto test
    word_f, char_f, x_full = ekpaideuse_tfidf(x_full_text)
    x_te = efarmose_tfidf(word_f, char_f, x_te_text)

    # swzw kai to montelo gia na yparxei artifact sthn paradosi (reproducibility)
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
            "text_builder": "src.preprocess.ftiakse_keimeno(include_metadata=True)",
        },
        model_path,
    )
    # csv pou anevainw sto Kaggle
    sub_path = grapse_submission(
        test,
        haz.predict(x_te),
        prod.predict(x_te),
        PRED_DIR / f"submission_sota_{best['label']}.csv",
    )
    print(f"model -> {model_path}")
    print(f"submission -> {sub_path}")


if __name__ == "__main__":
    main()
