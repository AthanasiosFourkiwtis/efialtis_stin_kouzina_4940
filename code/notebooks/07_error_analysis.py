# aplo error analysis gia to LinearSVC setup. To script ftiaxnei class reports
# gia hazard kai product, ena product report mono sta rows opou to hazard einai
# swsto, confusion pairs gia ta hazard labels kai merika paradeigmata lathwn gia
# qualitative review. Ta arxeia pou paragontai xrhsimopoiountai sthn anafora gia
# na fanei poies klaseis mperdeuontai kai giati to official ST1 einai dyskolo.

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# prosthetw to project root sto sys.path gia na vrei to src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io_utils import fortwse_dedomena
from src.models import ekpaideuse_tfidf, efarmose_tfidf
from sklearn.svm import LinearSVC
from src.preprocess import ftiakse_keimeno
from src.scoring import metrhse_st1


# pou tha sothoun ta CSVs
OUT_DIR = ROOT / "results" / "analysis"


def report_to_csv(y_true, y_pred, path):
    # grafw precision/recall/f1 ana class se CSV file.
    #
    # xrhsimopoiw to sklearn.classification_report me output_dict=True,
    # metafrazei to dict se rows kai sose to se CSV gia eukolo opening sto Excel
    # gia diavasma sthn anafora.
    #
    # kathe row tou CSV exei: class, precision, recall, f1, support
    # output_dict=True epistrefei dict me kathe class kai aggregate scores
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for label, vals in rep.items():
        # "accuracy" tou rep einai float, oxi dict - to skippeoume
        if not isinstance(vals, dict):
            continue
        rows.append(
            {
                "class": label,
                "precision": vals["precision"],
                "recall": vals["recall"],
                "f1": vals["f1-score"],
                "support": vals["support"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def confusion_to_csv(y_true, y_pred, path):
    # grafw MONO ta lathos zeugaria sto confusion matrix se csv.
    #
    # anti gia tin pliro NxN confusion matrix (pou exei pollous mhdenikous),
    # krataei mono tis "ektos diagwniou" entries (true != predicted) me count > 0.
    # tasinomei me megalitero count proti, etsi vlepw amesa poia
    # mperdeumata einai pio sixna.
    #
    # format: true, predicted, count
    # tetagmenh seira twn unique labels gia consistent confusion matrix
    labels = sorted(pd.Series(y_true).unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    # kanw iterate panw apo tin NxN matrix
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            count = int(cm[i, j])
            # krata mono ta cells: diagonal (krata sosta) den thelw,
            # zeros den thelw, mono off-diagonal me count > 0
            if true_label != pred_label and count > 0:
                rows.append(
                    {
                        "true": true_label,
                        "predicted": pred_label,
                        "count": count,
                    }
                )
    # sort fthinontas kata count: ta pio sixna lathi prota
    pd.DataFrame(rows).sort_values("count", ascending=False).to_csv(path, index=False)


def main():
    # ksanatrexw to final montelo sto validation gia na parw predictions
    train, valid, _ = fortwse_dedomena()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train_text = ftiakse_keimeno(train, include_metadata=True)
    x_valid_text = ftiakse_keimeno(valid, include_metadata=True)

    word, char, x_train = ekpaideuse_tfidf(x_train_text)
    x_valid = efarmose_tfidf(word, char, x_valid_text)

    # final setup: TF-IDF + LinearSVC C=1/C=2 gia hazard kai product
    haz = LinearSVC(C=1, class_weight="balanced", max_iter=5000)
    haz.fit(x_train, train["hazard-category"])
    prod = LinearSVC(C=2, class_weight="balanced", max_iter=5000)
    prod.fit(x_train, train["product-category"])
    pred_h = haz.predict(x_valid)
    pred_p = prod.predict(x_valid)
    parts = metrhse_st1(valid["hazard-category"], pred_h,
                      valid["product-category"], pred_p, return_components=True)

    pred_haz = haz.predict(x_valid)
    pred_prod = prod.predict(x_valid)

    # ftiaxnw class reports gia hazard kai product
    report_to_csv(
        valid["hazard-category"],
        pred_haz,
        OUT_DIR / "hazard_class_report.csv",
    )
    report_to_csv(
        valid["product-category"],
        pred_prod,
        OUT_DIR / "product_class_report.csv",
    )
    # product report mono sta rows pou to hazard einai swsto
    haz_ok = valid["hazard-category"].to_numpy() == pred_haz
    report_to_csv(
        valid["product-category"].to_numpy()[haz_ok],
        pred_prod[haz_ok],
        OUT_DIR / "product_class_report_hazard_correct.csv",
    )
    # poia hazard labels mperdeuei metaksy tous
    confusion_to_csv(
        valid["hazard-category"],
        pred_haz,
        OUT_DIR / "hazard_confusions.csv",
    )

    # krataw merika paradeigmata lathwn gia na ta valw/anferw sthn anafora
    errors = valid.copy()
    errors["pred_hazard"] = pred_haz
    errors["pred_product"] = pred_prod
    errors = errors[
        (errors["hazard-category"] != errors["pred_hazard"])
        | (errors["product-category"] != errors["pred_product"])
    ]
    errors[
        [
            "title",
            "hazard-category",
            "pred_hazard",
            "product-category",
            "pred_product",
        ]
    ].head(30).to_csv(OUT_DIR / "sample_errors.csv", index=False)

    print(
        "final valid: "
        f"st1={parts['st1']:.4f} "
        f"haz={parts['f1_hazard']:.4f} "
        f"prod={parts['f1_product_cond']:.4f} "
        f"haz_correct={parts['n_hazard_correct']}/{parts['n_total']}"
    )
    print(f"analysis files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
