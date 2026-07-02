# Simple error analysis for the LinearSVC setup. The script builds class reports
# for hazard and product, a product report restricted to the rows where the hazard
# is correct, confusion pairs for the hazard labels, and a few misclassified
# examples for qualitative review. The generated files feed the report,
# showing which classes get confused and why the official ST1 is hard.

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# add the project root to sys.path so src/ resolves
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io_utils import load_data
from src.models import fit_tfidf, apply_tfidf
from sklearn.svm import LinearSVC
from src.preprocess import build_text
from src.scoring import score_st1


# where the CSVs get saved
OUT_DIR = ROOT / "results" / "analysis"


def report_to_csv(y_true, y_pred, path):
    # writes per-class precision/recall/f1 into a CSV file.
    #
    # Uses sklearn.classification_report with output_dict=True,
    # turns the dict into rows, and saves it as CSV so it opens easily in Excel
    # when reading through it for the report.
    #
    # every CSV row holds: class, precision, recall, f1, support
    # output_dict=True returns a dict with every class plus the aggregate scores
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for label, vals in rep.items():
        # the report's "accuracy" entry is a float, not a dict - skip it
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
    # writes ONLY the mistaken pairs of the confusion matrix into a csv.
    #
    # instead of the full NxN confusion matrix (mostly zeros),
    # it keeps only the off-diagonal entries (true != predicted) with count > 0.
    # sorted with the largest count first, so the most frequent
    # mix-ups are immediately visible.
    #
    # format: true, predicted, count
    # unique labels in sorted order for a consistent confusion matrix
    labels = sorted(pd.Series(y_true).unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    # iterate over the NxN matrix
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            count = int(cm[i, j])
            # keep only the interesting cells: not the diagonal (correct ones),
            # not the zeros — only off-diagonal with count > 0
            if true_label != pred_label and count > 0:
                rows.append(
                    {
                        "true": true_label,
                        "predicted": pred_label,
                        "count": count,
                    }
                )
    # sort by count descending: the most frequent mistakes first
    pd.DataFrame(rows).sort_values("count", ascending=False).to_csv(path, index=False)


def main():
    # re-run the final model on validation to obtain predictions
    train, valid, _ = load_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train_text = build_text(train, include_metadata=True)
    x_valid_text = build_text(valid, include_metadata=True)

    word, char, x_train = fit_tfidf(x_train_text)
    x_valid = apply_tfidf(word, char, x_valid_text)

    # final setup: TF-IDF + LinearSVC C=1/C=2 for hazard and product
    haz = LinearSVC(C=1, class_weight="balanced", max_iter=5000)
    haz.fit(x_train, train["hazard-category"])
    prod = LinearSVC(C=2, class_weight="balanced", max_iter=5000)
    prod.fit(x_train, train["product-category"])
    pred_h = haz.predict(x_valid)
    pred_p = prod.predict(x_valid)
    parts = score_st1(valid["hazard-category"], pred_h,
                      valid["product-category"], pred_p, return_components=True)

    pred_haz = haz.predict(x_valid)
    pred_prod = prod.predict(x_valid)

    # build class reports for hazard and product
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
    # product report restricted to the rows where the hazard is correct
    haz_ok = valid["hazard-category"].to_numpy() == pred_haz
    report_to_csv(
        valid["product-category"].to_numpy()[haz_ok],
        pred_prod[haz_ok],
        OUT_DIR / "product_class_report_hazard_correct.csv",
    )
    # which hazard labels get confused with each other
    confusion_to_csv(
        valid["hazard-category"],
        pred_haz,
        OUT_DIR / "hazard_confusions.csv",
    )

    # keep a few misclassified examples to include/cite in the report
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
