# SemEval-2025 Task 9 ST1 metric:
# (macroF1(hazard) + macroF1(product | hazard correct)) / 2
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def metrhse_st1(y_haz_true, y_haz_pred, y_prod_true, y_prod_pred, return_components=False):
    y_haz_true = np.asarray(y_haz_true)
    y_haz_pred = np.asarray(y_haz_pred)
    y_prod_true = np.asarray(y_prod_true)
    y_prod_pred = np.asarray(y_prod_pred)

    f1_haz = f1_score(y_haz_true, y_haz_pred, average="macro", zero_division=0)

    # Product is scored only where the hazard prediction is correct.
    mask = y_haz_true == y_haz_pred
    if mask.sum() == 0:
        f1_prod = 0.0
    else:
        f1_prod = f1_score(y_prod_true[mask], y_prod_pred[mask], average="macro", zero_division=0)

    score = (f1_haz + f1_prod) / 2.0

    if return_components:
        return {
            "st1": score,
            "f1_hazard": f1_haz,
            "f1_product_cond": f1_prod,
            "n_hazard_correct": int(mask.sum()),
            "n_total": int(len(mask)),
        }
    return score


def report_ana_klash(y_true, y_pred, labels=None):
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def pinakas_sygxysis(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels)
