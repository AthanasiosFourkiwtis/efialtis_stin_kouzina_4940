# 02_classical.py
# Builds the project's first classical baseline. Joins title and text,
# fits TF-IDF with word and character features on the train set, and tries three
# linear classifiers for hazard and product. The script keeps the best
# validation score, writes a first Kaggle submission, and logs all the
# results to eval_log.csv. LinearSVC is the best of these
# initial experiments, with ST1 0.7599 on validation.

import sys
from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

# add the root to the path so src.scoring can be found
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
DATA_DIR = ROOT.parent / "data" / "raw"
if not DATA_DIR.exists():
    DATA_DIR = ROOT / "data" / "raw"
from src.scoring import score_st1

# load the data (the first column in train/valid is an unnamed index)
train = pd.read_csv(DATA_DIR / "train.csv", index_col=0)
valid = pd.read_csv(DATA_DIR / "valid.csv", index_col=0)
test  = pd.read_csv(DATA_DIR / "test.csv")
print("shapes:", train.shape, valid.shape, test.shape)

# join title + text into one field and lowercase it
X_tr_raw = (train["title"].fillna("") + " " + train["text"].fillna("")).str.lower()
X_va_raw = (valid["title"].fillna("") + " " + valid["text"].fillna("")).str.lower()
X_te_raw = (test["title"].fillna("")  + " " + test["text"].fillna("")).str.lower()

y_haz_tr  = train["hazard-category"].values
y_prod_tr = train["product-category"].values
y_haz_va  = valid["hazard-category"].values
y_prod_va = valid["product-category"].values

# tfidf over words (1-2 grams)
print("fitting word tfidf")
tf_word = TfidfVectorizer(
    ngram_range=(1, 2), min_df=3, max_df=0.95,
    sublinear_tf=True, max_features=200_000,
)
Xw_tr = tf_word.fit_transform(X_tr_raw)
Xw_va = tf_word.transform(X_va_raw)
Xw_te = tf_word.transform(X_te_raw)

# tfidf over characters (3-5 grams), helps with noisy / misspelled text
print("fitting char_wb tfidf")
tf_char = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), min_df=3,
    sublinear_tf=True, max_features=200_000,
)
Xc_tr = tf_char.fit_transform(X_tr_raw)
Xc_va = tf_char.transform(X_va_raw)
Xc_te = tf_char.transform(X_te_raw)

# join the two feature sets
X_tr = hstack([Xw_tr, Xc_tr]).tocsr()
X_va = hstack([Xw_va, Xc_va]).tocsr()
X_te = hstack([Xw_te, Xc_te]).tocsr()
print("feature shape:", X_tr.shape)

# try 3 models for both hazard and product (independent problems)
results = []
best_score = -1.0
best_name = None
best_pred_h_te = None
best_pred_p_te = None

for name in ["logreg", "linsvc", "cnb"]:
    print(f"\n>>> {name}")
    if name == "logreg":
        clf_h = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf_p = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif name == "linsvc":
        clf_h = LinearSVC(C=1.0, class_weight="balanced")
        clf_p = LinearSVC(C=1.0, class_weight="balanced")
    else:
        # ComplementNB does well on imbalanced text
        clf_h = ComplementNB(alpha=0.3)
        clf_p = ComplementNB(alpha=0.3)

    # train hazard
    clf_h.fit(X_tr, y_haz_tr)
    pred_h_va = clf_h.predict(X_va)

    # train product
    clf_p.fit(X_tr, y_prod_tr)
    pred_p_va = clf_p.predict(X_va)

    # evaluation with the official score
    parts = score_st1(y_haz_va, pred_h_va, y_prod_va, pred_p_va, return_components=True)
    print(f"  st1={parts['st1']:.4f}  f1_haz={parts['f1_hazard']:.4f}  f1_prod_cond={parts['f1_product_cond']:.4f}  haz_correct={parts['n_hazard_correct']}/{parts['n_total']}")
    results.append((name, parts))

    # keep the best one for the kaggle submission
    if parts["st1"] > best_score:
        best_score = parts["st1"]
        best_name = name
        best_pred_h_te = clf_h.predict(X_te)
        best_pred_p_te = clf_p.predict(X_te)

# print a summary
print("\n=== summary on valid ===")
for name, parts in results:
    print(f"{name:8s} st1={parts['st1']:.4f}  haz={parts['f1_hazard']:.4f}  prod={parts['f1_product_cond']:.4f}")
print(f"best: {best_name} ({best_score:.4f})")

# write the submission for the best model (id, hazard-category, product-category)
sub_dir = ROOT / "results/predictions"
sub_dir.mkdir(parents=True, exist_ok=True)
sub = pd.DataFrame({
    "id": test["id"].values,
    "hazard-category": best_pred_h_te,
    "product-category": best_pred_p_te,
})
sub_path = sub_dir / f"submission_classical_{best_name}.csv"
sub.to_csv(sub_path, index=False)
print(f"submission -> {sub_path}")

# keep a log of the experiments so they can be shown in the report
log_path = ROOT / "results/eval_log.csv"
write_header = not log_path.exists()
with open(log_path, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["timestamp", "notebook", "model", "features", "st1", "f1_haz", "f1_prod_cond"])
    ts = datetime.now().isoformat(timespec="seconds")
    for name, parts in results:
        w.writerow([
            ts, "02_classical", name, "tfidf_word12+char35",
            f"{parts['st1']:.4f}",
            f"{parts['f1_hazard']:.4f}",
            f"{parts['f1_product_cond']:.4f}",
        ])
print(f"log -> {log_path}")
