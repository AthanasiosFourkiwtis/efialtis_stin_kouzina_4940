# 03_dim_red.py
# elegxw an h meiwsh diastasewn me Truncated SVD voithaei panw sto TF-IDF.
# idea einai oti ta polla sparse features isws exoun thoryvo kai oti ena LSA
# representation mporei na krathsei pio katharo semantic signal. Sthn praksh to
# sVD xeirotereuei to ST1, giati xanontai spanies alla diakritikes lekseis. To
# kalytero run einai SVD600 me LinearSVC, alla menei katw apo to plain classical
# baseline tou 02_classical.py.

import sys
from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
DATA_DIR = ROOT.parent / "data" / "raw"
if not DATA_DIR.exists():
    DATA_DIR = ROOT / "data" / "raw"
from src.scoring import metrhse_st1

# fortwnw ta data
train = pd.read_csv(DATA_DIR / "train.csv", index_col=0)
valid = pd.read_csv(DATA_DIR / "valid.csv", index_col=0)
test  = pd.read_csv(DATA_DIR / "test.csv")

X_tr_raw = (train["title"].fillna("") + " " + train["text"].fillna("")).str.lower()
X_va_raw = (valid["title"].fillna("") + " " + valid["text"].fillna("")).str.lower()
X_te_raw = (test["title"].fillna("")  + " " + test["text"].fillna("")).str.lower()

y_haz_tr  = train["hazard-category"].values
y_prod_tr = train["product-category"].values
y_haz_va  = valid["hazard-category"].values
y_prod_va = valid["product-category"].values

# idia tfidf opws sto 02 (word 1-2 + char_wb 3-5)
print("kanoume tfidf")
tf_word = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True, max_features=200_000)
Xw_tr = tf_word.fit_transform(X_tr_raw); Xw_va = tf_word.transform(X_va_raw); Xw_te = tf_word.transform(X_te_raw)
tf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3, sublinear_tf=True, max_features=200_000)
Xc_tr = tf_char.fit_transform(X_tr_raw); Xc_va = tf_char.transform(X_va_raw); Xc_te = tf_char.transform(X_te_raw)

X_tr = hstack([Xw_tr, Xc_tr]).tocsr()
X_va = hstack([Xw_va, Xc_va]).tocsr()
X_te = hstack([Xw_te, Xc_te]).tocsr()
print("tfidf shape:", X_tr.shape)

# dokimazw ligous diaforetikous ari8mous diastatasewn
results = []
best_score = -1.0
best_label = None
best_pred_h_te = None
best_pred_p_te = None

for n_comp in [100, 300, 600]:
    print(f"\n>>> SVD n_components={n_comp}")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Z_tr = svd.fit_transform(X_tr)
    Z_va = svd.transform(X_va)
    Z_te = svd.transform(X_te)
    # meta to SVD kanw L2-normalize ta dianysmata (LSA standard)
    nrm = Normalizer(copy=False)
    Z_tr = nrm.fit_transform(Z_tr); Z_va = nrm.transform(Z_va); Z_te = nrm.transform(Z_te)
    expl = float(svd.explained_variance_ratio_.sum())
    print(f"  explained variance: {expl:.3f}")

    for clf_name in ["logreg", "linsvc"]:
        if clf_name == "logreg":
            clf_h = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf_p = LogisticRegression(max_iter=2000, class_weight="balanced")
        else:
            clf_h = LinearSVC(C=1.0, class_weight="balanced")
            clf_p = LinearSVC(C=1.0, class_weight="balanced")

        clf_h.fit(Z_tr, y_haz_tr)
        clf_p.fit(Z_tr, y_prod_tr)
        pred_h_va = clf_h.predict(Z_va)
        pred_p_va = clf_p.predict(Z_va)

        parts = metrhse_st1(y_haz_va, pred_h_va, y_prod_va, pred_p_va, return_components=True)
        label = f"svd{n_comp}_{clf_name}"
        print(f"  {clf_name:6s} st1={parts['st1']:.4f}  haz={parts['f1_hazard']:.4f}  prod={parts['f1_product_cond']:.4f}")
        results.append((label, n_comp, expl, parts))

        if parts["st1"] > best_score:
            best_score = parts["st1"]
            best_label = label
            best_pred_h_te = clf_h.predict(Z_te)
            best_pred_p_te = clf_p.predict(Z_te)

print("\n=== summary on valid ===")
for label, n_comp, expl, parts in results:
    print(f"{label:18s} expl={expl:.3f}  st1={parts['st1']:.4f}  haz={parts['f1_hazard']:.4f}  prod={parts['f1_product_cond']:.4f}")
print(f"best: {best_label} ({best_score:.4f})")

# grafw submission gia to best
sub = pd.DataFrame({
    "id": test["id"].values,
    "hazard-category": best_pred_h_te,
    "product-category": best_pred_p_te,
})
sub_path = ROOT / "results/predictions" / f"submission_{best_label}.csv"
sub.to_csv(sub_path, index=False)
print(f"submission -> {sub_path}")

# grafw log
log_path = ROOT / "results/eval_log.csv"
write_header = not log_path.exists()
with open(log_path, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["timestamp","notebook","model","features","st1","f1_haz","f1_prod_cond"])
    ts = datetime.now().isoformat(timespec="seconds")
    for label, n_comp, expl, parts in results:
        w.writerow([ts, "03_dim_red", label, f"tfidf+svd{n_comp}",
                    f"{parts['st1']:.4f}", f"{parts['f1_hazard']:.4f}", f"{parts['f1_product_cond']:.4f}"])
print(f"log -> {log_path}")
