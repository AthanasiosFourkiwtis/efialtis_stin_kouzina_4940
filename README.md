# Efialtis Stin Kouzina

Food Hazard Detection — SemEval-2025 Task 9, Subtask 1 (ST1).
Project for the NLP 053 course, CSE UOI 2026. Student ID 4940, Athanasios Fourkiotis.

For every food recall we want to predict two labels from the text:
the `hazard-category` (10 classes) and the `product-category`
(22 classes).

## The name

**Efialtis Stin Kouzina** (Greek for "Kitchen Nightmare"): the system spots the
"nightmares" hiding inside food recalls — allergens, contaminations, foreign
bodies — before they reach the plate. It works in two steps: first it detects
the hazard, then it uses that prediction to help predict the
product.

## The metric

The official ST1 score is not a simple average of two classifiers:

```
(macroF1(hazard) + macroF1(product, only where the hazard is correct)) / 2
```

In other words, if the hazard comes out wrong, a correct product prediction
does not count for that example. The problem is hard because the classes are
heavily imbalanced (long-tail) and macro-F1 gives equal weight to the rare ones.

## Exploratory data analysis (EDA)

The categories are strongly imbalanced — a few classes (allergens, biological)
hold most of the examples, while several have barely any. That is what makes
macro-F1 difficult.

![Hazard category distribution](code/results/figures/hazard_dist.png)

![Product category distribution](code/results/figures/product_dist.png)

The hazard–product relationship is not one-to-one: the same hazard shows up in
many products and vice versa, so both the text and the hazard
signal are needed.

![Hazard × product relationship](code/results/figures/joint_haz_prod.png)

The documents vary a lot in length (which is why character
n-grams were also used), and the per-year distribution is not uniform.

![Document length](code/results/figures/doc_length.png)

![Reports per year](code/results/figures/year_dist.png)

## The approach

1. **TF-IDF** over `title + text + metadata` (word 1–2 grams + char_wb 3–5 grams).
   The metadata (country, year, month) go in as plain text tokens.
2. **MiniLM** sentence embeddings (384-dim, L2-normalized), stacked onto the TF-IDF
   with scale=0.7 — found via a sweep; it is "U-shaped" (too much or too little hurts).
3. **Hazard** LinearSVC on the stacked feature space.
4. **Out-of-fold** (5-fold) hazard predictions on the train set: every row gets
   a prediction from a model that never saw it, so the product learns a
   realistic (~94%) hazard signal without leakage.
5. The OOF hazard goes in as a one-hot feature and the **product** LinearSVC
   is trained on the final feature space.
6. Final submission: **stacking ensemble** (TF-IDF/OOF + MiniLM) with weights
   selected through cross-validation.

## Results

| Model | Validation ST1 | Kaggle public |
| --- | ---: | ---: |
| TF-IDF + LinearSVC | 0.7599 | — |
| Efialtis Stin Kouzina (TF-IDF + OOF hazard) | 0.7623 | 0.7573 |
| + MiniLM embeddings (scale=0.7) | 0.7737 | 0.7512 |
| **Stacking ensemble (final)** | — | **0.7775** |

Key lesson: the single validation split (565 samples) was unreliable.
With 5-fold CV the ST1 came out at 0.7036 ± 0.0516 — "improvements" below ~0.05
were lost in the variance. Only the stacking survived both the CV and
Kaggle. A fine-tuned DistilBERT was also tried, but on this small dataset with
macro-F1 the classic TF-IDF + LinearSVC (class_weight=balanced) proved
better.

## Layout

```
code/            code (notebooks 01–12, src/, tests, main.py)
  notebooks/     EDA -> classical -> SOTA -> embeddings -> CV -> stacking
  src/           helpers (preprocess, scoring, io, models)
  results/       figures, analysis, predictions, logs
data/raw/        train.csv, valid.csv, test.csv
report.pdf       report (15 sections)
presentation.pdf presentation
```

Detailed run instructions and the full experiment history (including what
**didn't** work) live in [`code/README.md`](code/README.md) and in `report.pdf`.

## Quick start

```powershell
cd code
pip install -r requirements.txt
python main.py            # produces the final submission_stacking.csv
```

## Data

The data comes from SemEval-2025 Task 9 / Food Recall Incidents
(CC BY-NC-SA 4.0) and is used for educational purposes only.
