# Efialtis Stin Kouzina — Food Hazard Detection (SemEval-2025 Task 9, ST1)

Project for the Kaggle challenge of SemEval-2025 Task 9, ST1. For every food
recall we want to predict the `hazard-category` and the `product-category`.

The name **Efialtis Stin Kouzina** (Greek for "Kitchen Nightmare") describes
what the system does: it finds the "nightmares" hiding in food recalls
before they reach the plate. The method works in two steps: first we find
the hazard (what's wrong with the recall), and then the product
classifier uses that signal together with the feature space to
decide the product.

The score is:

```text
(macroF1(hazard) + macroF1(product only when the hazard is correct)) / 2
```

## Folders

`data/raw/` has `train.csv`, `valid.csv` and `test.csv`.
`src/` has the basic helper functions. `notebooks/` has the
scripts `01`–`12`: `01`–`09` are the local pipeline (EDA →
classical → SOTA → embeddings) and `10`–`12` are the CV evaluation
(`10_cv_eval.py`), the threshold tuning (`11_threshold_tune_cv.py`) and the
final stacking ensemble (`12_stacking_ensemble.py`).

The report plots are in `results/figures/`, and the class reports, the
confusions and the error examples are in `results/analysis/`. The Kaggle
submissions are in `results/predictions/`.

The report (`report.docx`/`report.pdf`) and the presentation
(`presentation.pptx`/`presentation.pdf`) are in the parent folder of the
deliverable, and the simple unit tests in `tests/test_core.py`.

`notebooks/10_distilbert_colab.ipynb` is the **DistilBERT fine-tune
notebook for Colab GPU** (neural baseline, section 12 of the report) — see
the instructions below.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## To run everything

```powershell
python notebooks/01_eda.py
python notebooks/02_classical.py
python notebooks/03_dim_red.py
python notebooks/04_neural.py
python notebooks/05_sota.py --best-c-hazard 1 --best-c-product 2
python notebooks/06_eval.py
python notebooks/07_error_analysis.py
python notebooks/08_efialtis_kouzina.py
python notebooks/09_embeddings.py
python -m unittest discover -s tests
```

## For just the final submission

```powershell
python main.py --submit
python notebooks/06_eval.py
python -m unittest discover -s tests
```

(`main.py` calls `notebooks/12_stacking_ensemble.py`.)

The final submission is
`results/predictions/submission_stacking.csv`, with best confirmed Kaggle
public score `0.77750`.

## What I tried

| Notebook | Model | Validation ST1 |
| --- | --- | ---: |
| 02_classical | TF-IDF + LogReg | 0.6875 |
| 02_classical | TF-IDF + LinearSVC | 0.7599 |
| 02_classical | TF-IDF + ComplementNB | 0.5139 |
| 03_dim_red | TF-IDF + SVD600 + LinearSVC | 0.6771 |
| 04_neural | TF-IDF + SVD300 + MLP | 0.5369 |
| 05_sota | TF-IDF + metadata + LinearSVC (C tuning) | 0.7612 |
| 08_efialtis_kouzina | TF-IDF + OOF hazard + LinearSVC | 0.7623 |
| **09_embeddings** | **TF-IDF + MiniLM (scale=0.7) + OOF hazard + LinearSVC** | **0.7737** |

The best single validation score in the local log is `0.7737`, from the
MiniLM run. The best Kaggle public score is `0.77750`, from
`submission_stacking.csv`.

## How Efialtis Stin Kouzina works

1. TF-IDF (word 1-2 + char_wb 3-5) on `title + text + metadata`.
2. **Sentence embeddings** (MiniLM-L6-v2, 384-dim, L2-normalized) for every
   text. Cached in `results/cache/emb_minilm_*.npy`.
3. Stack TF-IDF + (embeddings × scale=0.7). The scale was found with a sweep
   over `[0, 0.2, 0.3, 0.5, 0.7, 1.0]` and it's a **U-shape**: too much or too
   little hurts — the embeddings have to contribute without drowning out
   the 160k TF-IDF features.
4. The hazard LinearSVC trains on the stacked space.
5. **5-fold out-of-fold** hazard predictions on the train set (a realistic
   ~94% accurate signal, no leakage).
6. The OOF hazard goes in as a sparse **10-dim one-hot**, stacked on the
   TF-IDF+embeddings.
7. The product LinearSVC trains on the final feature space.
8. On the test set, the hazard comes from the full-train hazard model (not OOF).

## Failed experiments

Early on:

- Per-hazard product models (08_efialtis_kouzina attempt, ST1=0.7311): a
  separate LinearSVC per hazard. It fell apart because the product labels show
  up across many hazards and the macro F1 loses coherence.
- MiniLM embeddings at raw scale=1.0 (ST1=0.7448): the dense embeddings
  dominated the sparse TF-IDF. Fix: scale=0.7.

Later (after the stacking ensemble hit 0.77750 on Kaggle), I tried 6
more approaches and none of them moved me up:

| Experiment | CV gain | Folds + | Kaggle delta |
|---|---:|:---:|---:|
| 3-way stacking v2 (single-split tuning) | — | — | −0.012 |
| DistilBERT v3 single fine-tune | — | — | −0.041 |
| 3-way stacking v3 (proper CV OOF) | +0.018 | 4/5 | −0.013 |
| DeBERTa-v3 OOF | — | — | NaN (not submitted) |
| mpnet swap (× 0.7) | +0.031 | 5/5 | −0.015 |
| Hybrid weighted majority vote | — | — | −0.010 |

A note on DistilBERT: the v2 fine-tune gave ST1 `0.8071` on the single
validation split (565 samples) — tempting, but the single split
overestimates. No BERT version (v2/v3) beat the TF-IDF + MiniLM
stack in proper CV or on Kaggle, so the neural models stayed as a
documented baseline and didn't go into the final pipeline.

0.77750 is the best I got with everything I tried. More in
sections 14-15 of `report.docx`.

## DistilBERT on Colab (neural baseline)

For `notebooks/10_distilbert_colab.ipynb` (two separate DistilBERT models,
one for hazard and one for product; documented in section 12 of the report):

1. Go to [Colab](https://colab.research.google.com) → New notebook.
2. **Runtime → Change runtime type → GPU** (a free T4 is enough).
3. Open `10_distilbert_colab.ipynb` (File → Upload notebook).
4. In the Files panel (left), upload `data/raw/train.csv`,
   `data/raw/valid.csv`, `data/raw/test.csv`.
5. **Runtime → Run all**. Takes ~30-45 minutes.
6. Download `submission_distilbert.csv` and upload it to Kaggle.

It also produces `logits_haz_*.npy`, `logits_prod_*.npy` and `label_maps.json`
in case you want to ensemble with Efialtis Stin Kouzina locally later.

## Final CSVs

In `results/predictions/` you'll find the submissions from every stage:

- `submission_stacking.csv` — **final / best**, Kaggle public score
  `0.77750`.
- `submission_efialtis_kouzina.csv`, `submission_efialtis_kouzina_v2.csv`,
  `submission_sota_linsvc_ch1_cp2.csv`, `submission_classical_linsvc.csv`,
  `submission_svd600_linsvc.csv`, `submission_mlp_svd300.csv` — intermediate
  baseline submissions from sections 4-9.

The failed experiments of section 15 (3-way stacking, mpnet swap, hybrid
vote etc.) aren't kept as files; their Kaggle scores and the analysis
are in section 15 of the report and in the "Failed experiments" table
above.

## Deliverables

- `main.py` — entry point, produces `submission_stacking.csv` (Kaggle 0.77750)
- `report.docx` / `report.pdf` — 15-section report (in Greek)
- `presentation.pptx` / `presentation.pdf` — presentation (15 slides)
- `notebooks/` — full experiment history (production + documented failures)
