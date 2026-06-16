# Efialtis Stin Kouzina — Food Hazard Detection (SemEval-2025 Task 9, ST1)

Project για το Kaggle challenge στο SemEval-2025 Task 9, ST1. Για κάθε food
recall θέλουμε να προβλέψουμε το `hazard-category` και το `product-category`.

Το όνομα **Efialtis Stin Kouzina** (Εφιάλτης στην Κουζίνα) περιγράφει τι
κάνει το σύστημα: εντοπίζει τους «εφιάλτες» που κρύβονται στις ανακλήσεις
τροφίμων πριν φτάσουν στο πιάτο. Η μέθοδος δουλεύει σε δύο βήματα: πρώτα
εντοπίζουμε το hazard (τι πρόβλημα έχει το recall), και μετά ο product
ταξινομητής χρησιμοποιεί αυτό το signal μαζί με το feature space για να
αποφασίσει το product.

Το score είναι:

```text
(macroF1(hazard) + macroF1(product μόνο όταν το hazard είναι σωστό)) / 2
```

## Φάκελοι

Ο φάκελος `data/raw/` έχει τα `train.csv`, `valid.csv` και `test.csv`. Ο
φάκελος `src/` έχει τα βασικά helper functions. Ο φάκελος `notebooks/` έχει
τα scripts `01`–`12`: τα `01`–`09` είναι το τοπικό pipeline (EDA →
classical → SOTA → embeddings) και τα `10`–`12` το CV evaluation
(`10_cv_eval.py`), το threshold tuning (`11_threshold_tune_cv.py`) και το
τελικό stacking ensemble (`12_stacking_ensemble.py`).

Τα plots της αναφοράς είναι στο `results/figures/`, ενώ τα class reports, τα
confusions και τα παραδείγματα λαθών είναι στο `results/analysis/`. Τα Kaggle
submissions είναι στο `results/predictions/`.

Η αναφορά (`report.docx`/`report.pdf`) και η παρουσίαση
(`presentation.pptx`/`presentation.pdf`) βρίσκονται στον γονικό φάκελο της
παράδοσης, και τα απλά unit tests στο `tests/test_core.py`.

Το `notebooks/10_distilbert_colab.ipynb` είναι το **DistilBERT fine-tune
notebook για Colab GPU** (neural baseline, ενότητα 12 του report) — δες
οδηγίες παρακάτω.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Για να τρέξει όλο

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

## Για μόνο το τελικό submission

```powershell
python main.py --submit
python notebooks/06_eval.py
python -m unittest discover -s tests
```

(Το `main.py` καλεί το `notebooks/12_stacking_ensemble.py`.)

Το τελικό submission είναι
`results/predictions/submission_stacking.csv` με καλύτερο επιβεβαιωμένο Kaggle
public score `0.77750`.

## Τι δοκίμασα

| Notebook | Μοντέλο | Validation ST1 |
| --- | --- | ---: |
| 02_classical | TF-IDF + LogReg | 0.6875 |
| 02_classical | TF-IDF + LinearSVC | 0.7599 |
| 02_classical | TF-IDF + ComplementNB | 0.5139 |
| 03_dim_red | TF-IDF + SVD600 + LinearSVC | 0.6771 |
| 04_neural | TF-IDF + SVD300 + MLP | 0.5369 |
| 05_sota | TF-IDF + metadata + LinearSVC (C tuning) | 0.7612 |
| 08_efialtis_kouzina | TF-IDF + OOF hazard + LinearSVC | 0.7623 |
| **09_embeddings** | **TF-IDF + MiniLM (scale=0.7) + OOF hazard + LinearSVC** | **0.7737** |

Το καλύτερο single validation score στο τοπικό log είναι `0.7737` για το
MiniLM run. Το καλύτερο Kaggle public score είναι `0.77750` από το
`submission_stacking.csv`.

## Πώς δουλεύει το Efialtis Stin Kouzina

1. TF-IDF (word 1-2 + char_wb 3-5) πάνω σε `title + text + metadata`.
2. **Sentence embeddings** (MiniLM-L6-v2, 384-dim, L2-normalized) για κάθε
   κείμενο. Cached στο `results/cache/emb_minilm_*.npy`.
3. Stack TF-IDF + (embeddings × scale=0.7). Το scale βρέθηκε με sweep
   `[0, 0.2, 0.3, 0.5, 0.7, 1.0]` και είναι **U-shape**: πολύ ή πολύ λίγο
   βλάπτει — τα embeddings πρέπει να συνεισφέρουν αλλά να μη καταπνίξουν
   τα 160k TF-IDF features.
4. Hazard LinearSVC εκπαιδεύεται στο stacked space.
5. **5-fold out-of-fold** hazard predictions πάνω στο train (ρεαλιστικό
   ~94% accurate signal, χωρίς leakage).
6. Το OOF hazard μπαίνει σαν sparse **one-hot 10 διαστάσεων**, στοιβαγμένο
   στο stacked TF-IDF+embeddings.
7. Product LinearSVC εκπαιδεύεται πάνω στο final feature space.
8. Στο test, το hazard έρχεται από το full-train hazard μοντέλο (όχι OOF).

## Failed experiments

Νωρίς:

- Per-hazard product models (08_efialtis_kouzina try, ST1=0.7311): ένα
  ξεχωριστό LinearSVC ανά hazard. Έπεσε γιατί τα product labels εμφανίζονται
  σε πολλές hazards και το macro F1 χάνει συνοχή.
- MiniLM embeddings στο raw scale=1.0 (ST1=0.7448): τα dense embeddings
  κυριαρχούσαν πάνω στο sparse TF-IDF. Λύση: scale=0.7.

Αργότερα (μετά το stacking ensemble στο 0.77750 στο Kaggle), δοκίμασα 6
ακόμα προσεγγίσεις και καμία δεν με ανέβασε:

| Πείραμα | CV gain | Folds + | Kaggle Δέλτα |
|---|---:|:---:|---:|
| 3-way stacking v2 (single-split tuning) | — | — | −0.012 |
| DistilBERT v3 single fine-tune | — | — | −0.041 |
| 3-way stacking v3 (proper CV OOF) | +0.018 | 4/5 | −0.013 |
| DeBERTa-v3 OOF | — | — | NaN (μη υποβληθέν) |
| mpnet swap (× 0.7) | +0.031 | 5/5 | −0.015 |
| Hybrid weighted majority vote | — | — | −0.010 |

Σημείωση για τα DistilBERT: το fine-tune v2 έβγαλε ST1 `0.8071` στο single
validation split (565 δείγματα) — δελεαστικό, αλλά το single-split
υπερεκτιμά. Καμία εκδοχή του BERT (v2/v3) δεν ξεπέρασε το TF-IDF + MiniLM
stack σε proper CV ή στο Kaggle, οπότε τα neural μοντέλα έμειναν
τεκμηριωμένο baseline και δεν μπήκαν στο τελικό pipeline.

Το 0.77750 είναι το καλύτερο που πέτυχα με όσα δοκίμασα. Περισσότερα στις
ενότητες 14-15 του `report.docx`.

## DistilBERT στο Colab (neural baseline)

Για το `notebooks/10_distilbert_colab.ipynb` (δύο ξεχωριστά DistilBERT models,
ένα για hazard και ένα για product· τεκμηρίωση στην ενότητα 12 του report):

1. Πήγαινε στο [Colab](https://colab.research.google.com) → New notebook.
2. **Runtime → Change runtime type → GPU** (T4 free αρκεί).
3. Άνοιξε το `10_distilbert_colab.ipynb` (File → Upload notebook).
4. Στο Files panel (αριστερά), ανέβασε τα `data/raw/train.csv`,
   `data/raw/valid.csv`, `data/raw/test.csv`.
5. **Runtime → Run all**. Περιμένεις ~30-45 λεπτά.
6. Κατέβα το `submission_distilbert.csv` και ανέβασέ το στο Kaggle.

Παράγει επίσης `logits_haz_*.npy`, `logits_prod_*.npy` και `label_maps.json`
αν θες αργότερα να κάνεις ensemble με το Efialtis Stin Kouzina τοπικά.

## Τελικά CSVs

Στο `results/predictions/` υπάρχουν τα submissions όλων των σταδίων:

- `submission_stacking.csv` — **τελικό / καλύτερο**, Kaggle public score
  `0.77750`.
- `submission_efialtis_kouzina.csv`, `submission_efialtis_kouzina_v2.csv`,
  `submission_sota_linsvc_ch1_cp2.csv`, `submission_classical_linsvc.csv`,
  `submission_svd600_linsvc.csv`, `submission_mlp_svd300.csv` — ενδιάμεσα
  baseline submissions των ενοτήτων 4-9.

Τα αποτυχημένα πειράματα της ενότητας 15 (3-way stacking, mpnet swap, hybrid
vote κ.λπ.) δεν κρατούνται ως αρχεία· τα Kaggle scores τους και η ανάλυση
είναι στην ενότητα 15 του report και στον πίνακα «Failed experiments»
παραπάνω.

## Παραδοτέα

- `main.py` — entry point, παράγει `submission_stacking.csv` (Kaggle 0.77750)
- `report.docx` / `report.pdf` — αναφορά 15 ενοτήτων (Ελληνικά)
- `presentation.pptx` / `presentation.pdf` — παρουσίαση (15 slides)
- `notebooks/` — πλήρης ιστορία πειραμάτων (production + documented failures)
