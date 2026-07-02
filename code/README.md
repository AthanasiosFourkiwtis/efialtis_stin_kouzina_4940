# Efialtis Stin Kouzina, Food Hazard Detection (SemEval-2025 Task 9, ST1)

Project for the Kaggle challenge of SemEval-2025 Task 9, ST1. For every food recall we predict the hazard category and the product category. The name is Greek for Kitchen Nightmare, and it describes what the system does: it finds the nightmares hiding in food recalls before they reach the plate. The method works in two steps, first the hazard gets detected, and then the product classifier uses that signal together with the feature space to decide the product.

The score is the average of the hazard macro F1 and the product macro F1 counted only where the hazard is correct.

## Folders

The raw data, train.csv, valid.csv and test.csv, sits in data/raw. The basic helper functions live in src. The notebooks folder has scripts 01 to 12, where 01 to 09 form the local pipeline going from the EDA through the classical models to the SOTA setup and the embeddings, and 10 to 12 handle the cross-validation evaluation, the threshold tuning and the final stacking ensemble. The report plots are in results/figures, the class reports, confusions and error examples in results/analysis, and the Kaggle submissions in results/predictions. The report and the presentation are in the parent folder of the deliverable, and the simple unit tests in tests/test_core.py. The notebook 10_distilbert_colab.ipynb is the DistilBERT fine-tune for a Colab GPU, the neural baseline of section 12 of the report, with instructions further down.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running everything

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

If you just want the final submission:

```powershell
python main.py --submit
python notebooks/06_eval.py
python -m unittest discover -s tests
```

main.py calls notebooks/12_stacking_ensemble.py under the hood. The final submission is results/predictions/submission_stacking.csv, with a best confirmed Kaggle public score of 0.77750.

## What I tried

The first classical pass in 02 gave 0.6875 with logistic regression, 0.7599 with LinearSVC and 0.5139 with ComplementNB, all on TF-IDF, so LinearSVC won early and stayed. Dimensionality reduction in 03 hurt, TF-IDF with SVD600 and LinearSVC only reached 0.6771, and the MLP on SVD300 in 04 did even worse at 0.5369, which told me the sparse features matter and the dataset is too small for neural nets. Adding metadata and tuning C in 05 nudged the score to 0.7612. The conditional setup with the out-of-fold hazard feature in 08 reached 0.7623, and stacking MiniLM embeddings on top in 09 gave the best single validation score, 0.7737. The best Kaggle public score, 0.77750, came from the stacking ensemble.

## How Efialtis Stin Kouzina works

TF-IDF with word 1-2 grams and character 3-5 grams runs over title plus text plus metadata. Sentence embeddings from MiniLM-L6-v2, 384 dimensions and L2 normalized, get computed for every document and cached in results/cache so they never have to be recomputed. They stack onto the TF-IDF multiplied by 0.7, a scale found with a sweep over several values which turned out U-shaped, since the embeddings have to contribute without drowning out the 160 thousand TF-IDF features. The hazard LinearSVC trains on that stacked space. Then come the 5-fold out-of-fold hazard predictions over the train set, giving a realistic signal of about 94% accuracy with no leakage, which enter the product model as a sparse 10-dimensional one-hot. The product LinearSVC trains on this final space, and at test time the hazard comes from the full-train hazard model rather than the out-of-fold one.

## Failed experiments

Two things failed early. Per-hazard product models, one separate LinearSVC per hazard, only got to 0.7311 because the product labels show up across many hazards and the macro F1 loses coherence. And MiniLM embeddings at their raw scale reached just 0.7448, since the dense embeddings dominated the sparse TF-IDF, which is what led to the 0.7 scale.

Later, after the stacking ensemble hit 0.77750 on Kaggle, I tried six more things and none of them moved me up. A 3-way stacking tuned on the single split lost 0.012 on Kaggle. A single DistilBERT fine-tune lost 0.041. The same 3-way stacking redone with proper cross-validation looked like a gain of 0.018 locally and still lost 0.013 on Kaggle. A DeBERTa-v3 attempt produced NaNs and never got submitted. Swapping MiniLM for mpnet looked even better locally, a gain of 0.031 across all five folds, and still dropped 0.015 on the public board. A hybrid weighted majority vote lost 0.010.

A note on DistilBERT, because it's the tempting one: the v2 fine-tune scored 0.8071 on the single validation split of 565 samples. But the single split overestimates, and no BERT variant beat the TF-IDF and MiniLM stack under proper cross-validation or on Kaggle, so the neural models remained a documented baseline and stayed out of the final pipeline. 0.77750 is the best I got with everything I tried, and sections 14 and 15 of the report have the full story.

## DistilBERT on Colab

For notebooks/10_distilbert_colab.ipynb, which trains two separate DistilBERT models, one for hazard and one for product, open Colab, switch the runtime to GPU (a free T4 is enough), upload the notebook, upload the three csv files from data/raw in the files panel, and hit run all. Expect around 30 to 45 minutes. At the end you download submission_distilbert.csv and submit it to Kaggle. It also saves the logits and the label maps in case you want to ensemble with the main pipeline locally later.

## Final CSVs

results/predictions holds the submissions of every stage. submission_stacking.csv is the final and best one with the 0.77750 public score, and next to it sit the intermediate baselines from the earlier sections, the classical LinearSVC, the SVD600 one, the MLP one, the SOTA tuned one and the two efialtis_kouzina versions. The failed experiments of section 15 aren't kept as files, their Kaggle scores and the analysis live in the report and in the failed experiments section above.

## Deliverables

main.py is the entry point and produces submission_stacking.csv. The report, in Greek, is report.docx and report.pdf, the presentation is presentation.pptx and presentation.pdf, and the notebooks folder holds the full experiment history, both the production path and the documented failures.
