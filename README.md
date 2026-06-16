# Efialtis Stin Kouzina

Food Hazard Detection — SemEval-2025 Task 9, Subtask 1 (ST1).
Εργασία στο μάθημα NLP 053, CSE UOI 2026. AM 4940, Φουρκιώτης Αθανάσιος.

Για κάθε ανάκληση τροφίμου (food recall) θέλουμε να προβλέψουμε δύο labels από
το κείμενο: το `hazard-category` (10 κλάσεις) και το `product-category`
(22 κλάσεις).

## Το όνομα

**Efialtis Stin Kouzina** (Εφιάλτης στην Κουζίνα): το σύστημα εντοπίζει τους
«εφιάλτες» που κρύβονται στις ανακλήσεις τροφίμων — αλλεργιογόνα, μολύνσεις,
ξένα σώματα — πριν φτάσουν στο πιάτο. Δουλεύει σε δύο βήματα: πρώτα βρίσκει το
hazard, και μετά χρησιμοποιεί αυτή την πρόβλεψη για να βοηθήσει την πρόβλεψη του
product.

## Η μετρική

Το επίσημο ST1 δεν είναι απλός μέσος όρος δύο classifiers:

```
(macroF1(hazard) + macroF1(product, μόνο όπου το hazard είναι σωστό)) / 2
```

Δηλαδή αν το hazard βγει λάθος, η σωστή πρόβλεψη του product δεν μετράει σε
εκείνο το παράδειγμα. Το πρόβλημα είναι δύσκολο γιατί οι κλάσεις είναι πολύ
ανισόρροπες (long-tail) και το macro-F1 δίνει ίσο βάρος και στις σπάνιες.

## Ανάλυση δεδομένων (EDA)

Οι κατηγορίες είναι έντονα ανισόρροπες — λίγες κλάσεις (allergens, biological)
έχουν τα περισσότερα παραδείγματα, ενώ αρκετές έχουν ελάχιστα. Αυτό κάνει το
macro-F1 δύσκολο.

![Κατανομή hazard categories](code/results/figures/hazard_dist.png)

![Κατανομή product categories](code/results/figures/product_dist.png)

Η σχέση hazard–product δεν είναι ένα-προς-ένα: το ίδιο hazard εμφανίζεται σε
πολλά προϊόντα και αντίστροφα, οπότε χρειάζεται και το κείμενο και το hazard
signal.

![Σχέση hazard × product](code/results/figures/joint_haz_prod.png)

Τα κείμενα δεν έχουν όλα το ίδιο μήκος (γι' αυτό χρησιμοποίησα και character
n-grams), και η κατανομή ανά έτος δεν είναι ομοιόμορφη.

![Μήκος κειμένων](code/results/figures/doc_length.png)

![Reports ανά έτος](code/results/figures/year_dist.png)

## Η μέθοδος

1. **TF-IDF** σε `title + text + metadata` (word 1–2 grams + char_wb 3–5 grams).
   Τα metadata (country, year, month) μπαίνουν σαν απλά text tokens.
2. **MiniLM** sentence embeddings (384-dim, L2-normalized), stacked στα TF-IDF
   με scale=0.7 — βρέθηκε με sweep, είναι "U-shape" (πολύ ή πολύ λίγο βλάπτει).
3. **Hazard** LinearSVC πάνω στο stacked feature space.
4. **Out-of-fold** (5-fold) hazard predictions στο train: κάθε row παίρνει
   πρόβλεψη από μοντέλο που δεν το είδε, ώστε το product να μάθει έναν
   ρεαλιστικό (~94%) hazard signal χωρίς leakage.
5. Το OOF hazard μπαίνει σαν one-hot feature και το **product** LinearSVC
   εκπαιδεύεται πάνω στο τελικό feature space.
6. Τελικό submission: **stacking ensemble** (TF-IDF/OOF + MiniLM) με βάρη που
   επιλέχθηκαν μέσω cross-validation.

## Αποτελέσματα

| Μοντέλο | Validation ST1 | Kaggle public |
| --- | ---: | ---: |
| TF-IDF + LinearSVC | 0.7599 | — |
| Efialtis Stin Kouzina (TF-IDF + OOF hazard) | 0.7623 | 0.7573 |
| + MiniLM embeddings (scale=0.7) | 0.7737 | 0.7512 |
| **Stacking ensemble (τελικό)** | — | **0.7775** |

Σημαντικό μάθημα: το single validation split (565 δείγματα) ήταν αναξιόπιστο.
Με 5-fold CV το ST1 βγήκε 0.7036 ± 0.0516 — οι «βελτιώσεις» κάτω από ~0.05
χάνονταν στη διακύμανση. Μόνο το stacking πέρασε και από το CV και από το
Kaggle. Δοκίμασα επίσης fine-tuned DistilBERT, αλλά σε αυτό το μικρό dataset με
macro-F1 το κλασικό TF-IDF + LinearSVC (class_weight=balanced) αποδείχθηκε
καλύτερο.

## Δομή

```
code/            κώδικας (notebooks 01–12, src/, tests, main.py)
  notebooks/     EDA -> classical -> SOTA -> embeddings -> CV -> stacking
  src/           helpers (preprocess, scoring, io, models)
  results/       figures, analysis, predictions, logs
data/raw/        train.csv, valid.csv, test.csv
report.pdf       αναφορά (15 ενότητες)
presentation.pdf παρουσίαση
```

Αναλυτικές οδηγίες εκτέλεσης και πλήρης ιστορία πειραμάτων (μαζί με όσα **δεν**
δούλεψαν) στο [`code/README.md`](code/README.md) και στο `report.pdf`.

## Γρήγορη εκτέλεση

```powershell
cd code
pip install -r requirements.txt
python main.py            # παράγει το τελικό submission_stacking.csv
```

## Δεδομένα

Τα δεδομένα προέρχονται από το SemEval-2025 Task 9 / Food Recall Incidents
(CC BY-NC-SA 4.0) και χρησιμοποιούνται μόνο για εκπαιδευτικούς σκοπούς.
