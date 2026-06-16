# kanw 01_eda: Exploratory Data Analysis gia tin anafora.
#
# ftiaxnw ta vasika plots pou deixnoun tin katanomi twn hazard categories,
# tin katanomi twn product categories, ta mhkh twn keimenwn, tin xronologiki
# katanomi kai tin sxesh hazard me product. Auta ta figures mpainoun sto report
# gia na kalyptei i ergasia tin enothta analysis dedomenwn tou rubric.
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# prosthetw to project root sto sys.path gia na vroume to src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io_utils import FIG_DIR, fortwse_dedomena
from src.preprocess import ftiakse_keimeno


def swse_bar(series, title, path, top=None):
    # bar plot helper. An doseis top=N, krataei mono tis top N kathgories
    # (xrhsimo gia products pou exoun 22 kai den xwran ola).
    counts = series.value_counts()
    if top is not None:
        # krata mono ta top N (xrhsimo gia products me 22 classes)
        counts = counts.head(top)
    # figure height dynamics se sxesh me to n_classes (gia na min einai squashed)
    plt.figure(figsize=(10, max(4, 0.28 * len(counts))))
    # horizontal bar plot: counts sto x-axis, class names sto y-axis
    sns.barplot(x=counts.values, y=counts.index, color="#4c78a8")
    plt.title(title)
    plt.xlabel("count")
    plt.ylabel("")  # to y-axis labels einai ta class names, kaluterh xwris extra label
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()  # eleutheronei memory


def main():
    # fortwnw train+valid giati EXOUN labels kai mporw na kanw analysis
    # test den exei labels, ara den symvalei sto EDA
    train, valid, _ = fortwse_dedomena()
    # enwnoume train + valid me ena "split" column gia identification
    labelled = pd.concat([train.assign(split="train"), valid.assign(split="valid")], ignore_index=True)
    # auto-creator tou results/figures fakelou
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # plot: posa paradeigmata exei kathe hazard class
    # anamenoume megalh imbalance: allergens kai biological exoun pio polla,
    # packaging defect, organoleptic etc exoun ligotera. Auto fanei kai sto plot.
    swse_bar(labelled["hazard-category"], "Hazard category distribution", FIG_DIR / "hazard_dist.png")

    # plot: products (top 30 mono)
    # 22 product classes alla deixnoume top 30 (ola edw, exoun 22)
    swse_bar(
        labelled["product-category"],
        "Product category distribution",
        FIG_DIR / "product_dist.png",
        top=30,
    )

    # plot: posa tokens exei kathe recall text
    # ftiakse_keimeno(include_metadata=False) dinw mono title+text (no metadata)
    # .str.split() = sxizoume sto whitespace (proxy gia word tokens)
    # .map(len) = mhkos kathe split
    text_len = ftiakse_keimeno(labelled, include_metadata=False).str.split().map(len)
    plt.figure(figsize=(8, 4))
    sns.histplot(text_len, bins=40, color="#59a14f")
    plt.title("Document length")
    plt.xlabel("tokens")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "doc_length.png", dpi=160)
    plt.close()

    # plot: distribution ana xronia
    # to dataset exei reports apo ~1994 mexri 2022, me anisi katanomi ana xronia
    plt.figure(figsize=(9, 4))
    sns.countplot(data=labelled, x="year", color="#f28e2b")
    plt.title("Reports by year")
    # rotation 45 gia na min epikaluptountai ta year labels sto x-axis
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "year_dist.png", dpi=160)
    plt.close()

    # plot: hazard x product heatmap
    # pinakas hazard x product gia na dw poia zeugaria emfanizontai syxna.
    # crosstab = enas pinakas tipou confusion (hazard rows, product cols)
    # klassiko pattern: biological -> meat/dairy, allergens -> cereals, etc.
    joint = pd.crosstab(labelled["hazard-category"], labelled["product-category"])
    plt.figure(figsize=(13, 6))
    sns.heatmap(joint, cmap="viridis")
    plt.title("Hazard vs product category")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "joint_haz_prod.png", dpi=160)
    plt.close()
    print(f"figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
