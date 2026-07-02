# 01_eda: Exploratory Data Analysis for the report.
#
# Builds the core plots showing the distribution of hazard categories,
# the distribution of product categories, the document lengths, the yearly
# distribution, and the hazard-product relationship. These figures go into the
# report so the project covers the data-analysis section of the rubric.
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# add the project root to sys.path so src/ can be found
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.io_utils import FIG_DIR, load_data
from src.preprocess import build_text


def save_bar(series, title, path, top=None):
    # bar plot helper. Passing top=N keeps only the top N categories
    # (useful for products, which have 22 and don't all fit).
    counts = series.value_counts()
    if top is not None:
        # keep only the top N (useful for products with 22 classes)
        counts = counts.head(top)
    # figure height scales with n_classes (so nothing gets squashed)
    plt.figure(figsize=(10, max(4, 0.28 * len(counts))))
    # horizontal bar plot: counts on the x-axis, class names on the y-axis
    sns.barplot(x=counts.values, y=counts.index, color="#4c78a8")
    plt.title(title)
    plt.xlabel("count")
    plt.ylabel("")  # the y-axis labels are the class names; cleaner without an extra label
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()  # frees memory


def main():
    # load train+valid because they HAVE labels and can be analyzed
    # test has no labels, so it contributes nothing to the EDA
    train, valid, _ = load_data()
    # join train + valid with a "split" column for identification
    labelled = pd.concat([train.assign(split="train"), valid.assign(split="valid")], ignore_index=True)
    # auto-creates the results/figures folder
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # plot: how many examples each hazard class has
    # heavy imbalance is expected: allergens and biological have the most,
    # packaging defect, organoleptic etc. have fewer. The plot shows it too.
    save_bar(labelled["hazard-category"], "Hazard category distribution", FIG_DIR / "hazard_dist.png")

    # plot: products (top 30 only)
    # 22 product classes, but top 30 is shown (which covers all 22 here)
    save_bar(
        labelled["product-category"],
        "Product category distribution",
        FIG_DIR / "product_dist.png",
        top=30,
    )

    # plot: how many tokens each recall text has
    # build_text(include_metadata=False) gives just title+text (no metadata)
    # .str.split() = split on whitespace (a proxy for word tokens)
    # .map(len) = length of each split
    text_len = build_text(labelled, include_metadata=False).str.split().map(len)
    plt.figure(figsize=(8, 4))
    sns.histplot(text_len, bins=40, color="#59a14f")
    plt.title("Document length")
    plt.xlabel("tokens")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "doc_length.png", dpi=160)
    plt.close()

    # plot: distribution per year
    # the dataset holds reports from ~1994 up to 2022, unevenly spread across years
    plt.figure(figsize=(9, 4))
    sns.countplot(data=labelled, x="year", color="#f28e2b")
    plt.title("Reports by year")
    # rotate 45 degrees so the year labels don't overlap on the x-axis
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "year_dist.png", dpi=160)
    plt.close()

    # plot: hazard x product heatmap
    # a hazard x product table showing which pairs co-occur frequently.
    # crosstab = a confusion-style table (hazard rows, product cols)
    # classic pattern: biological -> meat/dairy, allergens -> cereals, etc.
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
