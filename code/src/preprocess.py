# helpers for text preprocessing before TF-IDF / embeddings.
import re
import pandas as pd


_SPACE_RE = re.compile(r"\s+")


def clean_text(value):
    # lowercases and collapses repeated spaces/newlines into a single space.
    if pd.isna(value):
        return ""
    return _SPACE_RE.sub(" ", str(value).lower()).strip()


def build_text(df, include_metadata=True):
    # Joins title+text. If include_metadata, appends country/year/month
    # as tokens so TF-IDF sees them as ordinary words.
    title = df.get("title", "").fillna("").astype(str)
    text = df.get("text", "").fillna("").astype(str)
    combined = title + " " + text

    if include_metadata:
        # metadata go in as plain text tokens, not as separate numeric features.
        country = df.get("country", "").fillna("").astype(str)
        year = df.get("year", "").fillna("").astype(str)
        month = df.get("month", "").fillna("").astype(str)
        combined = "country_" + country + " year_" + year + " month_" + month + " " + combined

    return combined.map(clean_text)
