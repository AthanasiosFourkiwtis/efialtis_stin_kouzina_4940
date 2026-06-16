# helpers gia text preprocessing prin to TF-IDF / embeddings.
import re
import pandas as pd


_SPACE_RE = re.compile(r"\s+")


def katharise_keimeno(value):
    # kanei lowercase kai mazeuei polla kena/newlines se ena keno.
    if pd.isna(value):
        return ""
    return _SPACE_RE.sub(" ", str(value).lower()).strip()


def ftiakse_keimeno(df, include_metadata=True):
    # Enwnw title+text. An include_metadata, prosthetw country/year/month
    # san tokens wste to TF-IDF na ta dei san kanonikes lekseis.
    title = df.get("title", "").fillna("").astype(str)
    text = df.get("text", "").fillna("").astype(str)
    combined = title + " " + text

    if include_metadata:
        # metadata mpainoun san apla text tokens, oxi san ksexwrista numeric features.
        country = df.get("country", "").fillna("").astype(str)
        year = df.get("year", "").fillna("").astype(str)
        month = df.get("month", "").fillna("").astype(str)
        combined = "country_" + country + " year_" + year + " month_" + month + " " + combined

    return combined.map(katharise_keimeno)
