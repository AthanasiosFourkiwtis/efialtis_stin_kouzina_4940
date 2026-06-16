# Reusable TF-IDF helpers for the notebook-style experiment scripts.

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


def ekpaideuse_tfidf(train_text, word_max_features=250_000, char_max_features=250_000):
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        max_features=word_max_features,
    )
    # Character n-grams help with brands, typos and short product names.
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=char_max_features,
    )
    Xw = word.fit_transform(train_text)
    Xc = char.fit_transform(train_text)
    return word, char, hstack([Xw, Xc]).tocsr()


def efarmose_tfidf(word, char, text):
    return hstack([word.transform(text), char.transform(text)]).tocsr()
