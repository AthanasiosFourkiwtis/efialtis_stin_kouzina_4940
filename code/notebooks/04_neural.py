# kanw 04_neural: aplo neural baseline me TF-IDF, SVD kai MLPClassifier.
#
# kaluptw to rubric pou zita neural baseline, opote edw xrhsimopoiw ena sklearn MLP. Epeidh
# MLP thelei dense input, prwta metatrepw to sparse TF-IDF se 300 SVD
# dimensions kai meta kanw L2 normalization. To score menei arketa katw apo ta
# classical models, kati pou deixnei oti i meiwsh diastasewn xanei simantikh
# pliroforia kai oti ta dedomena einai liga gia na doulepsei kala ena MLP.

import sys
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

# prosthetw to project root sto sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.eval_log import grapse_eval
from src.io_utils import PRED_DIR, fortwse_dedomena, grapse_submission
from src.models import ekpaideuse_tfidf, efarmose_tfidf
from src.preprocess import ftiakse_keimeno
from src.scoring import metrhse_st1


def main():
    # fortwnw ta 3 splits (train + valid me labels, test xwris labels)
    train, valid, test = fortwse_dedomena()
    # text = title + text + metadata tokens, lowercase normalized
    x_tr_text = ftiakse_keimeno(train)
    x_va_text = ftiakse_keimeno(valid)
    x_te_text = ftiakse_keimeno(test)

    # ftiaxnw TF-IDF, alla me ligotera max features gia na trexw pio aneta
    # (120k anti gia 250k - to MLP einai pio euaisthito sto megethos edw)
    word, char, x_tr = ekpaideuse_tfidf(x_tr_text, word_max_features=120_000, char_max_features=120_000)
    x_va = efarmose_tfidf(word, char, x_va_text)
    x_te = efarmose_tfidf(word, char, x_te_text)

    # MLP thelei dense/small vectors, opote kanw SVD prwta gia na meiwsoume
    # dimensionality. 300 components = LSA standard.
    # xrhsimopoiw Normalizer L2: kanw kathe row norm=1 (standard meta to LSA)
    # make_pipeline: enwnei ta dyo se ena pipeline pou kanw fit/transform mazi
    reducer = make_pipeline(TruncatedSVD(n_components=300, random_state=42), Normalizer(copy=False))
    z_tr = reducer.fit_transform(x_tr)   # (5082, 300) dense float
    z_va = reducer.transform(x_va)        # (565, 300)
    z_te = reducer.transform(x_te)        # (997, 300)

    # idies ry8miseis kai gia hazard kai gia product MLP
    # hidden_layer_sizes=(256,) = 1 hidden layer me 256 neurons
    # alpha=1e-4 = L2 regularization weight
    # max_iter=80 = mexri 80 epochs (sklearn MLP)
    # early_stopping=True = stamatei an to validation den veltiopoietai gia patience
    common = dict(
        hidden_layer_sizes=(256,),
        alpha=1e-4,
        max_iter=80,
        early_stopping=True,
        random_state=42,
    )
    # xrhsimopoiw LabelEncoder: gyrizei string labels se int 0..N-1 (apaiteitai apo sklearn)
    haz_encoder = LabelEncoder()
    prod_encoder = LabelEncoder()
    y_haz = haz_encoder.fit_transform(train["hazard-category"])
    y_prod = prod_encoder.fit_transform(train["product-category"])

    # ekpaideuw ena MLP gia hazard kai ena gia product (anekartita)
    haz = MLPClassifier(**common)
    prod = MLPClassifier(**common)
    haz.fit(z_tr, y_haz)
    prod.fit(z_tr, y_prod)

    # gyrnaw apo numeric labels pisw sta arxika string labels gia to metrhse_st1
    pred_h = haz_encoder.inverse_transform(haz.predict(z_va))
    pred_p = prod_encoder.inverse_transform(prod.predict(z_va))
    # official ST1 me return_components gia logging
    parts = metrhse_st1(
        valid["hazard-category"], pred_h, valid["product-category"], pred_p, return_components=True
    )
    print(
        f"mlp_svd300 st1={parts['st1']:.4f} "
        f"haz={parts['f1_hazard']:.4f} prod={parts['f1_product_cond']:.4f}"
    )
    grapse_eval("04_neural", "mlp_svd300", "tfidf_word_char+svd300", parts)

    # grafw kai submission gia na yparxei baseline csv (gia consistency me ta alla)
    sub_path = grapse_submission(
        test,
        haz_encoder.inverse_transform(haz.predict(z_te)),
        prod_encoder.inverse_transform(prod.predict(z_te)),
        PRED_DIR / "submission_mlp_svd300.csv",
    )
    print(f"submission -> {sub_path}")


if __name__ == "__main__":
    main()
