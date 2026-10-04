import numpy as np
from sklearn.metrics import balanced_accuracy_score

from src.common import backbone_key, choose_threshold, tab_transformer
from src.prepare_data import encode_binary, encode_gender, id_key


def test_choose_threshold_separates_perfectly_separable_data():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)]
    t = choose_threshold(y, p)
    assert balanced_accuracy_score(y, (p >= t).astype(int)) == 1.0


def test_tab_transformer_uses_training_statistics_only():
    X_train = np.array([[0, 1, 20.0], [1, 0, 40.0], [1, 1, np.nan]])
    X_test = np.array([[np.nan, 0, 30.0]])
    tt = tab_transformer(n_bin=2).fit(X_train)
    out = tt.transform(X_test)
    assert np.isfinite(out).all()
    # age is standardised with the training mean (30, after median imputation of the NaN)
    assert np.isclose(out[0, -2], 0.0)


def test_id_key_ignores_leading_zeros_and_case():
    assert id_key("001") == id_key("1") == "1"
    assert id_key(" P12 ") == "p12"
    assert id_key("0") == "0"


def test_encoders():
    import pandas as pd
    assert encode_gender(pd.Series(["Male", "Female", None])).tolist()[:2] == [1.0, 0.0]
    assert encode_binary(pd.Series(["yes", "no", 1, 0]), "x").tolist() == [1.0, 0.0, 1.0, 0.0]


def test_backbone_key():
    assert backbone_key("convnext_tiny.fb_in22k_ft_in1k") == "convnext_tiny"
    assert backbone_key("densenet121") == "densenet121"
