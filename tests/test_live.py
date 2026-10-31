"""La prédiction en direct doit reproduire exactement les variables d'entraînement (même code, passé strict)."""
import numpy as np
import pandas as pd

from fraudlib.data import load_raw
from fraudlib.features import RAW_TX_FIELDS, build_features
from fraudlib.live import card_profile, features_for
from fraudlib.synthetic import make_synthetic


def test_live_features_match_training(tmp_path):
    d = make_synthetic(n_cards=30, days=40, seed=1)
    p = tmp_path / "d.csv"
    d.to_csv(p, index=False)
    df = load_raw(str(p))
    X, cats = build_features(df)
    cc = df["cc_num"].value_counts().index[0]
    card = df[df["cc_num"] == cc].reset_index()
    for pos in (5, len(card) // 2, len(card) - 1):
        tx = card.iloc[pos][RAW_TX_FIELDS].to_dict()
        past = card.iloc[:pos]
        got = features_for(tx, past[RAW_TX_FIELDS], cats, list(X.columns))
        want = X.loc[card.loc[pos, "index"], list(X.columns)]
        np.testing.assert_allclose(got.iloc[0].to_numpy(float), want.to_numpy(float), rtol=1e-4, atol=1e-4)


def test_future_history_is_ignored():
    d = make_synthetic(n_cards=10, days=20, seed=2)
    cc = d["cc_num"].iloc[0]
    card = d[d["cc_num"] == cc].reset_index(drop=True)
    tx = card.iloc[3][RAW_TX_FIELDS].to_dict()
    _, cats = build_features(d.assign(ts=pd.to_datetime(d["trans_date_trans_time"])))
    X, _ = build_features(d.assign(ts=pd.to_datetime(d["trans_date_trans_time"])), cats)
    a = features_for(tx, card[RAW_TX_FIELDS].iloc[:3], cats, list(X.columns))
    b = features_for(tx, card[RAW_TX_FIELDS], cats, list(X.columns))  # historique complet, dont le futur
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float))


def test_plain_explanation_alert_and_ok():
    from fraudlib.explain import plain_explanation

    up = [{"label": "Montant", "value": 900.0, "contribution": 2.1}]
    down = [{"label": "Week-end", "value": 0.0, "contribution": -0.4}]
    assert plain_explanation(0.9, 0.5, up, down).startswith("Alerte")
    assert plain_explanation(0.1, 0.5, up, down).startswith("Transaction acceptée")
