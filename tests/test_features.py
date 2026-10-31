"""Vérifie l'absence de fuite : les variables d'une transaction ne dépendent que du passé."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudlib.data import chronological_split  # noqa: E402
from fraudlib.features import build_features  # noqa: E402
from fraudlib.synthetic import make_synthetic  # noqa: E402
import pandas as pd  # noqa: E402


def _df():
    d = make_synthetic(n_cards=40, days=30, seed=3)
    d["ts"] = pd.to_datetime(d["trans_date_trans_time"])
    return d.sort_values("ts", kind="stable").reset_index(drop=True)


def test_features_do_not_use_the_future():
    df = _df()
    full, cats = build_features(df)
    m = int(len(df) * 0.7)
    part, _ = build_features(df.iloc[:m].copy(), cats)
    assert np.allclose(full.iloc[:m].to_numpy(), part.to_numpy(), equal_nan=True)


def test_current_transaction_excluded_from_its_own_window():
    df = _df()
    X, _ = build_features(df)
    first = df.groupby("cc_num").head(1).index
    assert (X.loc[first, "n_tx_30d"] == 0).all()
    assert (X.loc[first, "is_first_tx"] == 1).all()


def test_chronological_split_is_ordered():
    df = _df()
    idx = chronological_split(df)
    assert df["ts"].iloc[idx["train"]].max() <= df["ts"].iloc[idx["val"]].min()
    assert df["ts"].iloc[idx["val"]].max() <= df["ts"].iloc[idx["test"]].min()
    assert sum(len(v) for v in idx.values()) == len(df)
