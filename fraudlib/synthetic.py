"""Générateur de données « façon Sparkov » pour tester le pipeline sans télécharger le vrai jeu.

Ne remplace PAS le jeu Kaggle : sert aux tests rapides et au développement hors ligne.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATS = ["grocery_pos", "gas_transport", "shopping_pos", "shopping_net", "misc_net", "misc_pos",
        "food_dining", "home", "entertainment", "health_fitness", "personal_care",
        "kids_pets", "travel", "grocery_net"]
RISKY = ["shopping_net", "misc_net", "grocery_pos", "gas_transport"]
HOUR_P = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2], float)
HOUR_P /= HOUR_P.sum()


def make_synthetic(n_cards=600, days=200, seed=0, start="2019-06-01", fraud_rate=0.006) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start).value // 10**9

    home_lat = rng.uniform(26, 48, n_cards)
    home_lon = rng.uniform(-123, -70, n_cards)
    pop = np.exp(rng.normal(9, 1.6, n_cards)).astype(int) + 50
    gender = rng.choice(["F", "M"], n_cards)
    dob = pd.to_datetime("1950-01-01") + pd.to_timedelta(rng.integers(0, 365 * 50, n_cards), unit="D")
    rate = rng.gamma(2.0, 0.6, n_cards)  # transactions / jour / carte
    cc = 4_000_000_000_000_000 + rng.integers(0, 10**9, n_cards)

    n_per = rng.poisson(rate * days)
    card = np.repeat(np.arange(n_cards), n_per)
    n = len(card)
    sec = (rng.integers(0, days, n) * 86400
           + rng.choice(24, n, p=HOUR_P) * 3600 + rng.integers(0, 3600, n))
    cat = rng.choice(CATS, n)
    amt = np.clip(np.exp(rng.normal(3.3, 0.9, n)), 1, 3000)
    mlat = home_lat[card] + rng.normal(0, 0.4, n)
    mlon = home_lon[card] + rng.normal(0, 0.4, n)
    legit = pd.DataFrame({"card": card, "sec": sec, "category": cat, "amt": amt,
                          "merch_lat": mlat, "merch_long": mlon, "is_fraud": 0})

    # épisodes de fraude : rafales nocturnes, montants élevés, marchands lointains
    n_ep = max(1, int(n * fraud_rate / 2.5))
    rows = []
    for _ in range(n_ep):
        c = rng.integers(0, n_cards)
        base = rng.integers(0, days) * 86400 + rng.choice([22, 23, 0, 1, 2, 3]) * 3600
        t = base
        for _k in range(rng.integers(1, 6)):
            t += int(rng.uniform(60, 2400))
            rows.append((c, t, rng.choice(RISKY), float(np.exp(rng.normal(6.0, 0.5))),
                         rng.uniform(26, 48), rng.uniform(-123, -70), 1))
    fraud = pd.DataFrame(rows, columns=legit.columns)

    d = pd.concat([legit, fraud], ignore_index=True)
    ts = pd.to_datetime(start_ts + d["sec"].to_numpy(), unit="s")
    c = d["card"].to_numpy()
    out = pd.DataFrame({
        "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": cc[c],
        "merchant": "fraud_Merchant" + pd.Series(rng.integers(0, 600, len(d))).astype(str),
        "category": d["category"].to_numpy(),
        "amt": d["amt"].round(2).to_numpy(),
        "gender": gender[c],
        "lat": home_lat[c], "long": home_lon[c], "city_pop": pop[c],
        "dob": dob[c].strftime("%Y-%m-%d"),
        "trans_num": [f"{i:032x}" for i in rng.permutation(len(d))],
        "merch_lat": d["merch_lat"].to_numpy(), "merch_long": d["merch_long"].to_numpy(),
        "is_fraud": d["is_fraud"].to_numpy(),
    })
    return out.sort_values("trans_date_trans_time").reset_index(drop=True)
