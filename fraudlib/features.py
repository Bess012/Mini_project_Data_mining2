"""Module 2 : variables temporelles et comportementales, sans fuite de données.

Toutes les variables comportementales (vélocité, écart au profil de la carte) sont calculées
uniquement à partir des transactions STRICTEMENT ANTÉRIEURES de la même carte : la transaction
courante n'entre jamais dans sa propre fenêtre. Calculer ces variables sur l'ensemble du flux
avant la séparation est donc sans fuite (en production, on ferait exactement la même chose).
Le test tests/test_features.py vérifie cette propriété.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86_400
BIG = 1 << 33  # > plage des timestamps + fenêtre : isole chaque carte dans la clé composite

RAW_TX_FIELDS = [
    "cc_num", "trans_date_trans_time", "merchant", "category", "amt", "gender",
    "lat", "long", "city_pop", "dob", "merch_lat", "merch_long",
]

BASE_FEATURES = [
    "amt", "log_amt",
    "hour_sin", "hour_cos", "dow", "is_weekend", "is_night",
    "age", "log_city_pop", "gender_M", "dist_home_merch_km",
    "n_tx_1h", "n_tx_24h", "n_tx_30d", "amt_sum_24h", "amt_ratio_30d",
    "log_secs_since_prev", "dist_prev_merch_km", "speed_kmh", "is_first_tx",
]

FEATURE_LABELS = {
    "amt": "Montant",
    "log_amt": "Montant (log)",
    "hour_sin": "Heure (sin)",
    "hour_cos": "Heure (cos)",
    "dow": "Jour de la semaine",
    "is_weekend": "Week-end",
    "is_night": "Transaction de nuit",
    "age": "Âge du client",
    "log_city_pop": "Population de la ville (log)",
    "gender_M": "Client de genre masculin",
    "dist_home_merch_km": "Distance domicile-marchand (km)",
    "n_tx_1h": "Nb de transactions sur 1 h",
    "n_tx_24h": "Nb de transactions sur 24 h",
    "n_tx_30d": "Nb de transactions sur 30 j",
    "amt_sum_24h": "Montant cumulé sur 24 h",
    "amt_ratio_30d": "Montant / moyenne 30 j de la carte",
    "log_secs_since_prev": "Temps depuis transaction précédente (log s)",
    "dist_prev_merch_km": "Distance au marchand précédent (km)",
    "speed_kmh": "Vitesse implicite entre marchands (km/h)",
    "is_first_tx": "Première transaction connue de la carte",
}


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("cat_"):
        return f"Catégorie : {name[4:]}"
    return name


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _past_window(key: np.ndarray, values: np.ndarray, window: int):
    """Pour chaque ligne (triée par carte puis temps) : nombre et somme des transactions de la
    même carte dans [t - window, t[ — la ligne courante est exclue."""
    n = len(key)
    idx = np.arange(n)
    left = np.searchsorted(key, key - window, side="left")
    cnt = idx - left
    cs = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    return cnt.astype(np.float64), cs[idx] - cs[left]


def build_features(df: pd.DataFrame, categories: list[str] | None = None):
    """Retourne (X, categories). X : DataFrame float32 aligné sur l'index de df.

    `categories` doit être figé à l'entraînement et réutilisé à l'inférence.
    """
    n = len(df)
    ts = df["ts"] if "ts" in df.columns else pd.to_datetime(df["trans_date_trans_time"])
    ts = pd.to_datetime(ts)
    t = ts.to_numpy().astype("datetime64[s]").astype(np.int64)

    amt = df["amt"].to_numpy(float)
    lat, lon = df["lat"].to_numpy(float), df["long"].to_numpy(float)
    mlat, mlon = df["merch_lat"].to_numpy(float), df["merch_long"].to_numpy(float)

    # --- temporelles
    hour = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy(float)
    dow = ts.dt.dayofweek.to_numpy(float)
    feats = {
        "amt": amt,
        "log_amt": np.log1p(amt),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow": dow,
        "is_weekend": (dow >= 5).astype(float),
        "is_night": ((hour >= 22) | (hour < 4)).astype(float),
        "age": ((ts - pd.to_datetime(df["dob"])).dt.days / 365.25).to_numpy(float),
        "log_city_pop": np.log1p(df["city_pop"].to_numpy(float)),
        "gender_M": (df["gender"].to_numpy() == "M").astype(float),
        "dist_home_merch_km": haversine_km(lat, lon, mlat, mlon),
    }

    # --- comportementales (tri par carte puis temps)
    codes, _ = pd.factorize(df["cc_num"])
    order = np.lexsort((t, codes))
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)

    t_s, c_s = t[order], codes[order].astype(np.int64)
    a_s = amt[order]
    key = c_s * BIG + t_s

    n1, _ = _past_window(key, a_s, 3600)
    n24, s24 = _past_window(key, a_s, DAY)
    n30, s30 = _past_window(key, a_s, 30 * DAY)
    mean30 = s30 / np.maximum(n30, 1)
    ratio = np.where(n30 > 0, a_s / np.maximum(mean30, 1e-6), 1.0)

    same = np.zeros(n, dtype=bool)
    same[1:] = c_s[1:] == c_s[:-1]
    dt = np.full(n, 30.0 * DAY)
    dt[1:] = np.where(same[1:], t_s[1:] - t_s[:-1], 30.0 * DAY)
    ml_s, mo_s = mlat[order], mlon[order]
    dprev = np.zeros(n)
    dprev[1:] = np.where(same[1:], haversine_km(ml_s[:-1], mo_s[:-1], ml_s[1:], mo_s[1:]), 0.0)
    speed = np.where(same, dprev / (np.maximum(dt, 60.0) / 3600.0), 0.0)

    feats["n_tx_1h"] = n1[inv]
    feats["n_tx_24h"] = n24[inv]
    feats["n_tx_30d"] = n30[inv]
    feats["amt_sum_24h"] = s24[inv]
    feats["amt_ratio_30d"] = np.clip(ratio, 0, 100)[inv]
    feats["log_secs_since_prev"] = np.log1p(dt)[inv]
    feats["dist_prev_merch_km"] = dprev[inv]
    feats["speed_kmh"] = np.clip(speed, 0, 3000)[inv]
    feats["is_first_tx"] = (~same).astype(float)[inv]

    X = pd.DataFrame({k: feats[k] for k in BASE_FEATURES}, index=df.index)

    # --- catégorie du marchand (one-hot, liste figée)
    if categories is None:
        categories = sorted(df["category"].dropna().unique().tolist())
    cat_values = df["category"].to_numpy()
    for c in categories:
        X[f"cat_{c}"] = (cat_values == c).astype(float)

    return X.astype("float32"), categories
