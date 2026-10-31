"""Prédiction en direct : transaction saisie + historique de la carte -> variables -> score.

Réutilise `build_features` (même code qu'à l'entraînement et que l'API) : les variables de vélocité
sont calculées sur les transactions STRICTEMENT antérieures de la carte.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .features import RAW_TX_FIELDS, build_features

RISK_COLORS = {"faible": "#10b981", "moyen": "#f59e0b", "élevé": "#f97316", "critique": "#dc2626"}


def risk_level(score: float, thr: float) -> str:
    """Même grille que l'API."""
    if score >= 0.9:
        return "critique"
    if score >= thr:
        return "élevé"
    if score >= thr / 3:
        return "moyen"
    return "faible"


def card_profile(hist_card: pd.DataFrame) -> dict:
    last = hist_card.sort_values("trans_date_trans_time").iloc[-1]
    return {
        "cc_num": int(last["cc_num"]), "gender": str(last["gender"]), "lat": float(last["lat"]),
        "long": float(last["long"]), "city_pop": int(last["city_pop"]), "dob": str(last["dob"]),
        "last_ts": pd.to_datetime(hist_card["trans_date_trans_time"]).max(), "n_hist": len(hist_card),
        "avg_amt": float(hist_card["amt"].mean()),
    }


def offset_point(lat: float, lon: float, km: float, bearing_deg: float = 45.0):
    """Point situé à `km` kilomètres de (lat, lon) dans la direction `bearing_deg` (approximation plane)."""
    b = math.radians(bearing_deg)
    dlat = km * math.cos(b) / 111.0
    dlon = km * math.sin(b) / (111.0 * max(math.cos(math.radians(lat)), 0.2))
    return lat + dlat, lon + dlon


def build_tx(profile: dict, *, ts: pd.Timestamp, amt: float, category: str, merchant: str,
             dist_km: float, bearing: float = 45.0) -> dict:
    mlat, mlon = offset_point(profile["lat"], profile["long"], dist_km, bearing)
    return {
        "cc_num": profile["cc_num"], "trans_date_trans_time": str(pd.Timestamp(ts)), "merchant": merchant,
        "category": category, "amt": float(amt), "gender": profile["gender"], "lat": profile["lat"],
        "long": profile["long"], "city_pop": profile["city_pop"], "dob": profile["dob"],
        "merch_lat": mlat, "merch_long": mlon,
    }


def burst_rows(tx: dict, n: int, gap_min: float = 7.0) -> list[dict]:
    """n achats simulés juste avant `tx` (même carte, même zone) pour tester les variables de vélocité."""
    t0 = pd.Timestamp(tx["trans_date_trans_time"])
    rows = []
    for i in range(1, n + 1):
        r = dict(tx)
        r["trans_date_trans_time"] = str(t0 - pd.Timedelta(minutes=gap_min * i))
        rows.append(r)
    return rows


def features_for(tx: dict, hist_card: pd.DataFrame | None, categories: list[str], feature_names: list[str],
                 extra_prev: list[dict] | None = None) -> pd.DataFrame:
    """Vecteur de variables (1 ligne) d'une transaction, calculé avec l'historique antérieur de la carte."""
    row = pd.DataFrame([tx])[RAW_TX_FIELDS]
    row["ts"] = pd.to_datetime(row["trans_date_trans_time"])
    parts = []
    if hist_card is not None and len(hist_card):
        h = hist_card[RAW_TX_FIELDS].copy()
        h["ts"] = pd.to_datetime(h["trans_date_trans_time"])
        parts.append(h[h["ts"] < row["ts"].iloc[0]])
    if extra_prev:
        e = pd.DataFrame(extra_prev)[RAW_TX_FIELDS]
        e["ts"] = pd.to_datetime(e["trans_date_trans_time"])
        parts.append(e)
    frame = pd.concat(parts + [row], ignore_index=True)
    X, _ = build_features(frame, categories)
    return X.iloc[[-1]][feature_names].reset_index(drop=True)


def sweep(params: dict, name: str, values, profile: dict, hist_card, categories, feature_names, model,
          n_burst: int = 0) -> np.ndarray:
    """Score en faisant varier un seul paramètre (amt, dist_km ou hour) : analyse « et si ? »."""
    Xs = []
    for v in values:
        p = dict(params)
        if name == "hour":
            d = pd.Timestamp(p["ts"]).normalize()
            p["ts"] = d + pd.Timedelta(hours=float(v))
        else:
            p[name] = float(v)
        tx = build_tx(profile, **p)
        Xs.append(features_for(tx, hist_card, categories, feature_names,
                               burst_rows(tx, n_burst) if n_burst else None))
    return model.predict_proba(pd.concat(Xs, ignore_index=True))[:, 1]
