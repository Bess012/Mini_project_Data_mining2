"""Module 1 : chargement, séparation chronologique, statistiques exploratoires."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SPLIT, TARGET
from .features import haversine_km

USECOLS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "lat", "long", "city_pop", "dob", "trans_num", "merch_lat", "merch_long", TARGET,
]
REQUIRED = [c for c in USECOLS if c != "trans_num"]


def load_raw(sources) -> pd.DataFrame:
    """Charge un ou plusieurs CSV Sparkov (fraudTrain.csv + fraudTest.csv), ou des fichiers importés.

    Les deux fichiers Kaggle sont concaténés puis retriés par date : la séparation
    60/20/20 est refaite par nos soins, chronologiquement.
    """
    if not isinstance(sources, (list, tuple)):
        sources = [sources]
    frames = [pd.read_csv(s, usecols=lambda c: c in USECOLS) for s in sources]
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour le jeu Sparkov : {missing}")
    if "trans_num" not in df.columns:
        df["trans_num"] = [f"row{i}" for i in range(len(df))]
    df = df.drop_duplicates("trans_num")
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    return df


def _advance_past_ties(ts: np.ndarray, i: int) -> int:
    """Décale la frontière pour ne pas couper un groupe de transactions de même horodatage."""
    while 0 < i < len(ts) and ts[i] == ts[i - 1]:
        i += 1
    return i


def chronological_split(df: pd.DataFrame, fractions=SPLIT) -> dict[str, np.ndarray]:
    """60 % entraînement / 20 % validation / 20 % test, dans l'ordre du temps."""
    if "ts" not in df.columns:
        raise ValueError("Pas d'information temporelle : utiliser random_stratified_split (à signaler).")
    ts = df["ts"].to_numpy()
    n = len(df)
    i1 = _advance_past_ties(ts, int(n * fractions[0]))
    i2 = _advance_past_ties(ts, int(n * (fractions[0] + fractions[1])))
    idx = np.arange(n)
    return {"train": idx[:i1], "val": idx[i1:i2], "test": idx[i2:]}


def random_stratified_split(y: np.ndarray, fractions=SPLIT, seed: int = 42) -> dict[str, np.ndarray]:
    """À n'utiliser que si le jeu n'a AUCUNE information temporelle (non le cas de Sparkov)."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=fractions[0], stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=fractions[1] / (fractions[1] + fractions[2]),
                              stratify=y[rest], random_state=seed)
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def split_summary(df: pd.DataFrame, idx: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, ix in idx.items():
        part = df.iloc[ix]
        rows.append({
            "jeu": name,
            "début": part["ts"].min(),
            "fin": part["ts"].max(),
            "transactions": len(part),
            "fraudes": int(part[TARGET].sum()),
            "taux de fraude (%)": round(100 * part[TARGET].mean(), 3),
        })
    return pd.DataFrame(rows)


def compute_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Agrégats exploratoires (utilisés par l'application, y compris sur un fichier importé)."""
    y = df[TARGET]
    ts = df["ts"]
    out: dict[str, pd.DataFrame] = {}

    out["summary"] = pd.DataFrame([{
        "transactions": len(df),
        "fraudes": int(y.sum()),
        "taux_fraude_pct": 100 * float(y.mean()),
        "cartes": int(df["cc_num"].nunique()),
        "début": ts.min(),
        "fin": ts.max(),
        "montant_moyen": float(df["amt"].mean()),
        "montant_moyen_fraude": float(df.loc[y == 1, "amt"].mean()) if y.sum() else np.nan,
    }])

    def rate(by, name):
        g = df.groupby(by)[TARGET].agg(["size", "sum", "mean"]).reset_index()
        g.columns = [name, "n", "n_fraud", "rate"]
        g["rate"] *= 100
        return g

    out["month"] = rate(ts.dt.to_period("M").dt.to_timestamp(), "mois")
    out["hour"] = rate(ts.dt.hour, "heure")
    out["dow"] = rate(ts.dt.dayofweek, "jour")
    out["category"] = rate(df["category"], "catégorie").sort_values("rate", ascending=False)
    out["gender"] = rate(df["gender"], "genre")

    lo, hi = np.log10(max(df["amt"].min(), 0.5)), np.log10(df["amt"].max())
    bins = np.linspace(lo, hi, 41)
    mids = 10 ** ((bins[:-1] + bins[1:]) / 2)
    parts = []
    for label, mask in (("Légitime", y == 0), ("Fraude", y == 1)):
        if mask.sum() == 0:
            continue
        h, _ = np.histogram(np.log10(df.loc[mask, "amt"].clip(lower=0.5)), bins=bins)
        parts.append(pd.DataFrame({"montant": mids, "classe": label, "part (%)": 100 * h / h.sum()}))
    out["amt_hist"] = pd.concat(parts, ignore_index=True)
    out.update(_extra_eda(df))
    return out


def _class_hist(values: pd.Series, y: pd.Series, bins, name: str) -> pd.DataFrame:
    parts = []
    mids = (bins[:-1] + bins[1:]) / 2
    for label, mask in (("Légitime", y == 0), ("Fraude", y == 1)):
        if mask.sum() == 0:
            continue
        h, _ = np.histogram(values[mask].clip(bins[0], bins[-1]), bins=bins)
        parts.append(pd.DataFrame({name: mids, "classe": label, "part (%)": 100 * h / max(h.sum(), 1)}))
    return pd.concat(parts, ignore_index=True)


def _extra_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Agrégats supplémentaires pour mieux comprendre le jeu (onglet Données)."""
    y, ts = df[TARGET], df["ts"]
    out: dict[str, pd.DataFrame] = {}

    # série quotidienne
    d = pd.DataFrame({"jour": ts.dt.floor("D"), "y": y}).groupby("jour")["y"].agg(["size", "sum"]).reset_index()
    d.columns = ["jour", "transactions", "fraudes"]
    out["daily"] = d

    # carte de chaleur jour de semaine x heure
    h = pd.DataFrame({"jour": ts.dt.dayofweek, "heure": ts.dt.hour, "y": y}).groupby(["jour", "heure"])["y"].agg(
        ["size", "sum"]).reset_index()
    h.columns = ["jour", "heure", "n", "n_fraud"]
    h["rate"] = 100 * h["n_fraud"] / h["n"]
    out["heat"] = h

    # distance domicile-marchand
    dist = pd.Series(haversine_km(df["lat"].to_numpy(float), df["long"].to_numpy(float),
                                  df["merch_lat"].to_numpy(float), df["merch_long"].to_numpy(float)), index=df.index)
    out["dist_hist"] = _class_hist(dist, y, np.linspace(0, float(np.nanpercentile(dist, 99.5)), 41), "distance (km)")

    # âge
    age = (ts - pd.to_datetime(df["dob"])).dt.days / 365.25
    out["age_hist"] = _class_hist(age, y, np.linspace(float(age.min()), float(age.max()), 31), "âge")
    band = pd.cut(age, [0, 25, 35, 45, 55, 65, 75, 120], right=False,
                  labels=["<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"])
    a = pd.DataFrame({"tranche": band, "y": y}).groupby("tranche", observed=True)["y"].agg(["size", "sum", "mean"]).reset_index()
    a.columns = ["tranche d'âge", "n", "n_fraud", "rate"]
    a["rate"] *= 100
    out["age_rate"] = a

    # taille de ville (déciles)
    try:
        q = pd.qcut(df["city_pop"], 8, duplicates="drop")
        c = pd.DataFrame({"q": q, "y": y}).groupby("q", observed=True)["y"].agg(["size", "sum", "mean"]).reset_index()
        c["taille de ville"] = [f"{int(i.left):,}–{int(i.right):,}".replace(",", " ") for i in c["q"]]
        c = c.drop(columns="q")
        c.columns = ["n", "n_fraud", "rate", "taille de ville"]
        c["rate"] *= 100
        out["city_rate"] = c
    except ValueError:
        pass

    # montant médian / moyen par catégorie et classe
    g = df.groupby(["category", TARGET])["amt"].agg(["median", "mean", "size"]).reset_index()
    g["classe"] = g[TARGET].map({0: "Légitime", 1: "Fraude"})
    out["cat_amt"] = g.drop(columns=TARGET).rename(columns={"category": "catégorie", "median": "médiane", "mean": "moyenne",
                                                           "size": "n"})

    # corrélation (Pearson) de quelques variables simples avec la fraude
    num = pd.DataFrame({
        "Montant": df["amt"], "Montant (log)": np.log1p(df["amt"]), "Heure": ts.dt.hour,
        "Nuit (22h-4h)": ((ts.dt.hour >= 22) | (ts.dt.hour < 4)).astype(float), "Week-end": (ts.dt.dayofweek >= 5).astype(float),
        "Âge": age, "Population ville (log)": np.log1p(df["city_pop"]), "Distance domicile-marchand": dist,
        "Fraude": y.astype(float),
    })
    out["corr"] = num.corr().round(3)
    return out
