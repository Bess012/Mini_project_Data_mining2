"""Module 6 : API FastAPI de scoring et d'explication des transactions.

Lancement (depuis la racine du projet) :
    uvicorn api.main:app --reload --port 8000
Documentation interactive : http://localhost:8000/docs
"""
from __future__ import annotations

import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudlib.config import ARTIFACTS  # noqa: E402
from fraudlib.explain import make_explainer, plain_explanation, reason_codes, shap_values  # noqa: E402
from fraudlib.features import RAW_TX_FIELDS, build_features, feature_label  # noqa: E402

STATE: dict = {}
LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle_path = ARTIFACTS / "model_bundle.joblib"
    if not bundle_path.exists():
        raise RuntimeError(f"{bundle_path} introuvable : lancer d'abord `python scripts/train.py --data ...`")
    bundle = joblib.load(bundle_path)
    hist = pd.read_parquet(ARTIFACTS / "history.parquet")
    hist["ts"] = pd.to_datetime(hist["trans_date_trans_time"])
    STATE["bundle"] = bundle
    STATE["explainer"] = make_explainer(bundle["model"])
    STATE["threshold"] = float(bundle["threshold"])
    STATE["history"] = {cc: g.reset_index(drop=True) for cc, g in hist.groupby("cc_num")}
    yield
    STATE.clear()


app = FastAPI(
    title="API de détection de fraude explicable (Sparkov)",
    description="Score une transaction carte bancaire, décide l'alerte selon le seuil métier "
                "et fournit des codes motifs (SHAP).",
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------- schémas
class Transaction(BaseModel):
    cc_num: int = Field(..., description="Identifiant de la carte")
    trans_date_trans_time: str = Field(..., examples=["2020-12-30 23:41:12"])
    merchant: str
    category: str = Field(..., examples=["shopping_net"])
    amt: float = Field(..., gt=0)
    gender: str = Field(..., examples=["F"])
    lat: float
    long: float
    city_pop: int
    dob: str = Field(..., examples=["1985-03-14"])
    merch_lat: float
    merch_long: float
    trans_num: Optional[str] = None


class ReasonCode(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float = Field(..., description="Contribution SHAP (log-odds) ; > 0 pousse vers la fraude")


class Prediction(BaseModel):
    trans_num: Optional[str]
    score: float = Field(..., description="Probabilité estimée de fraude")
    threshold: float
    is_alert: bool
    risk_level: str
    reason_codes: list[ReasonCode]


class Explanation(Prediction):
    base_value: float
    contributions: list[ReasonCode]
    summary: str = Field("", description="Explication courte en français : pourquoi alerte / pas d'alerte")


class SimulationRequest(BaseModel):
    """Transaction + scénario « et si ? » : achats fictifs juste avant, seuil temporaire."""
    transaction: Transaction
    simulated_previous: list[Transaction] = Field(default_factory=list, description="Achats fictifs antérieurs (vélocité)")
    threshold: Optional[float] = Field(None, gt=0, lt=1, description="Seuil utilisé pour CET appel uniquement")


class FeatureVector(BaseModel):
    features: dict[str, float]


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., gt=0, lt=1)


# ----------------------------------------------------------------------------- logique
def _risk_level(score: float, thr: float) -> str:
    if score >= 0.9:
        return "critique"
    if score >= thr:
        return "élevé"
    if score >= thr / 3:
        return "moyen"
    return "faible"


def _features_for(tx: Transaction, extra_prev: Optional[list[Transaction]] = None) -> pd.DataFrame:
    """Reconstruit les variables de vélocité à partir de l'historique de la carte (même code qu'à l'entraînement)."""
    b = STATE["bundle"]
    row = pd.DataFrame([tx.model_dump()])[RAW_TX_FIELDS]
    row["ts"] = pd.to_datetime(row["trans_date_trans_time"])
    hist = STATE["history"].get(tx.cc_num)
    if hist is not None and tx.trans_num is not None:
        hist = hist[hist["trans_num"] != tx.trans_num]  # rejeu d'une transaction déjà connue : pas de doublon
    parts = []
    if hist is not None and len(hist):
        parts.append(hist[hist["ts"] < row["ts"].iloc[0]][RAW_TX_FIELDS + ["ts"]])  # passé strict uniquement
    if extra_prev:
        e = pd.DataFrame([t.model_dump() for t in extra_prev])[RAW_TX_FIELDS]
        e["ts"] = pd.to_datetime(e["trans_date_trans_time"])
        parts.append(e)
    frame = pd.concat(parts + [row], ignore_index=True)
    X, _ = build_features(frame, b["categories"])
    return X.iloc[[-1]][b["features"]]


def _commit(tx: Transaction):
    row = pd.DataFrame([tx.model_dump()])
    row["ts"] = pd.to_datetime(row["trans_date_trans_time"])
    hist = STATE["history"].get(tx.cc_num)
    if hist is not None and tx.trans_num is not None and (hist["trans_num"] == tx.trans_num).any():
        return  # déjà vue : pas de double comptage
    new = pd.concat([hist, row], ignore_index=True) if hist is not None else row
    cutoff = new["ts"].max() - pd.Timedelta(days=STATE["bundle"]["history_days"])
    STATE["history"][tx.cc_num] = new[new["ts"] >= cutoff].reset_index(drop=True)


def _score_vector(X: pd.DataFrame, full: bool = False, threshold: Optional[float] = None):
    b = STATE["bundle"]
    thr = STATE["threshold"] if threshold is None else float(threshold)
    score = float(b["model"].predict_proba(X)[0, 1])
    sv, base = shap_values(STATE["explainer"], X)
    codes = reason_codes(sv[0], X.iloc[0])
    out = {"score": score, "threshold": thr, "is_alert": score >= thr,
           "risk_level": _risk_level(score, thr), "reason_codes": codes}
    if full:
        neg = [{"feature": X.columns[i], "label": feature_label(X.columns[i]), "value": float(X.iloc[0, i]),
                "contribution": float(sv[0, i])} for i in np.argsort(sv[0])[:3]]
        out["summary"] = plain_explanation(score, thr, codes, neg)
        order = np.argsort(-np.abs(sv[0]))[:15]
        out["base_value"] = base
        out["contributions"] = [{"feature": X.columns[i], "label": feature_label(X.columns[i]),
                                 "value": float(X.iloc[0, i]), "contribution": float(sv[0, i])} for i in order]
    return out


def _predict_one(tx: Transaction, commit: bool, full: bool = False, extra_prev: Optional[list[Transaction]] = None,
                 threshold: Optional[float] = None):
    with LOCK:
        X = _features_for(tx, extra_prev)
        res = _score_vector(X, full, threshold)
        if commit:
            _commit(tx)
    res["trans_num"] = tx.trans_num
    return res


# ----------------------------------------------------------------------------- routes
@app.get("/health", tags=["Service"])
def health():
    return {"status": "ok", "cartes_en_historique": len(STATE["history"])}


@app.get("/model", tags=["Service"])
def model_info():
    b = STATE["bundle"]
    return {
        "model": type(b["model"]).__name__, "imbalance_strategy": b["strategy"],
        "threshold": STATE["threshold"], "review_cost": b["review_cost"],
        "budget_per_day": b["budget_per_day"], "n_features": len(b["features"]),
        "features": [{"name": f, "label": feature_label(f)} for f in b["features"]],
        "trained_on": b["trained_on"],
    }


@app.get("/threshold", tags=["Seuil"])
def get_threshold():
    return {"threshold": STATE["threshold"], "default": float(STATE["bundle"]["threshold"])}


@app.put("/threshold", tags=["Seuil"])
def set_threshold(body: ThresholdUpdate):
    STATE["threshold"] = body.threshold
    return {"threshold": STATE["threshold"]}


@app.post("/predict", response_model=Prediction, tags=["Scoring"])
def predict(tx: Transaction, commit: bool = Query(True, description="Ajouter la transaction à l'historique de la carte")):
    """Score une transaction brute. Les variables de vélocité sont calculées côté serveur."""
    return _predict_one(tx, commit)


@app.post("/predict/batch", response_model=list[Prediction], tags=["Scoring"])
def predict_batch(txs: list[Transaction], commit: bool = True):
    """Score une liste (≤ 1000) de transactions, traitées dans l'ordre fourni."""
    if len(txs) > 1000:
        raise HTTPException(413, "1000 transactions maximum par appel")
    return [_predict_one(tx, commit) for tx in txs]


@app.post("/predict/features", response_model=Prediction, tags=["Scoring"])
def predict_features(body: FeatureVector):
    """Score un vecteur de variables déjà construit (les variables absentes valent 0)."""
    b = STATE["bundle"]
    unknown = set(body.features) - set(b["features"])
    if unknown:
        raise HTTPException(422, f"Variables inconnues : {sorted(unknown)}")
    X = pd.DataFrame([{f: body.features.get(f, 0.0) for f in b["features"]}], dtype="float32")
    with LOCK:
        res = _score_vector(X)
    res["trans_num"] = None
    return res


@app.post("/explain", response_model=Explanation, tags=["Explicabilité"])
def explain(tx: Transaction):
    """Explication détaillée d'une transaction (n'altère pas l'historique)."""
    return _predict_one(tx, commit=False, full=True)


@app.post("/explain/simulate", response_model=Explanation, tags=["Explicabilité"])
def explain_simulate(body: SimulationRequest):
    """Scénario « et si ? » : explication d'une transaction avec achats fictifs antérieurs et seuil temporaire.

    N'altère ni l'historique ni le seuil global du serveur.
    """
    return _predict_one(body.transaction, commit=False, full=True,
                        extra_prev=body.simulated_previous, threshold=body.threshold)
