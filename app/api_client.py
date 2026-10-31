"""Client HTTP de l'API de scoring (FastAPI). L'application Streamlit ne score plus en local : elle appelle l'API.

URL configurable : variable d'environnement FRAUD_API_URL (défaut http://localhost:8000).
"""
from __future__ import annotations

import os

import requests

API_URL = os.environ.get("FRAUD_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 20
_session = requests.Session()


class ApiError(RuntimeError):
    """API injoignable ou réponse en erreur (message lisible pour l'utilisateur)."""


def _call(method: str, path: str, **kw):
    try:
        r = _session.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kw)
    except requests.ConnectionError as e:
        raise ApiError(f"API injoignable sur {API_URL}") from e
    except requests.Timeout as e:
        raise ApiError(f"L'API n'a pas répondu en {TIMEOUT} s") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(f"Erreur API {r.status_code} : {detail}")
    return r.json()


def health() -> dict:
    return _call("GET", "/health")


def model_info() -> dict:
    return _call("GET", "/model")


def explain_simulate(transaction: dict, simulated_previous: list[dict] | None = None,
                     threshold: float | None = None) -> dict:
    """Score + décision + codes motifs + résumé en français (POST /explain/simulate)."""
    body = {"transaction": transaction, "simulated_previous": simulated_previous or []}
    if threshold is not None:
        body["threshold"] = threshold
    return _call("POST", "/explain/simulate", json=body)


def predict_features(features: dict[str, float]) -> dict:
    """Score d'un vecteur de variables déjà construit (POST /predict/features)."""
    return _call("POST", "/predict/features", json={"features": features})
