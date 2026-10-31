"""Client HTTP de l'API de scoring (FastAPI). L'application Streamlit ne score plus en local : elle appelle l'API.

URL configurable : variable d'environnement FRAUD_API_URL (défaut http://localhost:8000).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

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


_proc: subprocess.Popen | None = None


def ensure_api(wait: int = 120) -> dict:
    """Renvoie /health ; si l'API est locale et éteinte (cas d'un déploiement Streamlit Cloud), la démarre en
    arrière-plan (uvicorn) puis attend qu'elle réponde. Une API distante (FRAUD_API_URL) n'est jamais lancée ici."""
    global _proc
    try:
        return health()
    except ApiError:
        u = urlparse(API_URL)
        if u.hostname not in ("localhost", "127.0.0.1"):
            raise
    if _proc is None or _proc.poll() is not None:
        root = Path(__file__).resolve().parents[1]
        _proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1",
                                  "--port", str(u.port or 8000)], cwd=root)
    t0 = time.time()
    while time.time() - t0 < wait:
        if _proc.poll() is not None:
            raise ApiError("Le démarrage de l'API a échoué (artefacts manquants ? lancer scripts/train.py)")
        try:
            return health()
        except ApiError:
            time.sleep(1.5)
    raise ApiError(f"L'API locale n'a pas démarré en {wait} s")


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