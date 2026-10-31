"""Validation croisée chronologique (TimeSeriesSplit) sur le jeu d'entraînement.

Pour chaque méthode : AUC-ROC moyenne ± écart-type sur les plis. Les plis respectent l'ordre du temps
(fenêtre d'entraînement croissante, pli de validation toujours postérieur) : aucune fuite du futur.
Les méthodes non supervisées sont ajustées sans étiquette ; les étiquettes ne servent qu'à noter le pli.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .modeling import (AutoencoderScorer, IsolationForestScorer, LOFScorer, apply_imbalance,
                       make_logreg, make_xgb, proba)


def _cap(ix: np.ndarray, n: int, rng) -> np.ndarray:
    return ix if len(ix) <= n else np.sort(rng.choice(ix, n, replace=False))


def cross_validate_models(Xtr: pd.DataFrame, ytr: np.ndarray, strategy: str, n_splits: int = 3,
                          max_rows: int = 150_000, n_estimators: int = 400, lof_sample: int = 20_000,
                          seed: int = 42, log=print) -> pd.DataFrame:
    """Retourne un DataFrame (clé, cv_auc_mean, cv_auc_std, cv_folds)."""
    rng = np.random.default_rng(seed)
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(len(Xtr))))
    scores: dict[str, list[float]] = {k: [] for k in ("xgb", "lr", "if", "lof", "ae")}

    for f, (a, b) in enumerate(folds, 1):
        a, b = _cap(a, max_rows, rng), _cap(b, max_rows, rng)
        Xa, ya, Xb, yb = Xtr.iloc[a], ytr[a], Xtr.iloc[b], ytr[b]
        if yb.sum() == 0 or ya.sum() == 0:
            log(f"  pli {f} ignoré (aucune fraude)")
            continue
        Xs, ys, spw = apply_imbalance(Xa, ya, strategy, seed)
        out = {
            "xgb": proba(make_xgb(spw, n_estimators=n_estimators, seed=seed).fit(Xs, ys), Xb),
            "lr": proba(make_logreg(seed).fit(Xa, ya), Xb),
            "if": IsolationForestScorer(seed=seed).fit(Xa).score(Xb),
            "lof": LOFScorer(n_sample=lof_sample, seed=seed).fit(Xa).score(Xb),
            "ae": AutoencoderScorer(seed=seed).fit(Xa).score(Xb),
        }
        for k, s in out.items():
            scores[k].append(float(roc_auc_score(yb, s)))
        log(f"  pli {f}/{len(folds)} : " + "  ".join(f"{k}={scores[k][-1]:.3f}" for k in out))

    rows = [{"clé": k, "cv_auc_mean": float(np.mean(v)) if v else np.nan,
             "cv_auc_std": float(np.std(v)) if v else np.nan, "cv_folds": len(v)} for k, v in scores.items()]
    return pd.DataFrame(rows)
