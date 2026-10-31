"""Module 3 : modèles supervisés, stratégies de déséquilibre, détecteurs d'anomalies non supervisés."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- déséquilibre
STRATEGIES = {
    "aucune": "Aucune (poids uniformes)",
    "poids_racine": "Poids de classe atténué (√(neg/pos))",
    "poids_complet": "Poids de classe complet (neg/pos)",
    "sous_echantillonnage": "Sous-échantillonnage des légitimes (10 pour 1)",
}


def apply_imbalance(X, y, strategy: str, seed: int = 42):
    """Appliqué au SEUL jeu d'entraînement. Retourne (X, y, scale_pos_weight)."""
    pos = int(y.sum())
    neg = len(y) - pos
    if strategy == "aucune":
        return X, y, 1.0
    if strategy == "poids_racine":
        return X, y, float(np.sqrt(neg / max(pos, 1)))
    if strategy == "poids_complet":
        return X, y, float(neg / max(pos, 1))
    if strategy == "sous_echantillonnage":
        rng = np.random.default_rng(seed)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), 10 * len(pos_idx)), replace=False)
        keep = np.sort(np.concatenate([pos_idx, keep_neg]))
        return X.iloc[keep], y[keep], 1.0
    raise ValueError(strategy)


# ---------------------------------------------------------------- supervisés
def make_xgb(scale_pos_weight=1.0, n_estimators=400, seed=42, n_jobs=-1):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight, n_jobs=n_jobs, random_state=seed,
    )


def make_logreg(seed=42):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=300, C=1.0, random_state=seed),
    )


def proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float64)


# ---------------------------------------------------------------- non supervisés
class _Scorer:
    """Interface commune : fit(X) sans étiquettes, score(X) -> plus grand = plus anormal."""

    name = ""


def _sample(X, n, seed):
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=seed)


class IsolationForestScorer(_Scorer):
    name = "Isolation Forest"

    def __init__(self, n_sample=200_000, seed=42):
        self.n_sample, self.seed = n_sample, seed
        self.model = IsolationForest(n_estimators=300, max_samples=512, random_state=seed, n_jobs=-1)

    def fit(self, X):
        self.model.fit(_sample(X, self.n_sample, self.seed))
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class LOFScorer(_Scorer):
    name = "Local Outlier Factor"

    def __init__(self, n_sample=20_000, n_neighbors=20, seed=42):
        self.n_sample, self.seed = n_sample, seed
        self.pipe = make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1))

    def fit(self, X):
        self.pipe.fit(_sample(X, self.n_sample, self.seed))
        return self

    def score(self, X, chunk=50_000):
        sc, lof = self.pipe[0], self.pipe[1]
        out = []
        for i in range(0, len(X), chunk):
            Z = np.clip(sc.transform(X.iloc[i:i + chunk]), -8, 8)
            out.append(-lof.score_samples(Z))
        return np.concatenate(out)


class AutoencoderScorer(_Scorer):
    """Autoencodeur dense (MLP qui reconstruit son entrée) ; score = erreur de reconstruction."""

    name = "Autoencodeur"

    def __init__(self, n_sample=200_000, seed=42):
        self.n_sample, self.seed = n_sample, seed
        self.scaler = StandardScaler()
        self.net = MLPRegressor(hidden_layer_sizes=(24, 8, 24), activation="relu", max_iter=40,
                                batch_size=512, learning_rate_init=1e-3, random_state=seed)

    def _z(self, X):
        return np.clip(self.scaler.transform(X), -8, 8)

    def fit(self, X):
        Xs = _sample(X, self.n_sample, self.seed)
        self.scaler.fit(Xs)
        Z = self._z(Xs)
        self.net.fit(Z, Z)
        return self

    def score(self, X):
        Z = self._z(X)
        return ((Z - self.net.predict(Z)) ** 2).mean(axis=1)
