"""Module 5 : explicabilité SHAP (globale, locale, codes motifs)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_label


def make_explainer(model):
    import shap

    return shap.TreeExplainer(model)


def shap_values(explainer, X: pd.DataFrame):
    """Contributions en log-odds, shape (n, p), et valeur de base."""
    sv = np.asarray(explainer.shap_values(X))
    if sv.ndim == 3:  # certaines versions renvoient (n, p, 2)
        sv = sv[..., 1]
    base = float(np.ravel(explainer.expected_value)[-1])
    return sv, base


def to_explanation(sv, base, X: pd.DataFrame):
    import shap

    return shap.Explanation(
        values=sv, base_values=np.full(len(X), base), data=X.to_numpy(),
        feature_names=[feature_label(c) for c in X.columns],
    )


def reason_codes(sv_row, x_row: pd.Series, top: int = 5, min_codes: int = 3):
    """Les 3 à 5 variables qui ont le plus POUSSÉ le score vers 'fraude' (contribution > 0)."""
    order = np.argsort(-sv_row)
    chosen = [i for i in order[:top] if sv_row[i] > 0]
    if len(chosen) < min_codes:  # peu de contributions positives : compléter par |contribution|
        for i in np.argsort(-np.abs(sv_row)):
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= min_codes:
                break
    out = []
    for i in chosen:
        name = x_row.index[i]
        out.append({
            "feature": name,
            "label": feature_label(name),
            "value": float(x_row.iloc[i]),
            "contribution": float(sv_row[i]),
        })
    return out


def plain_explanation(score: float, thr: float, codes: list[dict], protective: list[dict] | None = None) -> str:
    """Courte explication en français : pourquoi cette transaction est (ou n'est pas) une alerte.

    `codes` = variables qui poussent vers la fraude ; `protective` = variables qui poussent vers légitime.
    """
    def fmt(c):
        return f"{c['label']} ({c['value']:.3g})"

    up = [c for c in codes if c["contribution"] > 0][:3]
    down = [c for c in (protective or []) if c["contribution"] < 0][:2]
    if score >= thr:
        head = f"Alerte : le score de {100 * score:.1f} % dépasse le seuil de {100 * thr:.1f} %."
        body = (" Principaux facteurs de risque : " + ", ".join(fmt(c) for c in up) + ".") if up else ""
        tail = (" Éléments rassurants : " + ", ".join(fmt(c) for c in down) + ".") if down else ""
    else:
        head = f"Transaction acceptée : le score de {100 * score:.1f} % reste sous le seuil de {100 * thr:.1f} %."
        body = (" Éléments qui rassurent le plus : " + ", ".join(fmt(c) for c in down) + ".") if down else ""
        tail = (" Points de vigilance : " + ", ".join(fmt(c) for c in up) + ".") if up else ""
    return head + body + tail


def global_importance(sv, columns) -> pd.DataFrame:
    imp = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({"feature": columns, "label": [feature_label(c) for c in columns], "mean_abs_shap": imp})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ---------- figures matplotlib ----------
def beeswarm_figure(expl, max_display=15):
    import matplotlib.pyplot as plt
    import shap

    plt.figure(figsize=(8, 5.5))
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def waterfall_figure(expl_row, max_display=12):
    import matplotlib.pyplot as plt
    import shap

    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(expl_row, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


# ---------- LIME ----------
def make_lime_explainer(X_ref: pd.DataFrame, seed: int = 0):
    """LIME tabulaire. `X_ref` fournit les statistiques de discrétisation / de perturbation
    (l'application utilise un échantillon du jeu de test, fraudes sur-représentées)."""
    from lime.lime_tabular import LimeTabularExplainer

    # les indicateurs 0/1 (week-end, nuit, catégorie, genre…) sont traités comme catégoriels : « Week-end = oui »
    binary = [i for i, c in enumerate(X_ref.columns) if set(np.unique(X_ref[c].to_numpy())) <= {0.0, 1.0}]
    return LimeTabularExplainer(
        X_ref.to_numpy(dtype=float), feature_names=[feature_label(c) for c in X_ref.columns],
        class_names=["légitime", "fraude"], mode="classification", discretize_continuous=True,
        categorical_features=binary, categorical_names={i: ["non", "oui"] for i in binary},
        random_state=seed,
    )


def lime_explain(explainer, model, x_row: pd.Series, num_features: int = 10, num_samples: int = 2000, seed: int = 0):
    """Explication LIME locale d'une transaction.

    Retourne (DataFrame[feature, label, rule, weight], infos) ; weight > 0 pousse vers « fraude »
    (en probabilité, pour le modèle surrogate linéaire local).
    """
    cols = list(x_row.index)

    def predict_fn(a):
        return model.predict_proba(pd.DataFrame(a, columns=cols).astype("float32"))

    np.random.seed(seed)
    exp = explainer.explain_instance(x_row.to_numpy(dtype=float), predict_fn, num_features=num_features,
                                     num_samples=num_samples, labels=(1,))
    rules = exp.as_list(label=1)
    idx_w = exp.as_map()[1]
    df = pd.DataFrame({
        "feature": [cols[i] for i, _ in idx_w],
        "label": [feature_label(cols[i]) for i, _ in idx_w],
        "rule": [r for r, _ in rules],
        "weight": [float(w) for _, w in idx_w],
    })
    info = {"local_pred": float(np.ravel(exp.local_pred)[0]), "r2": float(exp.score),
            "intercept": float(exp.intercept[1])}
    return df, info
