"""Application web interactive : détection de fraude explicable (Sparkov).

Lancement : streamlit run app/streamlit_app.py
Navigation par barre latérale ; une page par module.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fraudlib.config import ARTIFACTS  # noqa: E402
from fraudlib.data import compute_eda, load_raw  # noqa: E402
from fraudlib.evaluation import (best_cost_threshold, cost_by_k, eval_at_threshold,  # noqa: E402
                                 n_days, precision_at_k, recall_at_k, threshold_from_budget)
from fraudlib.explain import (beeswarm_figure, global_importance, lime_explain, make_explainer,  # noqa: E402
                              make_lime_explainer, reason_codes, shap_values, to_explanation,
                              waterfall_figure)
from fraudlib.features import feature_label  # noqa: E402
from fraudlib.live import RISK_COLORS, build_tx, burst_rows, card_profile  # noqa: E402
import api_client  # noqa: E402
from ui_theme import (CLASS_COLOR, COLOR, NEG, POS, confusion_fig, contribution_bars, hero,  # noqa: E402
                      inject_css, show)

st.set_page_config(page_title="Détection de fraude explicable", page_icon="🛡️", layout="wide")
inject_css()

# ------------------------------------------------------------------ chargement des artefacts
if not (ARTIFACTS / "report.json").exists():
    st.error("Artefacts introuvables. Lancer d'abord : `python scripts/train.py --data data/fraudTrain.csv data/fraudTest.csv`")
    st.stop()


@st.cache_data(show_spinner=False)
def load_tables():
    rep = json.loads((ARTIFACTS / "report.json").read_text(encoding="utf-8"))
    cv_path = ARTIFACTS / "cv.csv"
    return {
        "report": rep,
        "eda": joblib.load(ARTIFACTS / "eda.joblib"),
        "comparison": pd.read_csv(ARTIFACTS / "comparison.csv"),
        "cv": pd.read_csv(cv_path) if cv_path.exists() else None,
        "imbalance": pd.read_csv(ARTIFACTS / "imbalance.csv"),
        "split": pd.read_csv(ARTIFACTS / "split.csv"),
        "features": pd.read_csv(ARTIFACTS / "features.csv"),
        "val": pd.read_parquet(ARTIFACTS / "scores_val.parquet"),
        "test": pd.read_parquet(ARTIFACTS / "scores_test.parquet"),
    }


@st.cache_resource(show_spinner=False)
def load_model():
    bundle = joblib.load(ARTIFACTS / "model_bundle.joblib")
    return bundle, make_explainer(bundle["model"])


@st.cache_data(show_spinner=False)
def load_xtest():
    return pd.read_parquet(ARTIFACTS / "X_test.parquet")


@st.cache_data(show_spinner=False)
def load_history():
    h = pd.read_parquet(ARTIFACTS / "history.parquet")
    h["cc_num"] = h["cc_num"].astype("int64")
    return h


@st.cache_resource(show_spinner="Préparation de LIME…")
def get_lime():
    """Explicateur LIME ; statistiques de référence = échantillon du jeu de test (fraudes sur-représentées)."""
    X, y = load_xtest(), T["test"]["is_fraud"].to_numpy()
    rng = np.random.default_rng(0)
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    keep = np.concatenate([rng.choice(neg, min(6000, len(neg)), replace=False),
                           rng.choice(pos, min(1000, len(pos)), replace=False)]) if len(pos) else rng.choice(neg, 6000)
    return make_lime_explainer(X.iloc[np.sort(keep)])


T = load_tables()
REP = T["report"]
MODELS: dict[str, str] = REP["models"]
PAGES = ["📊 1 · Données", "🧪 2 · Préparation", "🤖 3 · Modèles", "🎯 4 · Évaluation & seuil",
         "🔍 5 · Explicabilité", "⚡ 6 · Prédiction en direct"]

# ------------------------------------------------------------------ barre latérale
with st.sidebar:
    st.markdown('<div class="side-brand">🛡️ Fraude<br>explicable</div>'
                '<div class="side-sub">Sparkov · Challenge 2 · Data Mining · ENSI</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        f'<div class="side-card">Modèle actif : <b>{MODELS[REP["final_model"]]}</b><br>'
        f'Seuil d\'alerte : <b>{REP["threshold"]:.4g}</b><br>'
        f'Budget : <b>{REP["budget_per_day"]:.0f}</b> alertes / jour<br>'
        f'Coût d\'analyse : <b>{REP["review_cost"]:.1f} $</b> / alerte</div>', unsafe_allow_html=True)
    st.caption("Classement des transactions par risque, seuil selon un coût métier, explication de chaque alerte.")


# =================================================================== utilitaires de courbes (cache)
@st.cache_data(show_spinner=False)
def pr_points(which: str, key: str):
    d = T[which]
    p, r, _ = precision_recall_curve(d["is_fraud"].to_numpy(), d[f"score_{key}"].to_numpy())
    sel = np.unique(np.linspace(0, len(p) - 1, 1500).astype(int))
    return r[sel], p[sel]


@st.cache_data(show_spinner=False)
def roc_points(which: str, key: str):
    d = T[which]
    f, t, _ = roc_curve(d["is_fraud"].to_numpy(), d[f"score_{key}"].to_numpy())
    sel = np.unique(np.linspace(0, len(f) - 1, 1500).astype(int))
    return f[sel], t[sel]


@st.cache_data(show_spinner=False)
def recall_k_points(which: str, key: str, kmax: int):
    d = T[which]
    y = d["is_fraud"].to_numpy().astype(int)
    order = np.argsort(-d[f"score_{key}"].to_numpy(), kind="stable")[:kmax]
    rec = np.cumsum(y[order]) / max(y.sum(), 1)
    sel = np.unique(np.linspace(0, len(rec) - 1, 1500).astype(int))
    return (sel + 1) / n_days(d["ts"]), rec[sel]


@st.cache_data(show_spinner=False)
def cost_points(which: str, key: str, rc: float, kmax: int):
    d = T[which]
    ks, c = cost_by_k(d[f"score_{key}"].to_numpy(), d["is_fraud"].to_numpy().astype(int), d["amt"].to_numpy(float), rc, kmax)
    sel = np.unique(np.linspace(0, len(ks) - 1, 1500).astype(int))
    return ks[sel] / n_days(d["ts"]), c[sel]


@st.cache_data(show_spinner=False)
def f1_table(mode: str) -> dict[str, float]:
    """F1 sur le test. mode = '0.50' (seuil de probabilité 0,5 — modèles supervisés seulement)
    ou 'opt' (seuil choisi sur la validation par minimisation du coût — toutes les méthodes)."""
    d, comp = T["test"], T["comparison"].set_index("clé")
    y = d["is_fraud"].to_numpy().astype(int)
    out = {}
    for k in MODELS:
        s = d[f"score_{k}"].to_numpy()
        if mode == "0.50":
            out[k] = float(f1_score(y, s >= 0.5, zero_division=0)) if k in ("xgb", "lr") else np.nan
        else:
            out[k] = float(f1_score(y, s >= comp.loc[k, "seuil (val)"], zero_division=0))
    return out


# =================================================================== 1 · DONNÉES
def get_eda(upload):
    if upload is not None:
        @st.cache_data(show_spinner="Calcul des statistiques…")
        def eda_from_upload(raw: bytes):
            return compute_eda(load_raw(io.BytesIO(raw)))
        return eda_from_upload(upload.getvalue()), None
    eda = T["eda"]
    if "heat" in eda:
        return eda, None

    @st.cache_data(show_spinner="Calcul des graphiques complémentaires…")
    def eda_fallback():  # ancien eda.joblib : on calcule les nouveaux graphiques sur validation + test
        return compute_eda(pd.concat([T["val"], T["test"]], ignore_index=True))
    return {**eda, **{k: v for k, v in eda_fallback().items() if k not in eda}}, \
        "Graphiques complémentaires calculés sur validation + test (40 % des données). Pour le jeu complet : " \
        "`python scripts/update_eda.py --data data/fraudTrain.csv data/fraudTest.csv`."


def page_data():
    hero("data", "Module 1 · Exploration des données",
         "Comprendre le jeu avant de modéliser : déséquilibre, montants, temps, profil client et géographie.")
    up = st.file_uploader("Importer un fichier CSV Sparkov (sinon : jeu utilisé à l'entraînement)", type="csv")
    eda, note = get_eda(up)
    if note:
        st.info(note)

    s = eda["summary"].iloc[0]
    c = st.columns(5)
    c[0].metric("Transactions", f"{int(s['transactions']):,}")
    c[1].metric("Fraudes", f"{int(s['fraudes']):,}")
    c[2].metric("Taux de fraude", f"{s['taux_fraude_pct']:.3f} %")
    c[3].metric("Cartes", f"{int(s['cartes']):,}")
    c[4].metric("Montant moyen (fraude)", f"{s['montant_moyen_fraude']:.0f} $", f"légitime : {s['montant_moyen']:.0f} $ en moyenne",
                delta_color="off")

    t1, t2, t3, t4, t5 = st.tabs(["🌐 Vue d'ensemble", "💰 Montants & catégories", "🕒 Temps", "👤 Profil & géographie",
                                  "🔗 Corrélations"])

    with t1:
        l, r = st.columns([2, 3])
        with l:
            fig = go.Figure(go.Pie(labels=["Légitimes", "Fraudes"], values=[s["transactions"] - s["fraudes"], s["fraudes"]],
                                   hole=0.62, marker_colors=[CLASS_COLOR["Légitime"], CLASS_COLOR["Fraude"]], pull=[0, 0.12],
                                   textinfo="percent", sort=False))
            fig.update_layout(title="Déséquilibre des classes", height=360,
                              annotations=[dict(text=f"{s['taux_fraude_pct']:.2f} %<br>fraude", showarrow=False, font_size=17)])
            show(fig)
        with r:
            m = eda["month"]
            fig = px.area(m, x="mois", y="rate", markers=True, title="Taux de fraude par mois (%)",
                          color_discrete_sequence=["#ec4899"], hover_data={"n": True, "n_fraud": True})
            fig.update_layout(yaxis_title="taux (%)", xaxis_title=None, height=360)
            show(fig)
        if "daily" in eda:
            d = eda["daily"]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=d["jour"], y=d["transactions"], name="Transactions / jour", fill="tozeroy",
                                     line=dict(color="#6366f1"), opacity=0.55), secondary_y=False)
            fig.add_trace(go.Bar(x=d["jour"], y=d["fraudes"], name="Fraudes / jour", marker_color="#ef4444"), secondary_y=True)
            fig.update_layout(title="Activité quotidienne : volume de transactions et fraudes", height=380,
                              legend=dict(orientation="h", y=1.12))
            fig.update_yaxes(title_text="transactions", secondary_y=False)
            fig.update_yaxes(title_text="fraudes", secondary_y=True)
            show(fig)

    with t2:
        l, r = st.columns(2)
        with l:
            fig = px.area(eda["amt_hist"], x="montant", y="part (%)", color="classe", log_x=True,
                          title="Distribution des montants par classe (échelle log)", color_discrete_map=CLASS_COLOR)
            fig.update_traces(opacity=0.55)
            show(fig)
        with r:
            fig = px.bar(eda["category"], x="catégorie", y="rate", color="rate", color_continuous_scale="Sunsetdark",
                         title="Taux de fraude par catégorie de marchand (%)", hover_data={"n": True, "n_fraud": True})
            fig.update_layout(yaxis_title="taux (%)", coloraxis_showscale=False)
            show(fig)
        if "cat_amt" in eda:
            ca = eda["cat_amt"]
            fig = px.bar(ca, x="catégorie", y="médiane", color="classe", barmode="group", color_discrete_map=CLASS_COLOR,
                         title="Montant médian par catégorie : fraudes vs transactions légitimes", hover_data=["moyenne", "n"])
            fig.update_layout(yaxis_title="montant médian ($)", legend=dict(orientation="h", y=1.12))
            show(fig)
        st.caption("Sparkov ne contient pas de variable de canal : la catégorie (ex. *shopping_net* vs *shopping_pos*) en tient lieu.")

    with t3:
        l, r = st.columns(2)
        with l:
            fig = px.bar(eda["hour"], x="heure", y="rate", color="rate", color_continuous_scale="Turbo",
                         title="Taux de fraude par heure (%)", hover_data={"n": True, "n_fraud": True})
            fig.update_layout(yaxis_title="taux (%)", coloraxis_showscale=False)
            show(fig)
        with r:
            dow = eda["dow"].copy()
            dow["jour"] = dow["jour"].map(dict(enumerate(["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"])))
            fig = px.bar(dow, x="jour", y="rate", color="rate", color_continuous_scale="Viridis",
                         title="Taux de fraude par jour de la semaine (%)", hover_data={"n": True, "n_fraud": True})
            fig.update_layout(yaxis_title="taux (%)", coloraxis_showscale=False)
            show(fig)
        if "heat" in eda:
            h = eda["heat"].pivot(index="jour", columns="heure", values="rate").reindex(range(7))
            h.index = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
            fig = px.imshow(h, aspect="auto", color_continuous_scale="YlOrRd", origin="upper",
                            labels=dict(x="heure", y="", color="taux (%)"), title="Carte de chaleur : taux de fraude, jour × heure")
            fig.update_layout(height=360)
            show(fig)

    with t4:
        if "age_hist" in eda:
            l, r = st.columns(2)
            with l:
                fig = px.area(eda["age_hist"], x="âge", y="part (%)", color="classe", color_discrete_map=CLASS_COLOR,
                              title="Distribution de l'âge par classe")
                fig.update_traces(opacity=0.55)
                show(fig)
            with r:
                fig = px.bar(eda["age_rate"], x="tranche d'âge", y="rate", color="rate", color_continuous_scale="Magma_r",
                             title="Taux de fraude par tranche d'âge (%)", hover_data={"n": True, "n_fraud": True})
                fig.update_layout(yaxis_title="taux (%)", coloraxis_showscale=False)
                show(fig)
            l, r = st.columns(2)
            with l:
                fig = px.area(eda["dist_hist"], x="distance (km)", y="part (%)", color="classe", color_discrete_map=CLASS_COLOR,
                              title="Distance domicile–marchand par classe")
                fig.update_traces(opacity=0.55)
                show(fig)
            with r:
                if "city_rate" in eda:
                    fig = px.bar(eda["city_rate"], x="taille de ville", y="rate", color="rate", color_continuous_scale="Tealgrn",
                                 title="Taux de fraude selon la taille de la ville (%)", hover_data={"n": True, "n_fraud": True})
                    fig.update_layout(yaxis_title="taux (%)", coloraxis_showscale=False, xaxis_tickangle=-30)
                    show(fig)
        g = eda["gender"]
        fig = px.bar(g, x="genre", y="rate", color="genre", title="Taux de fraude par genre (%)",
                     color_discrete_sequence=["#ec4899", "#3b82f6"], hover_data={"n": True, "n_fraud": True})
        fig.update_layout(yaxis_title="taux (%)", showlegend=False, height=320)
        show(fig)

    with t5:
        if "corr" in eda:
            cm = eda["corr"]
            l, r = st.columns([3, 2])
            with l:
                fig = px.imshow(cm, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                                title="Matrice de corrélation (Pearson)")
                fig.update_layout(height=520)
                show(fig)
            with r:
                cf = cm["Fraude"].drop("Fraude").sort_values()
                fig = go.Figure(go.Bar(x=cf.values, y=cf.index, orientation="h",
                                       marker_color=[NEG if v < 0 else POS for v in cf.values]))
                fig.update_layout(title="Corrélation avec la fraude", height=520, xaxis_title="r de Pearson")
                show(fig)
            st.caption("Corrélations linéaires simples avec la cible : elles sous-estiment les effets non linéaires et les interactions "
                       "(vélocité, voyage impossible) que les modèles à base d'arbres exploitent.")
        else:
            st.info("Relancer `scripts/update_eda.py` pour calculer les corrélations.")


# =================================================================== 2 · PRÉPARATION
def page_prep():
    hero("prep", "Module 2 · Préparation et variables",
         "Séparation chronologique sans fuite, variables comportementales et stratégie contre le déséquilibre.")
    st.subheader("Séparation chronologique 60 / 20 / 20")
    sp = T["split"].copy()
    sp["début"], sp["fin"] = pd.to_datetime(sp["début"]), pd.to_datetime(sp["fin"])
    fig = px.timeline(sp, x_start="début", x_end="fin", y="jeu", color="jeu", text="transactions",
                      color_discrete_map={"train": "#6366f1", "val": "#f59e0b", "test": "#10b981"},
                      title="Frises temporelles : chaque jeu est postérieur au précédent")
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(showlegend=False, height=260)
    show(fig)
    st.dataframe(T["split"], width="stretch", hide_index=True)
    st.caption("Séparation dans l'ordre du temps : le modèle est toujours évalué sur des transactions postérieures à celles "
               "de son entraînement. Le jeu Sparkov est horodaté : la séparation aléatoire stratifiée n'est donc pas utilisée.")

    st.subheader("Variables construites")
    st.markdown("""
- **Temporelles** : heure en codage cyclique (sin/cos), jour de la semaine, week-end, nuit.
- **Comportementales** (par carte, uniquement sur le **passé strict**) : nombre de transactions sur 1 h / 24 h / 30 j,
  montant cumulé sur 24 h, montant rapporté à la moyenne des 30 derniers jours, temps depuis la transaction précédente,
  distance au marchand précédent et vitesse implicite (voyage « impossible »).
- **Géographiques / profil** : distance domicile-marchand, âge, taille de la ville, genre, catégorie du marchand (one-hot).
""")
    st.dataframe(T["features"].rename(columns={"feature": "variable", "label": "libellé"}),
                 width="stretch", hide_index=True, height=300)

    st.subheader("Stratégie contre le déséquilibre (jeu d'entraînement uniquement)")
    imb = T["imbalance"].drop(columns=["clé"])
    chosen = REP["chosen_strategy"]
    chosen_name = T["imbalance"].set_index("clé").loc[chosen, "stratégie"]
    fig = px.bar(imb, x="PR-AUC (val)", y="stratégie", orientation="h", color="stratégie", text_auto=".4f",
                 title="PR-AUC en validation selon la stratégie",
                 color_discrete_map={n: ("#10b981" if n == chosen_name else "#c7d2fe") for n in imb["stratégie"]})
    fig.update_layout(showlegend=False, height=280, yaxis_title=None)
    show(fig)
    st.dataframe(imb.style.format({c: "{:.4f}" for c in imb.columns if "AUC" in c or "@k" in c} | {"coût min (val, $)": "{:,.0f}"})
                 .background_gradient(cmap="Greens", subset=["PR-AUC (val)"], low=0.05, high=0.25),
                 width="stretch", hide_index=True)
    st.success(f"Stratégie retenue (meilleure PR-AUC en validation) : **{chosen_name}**")


# =================================================================== 3 · MODÈLES
HIGHER = ["PR-AUC val", "PR-AUC test", "ROC-AUC test", "CV AUC (moy.)", "précision@k", "rappel@k", "précision", "rappel",
          "F2", "MCC", "économie (%)"]
LOWER = ["coût test ($)", "CV AUC (écart-type)"]


def perf_chart(comp: pd.DataFrame, cv: pd.DataFrame | None, f1_mode: str) -> go.Figure:
    """« CV AUC mean ± std vs test metrics » : barres groupées par modèle."""
    c = comp.set_index("clé")
    cvm = cv.set_index("clé") if cv is not None else None
    key_sort = (cvm["cv_auc_mean"] if cvm is not None else c["ROC-AUC test"]).reindex(c.index)
    keys = list(key_sort.sort_values(ascending=False).index)
    f1 = f1_table(f1_mode)
    names = [MODELS[k] for k in keys]
    bars = []
    if cvm is not None:
        bars.append(("CV AUC mean", [cvm.loc[k, "cv_auc_mean"] for k in keys], "#4c72b0",
                     dict(type="data", array=[cvm.loc[k, "cv_auc_std"] for k in keys], color="#111827", thickness=1.6, width=6)))
    bars += [("Test ROC-AUC", [c.loc[k, "ROC-AUC test"] for k in keys], "#dd8452", None),
             ("Test PR-AUC", [c.loc[k, "PR-AUC test"] for k in keys], "#55a868", None),
             ("F1@0.50" if f1_mode == "0.50" else "F1 (seuil optimal)", [f1[k] for k in keys], "#c44e52", None)]
    fig = go.Figure()
    for name, vals, col, err in bars:
        fig.add_trace(go.Bar(name=name, x=names, y=vals, marker_color=col, error_y=err, text=[f"{v:.2f}" if v == v else "n/a" for v in vals],
                             textposition="outside", cliponaxis=False))
    fig.update_layout(barmode="group", title="Tuned Model Performance — CV AUC mean ± std vs test metrics", yaxis_title="Score",
                      yaxis_range=[0, 1.12], height=470, legend=dict(orientation="h", y=-0.2), xaxis_title="Model")
    return fig


def page_models():
    hero("models", "Module 3 · Comparaison des méthodes",
         "Supervisé (XGBoost, régression logistique) contre non supervisé (Isolation Forest, LOF, autoencodeur).")
    comp = T["comparison"].copy()
    cv = T["cv"]

    # ---------------- tableau comparatif en dégradé de vert
    st.subheader("Tableau comparatif")
    show_df = comp.drop(columns=["clé"]).set_index("méthode")
    if cv is not None:
        m = cv.set_index("clé").reindex(comp["clé"])
        show_df.insert(2, "CV AUC (moy.)", m["cv_auc_mean"].to_numpy())
        show_df.insert(3, "CV AUC (écart-type)", m["cv_auc_std"].to_numpy())
    fmt = {c: "{:.3f}" for c in ["PR-AUC val", "PR-AUC test", "ROC-AUC test", "CV AUC (moy.)", "précision@k", "rappel@k",
                                  "précision", "rappel", "F2", "MCC"]}
    fmt |= {"CV AUC (écart-type)": "{:.4f}", "seuil (val)": "{:.4g}", "alertes/jour": "{:.1f}", "coût test ($)": "{:,.0f}",
            "coût sans modèle ($)": "{:,.0f}", "économie (%)": "{:.1f}"}
    hb = [c for c in HIGHER if c in show_df.columns]
    lb = [c for c in LOWER if c in show_df.columns]
    sty = (show_df.style.format(fmt, na_rep="–")
           .background_gradient(cmap="Greens", subset=hb, axis=0, low=0.04, high=0.3)
           .background_gradient(cmap="Greens_r", subset=lb, axis=0, low=0.04, high=0.3))
    st.dataframe(sty, width="stretch")
    st.caption("Dégradé de vert par colonne : **plus c'est foncé, meilleur est le résultat** (pour le coût et l'écart-type, "
               "la valeur la plus basse est la plus foncée). "
               f"k = budget de {REP['budget_per_day']:.0f} alertes/jour ≈ 1,5 × le nombre de fraudes quotidiennes. "
               f"Seuil de chaque méthode choisi sur la validation (coût d'analyse = {REP['review_cost']:.1f} $/alerte) ; "
               "colonnes « test » = confirmation sur le jeu de test.")

    # ---------------- graphique CV AUC ± écart-type vs métriques de test
    st.subheader("CV AUC (moyenne ± écart-type) vs métriques de test")
    f1_mode = st.radio("F1 affiché", ["0.50", "opt"], horizontal=True,
                       format_func=lambda x: {"0.50": "Seuil de probabilité 0,50 (modèles supervisés)",
                                              "opt": "Seuil optimal (coût, choisi sur la validation) — toutes les méthodes"}[x])
    show(perf_chart(comp, cv, f1_mode))
    if cv is None:
        st.warning("Barres de validation croisée absentes : lancer `python scripts/cv_report.py --data data/fraudTrain.csv "
                   "data/fraudTest.csv` (ou relancer `train.py`) pour créer `artifacts/cv.csv`.")
    else:
        nf = int(cv["cv_folds"].max())
        st.caption(f"Validation croisée **chronologique** ({nf} plis, fenêtre d'entraînement croissante) sur le jeu d'entraînement ; "
                   "barre d'erreur = écart-type de l'AUC-ROC entre plis. Les scores des méthodes non supervisées ne sont pas des "
                   "probabilités : le F1 à 0,50 n'a de sens que pour XGBoost et la régression logistique (« n/a » sinon).")

    # ---------------- courbes superposées
    st.subheader("Courbes superposées")
    which = st.radio("Jeu", ["test", "val"], horizontal=True, format_func=lambda x: {"test": "Test", "val": "Validation"}[x])
    nd = n_days(T[which]["ts"])
    kmax = int(min(len(T[which]), 10 * REP["budget_per_day"] * nd))
    tp, tr_, tk, tc = st.tabs(["Précision–rappel", "ROC", "Rappel selon le budget d'alertes", "Coût métier"])
    with tp:
        fig = go.Figure()
        for k, name in MODELS.items():
            rr, pp = pr_points(which, k)
            fig.add_trace(go.Scatter(x=rr, y=pp, mode="lines", name=name, line=dict(color=COLOR[k], width=3)))
        base = T[which]["is_fraud"].mean()
        fig.add_hline(y=base, line_dash="dot", annotation_text=f"hasard = {base:.4f}")
        fig.update_layout(title="Courbes précision-rappel", xaxis_title="Rappel", yaxis_title="Précision", height=480,
                          yaxis_range=[0, 1.02])
        show(fig, key=f"pr_{which}")
    with tr_:
        logx = st.checkbox("Échelle logarithmique pour le taux de faux positifs", value=False)
        fig = go.Figure()
        for k, name in MODELS.items():
            ff, tt = roc_points(which, k)
            fig.add_trace(go.Scatter(x=ff, y=tt, mode="lines", name=name, line=dict(color=COLOR[k], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="hasard", line=dict(color="#9ca3af", dash="dot")))
        fig.update_layout(title="Courbes ROC", xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs (rappel)",
                          height=480, yaxis_range=[0, 1.02])
        if logx:
            fig.update_xaxes(type="log", range=[-5, 0])
        show(fig, key=f"roc_{which}_{logx}")
    with tk:
        fig = go.Figure()
        for k, name in MODELS.items():
            xk, rk = recall_k_points(which, k, kmax)
            fig.add_trace(go.Scatter(x=xk, y=rk, mode="lines", name=name, line=dict(color=COLOR[k], width=3)))
        fig.add_vline(x=REP["budget_per_day"], line_dash="dash", line_color="#111827", annotation_text="budget d'alertes")
        fig.update_layout(title="Part des fraudes détectées selon le nombre d'alertes par jour", xaxis_title="Alertes / jour",
                          yaxis_title="Rappel", height=480, yaxis_range=[0, 1.02])
        show(fig, key=f"rk_{which}")
    with tc:
        fig = go.Figure()
        for k, name in MODELS.items():
            xc, cc = cost_points(which, k, float(REP["review_cost"]), kmax)
            fig.add_trace(go.Scatter(x=xc, y=cc, mode="lines", name=name, line=dict(color=COLOR[k], width=3)))
        fig.add_vline(x=REP["budget_per_day"], line_dash="dash", line_color="#111827", annotation_text="budget d'alertes")
        fig.update_layout(title="Coût total selon le nombre d'alertes par jour", xaxis_title="Alertes / jour", yaxis_title="Coût ($)",
                          height=480)
        show(fig, key=f"cost_{which}")

    # ---------------- matrices de confusion
    st.subheader("Matrices de confusion (jeu de test, seuil choisi sur la validation)")
    te = T["test"]
    y, amt, ndt = te["is_fraud"].to_numpy().astype(int), te["amt"].to_numpy(float), n_days(te["ts"])
    cs = comp.set_index("clé")
    keys = list(MODELS)
    for row in (keys[:3], keys[3:]):
        cols = st.columns(3)
        for col, k in zip(cols, row):
            r = eval_at_threshold(te[f"score_{k}"].to_numpy(), y, amt, cs.loc[k, "seuil (val)"], REP["review_cost"], ndt)
            with col:
                show(confusion_fig(r["tp"], r["fp"], r["fn"], r["tn"], f"{MODELS[k]} · {r['alerts_per_day']:.0f} alertes/j", 300),
                     key=f"cm_{k}")
    st.caption("Couleur = part de chaque ligne (vert : bien classé, rouge : erreur) ; le texte donne les effectifs. "
               "La classe fraude étant rare, la lecture par ligne rend visibles les fraudes manquées.")


# =================================================================== 4 · ÉVALUATION & SEUIL
def page_eval():
    hero("eval", "Module 4 · Évaluation et seuil d'alerte",
         "Pilotez le seuil ou le budget d'alertes et observez immédiatement le compromis fraudes manquées / coût d'analyse.")
    c1, c2, c3 = st.columns([2, 2, 2])
    mkey = c1.selectbox("Méthode", list(MODELS), index=list(MODELS).index("xgb"), format_func=lambda k: MODELS[k])
    rc = c2.number_input("Coût d'analyse d'une alerte ($)", 0.0, 500.0, float(REP["review_cost"]), 0.5)
    which = c3.radio("Jeu évalué", ["test", "val"], horizontal=True, key="ev_set",
                     format_func=lambda x: {"test": "Test", "val": "Validation"}[x])

    ev_df, va_df = T[which], T["val"]
    sc = ev_df[f"score_{mkey}"].to_numpy()
    y = ev_df["is_fraud"].to_numpy().astype(int)
    amt = ev_df["amt"].to_numpy(float)
    nd = n_days(ev_df["ts"])

    vsc = va_df[f"score_{mkey}"].to_numpy()
    vopt = best_cost_threshold(vsc, va_df["is_fraud"].to_numpy().astype(int), va_df["amt"].to_numpy(float), rc)
    thr_val = vopt["threshold"]

    mode = st.radio("Piloter par", ["Budget d'alertes par jour", "Seuil de score"], horizontal=True)
    kdef = float(REP["budget_per_day"])
    if mode.startswith("Budget"):
        kday = st.slider("Budget d'alertes / jour", 1.0, max(5 * kdef, 50.0), kdef, 1.0, key=f"kday_{mkey}")
        thr = threshold_from_budget(sc, round(kday * nd))
    else:
        lo, hi = float(np.min(sc)), float(np.max(sc))
        default = float(np.clip(thr_val, lo, hi))
        thr = st.slider("Seuil de score (alerte si score ≥ seuil)", lo, hi, default, (hi - lo) / 1000,
                        format="%.5f", key=f"thr_{mkey}_{which}_{rc}")

    res = eval_at_threshold(sc, y, amt, thr, rc, nd)
    k_alerts = res["alerts"]

    m = st.columns(5)
    m[0].metric("Alertes / jour", f"{res['alerts_per_day']:.1f}", f"{res['alerts']:,} au total", delta_color="off")
    m[1].metric("Précision", f"{100 * res['precision']:.1f} %")
    m[2].metric("Rappel", f"{100 * res['recall']:.1f} %", f"{res['tp']} / {res['tp'] + res['fn']} fraudes", delta_color="off")
    m[3].metric("F2", f"{res['f2']:.3f}")
    m[4].metric("MCC", f"{res['mcc']:.3f}")
    m = st.columns(5)
    m[0].metric("Coût total", f"{res['cost']:,.0f} $")
    m[1].metric("dont fraudes manquées", f"{res['missed_amount']:,.0f} $")
    m[2].metric("dont analyse des alertes", f"{res['review_cost_total']:,.0f} $")
    m[3].metric("Coût sans modèle", f"{res['cost_no_model']:,.0f} $")
    m[4].metric("Économie", f"{res['saving']:,.0f} $", f"{res['saving_pct']:.1f} %")
    st.caption(f"Seuil optimal sur la validation (coût minimal) : **{thr_val:.5g}** "
               f"→ {vopt['k'] / n_days(va_df['ts']):.1f} alertes/jour. "
               f"Précision@k / rappel@k pour k = {k_alerts:,} alertes : "
               f"{100 * precision_at_k(sc, y, k_alerts):.1f} % / {100 * recall_at_k(sc, y, k_alerts):.1f} %.")

    # ---------------- matrice de confusion + distribution des scores
    l, r = st.columns([2, 3])
    with l:
        show(confusion_fig(res["tp"], res["fp"], res["fn"], res["tn"], f"Matrice de confusion · {MODELS[mkey]}", 400),
             key=f"cm_eval_{mkey}_{which}")
    with r:
        fig = go.Figure()
        lo_s, hi_s = float(np.min(sc)), float(np.max(sc))
        bins = dict(start=lo_s, end=hi_s, size=(hi_s - lo_s) / 60 or 1)
        for lab, v in (("Légitime", 0), ("Fraude", 1)):
            fig.add_trace(go.Histogram(x=sc[y == v], name=lab, xbins=bins, histnorm="percent", marker_color=CLASS_COLOR[lab],
                                       opacity=0.65))
        fig.add_vline(x=thr, line_color="#111827", line_dash="dash", annotation_text="seuil")
        fig.update_layout(barmode="overlay", title="Distribution des scores par classe (axe vertical logarithmique)",
                          xaxis_title="Score", yaxis_title="% de la classe", yaxis_type="log", height=400,
                          legend=dict(orientation="h", y=1.12))
        show(fig, key=f"hist_{mkey}_{which}")

    kmax = int(min(len(sc), max(10 * kdef, 3 * res["alerts_per_day"]) * nd))
    ks, costs = cost_by_k(sc, y, amt, rc, kmax)
    sel = np.unique(np.linspace(0, len(ks) - 1, 2000).astype(int))
    fig = go.Figure(go.Scatter(x=ks[sel] / nd, y=costs[sel], mode="lines", name="Coût total", line=dict(color=COLOR[mkey], width=3),
                               fill="tozeroy", fillcolor="rgba(99,102,241,.10)"))
    fig.add_vline(x=res["alerts_per_day"], line_color="black", annotation_text="réglage actuel")
    fig.add_vline(x=vopt["k"] / n_days(va_df["ts"]), line_dash="dash", line_color="green", annotation_text="optimum validation",
                  annotation_position="bottom right")
    fig.update_layout(title="Coût total en fonction du nombre d'alertes par jour", xaxis_title="Alertes / jour",
                      yaxis_title="Coût ($)", height=380)
    show(fig, key=f"costk_{mkey}_{which}")

    st.subheader("Alertes en cours (les plus risquées en premier)")
    al = ev_df.loc[sc >= thr, ["ts", "trans_num", "category", "merchant", "amt", "is_fraud"]].copy()
    al["score"] = sc[sc >= thr]
    al = al.sort_values("score", ascending=False).head(200).rename(columns={"is_fraud": "fraude réelle"})
    st.dataframe(al.style.background_gradient(cmap="OrRd", subset=["score"]), width="stretch", hide_index=True, height=300)


# =================================================================== 5 · EXPLICABILITÉ
@st.cache_data(show_spinner="Calcul des valeurs SHAP…")
def global_shap(n=2500):
    _, explainer = load_model()
    Xte, test = load_xtest(), T["test"]
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(test["is_fraud"].to_numpy() == 1)
    neg = np.flatnonzero(test["is_fraud"].to_numpy() == 0)
    npos = min(len(pos), n // 5)
    ix = np.sort(np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, n - npos, replace=False)]))
    X = Xte.iloc[ix]
    sv, base = shap_values(explainer, X)
    return X, sv, base


@st.cache_data(show_spinner="Calcul de l'explication LIME…")
def lime_cached(i: int, nf: int, ns: int):
    bundle, _ = load_model()
    return lime_explain(get_lime(), bundle["model"], load_xtest().iloc[i], nf, ns)


def page_xai():
    hero("xai", "Module 5 · Explicabilité",
         "SHAP (attributions exactes pour les arbres) et LIME (modèle linéaire local) : pourquoi cette transaction est-elle suspecte ?")
    bundle, explainer = load_model()
    Xte, test = load_xtest(), T["test"]
    thr_model = float(bundle["threshold"])
    scx = test["score_xgb"].to_numpy()

    top = test.assign(_s=scx).sort_values("_s", ascending=False).head(300)
    options = {f"{r_.trans_num[:8]} · {r_.category} · {r_.amt:,.0f} $ · score {r_._s:.3f}{' · FRAUDE' if r_.is_fraud else ''}": i
               for i, r_ in top.iterrows()}
    t_glob, t_shap, t_lime, t_cmp = st.tabs(["🌍 Globale (SHAP)", "🧠 Alerte : SHAP", "🍋 Alerte : LIME", "⚖️ SHAP vs LIME"])

    with t_glob:
        Xs, svs, base = global_shap()
        l, r = st.columns([3, 2])
        with l:
            st.pyplot(beeswarm_figure(to_explanation(svs, base, Xs)))
            st.caption("Échantillon de 2 500 transactions du test, fraudes sur-représentées pour la lisibilité du graphique.")
        with r:
            imp = global_importance(svs, Xs.columns).head(15)
            fig = px.bar(imp[::-1], x="mean_abs_shap", y="label", orientation="h", color="mean_abs_shap",
                         color_continuous_scale="Plasma", title="Importance moyenne |SHAP|")
            fig.update_layout(yaxis_title=None, xaxis_title="mean |SHAP| (log-odds)", height=480, coloraxis_showscale=False)
            show(fig, key="imp_global")

    # sélection partagée par les trois onglets « alerte »
    with t_shap:
        choice = st.selectbox("Choisir une alerte (300 transactions les plus risquées du test)", list(options), key="xai_alert")
    i = options[choice]
    x_row = Xte.iloc[i]
    sv1, base1 = shap_values(explainer, Xte.iloc[[i]])
    score = float(scx[i])
    raw = test.iloc[i]

    with t_shap:
        a, b_ = st.columns([2, 3])
        with a:
            st.metric("Score de fraude", f"{score:.3f}", "ALERTE" if score >= thr_model else "sous le seuil",
                      delta_color="inverse" if score >= thr_model else "off")
            st.markdown("**Codes motifs** (variables qui poussent le plus vers « fraude »)")
            for n_, c_ in enumerate(reason_codes(sv1[0], x_row, top=5), 1):
                st.markdown(f'<div class="reason"><b>{n_}. {c_["label"]}</b> = <code>{c_["value"]:.3g}</code> '
                            f'&nbsp; (+{c_["contribution"]:.2f})</div>', unsafe_allow_html=True)
            st.caption(f"Transaction du {raw['ts']} · marchand {raw['merchant']} · réellement "
                       f"{'frauduleuse' if raw['is_fraud'] else 'légitime'}")
        with b_:
            st.pyplot(waterfall_figure(to_explanation(sv1, base1, Xte.iloc[[i]])[0]))

    with t_lime:
        st.markdown(f"Transaction sélectionnée : **{raw['category']} · {raw['amt']:,.0f} $** · score du modèle **{score:.3f}**")
        c1, c2 = st.columns(2)
        nf = c1.slider("Nombre de variables affichées", 5, 20, 10, key="lime_nf")
        ns = c2.select_slider("Échantillons de perturbation", [500, 1000, 2000, 5000], 2000, key="lime_ns")
        ldf, info = lime_cached(i, nf, ns)
        m = st.columns(3)
        m[0].metric("Score du modèle", f"{score:.3f}")
        m[1].metric("Prédiction du modèle local", f"{info['local_pred']:.3f}")
        m[2].metric("R² du modèle local", f"{info['r2']:.2f}")
        fig = contribution_bars(list(ldf["rule"]), ldf["weight"], "Contributions LIME (en probabilité)",
                                "poids dans le modèle linéaire local (> 0 : vers la fraude)")
        show(fig, key=f"lime_bars_{i}_{nf}_{ns}")
        st.caption("LIME perturbe la transaction, interroge le modèle sur ces voisins puis ajuste un modèle linéaire pondéré : "
                   "les règles (« distance > 67 km », « Week-end = non ») sont les conditions locales qui comptent. Un R² faible "
                   "signale que le comportement du modèle est peu linéaire autour de ce point : lire l'explication avec prudence.")

    with t_cmp:
        ldf, _ = lime_cached(i, 10, 2000)
        shap_s = pd.Series(sv1[0], index=Xte.columns)
        lime_s = ldf.set_index("feature")["weight"]
        feats = list(dict.fromkeys(list(lime_s.index) + list(shap_s.abs().sort_values(ascending=False).head(8).index)))[:12]
        sh = shap_s.reindex(feats).fillna(0.0)
        li = lime_s.reindex(feats).fillna(0.0)
        norm = lambda s: s / max(float(s.abs().max()), 1e-12)  # noqa: E731
        order = list((norm(sh).abs() + norm(li).abs()).sort_values().index)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=[feature_label(f) for f in order], x=norm(sh).reindex(order), orientation="h", name="SHAP",
                             marker_color="#6366f1"))
        fig.add_trace(go.Bar(y=[feature_label(f) for f in order], x=norm(li).reindex(order), orientation="h", name="LIME",
                             marker_color="#f59e0b"))
        fig.add_vline(x=0, line_color="#374151")
        fig.update_layout(barmode="group", title="Contributions relatives (÷ contribution maximale de chaque méthode)",
                          xaxis_title="contribution relative", height=max(420, 44 * len(order) + 120),
                          legend=dict(orientation="h", y=1.08))
        show(fig, key=f"cmp_{i}")
        top5_s = set(shap_s.abs().sort_values(ascending=False).head(5).index)
        top5_l = set(lime_s.abs().sort_values(ascending=False).head(5).index)
        agree = float(np.mean(np.sign(sh) == np.sign(li))) if len(feats) else 0.0
        m = st.columns(2)
        m[0].metric("Variables communes dans le top 5", f"{len(top5_s & top5_l)} / 5")
        m[1].metric("Accord de signe (variables comparées)", f"{100 * agree:.0f} %")
        st.caption("SHAP est exact pour XGBoost et additif (log-odds) ; LIME est une approximation locale en probabilité qui dépend "
                   "de l'échantillonnage. Un fort accord renforce la confiance dans l'explication ; un désaccord invite à "
                   "examiner la transaction à la main.")


# =================================================================== 6 · PRÉDICTION EN DIRECT
CAT_PRESET = {"normal": ["grocery_pos", "gas_transport", "food_dining"], "risk": ["shopping_net", "misc_net"]}


def _gauge(score: float, thr: float, color: str) -> go.Figure:
    b = [0.0, thr / 3, thr, max(0.9, thr), 1.0]
    cols = ["#bbf7d0", "#fde68a", "#fdba74", "#fca5a5"]
    steps = [{"range": [100 * b[j], 100 * b[j + 1]], "color": cols[j]} for j in range(4) if b[j + 1] > b[j]]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * score, number=dict(suffix=" %", valueformat=".2f", font=dict(color=color)),
        title=dict(text="Probabilité de fraude"),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color, thickness=0.32), steps=steps,
                   threshold=dict(line=dict(color="#111827", width=4), thickness=0.85, value=100 * thr))))
    fig.update_layout(height=290, margin=dict(l=25, r=25, t=60, b=10))
    return fig


def _pick_cat(categories: list[str], prefs: list[str]) -> str:
    return next((c for c in prefs if c in categories), categories[0])


@st.cache_data(show_spinner=False, max_entries=256)
def call_api(payload: str) -> dict:
    """Appel POST /explain/simulate (mis en cache : même saisie = même réponse, pas de rappel à chaque interaction)."""
    d = json.loads(payload)
    return api_client.explain_simulate(d["tx"], d["prev"], d["thr"])


def page_live():
    hero("live", "Module 6 · Prédiction en direct",
         "Saisissez une transaction : elle est envoyée à l'API de scoring (historique de la carte, score, décision, "
         "codes motifs) et une courte explication vous dit pourquoi elle est jugée frauduleuse ou non.")
    try:
        with st.spinner("Connexion à l'API de scoring (démarrage au premier appel)…"):
            h = api_client.ensure_api()
    except api_client.ApiError as e:
        st.error(f"{e}. Démarrer l'API dans un autre terminal : `uvicorn api.main:app --port 8000` "
                 "(URL modifiable via la variable FRAUD_API_URL).")
        st.stop()
    st.sidebar.success(f"API connectée · {h['cartes_en_historique']} cartes", icon="🔌")
    bundle, _ = load_model()  # métadonnées d'interface seulement (catégories, seuil par défaut) : le scoring passe par l'API
    hist = load_history()
    S = st.session_state
    cats = list(bundle["categories"])
    counts = hist["cc_num"].value_counts()
    cards = list(counts.index)

    def card_hist():
        return hist[hist["cc_num"] == S.lv_card]

    def reset_time():
        t = card_profile(card_hist())["last_ts"] + pd.Timedelta(minutes=30)
        S.lv_date, S.lv_time = t.date(), t.time().replace(second=0, microsecond=0)

    def preset(name):
        pr = card_profile(card_hist())
        base = {"normal": dict(lv_amt=float(np.clip(round(pr["avg_amt"]), 1, 5000)), lv_cat=_pick_cat(cats, CAT_PRESET["normal"]),
                               lv_time=dt.time(14, 30), lv_dist=5.0, lv_burst=0),
                "suspect": dict(lv_amt=350.0, lv_cat=_pick_cat(cats, CAT_PRESET["risk"]), lv_time=dt.time(23, 10),
                                lv_dist=120.0, lv_burst=1),
                "fraud": dict(lv_amt=900.0, lv_cat=_pick_cat(cats, CAT_PRESET["risk"]), lv_time=dt.time(2, 15),
                              lv_dist=800.0, lv_burst=4)}[name]
        S.update(base)

    if "lv_card" not in S:
        S.lv_card = cards[0]
        reset_time()
        preset("normal")
    S.setdefault("lv_merchant", "fraud_Live Merchant")
    S.setdefault("lv_bearing", 45)
    S.setdefault("lv_thr", float(np.clip(bundle["threshold"], 0.001, 0.999)))
    S.setdefault("lv_log", [])

    tab_form, tab_stream = st.tabs(["🎛️ Saisir une transaction", "📡 Simulation de flux en temps réel"])

    # ------------------------------------------------------------------ saisie manuelle
    with tab_form:
        st.markdown("**Scénarios rapides**")
        b1, b2, b3, _ = st.columns([1, 1, 1, 2])
        b1.button("🟢 Achat habituel", on_click=preset, args=("normal",), width="stretch")
        b2.button("🟠 Suspect", on_click=preset, args=("suspect",), width="stretch")
        b3.button("🔴 Attaque type fraude", on_click=preset, args=("fraud",), width="stretch")

        left, right = st.columns([5, 7], gap="large")
        with left:
            st.selectbox("Carte bancaire", cards, key="lv_card", on_change=reset_time,
                         format_func=lambda c: f"•••• {str(c)[-4:]} · {counts[c]} transactions (30 j)")
            pr = card_profile(card_hist())
            age = (pd.Timestamp(S.lv_date) - pd.Timestamp(pr["dob"])).days / 365.25
            st.caption(f"Profil : {'femme' if pr['gender'] == 'F' else 'homme'}, {age:.0f} ans · ville de {pr['city_pop']:,} hab. · "
                       f"montant moyen {pr['avg_amt']:.0f} $ · dernière transaction connue : {pr['last_ts']:%Y-%m-%d %H:%M}")
            st.slider("Montant ($)", 1.0, 5000.0, key="lv_amt", step=1.0)
            st.selectbox("Catégorie du marchand", cats, key="lv_cat")
            d1, d2 = st.columns(2)
            d1.date_input("Date", key="lv_date")
            d2.time_input("Heure", key="lv_time", step=300)
            st.slider("Distance domicile–marchand (km)", 0.0, 3000.0, key="lv_dist", step=5.0)
            st.slider("Achats déjà faits dans la dernière heure (simulation)", 0, 10, key="lv_burst",
                      help="Ajoute des transactions fictives juste avant celle-ci pour tester les variables de vélocité.")
            with st.expander("Options avancées"):
                st.text_input("Nom du marchand", key="lv_merchant")
                st.slider("Direction du marchand (°)", 0, 359, key="lv_bearing")
                st.slider("Seuil d'alerte", 0.001, 0.999, key="lv_thr", step=0.001, format="%.3f",
                          help="Par défaut : seuil minimisant le coût sur la validation.")

        # ---- calcul : la transaction est envoyée à l'API (aucun scoring local)
        ts = pd.Timestamp(dt.datetime.combine(S.lv_date, S.lv_time))
        params = dict(ts=ts, amt=S.lv_amt, category=S.lv_cat, merchant=S.lv_merchant, dist_km=S.lv_dist, bearing=S.lv_bearing)
        tx = build_tx(pr, **params)
        prev = burst_rows(tx, S.lv_burst) if S.lv_burst else []
        thr = float(S.lv_thr)
        try:
            res = call_api(json.dumps({"tx": tx, "prev": prev, "thr": thr}, sort_keys=True))
        except api_client.ApiError as e:
            with right:
                st.error(f"{e}. Lancer l'API : `uvicorn api.main:app --port 8000`")
            st.stop()
        score, thr = res["score"], res["threshold"]
        level, is_alert = res["risk_level"], res["is_alert"]
        color = RISK_COLORS[level]
        codes = res["reason_codes"]

        with right:
            bg = {"faible": "linear-gradient(120deg,#10b981,#34d399)", "moyen": "linear-gradient(120deg,#f59e0b,#fbbf24)",
                  "élevé": "linear-gradient(120deg,#f97316,#ef4444)", "critique": "linear-gradient(120deg,#b91c1c,#7f1d1d)"}[level]
            verdict = "🚨 ALERTE — transaction à vérifier" if is_alert else "✅ Transaction acceptée"
            st.markdown(f'<div class="verdict" style="background:{bg}">{verdict}'
                        f'<small>Niveau de risque : {level} · score {score:.4f} · seuil {thr:.3f}</small></div>', unsafe_allow_html=True)
            show(_gauge(score, thr, color), key="gauge")
            st.markdown("**Pourquoi ce résultat ?**")
            st.info(res["summary"])
            st.markdown("**Codes motifs** : ce qui pousse le plus vers « fraude »")
            for n_, c_ in enumerate(codes, 1):
                st.markdown(f'<div class="reason"><b>{n_}. {c_["label"]}</b> = <code>{c_["value"]:.3g}</code> '
                            f'&nbsp; <span style="color:{POS}">({c_["contribution"]:+.2f})</span></div>', unsafe_allow_html=True)
            st.caption("Explication globale du modèle (SHAP, LIME, comparaison) : page « 5 · Explicabilité ».")
            if st.button("➕ Ajouter au journal", key="lv_add"):
                S.lv_log.append({"heure": f"{ts:%m-%d %H:%M}", "carte": f"••{str(pr['cc_num'])[-4:]}", "catégorie": S.lv_cat,
                                 "montant ($)": S.lv_amt, "distance (km)": S.lv_dist, "achats 1 h": S.lv_burst,
                                 "score": round(score, 4), "risque": level, "décision": "ALERTE" if is_alert else "OK"})
                st.toast("Transaction ajoutée au journal", icon="📝")

        if S.lv_log:
            st.subheader("📝 Journal des prédictions")
            lg = pd.DataFrame(S.lv_log)
            rcol = {"faible": "#bbf7d0", "moyen": "#fef08a", "élevé": "#fed7aa", "critique": "#fecaca"}
            st.dataframe(lg.style.apply(lambda r: [f"background-color:{rcol[r['risque']]}"] * len(r), axis=1),
                         width="stretch", hide_index=True)
            c1, c2, _ = st.columns([1, 1, 4])
            if c1.button("🗑️ Vider le journal"):
                S.lv_log = []
                st.rerun()
            c2.download_button("⬇️ CSV", lg.to_csv(index=False).encode("utf-8"), "journal_predictions.csv", "text/csv")

    # ------------------------------------------------------------------ simulation de flux
    with tab_stream:
        st.markdown("Rejoue des transactions **réelles du jeu de test**, dans l'ordre chronologique : chacune est envoyée à "
                    "l'**API** (`POST /predict/features`) au fur et à mesure et comparée au seuil d'alerte.")
        Xte, test = load_xtest(), T["test"]
        c1, c2, c3, c4 = st.columns(4)
        n_tx = c1.slider("Transactions", 10, 80, 30, 5)
        pf = c2.slider("Part de fraudes (%)", 0, 60, 20, 5)
        delay = c3.slider("Pause (s / transaction)", 0.0, 1.0, 0.15, 0.05)
        seed = c4.number_input("Graine aléatoire", 0, 9999, 7)
        thr_s = float(bundle["threshold"])
        if st.button("▶️ Lancer la simulation"):
            rng = np.random.default_rng(int(seed))
            yv = test["is_fraud"].to_numpy()
            pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
            n_pos = min(len(pos), int(round(n_tx * pf / 100)))
            ix = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_tx - n_pos, replace=False)])
            ix = ix[np.argsort(test["ts"].to_numpy()[ix])]
            prog, ph_m, ph_t = st.progress(0.0), st.empty(), st.empty()
            rows = []
            tp = fp = fn = 0
            for step, j in enumerate(ix, 1):
                try:
                    out = api_client.predict_features({k: float(v) for k, v in Xte.iloc[j].items()})
                except api_client.ApiError as e:
                    st.error(str(e))
                    st.stop()
                p, thr_s = out["score"], out["threshold"]
                alert, real = bool(out["is_alert"]), bool(yv[j])
                tp += alert and real
                fp += alert and not real
                fn += (not alert) and real
                r = test.iloc[j]
                rows.append({"date": f"{r['ts']:%m-%d %H:%M}", "catégorie": r["category"], "montant ($)": float(r["amt"]),
                             "score": round(float(p), 4), "décision": "🚨 ALERTE" if alert else "✅ OK",
                             "réalité": "FRAUDE" if real else "légitime", "_k": "TP" if alert and real else "FP" if alert
                             else "FN" if real else "TN"})
                prog.progress(step / n_tx)
                with ph_m.container():
                    m = st.columns(5)
                    m[0].metric("Traitées", f"{step} / {n_tx}")
                    m[1].metric("Alertes", f"{tp + fp}")
                    m[2].metric("Fraudes détectées", f"{tp}")
                    m[3].metric("Fausses alertes", f"{fp}")
                    m[4].metric("Fraudes manquées", f"{fn}")
                view = pd.DataFrame(rows[-10:][::-1])
                colors = {"TP": "#bbf7d0", "FP": "#fde68a", "FN": "#fecaca", "TN": "#ffffff"}
                ph_t.dataframe(view.drop(columns="_k").style.apply(
                    lambda r_, v=view: [f"background-color:{colors[v.loc[r_.name, '_k']]}"] * len(r_), axis=1),
                    width="stretch", hide_index=True)
                if delay:
                    time.sleep(delay)
            full = pd.DataFrame(rows)
            fig = go.Figure()
            for lab, flag in (("légitime", False), ("FRAUDE", True)):
                d = full[full["réalité"] == lab]
                fig.add_trace(go.Scatter(x=d.index + 1, y=d["score"], mode="markers", name=lab,
                                         marker=dict(size=11, color=CLASS_COLOR["Fraude" if flag else "Légitime"],
                                                     symbol="diamond" if flag else "circle", line=dict(color="#fff", width=1))))
            fig.add_hline(y=thr_s, line_dash="dash", line_color="#111827", annotation_text="seuil")
            fig.update_layout(title="Scores du flux simulé", xaxis_title="ordre d'arrivée", yaxis_title="score",
                              yaxis_range=[-0.02, 1.02], height=380, legend=dict(orientation="h", y=1.12))
            show(fig, key="stream_chart")
            st.success(f"Terminé : {tp} fraude(s) détectée(s), {fn} manquée(s), {fp} fausse(s) alerte(s) sur {n_tx} transactions.")


# =================================================================== routage
{PAGES[0]: page_data, PAGES[1]: page_prep, PAGES[2]: page_models, PAGES[3]: page_eval,
 PAGES[4]: page_xai, PAGES[5]: page_live}[page]()