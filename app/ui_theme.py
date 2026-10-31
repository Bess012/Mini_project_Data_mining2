"""Thème visuel (CSS, gabarit Plotly) et petits composants d'interface de l'application."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# couleurs par modèle (cohérentes dans toute l'application)
COLOR = {"xgb": "#6366f1", "lr": "#10b981", "if": "#ef4444", "lof": "#a855f7", "ae": "#f59e0b"}
CLASS_COLOR = {"Légitime": "#3b82f6", "Fraude": "#ef4444"}
POS, NEG = "#ef4444", "#10b981"          # pousse vers fraude / vers légitime
PALETTE = ["#6366f1", "#ec4899", "#14b8a6", "#f59e0b", "#3b82f6", "#a855f7", "#ef4444", "#22c55e"]

# dégradé de la bannière de chaque page
HERO = {
    "data": ("📊", "linear-gradient(120deg,#4f46e5,#06b6d4)"),
    "prep": ("🧪", "linear-gradient(120deg,#059669,#84cc16)"),
    "models": ("🤖", "linear-gradient(120deg,#7c3aed,#ec4899)"),
    "eval": ("🎯", "linear-gradient(120deg,#f97316,#ef4444)"),
    "xai": ("🔍", "linear-gradient(120deg,#0ea5e9,#6366f1)"),
    "live": ("⚡", "linear-gradient(120deg,#db2777,#f59e0b)"),
}

pio.templates["fraud"] = go.layout.Template(layout=dict(
    colorway=PALETTE,
    font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=13, color="#1e1b4b"),
    title=dict(font=dict(size=16, color="#312e81"), x=0.01),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.75)",
    xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb"),
    yaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb"),
    hoverlabel=dict(bgcolor="#1e1b4b", font_color="#fff"),
    legend=dict(bgcolor="rgba(255,255,255,0.6)"),
    margin=dict(l=40, r=20, t=55, b=40),
))
pio.templates.default = "plotly_white+fraud"

CSS = """
<style>
:root { --c1:#6366f1; --c2:#ec4899; --c3:#14b8a6; --c4:#f59e0b; --c5:#3b82f6; }
.stApp { background: radial-gradient(1200px 600px at 100% -10%, #ede9fe 0%, transparent 60%),
                     radial-gradient(900px 500px at -10% 110%, #cffafe 0%, transparent 55%), #f6f7fb; }
.block-container { padding-top: 1.6rem; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- barre latérale ---------- */
[data-testid="stSidebar"] { background: linear-gradient(185deg,#312e81 0%,#6d28d9 55%,#be185d 100%); }
[data-testid="stSidebar"] * { color: #f5f3ff; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.25); }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display:none; }
[data-testid="stSidebar"] [data-testid="stRadioOption"], [data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.12); border-radius: 14px;
    padding: .6rem .85rem; margin-bottom: .4rem; width: 100%; cursor: pointer;
    transition: transform .18s ease, background .18s ease, box-shadow .18s ease; }
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover, [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,.22); transform: translateX(5px); }
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child { display:none; }
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #ffffff; box-shadow: 0 8px 22px rgba(0,0,0,.28); transform: translateX(7px); }
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) *,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * { color: #4c1d95 !important; font-weight: 700; }
.side-brand { font-size: 1.35rem; font-weight: 800; line-height: 1.15; letter-spacing: .2px; }
.side-sub { opacity: .8; font-size: .8rem; margin-bottom: .6rem; }
.side-card { background: rgba(255,255,255,.12); border-radius: 14px; padding: .7rem .9rem; font-size: .82rem; line-height: 1.55; }
.side-card b { color: #fde68a; }

/* ---------- bannière ---------- */
.hero { border-radius: 22px; padding: 1.35rem 1.7rem; color: #fff; margin-bottom: 1.1rem;
        box-shadow: 0 14px 34px rgba(79,70,229,.25); position: relative; overflow: hidden; }
.hero:after { content:""; position:absolute; right:-40px; top:-40px; width:210px; height:210px; border-radius:50%;
              background: rgba(255,255,255,.14); }
.hero h1 { margin:0; font-size: 1.85rem; font-weight: 800; color:#fff; padding:0; }
.hero p { margin:.25rem 0 0 0; opacity:.92; font-size:.97rem; max-width: 900px; }

/* ---------- cartes métriques ---------- */
[data-testid="stMetric"] { background:#fff; border-radius:16px; padding:.85rem 1rem; border-left:6px solid var(--c1);
    box-shadow:0 4px 16px rgba(99,102,241,.12); transition: transform .18s ease, box-shadow .18s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-4px); box-shadow:0 12px 26px rgba(99,102,241,.25); }
[data-testid="stColumn"]:nth-child(5n+2) [data-testid="stMetric"] { border-left-color: var(--c2); }
[data-testid="stColumn"]:nth-child(5n+3) [data-testid="stMetric"] { border-left-color: var(--c3); }
[data-testid="stColumn"]:nth-child(5n+4) [data-testid="stMetric"] { border-left-color: var(--c4); }
[data-testid="stColumn"]:nth-child(5n+5) [data-testid="stMetric"] { border-left-color: var(--c5); }
[data-testid="stMetricLabel"] p { font-weight: 600; color:#6b7280; }
[data-testid="stMetricValue"] { font-weight: 800; color:#1e1b4b; }

/* ---------- onglets internes, boutons, conteneurs ---------- */
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [role="tab"], .stTabs [data-baseweb="tab"] { background:#fff; border-radius: 12px 12px 0 0; padding:.45rem 1rem;
    font-weight:600; box-shadow: 0 2px 8px rgba(0,0,0,.05); }
.stTabs [role="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(120deg,#6366f1,#ec4899); }
.stTabs [role="tab"][aria-selected="true"] *, .stTabs [data-baseweb="tab"][aria-selected="true"] * { color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"] { display:none; }
.stButton > button, .stDownloadButton > button { border-radius: 12px; border: 0; font-weight: 700; color:#fff;
    background: linear-gradient(120deg,#6366f1,#ec4899); box-shadow:0 6px 16px rgba(236,72,153,.28);
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px) scale(1.02); filter: brightness(1.06);
    box-shadow:0 10px 22px rgba(236,72,153,.38); color:#fff; }
.stButton > button p { color:#fff; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { background: rgba(255,255,255,.7); border-radius: 16px;
    box-shadow: 0 4px 16px rgba(30,27,75,.07); padding: 4px; }
h2, h3 { color:#312e81; }
.pill { display:inline-block; padding:.25rem .8rem; border-radius:999px; color:#fff; font-weight:700; font-size:.85rem; }
.verdict { border-radius:18px; padding:1rem 1.3rem; color:#fff; font-weight:800; font-size:1.25rem;
    box-shadow:0 10px 26px rgba(0,0,0,.18); animation: pop .35s ease; }
.verdict small { display:block; font-weight:500; opacity:.92; font-size:.88rem; margin-top:.15rem; }
.reason { background:#fff; border-radius:12px; padding:.5rem .8rem; margin:.3rem 0; border-left:5px solid #ef4444;
    box-shadow:0 2px 8px rgba(0,0,0,.06); }
@keyframes pop { from { transform: scale(.96); opacity:.4 } to { transform: scale(1); opacity:1 } }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(key: str, title: str, subtitle: str):
    icon, grad = HERO[key]
    st.markdown(f'<div class="hero" style="background:{grad}"><h1>{icon} {title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def show(fig, key: str | None = None):
    """Affiche une figure Plotly avec NOTRE gabarit (theme=None : le thème Streamlit n'écrase pas les couleurs)."""
    st.plotly_chart(fig, width="stretch", theme=None, key=key)


def contribution_bars(names, values, title: str, xtitle: str, hover=None, height: int | None = None) -> go.Figure:
    """Barres horizontales rouge (vers fraude) / verte (vers légitime), triées par |contribution|."""
    values = np.asarray(values, float)
    order = np.argsort(np.abs(values))
    names = [names[i] for i in order]
    values = values[order]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=[POS if v > 0 else NEG for v in values],
        text=[f"{v:+.3f}" for v in values], textposition="outside", cliponaxis=False,
        hovertext=[hover[i] for i in order] if hover is not None else None,
        hovertemplate="%{y}<br>%{x:+.4f}<extra></extra>" if hover is None else "%{hovertext}<br>%{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#374151", line_width=1)
    fig.update_layout(title=title, xaxis_title=xtitle, height=height or max(320, 30 * len(names) + 110),
                      yaxis_title=None, showlegend=False)
    return fig


def confusion_fig(tp: int, fp: int, fn: int, tn: int, title: str = "", height: int = 330) -> go.Figure:
    """Matrice de confusion : couleur = part de la ligne (lisible malgré le déséquilibre), texte = effectifs."""
    z = np.array([[tn, fp], [fn, tp]], float)
    row = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
    text = [[f"<b>{int(z[i, j]):,}</b><br>{100 * row[i, j]:.1f} % de la ligne" for j in range(2)] for i in range(2)]
    labels = [["Vrais négatifs", "Faux positifs"], ["Fraudes manquées (FN)", "Fraudes détectées (VP)"]]
    text = [[f"{labels[i][j]}<br>{text[i][j]}" for j in range(2)] for i in range(2)]
    signed = np.array([[row[0, 0], -row[0, 1]], [-row[1, 0], row[1, 1]]])  # + = bien classé (vert), - = erreur (rouge)
    fig = go.Figure(go.Heatmap(
        z=signed, x=["Prédit légitime", "Prédit fraude"], y=["Réel légitime", "Réel fraude"],
        colorscale=[[0, "#b91c1c"], [0.25, "#fca5a5"], [0.5, "#ffffff"], [0.75, "#86efac"], [1, "#15803d"]],
        zmin=-1, zmax=1, showscale=False, text=text, texttemplate="%{text}", textfont=dict(size=12),
        hoverinfo="skip", xgap=4, ygap=4,
    ))
    fig.update_layout(title=title, height=height, yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=50, b=20))
    return fig
