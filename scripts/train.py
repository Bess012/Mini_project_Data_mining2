"""Entraînement complet : données -> variables -> split chrono -> déséquilibre -> modèles -> seuil -> artefacts.

Usage :
    python scripts/train.py --data data/fraudTrain.csv data/fraudTest.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudlib.config import ARTIFACTS, HISTORY_DAYS, REVIEW_COST, SEED, TARGET  # noqa: E402
from fraudlib.cv import cross_validate_models  # noqa: E402
from fraudlib.data import chronological_split, compute_eda, load_raw, split_summary  # noqa: E402
from fraudlib.evaluation import (best_cost_threshold, eval_at_threshold, n_days,  # noqa: E402
                                 summarize_scores)
from fraudlib.features import RAW_TX_FIELDS, build_features, feature_label  # noqa: E402
from fraudlib.modeling import (STRATEGIES, AutoencoderScorer, IsolationForestScorer,  # noqa: E402
                               LOFScorer, apply_imbalance, make_logreg, make_xgb, proba)

MODELS = {
    "lr": ("Régression logistique pondérée", "Supervisé"),
    "xgb": ("XGBoost", "Supervisé"),
    "if": ("Isolation Forest", "Non supervisé"),
    "lof": ("Local Outlier Factor", "Non supervisé"),
    "ae": ("Autoencodeur", "Non supervisé"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="CSV Sparkov (fraudTrain.csv fraudTest.csv)")
    ap.add_argument("--out", default=str(ARTIFACTS))
    ap.add_argument("--review-cost", type=float, default=REVIEW_COST, help="coût d'analyse d'une alerte ($)")
    ap.add_argument("--budget-per-day", type=float, default=None,
                    help="budget d'alertes/jour ; défaut = 1,5 x fraudes/jour observées en validation")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--lof-sample", type=int, default=20_000)
    ap.add_argument("--cv-folds", type=int, default=3, help="plis de la validation croisée chronologique (0 = désactivée)")
    ap.add_argument("--cv-max-rows", type=int, default=150_000, help="lignes max par pli (temps de calcul)")
    ap.add_argument("--max-rows", type=int, default=None, help="sous-échantillon chronologique (tests rapides)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ Module 1/2 : données
    log("Chargement des données")
    df = load_raw(args.data)
    if args.max_rows:
        df = df.iloc[: args.max_rows].reset_index(drop=True)
    log(f"{len(df):,} transactions, {int(df[TARGET].sum()):,} fraudes ({100 * df[TARGET].mean():.3f} %)")
    joblib.dump(compute_eda(df), out / "eda.joblib")

    log("Construction des variables (temporelles + comportementales sans fuite)")
    X, categories = build_features(df)
    y = df[TARGET].to_numpy(int)
    amt = df["amt"].to_numpy(float)

    idx = chronological_split(df)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    amt_va, amt_te = amt[va], amt[te]
    nd_va, nd_te = n_days(df["ts"].iloc[va]), n_days(df["ts"].iloc[te])
    split_df = split_summary(df, idx)
    log("\n" + split_df.to_string(index=False))

    budget = args.budget_per_day
    if budget is None:
        budget = max(1, math.ceil(1.5 * yva.sum() / nd_va))
    log(f"Budget d'alertes : {budget:.0f}/jour (≈ 1,5 x les fraudes quotidiennes de la validation)")
    rc = args.review_cost

    # ------------------------------------------------------------ déséquilibre (choix sur validation)
    log("Comparaison des stratégies de déséquilibre (XGBoost, entraînement seul)")
    imb_rows, xgb_models = [], {}
    for strat in STRATEGIES:
        Xs, ys, spw = apply_imbalance(Xtr, ytr, strat, SEED)
        m = make_xgb(spw, n_estimators=args.n_estimators, seed=SEED).fit(Xs, ys)
        p = proba(m, Xva)
        s = summarize_scores(p, yva, amt_va, nd_va, budget)
        b = best_cost_threshold(p, yva, amt_va, rc)
        imb_rows.append({"stratégie": STRATEGIES[strat], "clé": strat, "lignes d'entraînement": len(ys),
                         "PR-AUC (val)": s["pr_auc"], "ROC-AUC (val)": s["roc_auc"],
                         "précision@k (val)": s["precision_at_k"], "rappel@k (val)": s["recall_at_k"],
                         "coût min (val, $)": b["cost"]})
        xgb_models[strat] = m
        log(f"  {strat:22s} PR-AUC val = {s['pr_auc']:.4f}")
    imb = pd.DataFrame(imb_rows)
    best_strat = imb.sort_values("PR-AUC (val)", ascending=False).iloc[0]["clé"]
    log(f"Stratégie retenue : {best_strat}")

    # ------------------------------------------------------------ Module 3 : modèles
    log("Entraînement des modèles")
    scores_val, scores_te = {}, {}

    xgb = xgb_models[best_strat]
    scores_val["xgb"], scores_te["xgb"] = proba(xgb, Xva), proba(xgb, Xte)

    lr = make_logreg(SEED).fit(Xtr, ytr)
    scores_val["lr"], scores_te["lr"] = proba(lr, Xva), proba(lr, Xte)
    log("  régression logistique OK")

    for key, scorer in (("if", IsolationForestScorer(seed=SEED)),
                        ("lof", LOFScorer(n_sample=args.lof_sample, seed=SEED)),
                        ("ae", AutoencoderScorer(seed=SEED))):
        scorer.fit(Xtr)  # aucune étiquette utilisée
        scores_val[key], scores_te[key] = scorer.score(Xva), scorer.score(Xte)
        log(f"  {scorer.name} OK")

    # ------------------------------------------------------------ Module 4 : évaluation et seuil
    rows = []
    for key, (name, kind) in MODELS.items():
        sv, st = scores_val[key], scores_te[key]
        thr = best_cost_threshold(sv, yva, amt_va, rc)["threshold"]  # choisi sur la VALIDATION
        sum_va = summarize_scores(sv, yva, amt_va, nd_va, budget)
        sum_te = summarize_scores(st, yte, amt_te, nd_te, budget)
        ev = eval_at_threshold(st, yte, amt_te, thr, rc, nd_te)       # confirmé sur le TEST
        rows.append({
            "clé": key, "méthode": name, "type": kind,
            "PR-AUC val": sum_va["pr_auc"], "PR-AUC test": sum_te["pr_auc"], "ROC-AUC test": sum_te["roc_auc"],
            "précision@k": sum_te["precision_at_k"], "rappel@k": sum_te["recall_at_k"],
            "seuil (val)": thr, "alertes/jour": ev["alerts_per_day"],
            "précision": ev["precision"], "rappel": ev["recall"], "F2": ev["f2"], "MCC": ev["mcc"],
            "coût test ($)": ev["cost"], "coût sans modèle ($)": ev["cost_no_model"],
            "économie (%)": ev["saving_pct"],
        })
    comp = pd.DataFrame(rows)
    log("\n" + comp[["méthode", "PR-AUC test", "ROC-AUC test", "précision@k", "rappel@k",
                     "F2", "MCC", "économie (%)"]].round(3).to_string(index=False))

    # ------------------------------------------------------------ validation croisée chronologique
    if args.cv_folds >= 2:
        log(f"Validation croisée chronologique ({args.cv_folds} plis) sur l'entraînement")
        cv = cross_validate_models(Xtr, ytr, best_strat, args.cv_folds, args.cv_max_rows,
                                   args.n_estimators, args.lof_sample, SEED, log=log)
        cv.to_csv(out / "cv.csv", index=False)

    # ------------------------------------------------------------ artefacts
    final = "xgb"  # modèle à base d'arbres : requis pour SHAP
    thr_final = float(comp.set_index("clé").loc[final, "seuil (val)"])

    joblib.dump({
        "model": xgb, "features": list(X.columns), "categories": categories,
        "threshold": thr_final, "review_cost": rc, "budget_per_day": budget,
        "strategy": best_strat, "history_days": HISTORY_DAYS,
        "trained_on": {"rows": int(len(df)), "start": str(df["ts"].min()), "end": str(df["ts"].max())},
    }, out / "model_bundle.joblib", compress=3)

    meta_cols = ["ts", "trans_num", TARGET] + RAW_TX_FIELDS
    for name, ix, sc in (("val", va, scores_val), ("test", te, scores_te)):
        t = df.iloc[ix][meta_cols].reset_index(drop=True)
        for k in MODELS:
            t[f"score_{k}"] = sc[k]
        t.to_parquet(out / f"scores_{name}.parquet")
    Xte.reset_index(drop=True).to_parquet(out / "X_test.parquet")

    hist = df[df["ts"] >= df["ts"].max() - pd.Timedelta(days=HISTORY_DAYS)]
    hist[RAW_TX_FIELDS + ["trans_num"]].to_parquet(out / "history.parquet")

    imb.to_csv(out / "imbalance.csv", index=False)
    comp.to_csv(out / "comparison.csv", index=False)
    split_df.to_csv(out / "split.csv", index=False)
    pd.DataFrame({"feature": X.columns, "label": [feature_label(c) for c in X.columns]}).to_csv(
        out / "features.csv", index=False)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump({
            "models": {k: v[0] for k, v in MODELS.items()}, "final_model": final,
            "threshold": thr_final, "review_cost": rc, "budget_per_day": budget,
            "chosen_strategy": best_strat, "n_days_val": nd_va, "n_days_test": nd_te,
        }, f, indent=2, default=to_py, ensure_ascii=False)

    log(f"Terminé. Artefacts dans {out}")


if __name__ == "__main__":
    main()
