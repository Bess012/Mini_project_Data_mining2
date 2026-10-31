"""Module 4 : métriques pour événements rares, budget d'alertes, seuil selon un coût métier."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def n_days(ts) -> float:
    return max((ts.max() - ts.min()).total_seconds() / 86400.0, 1.0)


# ---------- métriques au seuil ----------
def prf2_mcc(tp, fp, fn, tn):
    tp, fp, fn, tn = float(tp), float(fp), float(fn), float(tn)
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) > 0 else 0.0
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return p, r, f2, mcc


def eval_at_threshold(scores, y, amt, thr, review_cost, ndays):
    """Toutes les quantités affichées par le curseur de l'application."""
    pred = scores >= thr
    pos = y == 1
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & ~pos).sum())
    p, r, f2, mcc = prf2_mcc(tp, fp, fn, tn)
    missed = float(amt[~pred & pos].sum())
    caught = float(amt[pred & pos].sum())
    review = float(review_cost) * (tp + fp)
    baseline = float(amt[pos].sum())  # aucun modèle : toutes les fraudes sont subies
    cost = missed + review
    return {
        "threshold": float(thr), "alerts": tp + fp, "alerts_per_day": (tp + fp) / ndays,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": p, "recall": r, "f2": f2, "mcc": mcc,
        "missed_amount": missed, "caught_amount": caught, "review_cost_total": review,
        "cost": cost, "cost_no_model": baseline,
        "saving": baseline - cost,
        "saving_pct": 100 * (baseline - cost) / baseline if baseline > 0 else 0.0,
    }


# ---------- budget d'alertes ----------
def top_k_mask(scores, k):
    n = len(scores)
    k = int(min(max(k, 0), n))
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    mask[np.argpartition(-scores, k - 1)[:k]] = True
    return mask


def precision_at_k(scores, y, k):
    m = top_k_mask(scores, k)
    return float(y[m].mean()) if m.any() else 0.0


def recall_at_k(scores, y, k):
    m = top_k_mask(scores, k)
    return float(y[m].sum() / y.sum()) if y.sum() > 0 else 0.0


def threshold_from_budget(scores, k):
    """Seuil tel que environ k transactions sont alertées (les k scores les plus élevés)."""
    n = len(scores)
    k = int(min(max(k, 1), n))
    return float(np.partition(scores, n - k)[n - k])


# ---------- coût métier ----------
def cost_by_k(scores, y, amt, review_cost, kmax=None):
    """Coût total si l'on alerte les k transactions les mieux classées, pour k = 0..kmax."""
    n = len(scores)
    kmax = n if kmax is None else int(min(kmax, n))
    order = np.argsort(-scores, kind="stable")[:kmax]
    fraud_amt = (amt * (y == 1))
    caught = np.concatenate([[0.0], np.cumsum(fraud_amt[order])])
    k = np.arange(kmax + 1)
    return k, fraud_amt.sum() - caught + review_cost * k


def best_cost_threshold(scores, y, amt, review_cost):
    """Seuil minimisant : montants des fraudes manquées + coût d'analyse des alertes."""
    k, cost = cost_by_k(scores, y, amt, review_cost)
    kstar = int(np.argmin(cost))
    if kstar == 0:
        thr = float(np.nextafter(scores.max(), np.inf))  # aucune alerte n'est rentable
    else:
        thr = float(np.sort(scores)[::-1][kstar - 1])
    return {"threshold": thr, "k": kstar, "cost": float(cost[kstar])}


# ---------- résumé de scores ----------
def summarize_scores(scores, y, amt, ndays, budget_per_day):
    k = int(round(budget_per_day * ndays))
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "k": k,
        "precision_at_k": precision_at_k(scores, y, k),
        "recall_at_k": recall_at_k(scores, y, k),
    }
