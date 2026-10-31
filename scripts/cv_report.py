"""Calcule uniquement la validation croisée chronologique (artifacts/cv.csv), sans réentraîner les modèles finaux.

Utile si `train.py` a déjà été lancé avant l'ajout de la courbe « CV AUC ± écart-type » dans l'application.
Usage : python scripts/cv_report.py --data data/fraudTrain.csv data/fraudTest.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudlib.config import ARTIFACTS, SEED, TARGET  # noqa: E402
from fraudlib.cv import cross_validate_models  # noqa: E402
from fraudlib.data import chronological_split, load_raw  # noqa: E402
from fraudlib.features import build_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--out", default=str(ARTIFACTS))
ap.add_argument("--cv-folds", type=int, default=3)
ap.add_argument("--cv-max-rows", type=int, default=150_000)
ap.add_argument("--n-estimators", type=int, default=400)
ap.add_argument("--lof-sample", type=int, default=20_000)
a = ap.parse_args()

out = Path(a.out)
strategy = json.loads((out / "report.json").read_text(encoding="utf-8"))["chosen_strategy"]
df = load_raw(a.data)
X, _ = build_features(df)
tr = chronological_split(df)["train"]
cv = cross_validate_models(X.iloc[tr], df[TARGET].to_numpy(int)[tr], strategy, a.cv_folds, a.cv_max_rows,
                           a.n_estimators, a.lof_sample, SEED)
cv.to_csv(out / "cv.csv", index=False)
print(cv.round(4).to_string(index=False))
