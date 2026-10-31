"""Recalcule artifacts/eda.joblib (nouveaux graphiques de l'onglet Données) sans réentraîner.
Usage : python scripts/update_eda.py --data data/fraudTrain.csv data/fraudTest.csv
"""
import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fraudlib.config import ARTIFACTS  # noqa: E402
from fraudlib.data import compute_eda, load_raw  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--out", default=str(ARTIFACTS))
a = ap.parse_args()
joblib.dump(compute_eda(load_raw(a.data)), Path(a.out) / "eda.joblib")
print("eda.joblib mis à jour")
