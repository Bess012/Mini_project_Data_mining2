"""Génère fraudTrain.csv / fraudTest.csv factices (format Sparkov) pour tester sans le vrai jeu."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fraudlib.synthetic import make_synthetic  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic")
ap.add_argument("--cards", type=int, default=600)
ap.add_argument("--days", type=int, default=200)
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
d = make_synthetic(a.cards, a.days)
cut = int(len(d) * 0.7)
d.iloc[:cut].to_csv(out / "fraudTrain.csv", index=False)
d.iloc[cut:].to_csv(out / "fraudTest.csv", index=False)
print(f"{len(d):,} lignes, taux de fraude {100 * d.is_fraud.mean():.2f} % -> {out}")
