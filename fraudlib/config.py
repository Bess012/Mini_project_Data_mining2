"""Paramètres globaux du projet (jeu Sparkov : Credit Card Transactions Fraud Detection)."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS = Path(os.environ.get("FRAUD_ARTIFACTS", ROOT / "artifacts"))

TARGET = "is_fraud"
SEED = 42

# Séparation chronologique 60 / 20 / 20
SPLIT = (0.6, 0.2, 0.2)

# Hypothèse métier (modifiable) : coût d'analyse d'une alerte par un analyste, en dollars.
# Coût total = somme des montants des fraudes manquées + REVIEW_COST x nombre d'alertes.
REVIEW_COST = 5.0

# Fenêtre d'historique conservée par l'API pour calculer les variables de vélocité
HISTORY_DAYS = 30
