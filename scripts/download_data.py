"""Télécharge le jeu Sparkov (kartik2112/fraud-detection) dans data/.

Prérequis : un compte Kaggle et ses identifiants (fichier ~/.kaggle/kaggle.json ou variables
KAGGLE_USERNAME / KAGGLE_KEY). Sans eux, téléchargez fraudTrain.csv et fraudTest.csv à la main
depuis la page Kaggle du jeu et placez-les dans data/.
"""
import shutil
import sys
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "data"
SLUG = "kartik2112/fraud-detection"


def main():
    try:
        import kagglehub
    except ImportError:
        sys.exit("pip install kagglehub")
    src = Path(kagglehub.dataset_download(SLUG))
    DEST.mkdir(exist_ok=True)
    for name in ("fraudTrain.csv", "fraudTest.csv"):
        found = list(src.rglob(name))
        if not found:
            sys.exit(f"{name} introuvable dans {src}")
        shutil.copy(found[0], DEST / name)
        print("copié :", DEST / name)


if __name__ == "__main__":
    main()
