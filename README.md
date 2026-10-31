# Détection de fraude explicable — jeu Sparkov

Projet Challenge 2 (Data Mining, ENSI). Jeu de données : *Credit Card Transactions Fraud Detection* (Sparkov, Kaggle `kartik2112/fraud-detection`).

## Installation et lancement

```bash
pip install -r requirements.txt

# 1. Données : python scripts/download_data.py   (identifiants Kaggle requis)
#    ou copier à la main fraudTrain.csv et fraudTest.csv dans data/

# 2. Entraînement (écrit artifacts/) — quelques minutes sur le jeu complet
python scripts/train.py --data data/fraudTrain.csv data/fraudTest.csv

# 3. Application web interactive (barre latérale : 6 pages)
streamlit run app/streamlit_app.py

# Si train.py a déjà été exécuté avant cette version : pas besoin de tout réentraîner
python scripts/cv_report.py  --data data/fraudTrain.csv data/fraudTest.csv   # -> artifacts/cv.csv (barres « CV AUC ± écart-type »)
python scripts/update_eda.py --data data/fraudTrain.csv data/fraudTest.csv   # -> nouveaux graphiques de l'onglet Données

# 4. API de scoring (docs : http://localhost:8000/docs)
uvicorn api.main:app --port 8000

# Tests (absence de fuite de données, ordre chronologique du split)
pytest -q
```

Sans accès à Kaggle, `python scripts/make_synthetic.py` génère des fichiers factices au même format pour tester
l'installation. **Ils sont trivialement séparables : ne jamais en tirer de conclusions sur les modèles.**

## Structure

| Chemin | Rôle |
|---|---|
| `fraudlib/data.py` | chargement, split chronologique 60/20/20, statistiques exploratoires |
| `fraudlib/features.py` | variables temporelles et comportementales (sans fuite) |
| `fraudlib/modeling.py` | stratégies de déséquilibre, XGBoost, régression logistique, IF / LOF / autoencodeur |
| `fraudlib/evaluation.py` | PR-AUC, ROC-AUC, précision@k, rappel@k, F2, MCC, coût métier et seuil optimal |
| `fraudlib/explain.py` | SHAP (beeswarm, waterfall, codes motifs) et LIME |
| `fraudlib/cv.py` | validation croisée chronologique (AUC moyenne ± écart-type par méthode) |
| `fraudlib/live.py` | prédiction en direct : variables d'une transaction saisie + historique de la carte |
| `scripts/train.py` | pipeline complet, écrit les artefacts |
| `app/streamlit_app.py` | application : une page (barre latérale) par module |
| `app/ui_theme.py` | thème (CSS, gabarit Plotly) et composants visuels |
| `scripts/cv_report.py`, `scripts/update_eda.py` | recalculent `cv.csv` / `eda.joblib` sans réentraîner |
| `api/main.py` | API FastAPI |

## Application : les 6 pages

1. **Données** : sous-onglets Vue d'ensemble (déséquilibre, série quotidienne), Montants & catégories, Temps (carte de chaleur jour × heure),
   Profil & géographie (âge, ville, distance), Corrélations.
2. **Préparation** : frise du découpage chronologique, variables, comparaison des stratégies de déséquilibre.
3. **Modèles** : tableau comparatif en dégradé de vert, graphique *CV AUC moyenne ± écart-type vs métriques de test*
   (ROC-AUC, PR-AUC, F1), courbes PR / ROC / rappel selon le budget / coût, matrices de confusion des 5 méthodes.
4. **Évaluation & seuil** : curseur de budget ou de seuil, matrice de confusion et distribution des scores en direct, coût métier.
5. **Explicabilité** : SHAP global, SHAP local, **LIME** local, comparaison SHAP vs LIME.
6. **Prédiction en direct** : saisie d'une transaction (scénarios rapides, curseurs), score en jauge, codes motifs, SHAP, LIME,
   analyse « et si… ? », journal des prédictions et simulation d'un flux de transactions réelles.
   Cette page **appelle l'API FastAPI en HTTP** (aucun scoring local) : lancer d'abord `uvicorn api.main:app --port 8000`
   (URL modifiable via `FRAUD_API_URL`). Elle affiche une courte explication du résultat ; l'explication globale (SHAP, LIME) est en page 5.

## Choix méthodologiques (à reprendre dans le rapport)

- **Fusion puis re-découpage** : `fraudTrain.csv` et `fraudTest.csv` sont concaténés, triés par date, puis découpés
  60/20/20 par ordre chronologique. Le découpage Kaggle d'origine n'est pas utilisé.
- **Pas de fuite** : chaque variable comportementale (nb de transactions sur 1 h / 24 h / 30 j, montant / moyenne 30 j,
  temps depuis la précédente, vitesse implicite) n'utilise que les transactions strictement antérieures de la même
  carte. Calculer ces variables sur tout le flux avant le découpage est donc légitime, et `tests/test_features.py`
  le vérifie. Les scalers sont ajustés sur l'entraînement seul.
- **Déséquilibre** : quatre stratégies comparées en validation (aucune, poids atténué, poids complet, sous-échantillonnage 10:1),
  appliquées au seul entraînement ; la meilleure PR-AUC de validation est retenue.
- **Non supervisé** : Isolation Forest, LOF (novelty, sur un sous-échantillon de 20 000 lignes pour le temps de calcul)
  et autoencodeur dense (MLP). Aucune étiquette n'est utilisée à l'entraînement.
- **Budget k** : par défaut 1,5 × le nombre moyen de fraudes par jour observé en validation (réglable : `--budget-per-day`).
- **Coût métier** : `coût = Σ montants des fraudes manquées + c × nombre d'alertes`, avec **c = 5 $ par alerte
  (hypothèse à justifier ou à modifier : `--review-cost`)**. Le seuil minimise ce coût sur la validation, puis est
  confirmé sur le test ; l'application recalcule l'optimum si l'on change c.
- **Validation croisée** : `TimeSeriesSplit` (3 plis, fenêtre croissante) sur l'entraînement seul, AUC-ROC moyenne ± écart-type ;
  les modèles non supervisés sont ajustés sans étiquette. Le F1 à 0,50 n'a de sens que pour les modèles qui sortent des probabilités.
- **LIME** : explicateur tabulaire ; les statistiques de perturbation viennent d'un échantillon du jeu de test (fraudes sur-représentées).
  C'est une approximation locale qui dépend de l'échantillonnage : à confronter à SHAP (exact pour XGBoost).
- **Modèle final** : XGBoost (modèle à base d'arbres, requis pour SHAP TreeExplainer) ; les autres méthodes servent à la comparaison.
- **Codes motifs** : les 3 à 5 variables dont la contribution SHAP pousse le plus vers « fraude ».

## API

| Route | Rôle |
|---|---|
| `GET /health`, `GET /model` | état, modèle, variables, seuil |
| `GET/PUT /threshold` | lire / modifier le seuil |
| `POST /predict` | transaction brute → score, alerte, niveau de risque, codes motifs |
| `POST /predict/batch` | liste de ≤ 1000 transactions |
| `POST /predict/features` | vecteur de variables déjà construit |
| `POST /explain` | contributions SHAP détaillées + résumé en français |
| `POST /explain/simulate` | idem avec achats fictifs antérieurs et seuil temporaire (utilisé par la page 6) |

L'API garde en mémoire les 30 derniers jours de transactions par carte pour recalculer les variables de vélocité avec
le **même code** qu'à l'entraînement. `commit=true` (défaut) ajoute la transaction à l'historique ; `commit=false` l'évalue
sans effet de bord. Le seuil modifié par `PUT /threshold` est perdu au redémarrage.
