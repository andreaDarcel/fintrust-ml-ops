# FinTrust ML Ops

Pipeline de préparation, validation et scoring de transactions, avec une API FastAPI.

## Logique

1. `data/raw/transaction.csv` et `customer.csv` sont lus par les chargeurs de `src/preprocessing.py`. Les en-têtes sont normalisés en `snake_case`, puis validés avec les schémas Pydantic de `src/data_validation.py`.
2. Les dates sont converties en variables d’heure, de jour de semaine et de week-end. Les identifiants et le label sont retirés des features. Les catégories sont encodées et les valeurs descriptives absentes sont imputées dans le pipeline sklearn.
3. `src/train.py` entraîne un Random Forest sur `Risk_Review_Flag` (`Yes`/`No`), avec un jeu de test stratifié. Un label absent ou non binaire interrompt l’entraînement ; aucune cible aléatoire n’est générée.
4. Le modèle et ses métadonnées sont écrits dans `models/fintrust_model.joblib` et `models/config.json`.
5. L’API charge ce modèle au démarrage. `/health` indique `degraded` sans artefact, `/info` retourne les métadonnées et `/predict` retourne une probabilité et `requires_review`.

Les données de transaction ne contiennent pas de fraude confirmée. `Risk_Review_Flag` est donc un proxy de revue, pas une vérité terrain de fraude. Sur le split local actuel, le F1 est d’environ 0,38 et l’AUC de 0,67 ; ces résultats ne suffisent pas à autoriser une décision de production.

## Démarrage

Depuis le dossier `fintrust-ml-ops` :

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pylint src tests
python -m src.train
python -m uvicorn src.api.main:app --reload
```

L’API écoute par défaut sur `http://127.0.0.1:8000`.

## CI/CD

`.github/workflows/ci.yml` exécute Pylint et les tests sur Python 3.12 et 3.13 pour les push et pull requests. Un tag `v*` lance `.github/workflows/cd.yml`, qui répète les contrôles, entraîne le modèle et publie le modèle et ses métadonnées comme artefact GitHub Actions. Le déploiement vers un environnement de production n’est pas configuré.
