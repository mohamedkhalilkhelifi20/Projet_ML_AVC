# Projet ML — Stroke Prediction AI
## Axe 2 : Classification de la Severite d'un AVC

**Domaine :** Intelligence Artificielle et Sante  
**Dataset :** International Stroke Trial (IST) — 19 435 patients, 112 variables  
**Annee :** 2025-2026  
**Auteur :** ML_Project — Axe 2

---

## Table des matieres

1. [Contexte et problematique](#1-contexte-et-problematique)
2. [Question clinique](#2-question-clinique)
3. [Dataset](#3-dataset)
4. [Structure du projet](#4-structure-du-projet)
5. [Pipeline ML — 4 versions](#5-pipeline-ml---4-versions)
6. [Resultats et performances](#6-resultats-et-performances)
7. [Fichiers de production](#7-fichiers-de-production)
8. [Installation et execution](#8-installation-et-execution)
9. [Integration interface](#9-integration-interface)

---

## 1. Contexte et problematique

L'accident vasculaire cerebral (AVC) est l'une des principales causes de mortalite et de handicap dans le monde. La classification precoce de la severite d'un AVC confirme est un enjeu medical majeur : elle permet aux cliniciens d'adapter la prise en charge et de prioriser les cas graves.

Ce projet constitue l'**Axe 2** d'une application intelligente multi-axe de prediction des AVC. Il repond a la question : **quelle est la severite de l'AVC d'un patient confirme ?**

---

## 2. Question clinique

```
Si AVC confirme → quelle est sa severite ?
    Classe 0 : Leger    (score composite 0-3)
    Classe 1 : Modere   (score composite 4-6)
    Classe 2 : Severe   (score composite 7+)
```

La variable cible **severite** n'existe pas directement dans l'IST. Elle est construite par feature engineering :

```
Score RDEF         = somme des deficits neurologiques RDEF1 a RDEF8 (Y=1, N=0)
RCONSC_factor      = F(alerte)=0, D(somnolent)=1, U(inconscient)=2
Score final        = Score RDEF + (RCONSC_factor x 2)
Classification     = 0-3 (leger), 4-6 (modere), 7+ (severe)
```

**Metrique principale : Recall severe**  
Justification clinique : un faux negatif (cas grave classe leger/modere) est plus dangereux qu'un faux positif. On minimise les cas graves rates.

---

## 3. Dataset

| Attribut | Valeur |
|----------|--------|
| Source | University of Edinburgh — IST (International Stroke Trial) |
| Patients | 19 435 |
| Variables | 112 colonnes |
| Suivi | Admission, 14 jours, 6 mois |
| Age moyen | 71.7 ans (min 16 — max 99) |
| Sexe | 53.5% hommes / 46.5% femmes |
| Licence | Open access — Edinburgh DataShare |

**Distribution des types d'AVC (STYPE) :**

| Type | Code | Effectif | Pourcentage |
|------|------|----------|-------------|
| Partial Anterior Circ. | PACS | 7 855 | 40.4% |
| Lacunar Stroke | LACS | 4 657 | 24.0% |
| Total Anterior Circ. | TACS | 4 638 | 23.9% |
| Posterior Circ. | POCS | 2 228 | 11.5% |

**Features utilisees (41 apres preprocessing) :**

- Numeriques : RDELAY, AGE, RSBP + scores RDEF computes
- Categorielles : SEX, RSLEEP, RATRIAL, RCT, RVISINF, RHEP24, RASP3, STYPE
- Features composites : n_deficits_confirmed, n_deficits_uncertain, deficit_ratio

---

## 4. Structure du projet

```
Projet_ML_AVC/
|
|-- Notebooks/
|   |-- 01_EDA_IST_Axe2_Severity_V1.ipynb
|   |-- 02_Preprocessing_IST_Axe2_Severity_V2.ipynb
|   |-- 03_Modeling_IST_Axe2_Severity_V2.ipynb
|   `-- 04_Modeling_IST_Axe2_Severity_V3.ipynb
|
|-- src/
|   `-- config.py
|
|-- artifacts/
|   `-- axe2/
|       |-- v1/
|       |   `-- test_results_v1.csv
|       |-- v2/
|       |   |-- gridsearch_results.csv
|       |   |-- test_results_all_models_v2.csv
|       |   |-- model_metadata_v2.json
|       |   |-- preprocessing_metadata_v2.json
|       |   `-- models/
|       |       `-- axe2_best_model_v2.pkl
|       `-- v3/
|           `-- models/
|               |-- pipeline_final.pkl      <- fichier principal
|               `-- features.json           <- infos pour interface
|
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

## 5. Pipeline ML — 4 versions

### Notebook 01 — EDA (Analyse Exploratoire)

**Fichier :** `01_EDA_IST_Axe2_Severity_V1.ipynb`

Objectif : comprendre la structure des donnees IST et valider la faisabilite de la classification de severite.

Contenu :
- Statistiques descriptives sur les 112 variables
- Analyse des valeurs manquantes
- Construction et validation du score de severite composite
- Distribution des classes (leger / modere / severe)
- Analyse des correlations entre les deficits neurologiques et la severite
- Selection des features d'admission uniquement (pas de leakage)
- Sauvegarde des features selectionnees dans `features_eda.json`

Resultats EDA cles :
- Fort desequilibre des classes : severe = ~8% des cas
- RCONSC et STYPE sont les variables les plus discriminantes
- AGE et RSBP ont une correlation moderee avec la severite

---

### Notebook 02 — Preprocessing V2

**Fichier :** `02_Preprocessing_IST_Axe2_Severity_V2.ipynb`

Objectif : produire des donnees propres et equilibrees pour la modelisation.

Etapes :
1. Corrections V2 par rapport a V1 :
   - RSBP = 0 remplace par NaN avant imputation (valeur impossible)
   - Valeur C dans RDEF → 0.5 + indicateur _uncertain (au lieu d'etre ignoree)
   - StandardScaler ajoute dans le pipeline numerique
   - AGE et RSBP conserves (retrait V1 etait incorrect)

2. Feature Engineering :
   - `RDEF1_v2` a `RDEF8_v2` : encodage Y=1, N=0, C=0.5
   - `RDEF1_uncertain` a `RDEF8_uncertain` : indicateur binaire pour C
   - `n_deficits_confirmed` : nombre de deficits confirmes
   - `n_deficits_uncertain` : nombre de deficits incertains
   - `deficit_ratio` : ratio deficits confirmes / total

3. Gestion du desequilibre :
   - SMOTE applique uniquement sur X_train
   - X_test jamais transforme par SMOTE (evaluation honnete)

Artefacts produits :
- `X_train_smote.csv` : donnees train apres SMOTE
- `X_test_preprocessed.csv` : donnees test
- `y_train_smote.csv` / `y_test.csv` : labels encodes
- `preprocessor_v2.pkl` : ColumnTransformer fitté
- `preprocessing_metadata_v2.json` : metadonnees

---

### Notebook 03 — Modelisation V2 (GridSearchCV)

**Fichier :** `03_Modeling_IST_Axe2_Severity_V2.ipynb`

Objectif : entrainer et optimiser 5 modeles avec GridSearchCV.

Modeles testes :
- LogisticRegression (lbfgs, multinomial, class_weight=balanced)
- DecisionTreeClassifier (max_depth=8, class_weight=balanced)
- RandomForestClassifier (200 estimateurs, class_weight=balanced)
- GradientBoostingClassifier (100 estimateurs, lr=0.1)
- XGBoostClassifier (200 estimateurs, lr=0.1)

Methodologie :
- Validation croisee : StratifiedKFold(5)
- Scoring : recall_macro (priorite clinique)
- GridSearchCV : grille d'hyperparametres pour chaque modele
- Evaluation sur test set : recall_severe, recall_macro, F1-macro, accuracy

Resultats GridSearch V2 (meilleur modele = LogisticRegression) :

| Modele | Recall Severe | Recall Macro | F1-macro |
|--------|--------------|--------------|----------|
| LogisticRegression | 0.8620 | 0.8508 | 0.7895 |
| RandomForest | 0.6350 | 0.8128 | 0.8016 |
| XGBoost | 0.6196 | 0.8144 | 0.8084 |
| GradientBoosting | 0.6043 | 0.8057 | 0.7973 |
| DecisionTree | 0.5307 | 0.7674 | 0.7650 |

Artefacts produits :
- `gridsearch_results.csv` : resultats pour comparaison V3
- `test_results_all_models_v2.csv` : metriques completes
- `model_metadata_v2.json` : metadonnees du meilleur modele
- `axe2_best_model_v2.pkl` : meilleur modele sauvegarde

---

### Notebook 04 — Modelisation V3 (Optimisation avancee)

**Fichier :** `04_Modeling_IST_Axe2_Severity_V3.ipynb`

Objectif : pousser les performances au maximum et rendre le modele explicable et deployable.

Innovations V3 par rapport a V2 :

**1. RandomizedSearchCV**
- Remplacement du GridSearchCV par RandomizedSearchCV
- Espace de recherche large avec distributions continues (scipy.stats)
- n_iter=10, CV=3 → 150 fits au total (contre 750 en V2)
- Checkpoint automatique par modele (evite de re-entrainer)
- Gain de temps : 2h30 → ~30-40 min

**2. Threshold Adjustment**
- Ajustement du seuil de decision pour la classe severe
- Si P(severe) > 0.30 → classifier severe (au lieu de 0.5 par defaut)
- Seuil optimal selectionne avec double contrainte :
  - F1-macro >= 95% de la baseline
  - Maximum 20% de cas predits severes
- Resultat : T = 0.30

**3. Calibration des probabilites**
- CalibratedClassifierCV avec Platt scaling et Isotonic regression
- Reliability diagram pour verifier la calibration
- Objectif : que P(severe) = 0.70 signifie vraiment 70% de chance

**4. SHAP Values — Explicabilite**
- TreeExplainer pour RF/XGBoost/GradientBoosting
- Summary plot (importance globale des features)
- Waterfall plot (explication d'un patient individuel)
- Justification : exige par le cahier des charges

**5. Analyse qualitative des erreurs**
- Reconstruction de STYPE et RCONSC depuis les colonnes OHE
- Taux d'erreur par type d'AVC (TACS, PACS, POCS, LACS)
- Profil des faux negatifs severes (cas graves rates)

**6. Validation croisee de stabilite**
- CV 5-fold sur le modele final
- Rapport moyenne +/- ecart-type du F1-macro

**7. Sauvegarde production-ready**
- Pipeline complet dans `pipeline_final.pkl`
- Fonction `predict_patient(patient_dict)` prete pour l'interface

Resultats finaux V3 :

| Metrique | V2 GridSearch | V3 RandomizedSearch |
|----------|--------------|---------------------|
| Recall Severe | 0.8620 | 0.9479 |
| Recall Macro | 0.8508 | 0.8486 |
| F1-macro | 0.7895 | 0.7548 |
| Threshold | 0.50 (defaut) | 0.30 (optimise) |

---

## 6. Resultats et performances

### Meilleur modele : LogisticRegression + Calibration Platt

```
Recall severe  : 0.9479   <- metrique principale (detecte 94.8% des cas graves)
Recall macro   : 0.8486   <- performance globale
F1-macro       : 0.7548   <- equilibre precision/recall
Accuracy       : 0.8113
Threshold      : 0.30
% predits severes : 20%   <- volume realiste
```

### Justification du choix du modele

La V3 est selectionnee comme version finale car :
- Le recall severe est superieur de +0.086 par rapport a V2
- Le threshold 0.30 est cliniquement justifie (mieux vaut sur-diagnostiquer que rater un cas grave)
- Le modele est calibre → les probabilites sont fiables
- Le pipeline est production-ready pour integration interface

### Comparaison V1 vs V2 vs V3

| Version | Optimisation | Recall Severe | F1-macro |
|---------|-------------|--------------|----------|
| V1 Baseline | Aucune | - | - |
| V2 GridSearchCV | GridSearch + SMOTE | 0.8620 | 0.7895 |
| V3 RandomizedSearch | RandomSearch + Threshold + Calibration | 0.9479 | 0.7548 |

---

## 7. Fichiers de production

### pipeline_final.pkl

Contient tout le necessaire pour la prediction :

```python
import joblib

artifact = joblib.load("artifacts/axe2/v3/models/pipeline_final.pkl")

artifact["model"]       # modele CalibratedClassifierCV pret a predire
artifact["features"]    # liste des 41 features attendues
artifact["threshold"]   # 0.30 (seuil de decision optimal)
artifact["label_map"]   # {0: "leger", 1: "modere", 2: "severe"}
artifact["metrics"]     # recall_severe, recall_macro, f1_macro, accuracy
artifact["metadata"]    # version, model_name, calibration, random_state
```

### Exemple de prediction

```python
import joblib, pandas as pd, numpy as np

artifact  = joblib.load("pipeline_final.pkl")
model     = artifact["model"]
threshold = artifact["threshold"]
label_map = artifact["label_map"]

# Donnees d'un patient (41 features)
patient = {
    "num__AGE": 72,
    "num__RSBP": 160,
    "num__RDELAY": 6,
    # ... autres features
}

df    = pd.DataFrame([patient])
proba = model.predict_proba(df)[0]

# Application du seuil
if proba[2] > threshold:
    pred = 2
else:
    pred = int(np.argmax(proba))

print("Severite :", label_map[pred])
print("P(leger)  :", round(proba[0], 4))
print("P(modere) :", round(proba[1], 4))
print("P(severe) :", round(proba[2], 4))
```

### features.json

```json
{
  "features": ["num__RDELAY", "num__AGE", "num__RSBP", ...],
  "n_features": 41,
  "label_map": {"0": "leger", "1": "modere", "2": "severe"},
  "threshold": 0.3,
  "model_name": "LogisticRegression"
}
```

---

## 8. Installation et execution

### Prerequis

```bash
pip install -r requirements.txt
```

### requirements.txt

```
pandas
numpy
scikit-learn==1.6.1
xgboost
imbalanced-learn
shap
matplotlib
seaborn
joblib
scipy
```

### Ordre d'execution des notebooks

```
1. 01_EDA_IST_Axe2_Severity_V1.ipynb
        Produit : features_eda.json

2. 02_Preprocessing_IST_Axe2_Severity_V2.ipynb
        Produit : X_train_smote.csv, X_test_preprocessed.csv,
                  y_train_smote.csv, y_test.csv, preprocessor_v2.pkl

3. 03_Modeling_IST_Axe2_Severity_V2.ipynb
        Produit : gridsearch_results.csv, axe2_best_model_v2.pkl

4. 04_Modeling_IST_Axe2_Severity_V3.ipynb
        Produit : pipeline_final.pkl, features.json
```

### Configuration Google Colab

Tous les notebooks utilisent `src/config.py` pour centraliser les chemins :

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/ML_Project")
DATA_PATH    = PROJECT_ROOT / "Data" / "IST_corrected.csv"
RANDOM_STATE = 42
```

### Optimisation du temps d'execution

Le notebook V3 utilise un systeme de checkpoints :
- Chaque modele RandomizedSearch est sauvegarde immediatement apres le fit
- A la re-execution, les modeles deja calcules sont charges depuis le `.pkl`
- Temps : 2h30 (premiere execution) → moins de 1 minute (re-execution)

Pour forcer le re-calcul :

```python
import shutil
shutil.rmtree(CHECKPOINT_DIR)
CHECKPOINT_DIR.mkdir()
```

---

## 9. Integration interface

Ce projet est prevu pour une integration avec une interface **Next.js** via une **API FastAPI**.

### Architecture

```
Next.js (frontend)
      |
      | HTTP POST (JSON patient)
      |
FastAPI (backend Python)
      |
      | joblib.load
      |
pipeline_final.pkl
```

### Fichiers a fournir au developpeur

```
artifacts/axe2/v3/models/pipeline_final.pkl   <- modele complet
artifacts/axe2/v3/models/features.json        <- features attendues
```

### Version scikit-learn requise

```bash
pip install scikit-learn==1.6.1
```

Le modele a ete entraine avec scikit-learn 1.6.1. Utiliser une version differente peut provoquer des avertissements de compatibilite.

---

## Notes techniques

- Le dataset IST n'est pas versionne sur GitHub (taille > 10 Mo)
- Les checkpoints de RandomizedSearch ne sont pas versionnels (regenerables)
- Les CSV intermediaires (X_train_smote, X_test_preprocessed) ne sont pas versionnels
- Seuls `pipeline_final.pkl` et `features.json` sont necessaires pour la production

---

*Projet ML Stroke Prediction 2025-2026 — Axe 2 : Classification de Severite*
