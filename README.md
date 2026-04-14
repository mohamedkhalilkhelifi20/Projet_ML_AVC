# 🧠 Projet ML — Prédiction Multi-Axe des AVC
### Stroke Prediction AI | IST Dataset + NHANES Dataset | ML 2025-2026

---

<p>
  <img src="https://img.shields.io/badge/Next.js-16.2.3-black?style=for-the-badge&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/React-19.2.4-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/TypeScript-5-3178C6?style=for-the-badge&logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-ML-AA4A44?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LightGBM-ML-9ACD32?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Imbalanced--Learn-SMOTE-FF6F00?style=for-the-badge" />
</p>

<p>
  <img src="https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Numpy-Data-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge" />
</p>

<p>
  <img src="https://img.shields.io/badge/SHAP-Explainability-FF5733?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Feature%20Engineering-Advanced-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Threshold%20Tuning-Optimisation-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Pipeline-Sklearn-6A5ACD?style=for-the-badge" />
</p>

<p>
  <img src="https://img.shields.io/badge/LightGBM-Axe%201-F4C430?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Logistic%20Regression-Axe%202%20%26%203-8B5CF6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-NHANES%20%26%20IST-0EA5E9?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Patients-19%20435-EF4444?style=for-the-badge" />
</p>

<p>
  <img src="https://img.shields.io/badge/status-en%20développement-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/ML%20Project-2025--2026-blueviolet?style=flat-square" />
</p>

---

## 👥 Équipe

| Membre | Rôle |
|--------|------|
| Mohamed Khalil Khelifi | Axe 1 — Risque AVC |
| Khouloud Lassoued | Axe 2 — Sévérité AVC |
| Sinda Graba | Axe 3 — Mortalité |

---


## 📊 Résultats clés

| Axe | Objectif | Meilleur modèle | Métrique principale |
|-----|----------|-----------------|---------------------|
| **Axe 1** | Risque d'AVC | XGBoost + SMOTE | AUC-ROC = — |
| **Axe 2** | Sévérité AVC | LogisticRegression (calibré) | Recall sévère = **0.9755** |
| **Axe 3** | Mortalité post-AVC | — | AUC-ROC = — |

---

## 📋 Table des matières

- [Description du projet](#-description-du-projet)
- [Datasets](#-datasets)
- [Structure du repository](#-structure-du-repository)
- [Axe 1 — Risque d'AVC](#-axe-1--risque-davc)
- [Axe 2 — Sévérité de l'AVC](#-axe-2--sévérité-de-lavc)
- [Axe 3 — Mortalité post-AVC](#-axe-3--mortalité-post-avc)
- [Interface Next.js](#-interface-nextjs)
- [Installation](#-installation)
- [Lancer l'application](#-lancer-lapplication)

---

## 📖 Description du projet

**Pipeline Machine Learning complet pour l'évaluation clinique du risque, de la sévérité et de la mortalité post-AVC.**  
*Basé sur les datasets NHANES (population générale US) et IST (International Stroke Trial — 19 435 patients).*

Application web intelligente de **prédiction multi-axe des AVC** basée sur le Machine Learning,
développée pour assister les cliniciens dans la prise de décision médicale.

**3 questions cliniques fondamentales :**
> 🔴 Q1 — Ce patient est-il à risque d'AVC ?
> 🟠 Q2 — Si AVC confirmé, quelle est sa sévérité ?
> 🟡 Q3 — Quel sera le devenir du patient ?

---

## 🗄️ Datasets

| Dataset | Source | Patients | Usage |
|---------|--------|----------|-------|
| **NHANES** | Mendeley — DOI: 10.17632/xggs239bnw.1 | 4 603 | Axe 1 |
| **IST** | Edinburgh DataShare | 19 435 | Axes 2 & 3 |

> ⚠️ Les fichiers CSV ne sont pas versionnés (trop lourds).
> Télécharger depuis les sources ci-dessus et placer dans `data/`.
⚠️ Le frontend est dans un repository séparé : ml-pipeline-frontend

---

## 📁 Structure du repository

```
Projet_ML_AVC/
│
├── 📓 Notebooks/
│   ├── axe1/
│   │   └── ...
│   ├── axe2/
│   │   ├── 01_EDA_IST_Axe2_Severity_V1.ipynb
│   │   ├── 02_Preprocessing_IST_Axe2_Severity_V1.ipynb
│   │   ├── 02_Preprocessing_IST_Axe2_Severity_V2.ipynb
│   │   ├── 03_Modeling_IST_Axe2_Severity_V1.ipynb
│   │   ├── 03_Modeling_IST_Axe2_Severity_V2.ipynb
│   │   └── 04_Modeling_IST_Axe2_Severity_V3.ipynb
│   └── axe3/
│       └── ...
│
├── 🐍 src/
│   ├── config.py
│   └── axe2_utils.py
│
├── 🤖 models/
│   └── axe2/
│       ├── axe2_best_model_v1.pkl
│       ├── axe2_best_model_v2.pkl
│       └── pipeline_final.pkl        ← modèle production V3
│
├── 🌐 app/
│   ├── main.py
│   └── pages/
│       ├── axe1_stroke_risk.py
│       ├── axe2_severity.py
│       └── axe3_mortality.py
│
├── 📊 data/                           ← non versionné (.gitignore)
│   ├── IST_corrected.csv
│   └── nhanes_stroke.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔵 Axe 1 — Risque d'AVC

**Dataset** : Ping Wang (2024), *Imbalanced Data-based Prediction and Risk Factor Analysis of Stroke*, Mendeley Data  
**DOI** : [10.17632/xggs239bnw.1](https://doi.org/10.17632/xggs239bnw.1)  
**Source** : NHANES (National Health and Nutrition Examination Survey)  
**Objectif** : Prédire le risque d'AVC à partir de données cliniques, démographiques et biologiques

---

## Contexte clinique

L'AVC est l'une des principales causes de mortalité et de handicap dans le monde. Ce projet développe un modèle de Machine Learning pour détecter les patients à risque avant la survenue d'un AVC.

**Métrique prioritaire : Recall**  
Un faux négatif (AVC non détecté) est cliniquement plus grave qu'un faux positif. Le Recall est donc la métrique principale tout au long du projet. L'AUC-ROC sert de critère secondaire pour mesurer la qualité de discrimination.

---

## Dataset

| Attribut | Valeur |
|---|---|
| Patients total | 4 603 |
| AVC (stroke=1) | 362 (7.86%) |
| Non-AVC (stroke=0) | 4 241 (92.14%) |
| Variables prédictives | 35 |
| Variable cible | `stroke` (0 = non, 1 = oui) |
| Valeurs manquantes | 0 |
| Doublons | 0 |

**Variables catégorielles (15)** : `gender`, `age`, `Race`, `Marital status`, `alcohol`, `smoke`, `sleep disorder`, `Health Insurance`, `General health condition`, `depression`, `diabetes`, `hypertension`, `high cholesterol`, `Coronary Heart Disease`, `Body Mass Index`

**Variables numériques (20)** : `sleep time`, `Minutes sedentary activity`, `Waist Circumference`, `Systolic blood pressure`, `Diastolic blood pressure`, `High-density lipoprotein`, `Triglyceride`, `Low-density lipoprotein`, `Fasting Glucose`, `Glycohemoglobin`, `energy`, `protein`, `Carbohydrate`, `Dietary fiber`, `Total fat`, `Total saturated fatty acids`, `Total monounsaturated fatty acids`, `Total polyunsaturated fatty acids`, `Potassium`, `Sodium`

---

## Structure du projet

```
stroke-ml-project/
├── dataset/
│   ├── data.csv                         ← Dataset brut NHANES
│   └── data_clean/
│       ├── df_clean.csv                 ← Dataset après preprocessing
│       ├── df_features.csv              ← Dataset après feature selection (V2/V3/V4)
│       ├── df_train_final.csv           ← Train set (version finale)
│       └── df_test_final.csv            ← Test set (version finale)
│
├── notebook_v1/
│   ├── src/
│   │   ├── config.py                    ← Configuration centrale (constantes + fonctions)
│   │   └── features.json               ← Listes CATEGORICAL_VARS / NUMERICAL_VARS
│   │
│   ├── 1_EDA_data_v1.ipynb             ← EDA baseline
│   ├── 2_Preprocessing_v1.ipynb        ← Preprocessing minimal
│   ├── 3_modeling_v1.ipynb             ← Modeling baseline
│   ├── pipeline_v1.pkl                 ← Modèle V1 sauvegardé
│   └── features_v1.pkl                 ← Features V1 (35 originales)
│
├── notebook_v2/
│   ├── 1_EDA_Data_v2.ipynb             ← EDA enrichie (outliers, skewness, VIF)
│   ├── 2_data_preprocessing_v2.ipynb   ← Winsorization + Log1p
│   ├── 3_modeling_v2.ipynb             ← Feature Selection + Pipeline
│   ├── pipeline_v2.pkl                 ← Meilleur modèle V2
│   └── features_v2.pkl                 ← Features sélectionnées V2
│
├── notebook_v3/
│   ├── 3_modeling_v3.ipynb             ← SMOTE + GridSearchCV + seuil optimal
│   ├── pipeline_v3.pkl                 ← Artifact {pipeline, threshold, features, model_name}
│   └── features_v3.pkl                 ← Features V3
│
├── notebook_v4/
│   ├── 3_modeling_v4.ipynb             ← ADASYN + LightGBM + seuil optimal
│   ├── pipeline_v4.pkl                 ← Artifact {pipeline, threshold, features, model_name}
│   └── features_v4.pkl                 ← Features V4
│
├── notebook_final/
│   ├── 1_EDA_Data.ipynb                ← EDA finale
│   ├── 2_data_preprocessing.ipynb      ← Preprocessing final
│   ├── 3_Feature_Engineering.ipynb     ← Feature Engineering + split unique
│   ├── 4_modeling.ipynb                ← Modeling final avec seuil OOF
│   ├── pipeline_final.pkl              ← Artifact final {pipeline, threshold, features, model_name}
│   └── selected_features.pkl           ← Features finales sélectionnées
│
└── README.md
```

---

## Architecture centrale — `config.py`

`config.py` est la **source de vérité unique** du projet. Il est importé par tous les notebooks.

```python
import sys
sys.path.insert(0, './src')
import importlib, config
importlib.reload(config)
from config import TARGET, RANDOM_STATE, check_data_quality, load_features
```

**Constantes** : `TARGET = 'stroke'`, `RANDOM_STATE = 42`, `DATA_PATH`, `CLEAN_PATH`, `FEATURES_PATH`, `FEATURES_JSON_PATH`

**Fonctions** :
- `check_data_quality(df)` — vérifie NaN et doublons, appelée dans chaque notebook
- `save_features(categorical_vars, numerical_vars)` — sauvegarde dans `features.json`, appelée en fin d'EDA
- `load_features()` — charge depuis `features.json`, appelée dans Preprocessing et Modeling

---

## Progression des versions

| Version | Recall stroke | AUC-ROC | Nouveauté principale |
|---|---|---|---|
| V1 — Baseline | ~0% | ~0.50 | 3 modèles par défaut, aucun traitement déséquilibre |
| V2 — Feature Selection | ~10–25% | ~0.59 | Winsorization + Log1p + corrélation + VIF itératif |
| V3 — SMOTE + Tuning | ~60–80% | ~0.54 | ImbPipeline SMOTE + GridSearchCV + seuil dynamique |
| V4 — ADASYN + LightGBM | ~60–95% | ~0.54 | ADASYN + LightGBM 4ème modèle |
| **Finale** | **~86%** | **~0.52** | Feature Engineering + split-first + seuil OOF |

> **Note** : L'AUC-ROC plafonne autour de 0.54 indépendamment de la configuration. C'est une limitation intrinsèque du dataset NHANES pour cette tâche, pas une erreur méthodologique.

---

## Version 1 — Baseline

**Notebooks** : `1_EDA_data_v1.ipynb` → `2_Preprocessing_v1.ipynb` → `3_modeling_v1.ipynb`

### Objectif
Établir une baseline sans aucun traitement du déséquilibre de classes. Justifie empiriquement les améliorations des versions suivantes.

### EDA V1
- Distributions des 35 variables (histogrammes, countplots)
- Visualisation du déséquilibre cible (92.14% vs 7.86%)
- Corrélation de toutes les variables avec `stroke`
- Heatmap de corrélation inter-features
- `save_features()` → génère `features.json`

### Preprocessing V1
- Chargement `data.csv` + `str.strip()` (corrige `'alcohol '` avec espace)
- `check_data_quality()` — confirmation 0 NaN, 0 doublons
- Sauvegarde `df_clean.csv` **sans transformation**

### Modeling V1
- Chargement `df_clean.csv` + `load_features()`
- Split 80/20 stratifié
- `StandardScaler` fitté sur `X_train[NUMERICAL_VARS]` uniquement
- 3 modèles par défaut : Logistic Regression, Random Forest, XGBoost
- **Résultats** : Accuracy ~92%, Recall ~0%, AUC ~0.50
- **Conclusion** : Sans rééquilibrage, les modèles ignorent la classe minoritaire → justifie V2

### Artefacts sauvegardés
- `pipeline_v1.pkl` — meilleur modèle selon Recall (pipeline simple sans scaler)
- `features_v1.pkl` — 35 features originales

---

## Version 2 — Feature Selection + Preprocessing avancé

**Notebooks** : `1_EDA_Data_v2.ipynb` → `2_data_preprocessing_v2.ipynb` → `3_modeling_v2.ipynb`

### Objectif
Améliorer la qualité des données et réduire le bruit par sélection de features. Justifie l'ajout de SMOTE en V3.

### EDA V2 (enrichie)
- `LABEL_MAPS` importé depuis `config.py` — utilisé directement dans les visualisations
- `CATEGORICAL_VARS` défini dynamiquement : `nunique() <= 6`
- **Section outliers** : quantification % par variable → justifie Winsorization
- **Section skewness** : tableau coloré (rouge = `|skew| > 2`) → justifie Log1p
- **Section corrélation** : toutes variables (cat + num) vs `stroke`, deux seuils visuels (0.05 / 0.10)
- **Heatmap complète** : inter-features → identifie paires multicolinéaires → justifie VIF

### Preprocessing V2
- `load_features()` chargé **avant** `pd.read_csv()` — ordre correct
- `str.strip()` sur les noms de colonnes
- **Winsorization IQR** : bornes calculées sur `df` original, appliquées sur `df_proc`
- **Log1p** : recalculé sur `df_proc` après Winsorization (variables avec `|skew| > 2`)
- Sauvegarde `df_clean.csv` avec transformations

### Modeling V2
- Chargement `df_clean.csv` (35 features transformées)
- Split 80/20 stratifié sur `ALL_FEATURES`
- **Feature Selection** :
  - Corrélation Pearson sur `X_train` : suppression si `|corr| < 0.01`
  - VIF itératif : suppression une feature à la fois (seuil VIF < 15) — évite l'élimination incorrecte de features importantes comme `age` et `Systolic blood pressure`
- `Pipeline(ColumnTransformer(StandardScaler sur NUM_SELECTED) → Model)`
- 3 modèles : Logistic Regression, Random Forest, XGBoost
- **Résultats** : Recall ~10–25%, **AUC ~0.59** (meilleur AUC de toutes les versions)
- Sauvegarde `df_features.csv` avec les features sélectionnées

### Artefacts sauvegardés
- `pipeline_v2.pkl` — meilleur modèle selon Recall
- `features_v2.pkl` — features sélectionnées (après corrélation + VIF)

---

## Version 3 — SMOTE + GridSearchCV + Seuil dynamique

**Notebook** : `3_modeling_v3.ipynb` (réutilise preprocessing V2)

### Objectif
Traiter le déséquilibre de classes par SMOTE et optimiser les hyperparamètres. Premier version avec recall cliniquement acceptable.

### Architecture pipeline
```
ImbPipeline(
    SMOTE(random_state=42),
    ColumnTransformer(StandardScaler sur NUM_SELECTED),
    Model
)
```

**Règles importantes** :
- SMOTE sans `class_weight='balanced'` — double correction interdite
- StandardScaler dans le pipeline — fitté sur `X_train` uniquement

### Optimisation
- `RandomizedSearchCV` + `GridSearchCV` (scoring='recall', cv=5)
- Modèles retenus : GridSearchCV
- 3 modèles : Logistic Regression, Random Forest, XGBoost

### Seuil dynamique
- Analyse `np.arange(0.1, 0.9, 0.05)` par modèle
- **Critère** : `Recall >= 0.60 ET Seuil >= 0.10` (évite les classifieurs dégénérés)
- Fallback si aucun candidat : seuil avec `Seuil >= 0.10` uniquement
- Sélection best model : **AUC-ROC max** parmi candidats valides

### Résultats
- Recall ~60–80%, AUC ~0.54
- Recall amélioré mais AUC quasi identique à V2 → limitation dataset

### Artefacts sauvegardés
```python
artifact = {
    'pipeline'  : best_model,
    'threshold' : best_threshold,
    'features'  : SELECTED_FEATURES,
    'model_name': best_name,
}
# pipeline_v3.pkl
```

---

## Version 4 — ADASYN + LightGBM

**Notebook** : `3_modeling_v4.ipynb` (réutilise preprocessing V2)

### Objectif
Tester ADASYN (génération adaptative) et LightGBM comme alternative à SMOTE + XGBoost.

### Différences vs V3
- `ADASYN` remplace `SMOTE` — génère plus d'exemples dans les zones difficiles
- **LightGBM** ajouté comme 4ème modèle
- `RandomizedSearchCV` uniquement (GridSearchCV supprimé)

### Règles respectées
- ADASYN sans `class_weight='balanced'` — pas de double correction
- LightGBM sans `is_unbalance=True` — pas de double correction
- Même critère seuil que V3 : `Recall >= 0.60 ET Seuil >= 0.10`

### Résultats
- Recall comparable à V3, AUC ~0.54
- ADASYN ne surpasse pas SMOTE de manière significative
- La limitation vient du dataset, pas de la méthode

### Artefacts sauvegardés
- `pipeline_v4.pkl` — dict artifact identique à V3
- `features_v4.pkl` — features sélectionnées

---

## Version Finale — Feature Engineering + Seuil OOF

**Notebooks** : `1_EDA_Data.ipynb` → `2_data_preprocessing.ipynb` → `3_Feature_Engineering.ipynb` → `4_modeling.ipynb`

### Objectif
Architecture complète et rigoureuse : split en premier, feature engineering clinique, seuil optimal sans data leakage.

### Feature Engineering (`3_Feature_Engineering.ipynb`)

**Le split 80/20 est effectué ICI et une seule fois** — avant toute sélection ou engineering. Toutes les transformations sont calculées sur `X_train` et appliquées à l'identique sur `X_test`.

**Étape 1 — Filtrage corrélation** : `CORR_THRESHOLD = 0.02` sur `X_train`

**Étape 2 — Features engineered** (créées sur `X_train`, appliquées sur `X_test`) :

| Feature | Formule | Justification clinique |
|---|---|---|
| `cardio_risk_score` | somme(hypertension, diabetes, high cholesterol, smoke) | Score de risque cardiovasculaire global |
| `age_CHD` | age × Coronary Heart Disease | Interaction âge × maladie coronarienne |
| `pulse_pressure` | Systolic BP − Diastolic BP | Marqueur rigidité artérielle |
| `hba1c_diabetes` | Glycohemoglobin × diabetes | Double marqueur glycémique |
| `ldl_hdl_ratio` | LDL / (HDL + 1e-6) | Profil lipidique cardiovasculaire |
| `depression_insurance` | depression × Health Insurance | Profil vulnérable sans accès aux soins |
| `fat_ratio` | sat. fatty acids / poly. fatty acids | Alimentation pro-inflammatoire |
| `fiber_cholesterol` | Dietary fiber × high cholesterol | Protection alimentaire |
| `smoke_hypertension` | smoke × hypertension | Double facteur de risque vasculaire |
| `smoke_diabetes` | smoke × diabetes | Cumul risques métaboliques |

**Sorties** : `df_train_final.csv`, `df_test_final.csv`, `selected_features.pkl`

### Modeling final (`4_modeling.ipynb`)

**Pipeline** :
```
ImbPipeline(
    SMOTE(random_state=42),
    StandardScaler(),
    Model
)
```

**Modèles** : Logistic Regression, XGBoost, LightGBM  
**Optimisation** : `GridSearchCV(scoring='recall', cv=StratifiedKFold(5))`

**Seuil OOF (Out-Of-Fold)** :
```
cross_val_predict(cv=5) sur X_train → probabilités OOF
→ X_test reste invisible → zéro data leakage
→ Filtre : Recall >= 0.60 ET Seuil >= 0.10
→ Meilleur F1 parmi candidats valides
```

**Sélection best model** : AUC-ROC max parmi `Recall >= 0.60 ET Seuil >= 0.10`

**Résultats** : Recall ~86%, AUC ~0.52, Seuil ~0.25

### Artefact final
```python
artifact = {
    'pipeline'  : best_pipeline,   # ImbPipeline entraîné
    'threshold' : best_threshold,  # seuil optimal OOF
    'features'  : FINAL_FEATURES,  # liste des features finales
    'model_name': best_model_name, # nom du meilleur modèle
}
# pipeline_final.pkl
```

### Chargement pour inférence
```python
import pickle

with open('pipeline_final.pkl', 'rb') as f:
    artifact = pickle.load(f)

pipeline   = artifact['pipeline']
threshold  = artifact['threshold']
features   = artifact['features']
model_name = artifact['model_name']

# Prédiction
y_proba = pipeline.predict_proba(X_new[features])[:, 1]
y_pred  = (y_proba >= threshold).astype(int)
```

---

## Principes techniques clés

| Principe | Règle |
|---|---|
| Recall prioritaire | Metric principale clinique — faux négatif = AVC manqué |
| str.strip() | Obligatoire après `pd.read_csv('data.csv')` — colonne `'alcohol '` avec espace |
| StandardScaler | Fitté sur `X_train` uniquement, dans le pipeline, sur `NUMERICAL_VARS` seulement |
| SMOTE | Jamais combiné avec `class_weight='balanced'` — double correction |
| LightGBM | Sans `is_unbalance=True` si SMOTE dans le pipeline |
| XGBoost | Sans `scale_pos_weight` si SMOTE dans le pipeline |
| VIF | Suppression itérative (une feature à la fois) — pas single-pass |
| Winsorization | Bornes calculées sur `df` original, appliquées sur `df_proc` |
| Seuil minimum | `>= 0.10` — évite les classifieurs dégénérés à recall artificiel |
| AUC plafond | ~0.54 — limitation du dataset, pas du code |

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn statsmodels joblib
```

### `requirements.txt`
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
imbalanced-learn
statsmodels
joblib
jupyter
```

---

## Ordre d'exécution

### Versions V1 → V4 (comparaison)
```
notebook_v1/1_EDA_data_v1.ipynb
notebook_v1/2_Preprocessing_v1.ipynb
notebook_v1/3_modeling_v1.ipynb

notebook_v2/1_EDA_Data_v2.ipynb
notebook_v2/2_data_preprocessing_v2.ipynb
notebook_v2/3_modeling_v2.ipynb

notebook_v3/3_modeling_v3.ipynb   ← réutilise df_features.csv de V2

notebook_v4/3_modeling_v4.ipynb   ← réutilise df_features.csv de V2
```

### Version finale
```
notebook_final/1_EDA_Data.ipynb
notebook_final/2_data_preprocessing.ipynb
notebook_final/3_Feature_Engineering.ipynb
notebook_final/4_modeling.ipynb
```

---

## Références

- Ping Wang (2024). *Imbalanced Data-based Prediction and Risk Factor Analysis of Stroke*. Mendeley Data. DOI: 10.17632/xggs239bnw.1
- NHANES — National Health and Nutrition Examination Survey, CDC
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.
- He et al. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning. IJCNN.


---

## 🟠 Axe 2 — Sévérité de l'AVC

> **"Quelle est la sévérité de l'AVC ?"**

### Dataset
- **Source :** IST — Edinburgh DataShare
- **Patients :** 19 435 | Variables : 112 | Période : 1991–1996

### Variable cible construite

La variable `severity_class` n'existe pas dans l'IST — elle est construite via un score composite :

```
RDEF_score     = Σ(RDEF1..8)      Y=1 | N=0 | C=0.5 (V2)
RCONSC_factor  = F→0 | D→1 | U→2
severity_score = RDEF_score + 2 × RCONSC_factor

léger : 0–3  |  modéré : 4–6  |  sévère : 7+
```

**Distribution :**

| Classe | Effectif | % |
|--------|----------|---|
| léger | 9 692 | 49.9% |
| modéré | 8 096 | 41.7% |
| **sévère** | **1 630** | **8.4%** |

### 🔄 Progression des versions

#### 📘 Version 1 — Baseline exploratoire

**Notebooks :**
- `01_EDA_IST_Axe2_Severity_V1.ipynb` — Analyse exploratoire complète
- `02_Preprocessing_IST_Axe2_Severity_V1.ipynb` — Preprocessing baseline
- `03_Modeling_IST_Axe2_Severity_V1.ipynb` — Modélisation comparative

**Ce qui a été fait :**

| Étape | Détail |
|-------|--------|
| EDA | Classification temporelle des 112 variables, détection leakage, analyse RDEF |
| Preprocessing | Pipeline `SimpleImputer + OneHotEncoder`, split 80/20 stratifié |
| Features | 20 features baseline (avec RCONSC + RDEF = leakage structurel démontré) |
| Modèles | LR, DT, RF, GB — 4 modèles, 3 familles |
| Optimisation | GridSearchCV — tous les modèles, scoring=`recall_macro`, CV=3 |
| Évaluation | F1-macro, Recall macro, **Recall sévère**, AUC-ROC par classe |

**Résultats V1 :**

| Scénario | Recall sévère | F1-macro | Accuracy |
|----------|---------------|----------|----------|
| V1 avec leakage (RF) | ≈ 0.98 | ≈ 0.98 | ≈ 0.98 |
| **V1 sans leakage (GridSearch)** | **0.72** | **0.5422** | **0.5857** |

> ⚠️ La chute F1 : 0.98 → 0.54 **prouve le data leakage structurel** (RCONSC + RDEF dans la formule cible).

---

#### 📗 Version 2 — Pipeline corrigé + SMOTE

**Notebooks :**
- `02_Preprocessing_IST_Axe2_Severity_V2.ipynb` — Preprocessing amélioré
- `03_Modeling_IST_Axe2_Severity_V2.ipynb` — Modélisation renforcée

**4 corrections majeures :**

| Problème V1 | Correction V2 |
|-------------|---------------|
| RSBP=0 valide | → NaN avant imputation |
| C dans RDEF → NaN | → 0.5 + variable `_uncertain` |
| Pas de StandardScaler | → Ajouté dans pipeline numérique |
| Pas de SMOTE | → Appliqué sur X_train uniquement |

**Feature Engineering V2 — 3 scores composites :**

| Feature | Construction |
|---------|-------------|
| `n_deficits_confirmed` | Σ(RDEF_v2 == 1.0) |
| `n_deficits_uncertain` | Σ(RDEF_uncertain == 1) |
| `deficit_ratio` | confirmed / total évaluable |

**SMOTE — équilibrage du train set :**

| Classe | Avant SMOTE | Après SMOTE |
|--------|-------------|-------------|
| léger | 7 753 (49.9%) | 7 753 (35.3%) |
| modéré | 6 477 (41.7%) | 6 477 (29.5%) |
| **sévère** | **1 304 (8.4%)** | **7 753 (35.3%)** |

**5 modèles testés :** LR, DT, RF, GB, **XGBoost** (ajouté en V2)

**Résultats V2 :**

| Modèle | Recall sévère | Recall macro | F1-macro | Accuracy |
|--------|---------------|--------------|----------|----------|
| **LogisticRegression** ⭐ | **0.8650** | **0.8534** | **0.7926** | **0.8568** |
| DecisionTree | 0.8344 | 0.8548 | 0.8025 | 0.8702 |
| RandomForest | 0.6411 | 0.8145 | 0.8019 | 0.8859 |
| GradientBoosting | — | — | — | — |
| XGBoost | — | — | — | — |

> 💡 La Régression Logistique surpasse RF grâce au StandardScaler + SMOTE.

**Après GridSearchCV :**

| Métrique | V1 | V2 | Gain |
|----------|----|----|------|
| Recall sévère | 0.72 | **0.8620** | **+0.142** |
| F1-macro | 0.5422 | **0.7895** | **+0.247** |
| Accuracy | 0.5857 | 0.8508 | +0.265 |

---

#### 📕 Version 3 — Optimisation avancée + Explicabilité

**Notebook :** `04_Modeling_IST_Axe2_Severity_V3.ipynb`

**Techniques avancées :**

| Technique | Objectif | Résultat |
|-----------|----------|----------|
| RandomizedSearchCV (n_iter=10, CV=3) | Espace large vs GridSearch V2 | Amélioration sur tous les modèles |
| Threshold Tuning (T=0.30) | Maximiser Recall sévère | +0.0829 points |
| Calibration Platt Scaling | Probabilités fiables | Recall sévère : 0.8650 → 0.9417 |
| SHAP Values | Explicabilité (cahier des charges) | Global + patient individuel |
| Analyse erreurs | Profiler les 43 FN sévères | Profils à risque identifiés |
| CV Stabilité | Vérifier robustesse | std=0.0025 → très stable |
| AUC-ROC multiclasse | Évaluation complète | One-vs-Rest par classe |

**Threshold Tuning — table des seuils :**

| Seuil T | Recall sévère | F1-macro | % prédit sévères |
|---------|---------------|----------|------------------|
| 0.15 | 0.9755 | 0.7142 | 24% |
| **0.30** ⭐ | **0.9479** | **0.7548** | **20%** |
| 0.50 | 0.8650 | 0.7927 | 15% |

**Calibration des probabilités :**

| Variante | Recall sévère | Recall macro | F1-macro |
|----------|---------------|--------------|----------|
| Non calibré | 0.8650 | 0.8534 | 0.7927 |
| **Platt scaling** ⭐ | **0.9417** | **0.8465** | **0.7536** |
| Isotonic regr. | 0.9417 | 0.8540 | 0.7651 |

**Stabilité CV 5-fold :**

| Métrique | Moyenne | Std |
|----------|---------|-----|
| F1-macro | 0.8698 | **0.0025** ← très stable |
| Recall macro | 0.8693 | 0.0024 |
| Accuracy | 0.8751 | 0.0024 |

**Résultats finaux V3 (pipeline_final.pkl — T=0.30) :**

| Métrique | Valeur |
|----------|--------|
| **Recall sévère** ⭐ | **0.9755** |
| Recall macro | 0.8236 |
| F1-macro | 0.7069 |
| Accuracy | 0.7853 |

### 📈 Progression globale Axe 2

```
Recall sévère :

V1 (baseline)          0.720  ████████████████████░░░░░░░░░░░░
V2 (SMOTE + GridSearch) 0.862  ███████████████████████████░░░░░
V3 (Platt calibration)  0.942  ██████████████████████████████░░
V3 final (T=0.30)       0.976  █████████████████████████████████

Gain total V1 → V3 : +0.256 (+35.5%)
```

### 🔑 Modèle de production

```
models/axe2/pipeline_final.pkl
  ├── preprocessor  : ColumnTransformer (StandardScaler + OneHotEncoder)
  ├── model         : CalibratedClassifierCV (Platt scaling)
  └── threshold     : T = 0.30 sur P(sévère)
```

---

## 🔴 Axe 3 — 🧠 Mortalité post-AVC

🎯 Objectif

Prédire la mortalité des patients :

DDEAD → décès à 14 jours
FDEAD → décès à 6 mois

👉 Objectif métier : détecter les patients à risque

🧩 Pipeline global
Preprocessing → SMOTE → Modeling → Threshold tuning → SHAP
🧹 1. Preprocessing
🔹 Version 1
Imputation simple (médiane / mode)
Encodage one-hot
Nettoyage basique

⚠️ Limites :

pas de feature engineering
pas de pipeline structuré
déséquilibre ignoré
🔹 Version 2
🚀 Améliorations
Feature Engineering :
RDEF_SCORE
transformation RCONSC
Pipeline sklearn
Sélection de variables
⚖️ 2. Gestion du déséquilibre
🔥 Problème

Classe minoritaire (décès) très faible

✅ Solution

👉 SMOTE (appliqué uniquement sur train)

🤖 3. Modélisation
Modèles testés
Logistic Regression
Random Forest
XGBoost (V2/V3)
📊 4. Comparaison des modèles
🔹 🔥 VERSION 1 (Baseline)
DDEAD
Modèle	Accuracy	Precision	Recall	F1-score	AUC
Logistic Regression	0.74	0.25	0.71	0.37	0.80
Random Forest	0.89	0.42	0.06	0.10	0.78

👉 ⚠️ RF ignore la classe 1

FDEAD
Modèle	Accuracy	Precision	Recall	F1-score	AUC
Logistic Regression	0.74	0.46	0.74	0.56	0.80
Random Forest	0.80	0.60	0.32	0.42	0.78
🔹 🚀 VERSION 2 (SMOTE + amélioration)
DDEAD
Modèle	Accuracy	Precision	Recall	F1-score	AUC
Logistic Regression	0.75	0.25	0.70	0.37	0.79
Random Forest	0.88	0.35	0.10	0.15	0.76
XGBoost	0.89	0.55	0.09	0.16	0.79
FDEAD
Modèle	Accuracy	Precision	Recall	F1-score	AUC
Logistic Regression	0.74	0.45	0.71	0.55	0.79
Random Forest	0.79	0.56	0.37	0.45	0.76
XGBoost	0.80	0.61	0.36	0.45	0.79
🔹 🔥 VERSION 3 (Threshold + tuning)
🎯 Seuils optimaux
Target	Seuil
DDEAD	0.2
FDEAD	0.25
DDEAD (final)
Modèle	Seuil	Accuracy	Precision	Recall	F1-score	AUC
Logistic Regression	0.5	0.75	0.25	0.70	0.37	0.79
Logistic Tuned	0.1	0.17	0.11	0.99	0.20	0.79
🔥 Final	0.2	0.43	0.15	0.92	0.25	0.79
FDEAD (final)
Modèle	Seuil	Accuracy	Precision	Recall	F1-score	AUC
Logistic Regression	0.5	0.74	0.45	0.71	0.55	0.79
Logistic Tuned	0.1	0.27	0.23	0.99	0.38	0.79
🔥 Final	0.25	0.48	0.29	0.93	0.45	0.79
🎯 5. Insight clé

👉 L’amélioration principale vient de :

gestion du déséquilibre (SMOTE)
optimisation du seuil (threshold tuning)
🔍 6. Interprétation (SHAP)

Variables les plus importantes :

AGE
RDEF_SCORE
RCONSC
RSBP
⚠️ 7. Data Leakage

Prévention :

split avant preprocessing
SMOTE uniquement sur train
suppression variables futures
🏁 Conclusion

👉 Le modèle final est :

✔ robuste
✔ interprétable
✔ adapté au contexte médical

👉 priorité :

🔥 maximiser le recall

---

## 🌐 Interface Next-js



## 🗺️ Vue d'ensemble

StrokeAI est un pipeline ML clinique en **3 axes indépendants** :

| Axe | Objectif | Modèle | Dataset | Cible |
|-----|----------|--------|---------|-------|
| **Axe 1** | Risque d'AVC | LightGBM | NHANES | Binaire (0/1) |
| **Axe 2** | Sévérité de l'AVC | CalibratedClassifierCV (LR) | IST | 3 classes |
| **Axe 3** | Mortalité post-AVC | CalibratedClassifierCV (LR) × 2 | IST | Binaire × 2 |

Chaque axe dispose de :
- Un **wizard de saisie** multi-étapes (3 à 4 étapes)
- Une **inférence backend** via FastAPI
- Une **page de résultats** avec jauge de risque, recommandations cliniques et export JSON

---

## 🎯 Axes de prédiction

### Axe 1 — Risque d'AVC (LightGBM)

> **Dataset :** NHANES — population générale américaine  
> **Threshold :** 0.25 (optimisé pour le rappel)

**4 étapes :** Profil démographique → Mode de vie → Santé & antécédents → Biologie & nutrition

**Features clés :**
- 37 features brutes NHANES (démographie, biologie, nutrition, comportement)
- Feature engineering : `pulse_pressure`, `fat_ratio`, `fat_sat_ratio`, etc.
- Données biologiques : LDL, glycémie à jeun, potassium, sodium
- Données nutritionnelles : énergie, protéines, glucides, lipides (7 types)

```
Inputs (37) → Feature Engineering → LightGBM → P(AVC) vs threshold 0.25 → Verdict
```

---

### Axe 2 — Sévérité de l'AVC (IST)

> **Dataset :** IST — 19 435 patients, 112 variables  
> **Threshold sévère :** P(sévère) ≥ 0.30 (classe prioritaire)

**3 étapes :** Démographie & admission → Déficits neurologiques → Variables cliniques

**Features clés :**
- 18 features IST à l'admission (T=0, sans data leakage)
- 8 déficits neurologiques RDEF1–RDEF8 encodés Y/N/C
- Feature engineering : `RDEF_score`, `RDEF_uncertain`, `deficit_ratio`
- Variables cliniques : STYPE (OCSP), RATRIAL, RCT, RVISINF, RHEP24, RASP3

**Cible construite :**
```
severity_score = RDEF_score + 2 × RCONSC_factor
léger: 0–3  |  modéré: 4–6  |  sévère: 7+
Distribution : 49.9% / 41.7% / 8.4%
```

---

### Axe 3 — Mortalité post-AVC (IST)

> **Dataset :** IST — mêmes 19 435 patients  
> **Deux modèles en parallèle :** DDEAD (14j, threshold 0.20) + FDEAD (6m, threshold 0.25)

**3 étapes :** Profil & admission → Bilan neurologique → Traitement (RXASP/RXHEP)

**Features clés :**
- AGE, SEX, RSBP, RDELAY (délai AVC → admission)
- RDEF_SCORE (0–8), RCONSC (F/D/U), RATRIAL, STYPE
- RXASP (aspirine Y/N), RXHEP (héparine H/L/M/N)

**Inférence parallèle :**
```javascript
const [fdead, ddead] = await Promise.all([
  POST /predict/axe3/fdead,
  POST /predict/axe3/ddead
])
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    FRONTEND                          │
│              Next.js 16 · React 19                   │
│                                                      │
│  / (Landing)  →  /axe1  →  /axe2  →  /axe3          │
│                                                      │
│  Sidebar partagée · MenuButton · SidebarContext      │
└──────────────────┬───────────────────────────────────┘
                   │  HTTP / JSON
                   ▼
┌──────────────────────────────────────────────────────┐
│                    BACKEND                           │
│               FastAPI · Python 3.10+                 │
│                                                      │
│  POST /predict/axe1          → LightGBM              │
│  POST /predict/axe2          → CalibratedClassifierCV│
│  POST /predict/axe3/fdead    → LogisticRegression    │
│  POST /predict/axe3/ddead    → LogisticRegression    │
│  GET  /health                → {"status": "ok"}      │
└──────────────────────────────────────────────────────┘
```

---

## 📁 Structure du projet

```
ml-pipeline-frontend/
│
├── app/
│   ├── layout.tsx              # Root layout — SidebarProvider + Sidebar
│   ├── page.tsx                # Landing page (3 cartes axes cliquables)
│   ├── axe1/page.tsx           # Wizard Axe 1 — 4 étapes
│   ├── axe2/page.tsx           # Wizard Axe 2 — 3 étapes
│   └── axe3/page.tsx           # Wizard Axe 3 — 3 étapes
│
├── components/
│   ├── Sidebar.tsx             # Sidebar navigation partagée
│   ├── MenuButton.tsx          # Bouton hamburger (toggle sidebar)
│   ├── axe1/                   # Composants Axe 1
│   │   ├── ProgressBar.tsx
│   │   ├── StepProfil.tsx
│   │   ├── StepVie.tsx
│   │   ├── StepSante.tsx
│   │   ├── StepBiologie.tsx
│   │   └── ResultCard.tsx
│   ├── axe2/                   # Composants Axe 2
│   │   ├── ProgressBarAxe2.tsx
│   │   ├── StepDemographie.tsx
│   │   ├── StepDeficits.tsx
│   │   ├── StepClinique.tsx
│   │   └── ResultAxe2.tsx
│   └── axe3/                   # Composants Axe 3
│       ├── ProgressBarAxe3.tsx
│       ├── StepProfilAxe3.tsx
│       ├── StepNeurologie.tsx
│       ├── StepTraitement.tsx
│       └── ResultAxe3.tsx
│
├── contexts/
│   └── SidebarContext.tsx      # État global sidebar (isOpen/toggle/close)
│
├── lib/
│   └── api.ts                  # Client FastAPI + types TypeScript
│
├── styles/
│   ├── home.css                # Landing page
│   ├── sidebar.css             # Sidebar + MenuButton
│   ├── wizard.css              # Layout partagé wizard
│   ├── progressbar.css         # Barre de progression
│   ├── result-card.css         # Résultats (partagé)
│   ├── step-profil.css         # Layout étapes (partagé)
│   ├── step-vie.css            # Toggles, sliders (partagé)
│   ├── axe1/                   # CSS spécifiques Axe 1
│   │   ├── step-sante.css
│   │   └── step-biologie.css
│   ├── axe2/                   # CSS spécifiques Axe 2
│   │   ├── step-demographie.css
│   │   ├── step-deficits.css
│   │   ├── step-clinique.css
│   │   └── result-axe2.css
│   └── axe3/                   # CSS spécifiques Axe 3
│       ├── step-profil-axe3.css
│       ├── step-neurologie.css
│       ├── step-traitement.css
│       └── result-axe3.css
```

---

## 🚀 Installation & Lancement

### Prérequis

- **Node.js** ≥ 18
- **Python** ≥ 3.10
- Backend FastAPI démarré sur `http://localhost:8000`

### Frontend

```bash
# Cloner le repo
git clone <url-repo>
cd ml-pipeline-frontend

# Installer les dépendances
npm install

# Lancer en développement
npm run dev
```

> Ouvrir [http://localhost:3000](http://localhost:3000)

### Backend (FastAPI)

```bash
# Depuis le dossier backend
cd backend/

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur
uvicorn main:app --reload --port 8000
```

> API disponible sur [http://localhost:8000](http://localhost:8000)  
> Documentation Swagger : [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 API Backend

### Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/health` | Vérification backend |
| `POST` | `/predict/axe1` | Prédiction risque AVC (LightGBM) |
| `POST` | `/predict/axe2` | Prédiction sévérité IST (3 classes) |
| `POST` | `/predict/axe3/fdead` | Prédiction mortalité 6 mois |
| `POST` | `/predict/axe3/ddead` | Prédiction mortalité 14 jours |

### Exemple de réponse `/predict/axe1`

```json
{
  "probability": 0.342,
  "prediction": 1,
  "threshold": 0.25,
  "verdict": "Risque élevé détecté",
  "engineered_features": {
    "pulse_pressure": 52,
    "fat_ratio": 0.38,
    "fat_sat_ratio": 0.31
  }
}
```

### Exemple de réponse `/predict/axe2`

```json
{
  "severity": "sévère",
  "severity_label": "Sévère",
  "prediction": 2,
  "probability_severe": 0.41,
  "probability_modere": 0.35,
  "probability_leger": 0.24,
  "threshold": 0.30,
  "deficit_summary": {
    "n_confirmed": 5,
    "n_uncertain": 2,
    "deficit_ratio": 0.625
  }
}
```

### Exemple de réponse `/predict/axe3/fdead`

```json
{
  "probability": 0.31,
  "prediction": 1,
  "threshold": 0.25,
  "verdict": "Risque de décès à 6 mois",
  "risk_level": "Élevé"
}
```

---

## 🛠️ Stack technique

### Frontend

| Technologie | Version | Usage |
|-------------|---------|-------|
| **Next.js** | 16.2.3 | Framework App Router |
| **React** | 19.2.4 | UI |
| **TypeScript** | 5 | Typage statique |
| **CSS Modules** | — | Styles par composant |
| **DM Sans** | — | Typographie (Google Fonts) |

### Backend

| Technologie | Version | Usage |
|-------------|---------|-------|
| **FastAPI** | ≥ 0.100 | API REST |
| **scikit-learn** | — | CalibratedClassifierCV, LogisticRegression |
| **LightGBM** | — | Modèle Axe 1 |
| **pandas / numpy** | — | Preprocessing |
| **joblib** | — | Sérialisation des modèles |
| **uvicorn** | — | Serveur ASGI |

---

## 📊 Datasets

### NHANES — National Health and Nutrition Examination Survey

- **Source :** CDC / US Population
- **Usage :** Axe 1 — Risque d'AVC
- **Features sélectionnées :** 37 (démographie, biologie, nutrition, comportement)
- **Target :** AVC déclaré (binaire 0/1)

### IST — International Stroke Trial

- **Référence :** Sandercock et al., 1997
- **Patients :** 19 435
- **Variables :** 112
- **Usage :** Axe 2 (sévérité) & Axe 3 (mortalité)
- **Split :** 80/20 stratifié → Train : 15 534 / Test : 3 884

```
Variables temporelles (éviter le data leakage) :
  T=0   (Admission)    → ✅ utilisables pour Axe 2 & 3
  T=14j (Sortie)       → ❌ leakage temporel
  T=6m  (Suivi)        → ❌ leakage temporel
```

---

<div align="center">

**StrokeAI** — Projet ML 2025–2026  
NHANES · IST · FastAPI · Next.js · LightGBM · Logistic Regression

</div>

## 📄 Licence

Dataset IST : Open Access — University of Edinburgh DataShare
Dataset NHANES : CC BY 4.0 — Mendeley Data

---

*Projet ML 2025-2026 — Université*





