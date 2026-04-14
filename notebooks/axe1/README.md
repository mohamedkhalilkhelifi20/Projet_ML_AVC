# Stroke Risk Prediction — Machine Learning Project (Axe 1)

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
