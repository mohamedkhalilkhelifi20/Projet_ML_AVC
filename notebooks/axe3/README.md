# 🧠 Axe 3 – Prédiction de la mortalité post-AVC


---


## 📌 Table des matières


1. Introduction  
2. Objectif  
3. Données utilisées  
4. Prétraitement des données  
5. Feature Engineering  
6. Gestion du déséquilibre (SMOTE)  
7. Modélisation  
8. Modèles testés  
9. Optimisation des modèles  
10. Seuils de décision (Threshold tuning)  
11. Résultats  
12. Modèle final retenu  
13. Interprétabilité (SHAP)  
14. Conclusion  


---


## 📖 Introduction


Dans cet axe, nous développons des modèles de machine learning pour prédire la **mortalité après un AVC**, à partir de données cliniques issues du dataset IST.


Deux horizons de prédiction sont étudiés :


- **DDEAD** : mortalité à court terme (14 jours)  
- **FDEAD** : mortalité à long terme (6 mois)  


👉 Il s’agit d’un problème de **classification binaire fortement déséquilibré**.


---


## 🎯 Objectif


Construire des modèles capables de :


- détecter efficacement les patients à risque  
- **maximiser le Recall (priorité médicale)**  
- maintenir un compromis avec la précision  
- fournir des résultats interprétables  


---


## 📊 Données utilisées


Le dataset IST contient :


- 19 435 patients  
- 112 variables  


Types de variables :


- données démographiques (AGE, SEX)  
- données cliniques (pression, état neurologique, traitements)  


Variables cibles :


- `DDEAD`  
- `FDEAD`  


---


## ⚙️ Prétraitement des données


Les principales étapes :


- Nettoyage des valeurs manquantes  
- Transformation des cibles (Y/N → 1/0)  
- Imputation :
  - médiane (variables numériques)
  - mode (variables catégorielles)
- Encodage des variables catégorielles (**One-Hot Encoding**)  
- Séparation train/test (80/20) avec stratification  


👉 Un **pipeline sklearn** a été utilisé pour garantir la reproductibilité et éviter le data leakage.


---


## 🧠 Feature Engineering


Deux transformations importantes :


### 🔹 RDEF_SCORE
- Agrégation des variables `RDEF1` à `RDEF8`
- Représente le niveau de déficit neurologique


### 🔹 RCONSC_NUM
- Transformation du niveau de conscience en variable numérique


👉 Objectif : introduire une information clinique synthétique et exploitable par le modèle.


---


## ⚖️ Gestion du déséquilibre (SMOTE)


Le dataset présente un fort déséquilibre :


- majorité : patients survivants  
- minorité : patients décédés  


### 🔧 Solution


👉 Utilisation de **SMOTE** (Synthetic Minority Over-sampling Technique)


- génération de données synthétiques  
- appliqué uniquement sur le train  


👉 Impact :


- amélioration du Recall  
- meilleure détection des cas critiques  


---


## 🤖 Modélisation


Plusieurs modèles ont été testés :


- Logistic Regression  
- Random Forest  
- XGBoost  


---


## 🔁 Modèles testés


| Modèle              | Observation |
|---------------------|------------|
| Random Forest       | Bonne performance globale mais Recall insuffisant pour la classe minoritaire |
| XGBoost             | Modèle performant mais n’apporte pas d’amélioration significative sur le Recall dans ce contexte |
| Logistic Regression | Meilleur compromis entre performance et interprétabilité après optimisation |


👉 Le choix est basé sur les résultats expérimentaux et l’objectif médical.


---


## 🔧 Optimisation des modèles


Les techniques suivantes ont été utilisées :


- **GridSearchCV** pour le tuning des hyperparamètres  
- **Validation croisée (Stratified K-Fold)**  
- optimisation basée sur le Recall  


👉 Permet d’obtenir un modèle robuste et stable.


---


## 🎚️ Seuils de décision (Threshold tuning)


Le seuil par défaut (0.5) n’est pas adapté.


### 🔥 Seuils optimaux retenus :


- **DDEAD → 0.20**  
- **FDEAD → 0.25**  


👉 Impact :


- augmentation significative du Recall  
- meilleure détection des patients à risque  


---


## 📈 Résultats


Métriques utilisées :


- Accuracy  
- Precision  
- Recall ⭐  
- F1-score  
- AUC-ROC  


### 📊 Observations


- SMOTE améliore la performance sur la classe minoritaire  
- le threshold tuning améliore fortement le Recall  
- les résultats sont cohérents avec l’objectif médical  


---


## 🏆 Modèle final retenu


### 🔹 DDEAD


- Logistic Regression + GridSearchCV + Threshold  
- Seuil : **0.20**


---


### 🔹 FDEAD


- Logistic Regression + GridSearchCV + Threshold  
- Seuil : **0.25**


---


### ✅ Justification


- priorité au Recall  
- modèle interprétable  
- pipeline structuré  
- performances robustes  


---


## 🔍 Interprétabilité (SHAP)


Utilisation de **SHAP** pour :


- analyser l’impact des variables  
- expliquer les prédictions  
- renforcer la confiance dans le modèle  


### 🔥 Variables importantes :


- AGE  
- RDEF_SCORE  
- RCONSC  
- RSBP  


---


## ✅ Conclusion


- Le modèle permet une **détection efficace des patients à risque**  
- Le pipeline est complet et reproductible  
- Le **threshold tuning est l’amélioration clé**  


👉 Le projet respecte les contraintes du domaine médical :


- priorité au Recall  
- interprétabilité  
- robustesse  


---


## 🚀 Perspectives




- modèles avancés (LightGBM)  
- Tester des modèles hybrides
- Enrichir les données


---

