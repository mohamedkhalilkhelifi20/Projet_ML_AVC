# 🧠 Axe 3 – Prédiction de la mortalité post-AVC

## 📌 Table des matières

1. [Introduction](#introduction)
2. [Objectif](#objectif)
3. [Données utilisées](#données-utilisées)
4. [Prétraitement des données](#prétraitement-des-données)
5. [Modélisation](#modélisation)
6. [Modèles testés (versions précédentes)](#modèles-testés-versions-précédentes)
7. [Optimisation des modèles](#optimisation-des-modèles)
8. [Seuils de décision (Threshold tuning)](#seuils-de-décision-threshold-tuning)
9. [Résultats](#résultats)
10. [Modèle final retenu](#modèle-final-retenu)
11. [Interprétabilité (SHAP)](#interprétabilité-shap)
12. [Conclusion](#conclusion)

---

## 📖 Introduction

Dans cet axe, nous développons des modèles de machine learning pour prédire la **mortalité après un AVC**, en nous basant sur des variables cliniques et biologiques.

Deux cibles ont été étudiées :

* **DDEAD** : mortalité à court terme
* **FDEAD** : mortalité à long terme

---

## 🎯 Objectif

Construire des modèles capables de :

* détecter au mieux les patients à risque
* **maximiser le Recall** (priorité médicale)
* fournir des résultats interprétables

---

## 📊 Données utilisées

Les données incluent :

* variables cliniques
* paramètres biologiques
* informations démographiques

Variables cibles :

* `DDEAD`
* `FDEAD`

---

## ⚙️ Prétraitement des données

* Nettoyage des valeurs manquantes
* Encodage des variables catégorielles
* Normalisation / standardisation
* Séparation **train / test**

---

## 🤖 Modélisation (Version finale)

Les modèles retenus dans cette version finale :

| Modèle                              | Description                      |
| ----------------------------------- | -------------------------------- |
| Baseline Logistic Regression        | Modèle de référence              |
| Logistic + Threshold tuning         | Ajustement du seuil              |
| Logistic + GridSearchCV             | Optimisation des hyperparamètres |
| Logistic + GridSearchCV + Threshold | ✅ Modèle final                   |

---

## 🔁 Modèles testés (versions précédentes)

Dans les versions antérieures de ce projet, plusieurs modèles ont été explorés :

| Modèle              | Statut | Observation                                  |
| ------------------- | ------ | -------------------------------------------- |
| Random Forest       | Testé  | Bonne performance mais moins interprétable   |
| XGBoost             | Testé  | Performant mais plus complexe                |
| Logistic Regression | Retenu | Bon compromis performance / interprétabilité |

👉 **Choix final : Logistic Regression**

* plus simple
* plus interprétable (important en médical)
* performances satisfaisantes après tuning

---

## 🔧 Optimisation des modèles

* Utilisation de **GridSearchCV**
* Validation croisée
* Optimisation des hyperparamètres

---

## 🎚️ Seuils de décision (Threshold tuning)

Optimisation basée sur le **Recall** :

* **DDEAD → seuil retenu = 0.20**
* **FDEAD → seuil retenu = 0.25**

👉 Impact :

* amélioration du Recall
* meilleure détection des patients à risque

---

## 📈 Résultats

Métriques utilisées :

* Accuracy
* Precision
* Recall ⭐ (prioritaire)
* F1-score
* AUC-ROC

### 📊 Observations

* Le tuning du seuil améliore significativement le Recall
* Le modèle optimisé offre le meilleur compromis
* Les performances sont cohérentes avec l’objectif clinique

---

## 🏆 Modèle final retenu

### 🔹 DDEAD

* Modèle : Logistic Regression + GridSearchCV + Threshold
* Seuil : **0.20**

### 🔹 FDEAD

* Modèle : Logistic Regression + GridSearchCV + Threshold
* Seuil : **0.25**

👉 Justification :

* priorité au Recall
* modèle interprétable
* performances robustes

---

## 🔍 Interprétabilité (SHAP)

Utilisation de **SHAP** pour :

* analyser l’impact des variables
* expliquer les prédictions
* renforcer la confiance clinique

---

## ✅ Conclusion

* Le modèle permet une **détection efficace des patients à risque**
* Le **threshold tuning est déterminant**
* Le choix du modèle privilégie l’**interprétabilité médicale**

---

## 🚀 Perspectives

* Tester des modèles hybrides
* Enrichir les données

---
