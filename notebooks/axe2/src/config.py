# =============================================================================
# CONFIG CENTRALISEE DU PROJET
# =============================================================================
# Utilisation dans les notebooks :
#
# import sys
# sys.path.insert(0, './src')
#
# import importlib, config
# importlib.reload(config)
#
# from config import (
#     TARGET,
#     RANDOM_STATE,
#     DATA_PATH,
#     CLEAN_DATA_PATH,
#     FEATURES_JSON_PATH,
#     check_data_quality,
#     save_features,
#     load_features
# )
# =============================================================================

import os
import json
from pathlib import Path

# =============================================================================
# 1. PARAMETRES GENERAUX
# =============================================================================

TARGET = "severity_class"
RANDOM_STATE = 42

# =============================================================================
# 2. RACINE DU PROJET
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# 3. DOSSIERS DU PROJET
# =============================================================================

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "Data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# =============================================================================
# 4. FICHIERS PRINCIPAUX
# =============================================================================

DATA_PATH = DATA_DIR / "IST_corrected.csv"
CLEAN_DATA_PATH = ARTIFACTS_DIR / "df_clean.csv"
FEATURES_JSON_PATH = SRC_DIR / "features.json"

# =============================================================================
# 3. CREATION AUTOMATIQUE DES DOSSIERS
# =============================================================================

def ensure_directories():
    """
    Crée automatiquement les dossiers nécessaires si inexistants.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 4. CONTROLE QUALITE DES DONNEES
# =============================================================================

def check_data_quality(df):
    """
    Vérifie :
    - valeurs manquantes
    - doublons
    - dimensions

    Appelée dans tous les notebooks.
    """
    print("=" * 50)
    print("📊 DATA QUALITY REPORT")
    print("=" * 50)

    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()

    print(f"\n🔎 Valeurs manquantes totales : {total_missing}")

    if total_missing > 0:
        print("\nColonnes avec valeurs manquantes :")
        print(missing[missing > 0].sort_values(ascending=False))
    else:
        print("✅ Aucune valeur manquante")

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\n🔁 Doublons : {duplicates}")

    if duplicates == 0:
        print("✅ Aucun doublon")
    else:
        print(f"⚠️ {duplicates} doublons détectés")

    # Shape
    print(f"\n📐 Shape : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print("=" * 50)

    return total_missing == 0 and duplicates == 0

# =============================================================================
# 5. SAUVEGARDE DES FEATURES (EDA → suite pipeline)
# =============================================================================

def save_features(categorical_vars, numerical_vars):
    """
    Sauvegarde les variables détectées en EDA.

    Sortie : src/features.json
    """
    features = {
        "target": TARGET,
        "categorical_vars": categorical_vars,
        "numerical_vars": numerical_vars,
        "all_features": categorical_vars + numerical_vars,
    }

    with open(FEATURES_JSON_PATH, "w") as f:
        json.dump(features, f, indent=4)

    print("\n✅ features.json sauvegardé")
    print(f"📍 path : {FEATURES_JSON_PATH}")
    print(f"🔹 categorical : {len(categorical_vars)}")
    print(f"🔹 numerical   : {len(numerical_vars)}")
    print(f"🔹 total       : {len(features['all_features'])}")

# =============================================================================
# 6. CHARGEMENT DES FEATURES (EDA → preprocessing → modeling)
# =============================================================================

def load_features():
    """
    Charge les variables depuis features.json.

    Retour :
        TARGET, categorical_vars, numerical_vars, all_features
    """
    if not os.path.exists(FEATURES_JSON_PATH):
        raise FileNotFoundError(
            f"{FEATURES_JSON_PATH} introuvable.\n"
            f"➡️ Exécute d'abord le notebook EDA."
        )

    with open(FEATURES_JSON_PATH, "r") as f:
        features = json.load(f)

    print("\n📦 features.json chargé")
    print(f"🔹 target      : {features['target']}")
    print(f"🔹 categorical : {len(features['categorical_vars'])}")
    print(f"🔹 numerical   : {len(features['numerical_vars'])}")

    return (
        features["target"],
        features["categorical_vars"],
        features["numerical_vars"],
        features["all_features"],
    )