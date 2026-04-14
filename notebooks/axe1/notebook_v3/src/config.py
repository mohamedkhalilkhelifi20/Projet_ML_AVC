import json
import os

# 1. Constantes statiques

TARGET        = 'stroke'
RANDOM_STATE  = 42
DATA_PATH     = '../dataset/data.csv'
CLEAN_PATH    = '../dataset/df_clean.csv'
FEATURES_PATH      = '../dataset/data_clean/df_features.csv'
FEATURES_JSON_PATH = './src/features.json'

# 2. Encodage NHANES — labels lisibles pour les graphiques
# Défini ici une seule fois, importé dans l'EDA pour les visualisations


LABEL_MAPS = {
    'gender'                   : {1: 'Male', 2: 'Female'},
    'age'                      : {1: '20-39', 2: '40-59', 3: '60+'},
    'Race'                     : {1: 'Mexican American', 2: 'Other Hispanic',
                                  3: 'Non-H White', 4: 'Non-H Black', 5: 'Other'},
    'Marital status'           : {1: 'Married', 2: 'Widowed', 3: 'Divorced',
                                  4: 'Separated', 5: 'Never Married', 6: 'Living w/ Partner'},
    'alcohol'                  : {0: 'No', 1: 'Yes'},
    'smoke'                    : {0: 'No', 1: 'Yes'},
    'sleep disorder'           : {1: 'Yes', 2: 'No'},
    'Health Insurance'         : {1: 'Yes', 2: 'No'},
    'General health condition' : {1: 'Excellent', 2: 'Very Good', 3: 'Good',
                                  4: 'Fair', 5: 'Poor'},
    'depression'               : {1: 'Never', 2: 'Several days', 3: 'More than half days'},
    'diabetes'                 : {0: 'No', 1: 'Yes'},
    'hypertension'             : {0: 'No', 1: 'Yes'},
    'high cholesterol'         : {0: 'No', 1: 'Yes'},
    'Coronary Heart Disease'   : {0: 'No', 1: 'Yes'},
    'Body Mass Index'          : {1: 'Underweight', 2: 'Normal',
                                  3: 'Overweight', 4: 'Obese'},
}

# Plages de valeurs valides selon codebook NHANES
# Utilisé dans check_data_quality() pour détecter les anomalies
EXPECTED_VALUES = {
    'stroke'                   : {0, 1},
    'gender'                   : {1, 2},
    'age'                      : {1, 2, 3},
    'Race'                     : {1, 2, 3, 4, 5},
    'Marital status'           : {1, 2, 3, 4, 5, 6},
    'alcohol'                  : {0, 1},
    'smoke'                    : {0, 1},
    'sleep disorder'           : {1, 2},
    'Health Insurance'         : {1, 2},
    'General health condition' : {1, 2, 3, 4, 5},
    'depression'               : {1, 2, 3},
    'diabetes'                 : {0, 1},
    'hypertension'             : {0, 1},
    'high cholesterol'         : {0, 1},
    'Coronary Heart Disease'   : {0, 1},
    'Body Mass Index'          : {1, 2, 3, 4},
}

# 3. Fonctions utilitaires — définies ici, appelées dans tous les notebooks
#    Evite de répéter les mêmes vérifications dans chaque notebook

def check_data_quality(df):
    """
    Vérifie la qualité du dataset : valeurs manquantes et doublons.
    Définie dans config.py — appelée dans EDA et Preprocessing.
    Reçoit df en argument → pas de dépendance au dataset global.

    Utilisation :
        from config import check_data_quality
        check_data_quality(df)
    """
    print('=' * 45)
    print('  Data Quality Report')
    print('=' * 45)

    # Valeurs manquantes
    missing = df.isnull().sum()
    missing_total = missing.sum()
    print(f'\nValeurs manquantes : {missing_total}')
    if missing_total > 0:
        print(missing[missing > 0].to_string())
    else:
        print('  Aucune valeur manquante.')

    # Doublons
    n_dup = df.duplicated().sum()
    print(f'\nDoublons : {n_dup}')
    if n_dup == 0:
        print('  Aucun doublon.')
    else:
        print(f'  {n_dup} doublons détectés.')

    # Shape
    print(f'\nShape : {df.shape[0]} lignes x {df.shape[1]} colonnes')
    print('=' * 45)

    return missing_total == 0 and n_dup == 0


def save_features(categorical_vars, numerical_vars):
    features = {
        'target'          : TARGET,
        'categorical_vars': categorical_vars,
        'numerical_vars'  : numerical_vars,
        'all_features'    : categorical_vars + numerical_vars,
    }
    with open(FEATURES_JSON_PATH, 'w') as f:  # ← FEATURES_JSON_PATH
        json.dump(features, f, indent=2)

    print(f'features.json sauvegardé :')
    print(f'  target           : {TARGET}')
    print(f'  categorical_vars : {len(categorical_vars)} variables')
    print(f'  numerical_vars   : {len(numerical_vars)} variables')
    print(f'  all_features     : {len(categorical_vars + numerical_vars)} variables')


def load_features():
    if not os.path.exists(FEATURES_JSON_PATH):  # ← FEATURES_JSON_PATH
        raise FileNotFoundError(
            f"'{FEATURES_JSON_PATH}' introuvable. "
            f"Exécute d'abord le notebook 1_EDA_Data.ipynb pour le générer."
        )

    with open(FEATURES_JSON_PATH) as f:  # ← FEATURES_JSON_PATH
        features = json.load(f)

    print(f'features.json chargé :')
    print(f'  target           : {features["target"]}')
    print(f'  categorical_vars : {len(features["categorical_vars"])} variables')
    print(f'  numerical_vars   : {len(features["numerical_vars"])} variables')

    return (
        features['target'],
        features['categorical_vars'],
        features['numerical_vars'],
        features['all_features'],
    )


