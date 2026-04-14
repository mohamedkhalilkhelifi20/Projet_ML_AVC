import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def build_severity_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rdef_cols = [f"RDEF{i}" for i in range(1, 9)]
    rdef_map = {"Y": 1, "N": 0, "C": np.nan}
    for col in rdef_cols:
        if col in df.columns:
            df[col + "_bin"] = df[col].map(rdef_map)
    rdef_bin_cols = [col + "_bin" for col in rdef_cols if col + "_bin" in df.columns]
    df["RDEF_score"] = df[rdef_bin_cols].sum(axis=1, min_count=1)
    df["RCONSC_factor"] = df["RCONSC"].map({"F": 0, "D": 1, "U": 2})
    df["severity_score"] = df["RDEF_score"] + 2 * df["RCONSC_factor"]
    df["severity_class"] = pd.cut(
        df["severity_score"],
        bins=[-np.inf, 3, 6, np.inf],
        labels=["leger", "modere", "severe"]
    )
    return df


def get_baseline_features(df: pd.DataFrame) -> list[str]:
    baseline_features = [
        "RDELAY", "RCONSC", "SEX", "AGE", "RSLEEP", "RATRIAL", "RCT", "RVISINF",
        "RHEP24", "RASP3", "RSBP", "RDEF1", "RDEF2", "RDEF3", "RDEF4",
        "RDEF5", "RDEF6", "RDEF7", "RDEF8", "STYPE"
    ]
    return [col for col in baseline_features if col in df.columns]


def get_feature_versions(baseline_features: list[str]) -> tuple[list[str], list[str]]:
    features_v1 = baseline_features.copy()
    strict_drop = ["RCONSC", "RDEF1", "RDEF2", "RDEF3", "RDEF4", "RDEF5", "RDEF6", "RDEF7", "RDEF8"]
    features_v2 = [col for col in baseline_features if col not in strict_drop]
    return features_v1, features_v2


def encode_rconsc_if_present(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "RCONSC" in X.columns:
        X["RCONSC"] = X["RCONSC"].map({"F": 0, "D": 1, "U": 2})
    return X


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def prepare_axe2_data(df: pd.DataFrame, version: str = "v1", test_size: float = 0.2, random_state: int = 42):
    df = build_severity_target(df)
    df_model = df.dropna(subset=["severity_class"]).copy()

    baseline_features = get_baseline_features(df_model)
    features_v1, features_v2 = get_feature_versions(baseline_features)

    selected_features = features_v2 if version.lower() == "v2" else features_v1

    X = df_model[selected_features].copy()
    y = df_model["severity_class"].copy()
    X = encode_rconsc_if_present(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    return X_train, X_test, y_train, y_test, preprocessor
