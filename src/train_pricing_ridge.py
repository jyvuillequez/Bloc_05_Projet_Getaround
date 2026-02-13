import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
import os

# Config
load_dotenv()

API_KEY_S3 = os.environ["AWS_ACCESS_KEY_ID"]
API_SECRET_KEY_S3 = os.environ["AWS_SECRET_ACCESS_KEY"]

DATA_PATH = "data/raw/get_around_pricing_project.csv"
TARGET = "rental_price_per_day"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://jyvuillequez-mlflow-server.hf.space")
EXPERIMENT_NAME = "getaround_pricing_ridge"

RANDOM_STATE = 42
TEST_SIZE = 0.2

ALPHA = 1.0                
USE_LOG_TARGET = False
CAP_OUTLIERS = False

EXPORT_DIR = "data/outputs/pricing"
MODEL_FILENAME = "pricing_ridge.joblib"
SCHEMA_FILENAME = "schema.json"


# Cleaning
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # index parasite
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # incohérences simples
    if "mileage" in df.columns:
        df = df[df["mileage"].isna() | (df["mileage"] >= 0)]
    if "engine_power" in df.columns:
        df = df[df["engine_power"].isna() | (df["engine_power"] > 0)]

    return df


def cap_outliers_p99(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in num_cols:
        p99 = df[c].quantile(0.99)
        df[c] = df[c].clip(upper=p99)
    return df


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


# Main
def main():
    # MLflow init
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load
    df = pd.read_csv(DATA_PATH)
    df = basic_clean(df)

    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' introuvable. Colonnes: {list(df.columns)}")

    # Split X/y
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET])

   
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if CAP_OUTLIERS and num_cols:
        X = cap_outliers_p99(X, num_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocess = build_preprocess(X_train)
    model = Ridge(alpha=ALPHA, random_state=RANDOM_STATE)

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

    # target
    if USE_LOG_TARGET:
        y_train_fit = np.log1p(y_train)
        y_test_eval = y_test
    else:
        y_train_fit = y_train
        y_test_eval = y_test

    with mlflow.start_run(run_name="ridge_baseline"):
        # Log params (traçabilité)
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", ALPHA)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("use_log_target", USE_LOG_TARGET)
        mlflow.log_param("cap_outliers_p99", CAP_OUTLIERS)
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_features", X.shape[1])

        # Fit
        pipe.fit(X_train, y_train_fit)

        # Predict
        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)

        if USE_LOG_TARGET:
            pred_train = np.expm1(pred_train)
            pred_test = np.expm1(pred_test)

        # Metriques
        metrics = {
            "rmse_train": rmse(y_train, pred_train),
            "rmse_test": rmse(y_test_eval, pred_test),
            "mae_test": float(mean_absolute_error(y_test_eval, pred_test)),
            "r2_test": float(r2_score(y_test_eval, pred_test)),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Export artifacts
        os.makedirs(EXPORT_DIR, exist_ok=True)

        model_path = os.path.join(EXPORT_DIR, MODEL_FILENAME)
        joblib.dump(pipe, model_path)

        schema = {
            "target": TARGET,
            "feature_columns": X.columns.tolist(),
        }
        schema_path = os.path.join(EXPORT_DIR, SCHEMA_FILENAME)
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        # Log artifacts MLflow
        mlflow.log_artifact(model_path, artifact_path="export")
        mlflow.log_artifact(schema_path, artifact_path="export")

        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print("MLflow tracking URI:", TRACKING_URI)
        print("Experiment:", EXPERIMENT_NAME)
        print("Metrics:", metrics)
        print("Saved:", model_path)
        print("Schema:", schema_path)


if __name__ == "__main__":
    main()
