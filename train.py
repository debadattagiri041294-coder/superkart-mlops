"""
train.py
--------
Step 2 of the SuperKart Sales Forecast MLOps pipeline.

What this script does:
1. Loads train/test data from the Hugging Face dataset space.
2. Builds a preprocessing + model pipeline (XGBoost Regressor).
3. Tunes hyperparameters with RandomizedSearchCV.
4. Evaluates the tuned model (RMSE, MAE, R2) on the held-out test set.
5. Registers (uploads) the best model to the Hugging Face model hub.

Usage:
    python train.py

Environment variables required:
    HF_TOKEN         - Hugging Face access token with write permission
    HF_DATASET_REPO  - e.g. "your-username/superkart-sales-data"
    HF_MODEL_REPO    - e.g. "your-username/superkart-sales-model"
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from huggingface_hub import HfApi, hf_hub_download, login

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "your-username/superkart-sales-data")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "your-username/superkart-sales-model")
TARGET_COL = "Product_Store_Sales_Total"
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age"]
CATEGORICAL_FEATURES = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]


def load_data():
    """Load train/test CSVs from Hugging Face; falls back to local ./data."""
    if HF_TOKEN:
        login(token=HF_TOKEN)
        train_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="train.csv", repo_type="dataset")
        test_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="test.csv", repo_type="dataset")
    else:
        print("HF_TOKEN not set — loading local CSVs instead of Hugging Face Hub.")
        train_path = os.path.join("data", "train.csv")
        test_path = os.path.join("data", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_pipeline() -> Pipeline:
    """Preprocessing + XGBoost regressor pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def tune_model(pipeline: Pipeline, X_train, y_train) -> RandomizedSearchCV:
    """Hyperparameter tuning with RandomizedSearchCV."""
    param_distributions = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    return search


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute RMSE, MAE, R2 on the test set."""
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    print(f"Test metrics: {metrics}")
    return metrics


def register_model(model, metrics: dict):
    """Save the model + metrics locally, and upload to the HF model hub."""
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    if not HF_TOKEN:
        print("HF_TOKEN not set — skipping upload to Hugging Face model hub.")
        return

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=MODEL_PATH, path_in_repo=MODEL_PATH, repo_id=HF_MODEL_REPO, repo_type="model")
    api.upload_file(path_or_fileobj=METRICS_PATH, path_in_repo=METRICS_PATH, repo_id=HF_MODEL_REPO, repo_type="model")
    print(f"Model + metrics uploaded -> {HF_MODEL_REPO}")


def main():
    print("Loading train/test data...")
    train_df, test_df = load_data()

    X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

    print("Building pipeline...")
    pipeline = build_pipeline()

    print("Tuning hyperparameters (RandomizedSearchCV, 5-fold CV)...")
    search = tune_model(pipeline, X_train, y_train)
    best_model = search.best_estimator_

    print("Evaluating best model on test set...")
    metrics = evaluate_model(best_model, X_test, y_test)
    metrics["best_params"] = search.best_params_

    print("Registering model to Hugging Face Hub...")
    register_model(best_model, metrics)

    print("Training complete.")


if __name__ == "__main__":
    main()
