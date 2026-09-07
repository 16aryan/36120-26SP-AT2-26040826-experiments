"""Training, evaluation, and serialization module for CCI and WHC models."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
import xgboost as xgb
import lightgbm as lgb

from src.config import (
    TRAIN_END_DATE,
    VAL_END_DATE,
    TEST_END_DATE,
    CCI_MODELS_DIR,
    WHC_MODELS_DIR,
    RANDOM_SEED,
    WHC_CLASSES,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Temporal Splitting
# =====================================================================

def temporal_train_val_test_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: str = TRAIN_END_DATE,
    val_end: str = VAL_END_DATE,
    test_end: str = TEST_END_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset temporally into train, validation, and test sets.

    Ensures zero temporal leakage across splits:
    - Train: observations <= train_end (e.g., 2015-2023)
    - Val:   train_end < observations <= val_end (e.g., 2024)
    - Test:  val_end < observations <= test_end (e.g., 2025)
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    train = df[df[date_col] <= pd.to_datetime(train_end)].copy()
    val = df[(df[date_col] > pd.to_datetime(train_end)) & (df[date_col] <= pd.to_datetime(val_end))].copy()
    test = df[(df[date_col] > pd.to_datetime(val_end)) & (df[date_col] <= pd.to_datetime(test_end))].copy()

    logger.info(
        "Temporal Split -> Train: %d rows (%s to %s) | Val: %d rows (%s to %s) | Test: %d rows (%s to %s)",
        len(train), train[date_col].min().strftime("%Y-%m-%d"), train[date_col].max().strftime("%Y-%m-%d"),
        len(val), val[date_col].min().strftime("%Y-%m-%d"), val[date_col].max().strftime("%Y-%m-%d"),
        len(test), test[date_col].min().strftime("%Y-%m-%d"), test[date_col].max().strftime("%Y-%m-%d"),
    )
    return train, val, test


# =====================================================================
# Target 1: CCI Regression Training & Evaluation
# =====================================================================

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate RMSE, MAE, and R2."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def train_and_eval_cci(
    model,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a multi-output regressor on (h1, h2, h3) and evaluate on test set."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {}
    horizons = ["h1", "h2", "h3"]
    for i, h in enumerate(horizons):
        metrics[h] = evaluate_regression(y_test.iloc[:, i].values, preds[:, i])

    metrics["mean_rmse"] = round(float(np.mean([metrics[h]["rmse"] for h in horizons])), 4)
    metrics["mean_mae"] = round(float(np.mean([metrics[h]["mae"] for h in horizons])), 4)
    metrics["mean_r2"] = round(float(np.mean([metrics[h]["r2"] for h in horizons])), 4)

    return model, metrics


# =====================================================================
# Target 2: WHC Classification Training & Evaluation
# =====================================================================

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate multi-class accuracy, balanced accuracy, macro F1, and per-class metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def train_and_eval_whc(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a classifier for +7 day WHC and evaluate."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate_classification(y_test.values, preds)
    return model, metrics


# =====================================================================
# Model Artifact Serialization & Metadata
# =====================================================================

def save_model_artifact(
    model: Any,
    metadata: Dict[str, Any],
    model_dir: Path,
    model_filename: str = "model.joblib",
    metadata_filename: str = "metadata.json",
) -> None:
    """Save trained model artefact and accompanying metadata JSON."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_filename
    metadata_path = model_dir / metadata_filename

    joblib.dump(model, model_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model to %s and metadata to %s", model_path, metadata_path)
