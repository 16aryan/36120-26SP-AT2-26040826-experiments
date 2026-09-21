"""Model experimentation and artifact generation script.

Trains and systematically evaluates multiple candidate algorithms for:
1. Target 1: Climate Comfort Index (CCI) - Multi-horizon regression (+1, +2, +3 days)
2. Target 2: Weather Hazard Category (WHC) - Multiclass classification (+7 days)
"""

import json
import logging
from pathlib import Path
import pickle
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb

from src.config import (
    PROCESSED_DATA_DIR,
    CCI_MODELS_DIR,
    WHC_MODELS_DIR,
    RANDOM_SEED,
    WHC_CLASSES,
)
from src.train import temporal_train_val_test_split, save_model_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_datasets():
    csv_path = PROCESSED_DATA_DIR / "sydney_features_targets.csv"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    # Feature columns (exclude raw dates, textual names, and targets)
    exclude_cols = [
        "date",
        "whc_name",
        "target_cci_h1",
        "target_cci_h2",
        "target_cci_h3",
        "target_whc_h7",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    train_df, val_df, test_df = temporal_train_val_test_split(df)
    return df, train_df, val_df, test_df, feature_cols


def run_cci_experiments(train_df, val_df, test_df, feature_cols):
    logger.info("=== Starting CCI Regression Experiments ===")
    target_cols = ["target_cci_h1", "target_cci_h2", "target_cci_h3"]

    # Drop NaNs in target columns (last 3 rows of dataset)
    tr = train_df.dropna(subset=target_cols).copy()
    vl = val_df.dropna(subset=target_cols).copy()
    ts = test_df.dropna(subset=target_cols).copy()

    # Fill any minor feature NaNs in lags
    X_train = tr[feature_cols].fillna(tr[feature_cols].median())
    y_train = tr[target_cols]

    X_val = vl[feature_cols].fillna(tr[feature_cols].median())
    y_val = vl[target_cols]

    X_test = ts[feature_cols].fillna(tr[feature_cols].median())
    y_test = ts[target_cols]

    models = {
        "Baseline (Dummy Mean)": MultiOutputRegressor(DummyRegressor(strategy="mean")),
        "Ridge Regression": MultiOutputRegressor(Ridge(alpha=10.0, random_state=RANDOM_SEED)),
        "Random Forest Regressor": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1)),
        "LightGBM Regressor": MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=120, max_depth=6, learning_rate=0.05, random_state=RANDOM_SEED, verbose=-1)),
        "XGBoost Regressor": MultiOutputRegressor(xgb.XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED, n_jobs=-1)),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds_test = model.predict(X_test)

        h1_rmse = np.sqrt(mean_squared_error(y_test["target_cci_h1"], preds_test[:, 0]))
        h2_rmse = np.sqrt(mean_squared_error(y_test["target_cci_h2"], preds_test[:, 1]))
        h3_rmse = np.sqrt(mean_squared_error(y_test["target_cci_h3"], preds_test[:, 2]))
        mean_rmse = np.mean([h1_rmse, h2_rmse, h3_rmse])

        mean_mae = np.mean([
            mean_absolute_error(y_test["target_cci_h1"], preds_test[:, 0]),
            mean_absolute_error(y_test["target_cci_h2"], preds_test[:, 1]),
            mean_absolute_error(y_test["target_cci_h3"], preds_test[:, 2]),
        ])

        mean_r2 = np.mean([
            r2_score(y_test["target_cci_h1"], preds_test[:, 0]),
            r2_score(y_test["target_cci_h2"], preds_test[:, 1]),
            r2_score(y_test["target_cci_h3"], preds_test[:, 2]),
        ])

        results.append({
            "model": name,
            "mean_rmse": round(float(mean_rmse), 4),
            "mean_mae": round(float(mean_mae), 4),
            "mean_r2": round(float(mean_r2), 4),
            "h1_rmse": round(float(h1_rmse), 4),
            "h2_rmse": round(float(h2_rmse), 4),
            "h3_rmse": round(float(h3_rmse), 4),
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("mean_rmse")
    logger.info("CCI Model Results:\n%s", results_df.to_string(index=False))

    # Dynamically select the best-performing model (lowest mean RMSE)
    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    logger.info("Selected best CCI model: %s (Mean RMSE: %.4f)", best_name, results_df.iloc[0]["mean_rmse"])

    # Compute top feature importances across horizons (handles trees and linear models)
    if hasattr(best_model.estimators_[0], "feature_importances_"):
        importances = np.mean([est.feature_importances_ for est in best_model.estimators_], axis=0)
    elif hasattr(best_model.estimators_[0], "coef_"):
        importances = np.mean([np.abs(est.coef_) for est in best_model.estimators_], axis=0)
    else:
        importances = np.ones(len(feature_cols))

    top_indices = np.argsort(importances)[::-1][:15]
    feature_importance_dict = {feature_cols[i]: round(float(importances[i]), 4) for i in top_indices}

    best_res = results_df[results_df["model"] == best_name].iloc[0].to_dict()
    metadata = {
        "target": "Climate Comfort Index",
        "algorithm": best_name,
        "forecast_horizon": "Next 3 Days (+1, +2, +3 days)",
        "prediction_type": "regression",
        "features": list(feature_importance_dict.keys()),
        "feature_importance": feature_importance_dict,
        "performance_metrics": {
            "mean_rmse": best_res["mean_rmse"],
            "mean_mae": best_res["mean_mae"],
            "mean_r2": best_res["mean_r2"],
            "h1_rmse": best_res["h1_rmse"],
            "h2_rmse": best_res["h2_rmse"],
            "h3_rmse": best_res["h3_rmse"],
        },
        "all_model_comparison": results,
    }

    # Save model.joblib and metadata.json
    save_model_artifact(
        model=best_model,
        metadata=metadata,
        model_dir=CCI_MODELS_DIR,
        model_filename="model.joblib",
        metadata_filename="metadata.json",
    )

    # Save cci_model.pkl dictionary artifact for FastAPI deployment compatibility
    cci_pkl_path = CCI_MODELS_DIR / "cci_model.pkl"
    with open(cci_pkl_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": None,
            "features_list": feature_cols,
            "algorithm": best_name,
            "target_cols": target_cols,
        }, f)
    logger.info("Saved CCI pickle artifact to %s", cci_pkl_path)

    return results_df, metadata


def run_whc_experiments(train_df, val_df, test_df, feature_cols):
    logger.info("=== Starting WHC Classification Experiments ===")
    target_col = "target_whc_h7"

    tr = train_df.dropna(subset=[target_col]).copy()
    vl = val_df.dropna(subset=[target_col]).copy()
    ts = test_df.dropna(subset=[target_col]).copy()

    X_train = tr[feature_cols].fillna(tr[feature_cols].median())
    y_train = tr[target_col].astype(int)

    X_val = vl[feature_cols].fillna(tr[feature_cols].median())
    y_val = vl[target_col].astype(int)

    X_test = ts[feature_cols].fillna(tr[feature_cols].median())
    y_test = ts[target_col].astype(int)

    models = {
        "Baseline (Dummy Stratified)": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=120, max_depth=8, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "LightGBM Classifier": lgb.LGBMClassifier(n_estimators=120, max_depth=6, class_weight="balanced", learning_rate=0.05, random_state=RANDOM_SEED, verbose=-1),
        "XGBoost Classifier": xgb.XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED, eval_metric="mlogloss", n_jobs=-1),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds_test = model.predict(X_test)

        acc = accuracy_score(y_test, preds_test)
        bal_acc = balanced_accuracy_score(y_test, preds_test)
        macro_f1 = f1_score(y_test, preds_test, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, preds_test, average="weighted", zero_division=0)

        results.append({
            "model": name,
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    logger.info("WHC Model Results:\n%s", results_df.to_string(index=False))

    # Best model (dynamically highest Macro F1)
    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    logger.info("Selected best WHC model: %s (Macro F1: %.4f)", best_name, results_df.iloc[0]["macro_f1"])

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.mean(np.abs(best_model.coef_), axis=0)
    else:
        importances = np.ones(len(feature_cols))

    top_indices = np.argsort(importances)[::-1][:15]
    feature_importance_dict = {feature_cols[i]: round(float(importances[i]), 4) for i in top_indices}

    best_res = results_df.iloc[0].to_dict()
    metadata = {
        "target": "Weather Hazard Category",
        "algorithm": best_name,
        "forecast_horizon": "Exactly +7 Days",
        "prediction_type": "multiclass classification",
        "classes": WHC_CLASSES,
        "features": list(feature_importance_dict.keys()),
        "feature_importance": feature_importance_dict,
        "performance_metrics": {
            "accuracy": best_res["accuracy"],
            "balanced_accuracy": best_res["balanced_accuracy"],
            "macro_f1": best_res["macro_f1"],
            "weighted_f1": best_res["weighted_f1"],
        },
        "all_model_comparison": results,
    }

    # Save model.joblib and metadata.json
    save_model_artifact(
        model=best_model,
        metadata=metadata,
        model_dir=WHC_MODELS_DIR,
        model_filename="model.joblib",
        metadata_filename="metadata.json",
    )

    # Save whc_model.pkl dictionary artifact for FastAPI deployment compatibility
    whc_pkl_path = WHC_MODELS_DIR / "whc_model.pkl"
    with open(whc_pkl_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": None,
            "features_list": feature_cols,
            "algorithm": best_name,
            "class_labels": WHC_CLASSES,
        }, f)
    logger.info("Saved WHC pickle artifact to %s", whc_pkl_path)

    return results_df, metadata


def main():
    _, train_df, val_df, test_df, feature_cols = prepare_datasets()
    run_cci_experiments(train_df, val_df, test_df, feature_cols)
    run_whc_experiments(train_df, val_df, test_df, feature_cols)

    # Sync fresh artifacts to API repo models directory if present
    api_models_dir = Path(__file__).resolve().parent.parent.parent / "36120-26SP-AT2-26040826-api" / "models"
    if api_models_dir.exists():
        logger.info("Synchronizing fresh model artifacts to FastAPI deployment repository: %s", api_models_dir)
        for sub, pkl_name in [("comfort_climate", "cci_model.pkl"), ("weather_hazard", "whc_model.pkl")]:
            src_dir = Path(__file__).resolve().parent.parent / "models" / sub
            dst_dir = api_models_dir / sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            for fname in [pkl_name, "metadata.json"]:
                s = src_dir / fname
                d = dst_dir / fname
                if s.exists():
                    shutil.copy2(s, d)
                    logger.info("  Synced %s -> %s", s.name, d)


if __name__ == "__main__":
    main()
