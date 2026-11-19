"""
XGBoost Regression Cross-Validation Pipeline for Turbofan RUL Prediction.
NASA C-MAPSS FD001 Dataset.

Implements Step 6.3C:
- 5-Fold GroupKFold partitioning on unit_number (identical to Linear Regression & RF).
- Original unscaled numerical features (gradient boosting uses split-finding on order).
- Fixed random_state=42 for reproducibility.
- Hyperparameters:
    n_estimators=300
    max_depth=4
    learning_rate=0.05
    subsample=0.8
    colsample_bytree=0.8
    objective='reg:squarederror'
- Out-of-fold prediction generation, error tracking, and metrics calculation.
- Feature importance extraction across folds (gain-based / weight-based mean ± std).
- Diagnostic visualization (Actual vs Predicted, Residual analysis, Feature Importance).
- Model artifact serialization with joblib.
- Does NOT touch or evaluate on the NASA test dataset.
"""

import sys
from typing import Dict, Any, List
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

from src.data_loader import load_train_data
from src.preprocessing import (
    create_rul_target,
    select_informative_features,
    separate_features_target,
    prepare_model_features,
    CANONICAL_15_FEATURES,
)
from src.splitting import create_engine_group_splits, get_fold_data
from src.evaluate import (
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
)


def run_xgboost_cv(
    n_splits: int = 5,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    objective: str = "reg:squarederror",
    random_state: int = 42,
    n_jobs: int = -1,
    reports_dir: str = "reports/individual_models/xgboost",
    models_dir: str = "models/individual/xgboost",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes 5-fold GroupKFold cross-validation for XGBoost Regression.

    Parameters
    ----------
    n_splits : int, default=5
        Number of cross-validation folds.
    n_estimators : int, default=300
        Number of boosting rounds.
    max_depth : int, default=4
        Maximum tree depth.
    learning_rate : float, default=0.05
        Boosting shrinkage parameter.
    subsample : float, default=0.8
        Row subsample ratio.
    colsample_bytree : float, default=0.8
        Column subsample ratio per tree.
    objective : str, default='reg:squarederror'
        Regression loss function.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel threads.
    reports_dir : str, default='reports/individual_models/xgboost'
        Output directory for reports and figures.
    models_dir : str, default='models/individual/xgboost'
        Directory to store trained model checkpoints.
    verbose : bool, default=True
        Whether to print fold progress.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'fold_metrics_df', 'summary_metrics',
        'oof_predictions_df', 'feature_importance_df', and 'models'.
    """
    rep_path = Path(reports_dir)
    mod_path = Path(models_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    mod_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data & engineer RUL target
    raw_train = load_train_data()
    train_with_rul = create_rul_target(raw_train)

    # 2. Select informative features
    proc_df, selected_features, _ = select_informative_features(
        train_with_rul,
        variance_threshold=1e-5,
        include_operational_settings=False,
        include_time_cycles=True,
    )

    canonical_features = [f for f in CANONICAL_15_FEATURES if f in selected_features]

    # 3. Separate features, target, and grouping units
    X, y, unit_ids = separate_features_target(proc_df, feature_cols=canonical_features)

    # 4. Create identical GroupKFold splits
    splits = create_engine_group_splits(X, y, unit_ids, n_splits=n_splits, verbose=verbose)

    fold_metrics = []
    oof_dfs = []
    trained_models = {}
    fold_importances = []

    if verbose:
        print("\n" + "=" * 80)
        print("             TRAINING XGBOOST REGRESSOR (5-FOLD GROUPKFOLD CV)")
        print("=" * 80)

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_dict = get_fold_data(X, y, unit_ids, train_idx, val_idx)
        X_tr = fold_dict["X_train"]
        y_tr = fold_dict["y_train"]
        X_va = fold_dict["X_val"]
        y_va = fold_dict["y_val"]
        units_va = fold_dict["unit_ids_val"]

        # Step 6.2: Unscaled numerical features for Tree models
        X_tr_xgb, X_va_xgb, _ = prepare_model_features(
            X_tr, X_va, model_type="xgboost", fold=fold_num
        )

        # Instantiate XGBoost model
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(X_tr_xgb, y_tr)

        # Predict RUL on validation engines
        val_preds = model.predict(X_va_xgb)

        # Calculate metrics for current fold
        metrics = compute_regression_metrics(y_va, val_preds)
        metrics["Fold"] = f"Fold {fold_num}"
        metrics["Train_Engines"] = len(set(fold_dict["unit_ids_train"]))
        metrics["Val_Engines"] = len(set(units_va))
        metrics["Val_Rows"] = len(y_va)
        fold_metrics.append(metrics)

        # Record out-of-fold prediction dataframe
        fold_oof_df = pd.DataFrame({
            "engine_id": units_va.values,
            "actual_RUL": y_va.values,
            "predicted_RUL": val_preds,
            "absolute_error": np.abs(y_va.values - val_preds),
            "fold": fold_num,
        }, index=y_va.index)
        oof_dfs.append(fold_oof_df)

        # Collect feature importance (gain-based)
        fold_importances.append(model.feature_importances_)

        # Save model checkpoint
        model_file = mod_path / f"xgboost_fold_{fold_num}.joblib"
        joblib.dump(model, model_file)
        trained_models[f"fold_{fold_num}"] = model

        if verbose:
            print(
                f"Fold {fold_num:2d} | MAE: {metrics['MAE']:6.2f} | "
                f"RMSE: {metrics['RMSE']:6.2f} | R²: {metrics['R2']:6.4f} | "
                f"Val Engines: {metrics['Val_Engines']:2d}"
            )

    # Aggregate out-of-fold predictions
    oof_predictions_df = pd.concat(oof_dfs, axis=0).sort_index()

    # Compile fold metrics dataframe
    fold_metrics_df = pd.DataFrame(fold_metrics)
    mean_mae = float(fold_metrics_df["MAE"].mean())
    std_mae = float(fold_metrics_df["MAE"].std())
    mean_rmse = float(fold_metrics_df["RMSE"].mean())
    std_rmse = float(fold_metrics_df["RMSE"].std())
    mean_r2 = float(fold_metrics_df["R2"].mean())
    std_r2 = float(fold_metrics_df["R2"].std())

    summary_metrics = {
        "model_name": "XGBoost",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "objective": objective,
        "random_state": random_state,
        "mean_MAE": mean_mae,
        "std_MAE": std_mae,
        "mean_RMSE": mean_rmse,
        "std_RMSE": std_rmse,
        "mean_R2": mean_r2,
        "std_R2": std_r2,
        "n_splits": n_splits,
    }

    if verbose:
        print("-" * 80)
        print(f"Mean ± Std Across {n_splits} Folds:")
        print(f"  MAE  : {mean_mae:6.2f} ± {std_mae:4.2f} cycles")
        print(f"  RMSE : {mean_rmse:6.2f} ± {std_rmse:4.2f} cycles")
        print(f"  R²   : {mean_r2:6.4f} ± {std_r2:4.4f}")
        print("=" * 80)

    # Save outputs
    oof_save_path = rep_path / "oof_predictions.csv"
    oof_predictions_df.to_csv(oof_save_path, index=False)

    metrics_save_path = rep_path / "fold_metrics.csv"
    fold_metrics_df.to_csv(metrics_save_path, index=False)

    summary_save_path = rep_path / "summary_metrics.json"
    pd.Series(summary_metrics).to_json(summary_save_path, indent=2)

    # Feature Importance aggregation across folds
    imp_matrix = np.array(fold_importances)
    feature_importance_df = pd.DataFrame({
        "feature": canonical_features,
        "importance_mean": imp_matrix.mean(axis=0),
        "importance_std": imp_matrix.std(axis=0),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    feature_importance_df.to_csv(rep_path / "feature_importance.csv", index=False)

    # Generate diagnostic plots
    plot_actual_vs_predicted(
        oof_predictions_df["actual_RUL"],
        oof_predictions_df["predicted_RUL"],
        model_name="XGBoost Regressor",
        save_path=rep_path / "actual_vs_predicted.png",
    )

    plot_residuals(
        oof_predictions_df["actual_RUL"],
        oof_predictions_df["predicted_RUL"],
        model_name="XGBoost Regressor",
        save_path=rep_path / "residual_plot.png",
    )

    plot_feature_importance(
        feature_importance_df,
        model_name="XGBoost Regressor",
        save_path=rep_path / "feature_importance.png",
        top_n=15,
    )

    if verbose:
        print(f"Outputs successfully saved to: {rep_path.resolve()}")

    return {
        "fold_metrics_df": fold_metrics_df,
        "summary_metrics": summary_metrics,
        "oof_predictions_df": oof_predictions_df,
        "feature_importance_df": feature_importance_df,
        "models": trained_models,
    }


if __name__ == "__main__":
    results = run_xgboost_cv(verbose=True)
