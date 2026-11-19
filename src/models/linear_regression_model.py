"""
Linear Regression Cross-Validation Pipeline for Turbofan RUL Prediction.
NASA C-MAPSS FD001 Dataset.

Implements Step 6.3A:
- 5-Fold GroupKFold partitioning on unit_number (zero engine overlap).
- StandardScaler fitted exclusively on each training fold.
- Out-of-fold prediction generation, error tracking, and metrics calculation.
- Diagnostic visualization (Actual vs Predicted, Residual analysis).
- Model artifact serialization with joblib.
- Does NOT touch or evaluate on the NASA test dataset.
"""

import sys
from typing import Dict, Any, Tuple
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

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
)


def run_linear_regression_cv(
    n_splits: int = 5,
    reports_dir: str = "reports/individual_models/linear_regression",
    models_dir: str = "models/individual/linear_regression",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes 5-fold GroupKFold cross-validation for Linear Regression.

    Parameters
    ----------
    n_splits : int, default=5
        Number of cross-validation folds.
    reports_dir : str, default='reports/individual_models/linear_regression'
        Directory to save OOF predictions, metrics, and diagnostic figures.
    models_dir : str, default='models/individual/linear_regression'
        Directory to store trained model checkpoints.
    verbose : bool, default=True
        Whether to print fold progress.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'fold_metrics_df', 'summary_metrics',
        'oof_predictions_df', and 'models'.
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

    # Ensure canonical feature ordering
    canonical_features = [f for f in CANONICAL_15_FEATURES if f in selected_features]

    # 3. Separate features, target, and grouping units
    X, y, unit_ids = separate_features_target(proc_df, feature_cols=canonical_features)

    # 4. Create GroupKFold splits
    splits = create_engine_group_splits(X, y, unit_ids, n_splits=n_splits, verbose=verbose)

    fold_metrics = []
    oof_dfs = []
    trained_models = {}

    if verbose:
        print("\n" + "=" * 80)
        print("          TRAINING LINEAR REGRESSION (5-FOLD GROUPKFOLD CV)")
        print("=" * 80)

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_dict = get_fold_data(X, y, unit_ids, train_idx, val_idx)
        X_tr = fold_dict["X_train"]
        y_tr = fold_dict["y_train"]
        X_va = fold_dict["X_val"]
        y_va = fold_dict["y_val"]
        units_va = fold_dict["unit_ids_val"]

        # Step 6.2: StandardScaler fit strictly on X_tr
        X_tr_scaled, X_va_scaled, scaler = prepare_model_features(
            X_tr,
            X_va,
            model_type="linear",
            fold=fold_num,
            scaler_dir=mod_path / "scalers",
        )

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_tr_scaled, y_tr)

        # Predict RUL on validation engines
        val_preds = model.predict(X_va_scaled)

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

        # Save model checkpoint
        model_file = mod_path / f"linear_regression_fold_{fold_num}.joblib"
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
        "model_name": "Linear Regression",
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

    # Generate diagnostic plots
    plot_actual_vs_predicted(
        oof_predictions_df["actual_RUL"],
        oof_predictions_df["predicted_RUL"],
        model_name="Linear Regression",
        save_path=rep_path / "actual_vs_predicted.png",
    )

    plot_residuals(
        oof_predictions_df["actual_RUL"],
        oof_predictions_df["predicted_RUL"],
        model_name="Linear Regression",
        save_path=rep_path / "residual_plot.png",
    )

    # Extract & save model coefficients across folds
    coefs = [trained_models[f"fold_{k}"].coef_ for k in range(1, n_splits + 1)]
    coef_df = pd.DataFrame(coefs, columns=canonical_features)
    coef_summary = pd.DataFrame({
        "feature": canonical_features,
        "coef_mean": coef_df.mean(axis=0),
        "coef_std": coef_df.std(axis=0),
    }).sort_values("coef_mean", key=abs, ascending=False).reset_index(drop=True)
    coef_summary.to_csv(rep_path / "coefficients_summary.csv", index=False)

    if verbose:
        print(f"Outputs successfully saved to: {rep_path.resolve()}")

    return {
        "fold_metrics_df": fold_metrics_df,
        "summary_metrics": summary_metrics,
        "oof_predictions_df": oof_predictions_df,
        "coefficients_summary": coef_summary,
        "models": trained_models,
    }


if __name__ == "__main__":
    results = run_linear_regression_cv(verbose=True)
