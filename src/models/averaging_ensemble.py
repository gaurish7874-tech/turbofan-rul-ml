"""
Simple Averaging Ensemble for Turbofan RUL Prediction.
NASA C-MAPSS FD001 Dataset.

Implements Step B1:
- Combines Random Forest and XGBoost out-of-fold predictions.
- Strictly aligns engine IDs, observation indices, and fold numbers.
- Computes simple average: (y_pred_rf + y_pred_xgb) / 2.
- Calculates MAE, RMSE, R² across folds and compares with individual baselines.
- Generates Actual vs Predicted and Residual diagnostic plots.
- Stores outputs under reports/ensemble/simple_averaging/.
- Does NOT touch the NASA test dataset.
"""

import sys
import json
from typing import Dict, Any
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluate import (
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
)


def run_simple_averaging_ensemble(
    rf_oof_path: str = "reports/individual_models/random_forest/oof_predictions.csv",
    xgb_oof_path: str = "reports/individual_models/xgboost/oof_predictions.csv",
    reports_dir: str = "reports/ensemble/simple_averaging",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Computes simple average ensemble of Random Forest and XGBoost out-of-fold predictions.

    Parameters
    ----------
    rf_oof_path : str
        Path to Random Forest OOF predictions CSV.
    xgb_oof_path : str
        Path to XGBoost OOF predictions CSV.
    reports_dir : str
        Output directory for predictions, metrics, and plots.
    verbose : bool, default=True
        Whether to print fold metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'fold_metrics_df', 'summary_metrics',
        'comparison_df', and 'ensemble_oof_df'.
    """
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    rf_oof = pd.read_csv(rf_oof_path)
    xgb_oof = pd.read_csv(xgb_oof_path)

    # 1. Rigorous alignment checks
    assert len(rf_oof) == len(xgb_oof), "Length mismatch between RF and XGBoost OOF predictions!"
    assert (rf_oof["actual_RUL"] == xgb_oof["actual_RUL"]).all(), "Ground truth RUL mismatch between RF and XGB!"
    assert (rf_oof["engine_id"] == xgb_oof["engine_id"]).all(), "Engine ID mismatch between RF and XGB!"
    assert (rf_oof["fold"] == xgb_oof["fold"]).all(), "Fold alignment mismatch between RF and XGB!"

    y_true = rf_oof["actual_RUL"].values
    rf_pred = rf_oof["predicted_RUL"].values
    xgb_pred = xgb_oof["predicted_RUL"].values
    folds = rf_oof["fold"].values
    engine_ids = rf_oof["engine_id"].values

    # 2. Simple Averaging Formula
    ensemble_preds = (rf_pred + xgb_pred) / 2.0

    # 3. Create OOF prediction dataframe
    ensemble_oof_df = pd.DataFrame({
        "engine_id": engine_ids,
        "actual_RUL": y_true,
        "rf_pred": rf_pred,
        "xgb_pred": xgb_pred,
        "predicted_RUL": ensemble_preds,
        "absolute_error": np.abs(y_true - ensemble_preds),
        "fold": folds,
    })

    # 4. Per-fold metric calculation
    fold_metrics = []
    n_splits = len(np.unique(folds))

    if verbose:
        print("=" * 80)
        print("          SIMPLE AVERAGING ENSEMBLE (RANDOM FOREST + XGBOOST)")
        print("=" * 80)

    for f in range(1, n_splits + 1):
        mask = (folds == f)
        y_f_true = y_true[mask]
        y_f_pred = ensemble_preds[mask]

        metrics = compute_regression_metrics(y_f_true, y_f_pred)
        metrics["Fold"] = f"Fold {f}"
        metrics["Val_Engines"] = len(np.unique(engine_ids[mask]))
        metrics["Val_Rows"] = int(np.sum(mask))
        fold_metrics.append(metrics)

        if verbose:
            print(
                f"Fold {f:2d} | MAE: {metrics['MAE']:6.2f} | "
                f"RMSE: {metrics['RMSE']:6.2f} | R²: {metrics['R2']:6.4f} | "
                f"Val Engines: {metrics['Val_Engines']:2d}"
            )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    mean_mae = float(fold_metrics_df["MAE"].mean())
    std_mae = float(fold_metrics_df["MAE"].std())
    mean_rmse = float(fold_metrics_df["RMSE"].mean())
    std_rmse = float(fold_metrics_df["RMSE"].std())
    mean_r2 = float(fold_metrics_df["R2"].mean())
    std_r2 = float(fold_metrics_df["R2"].std())

    summary_metrics = {
        "model_name": "Simple Averaging Ensemble (RF + XGB)",
        "weights": {"Random Forest": 0.5, "XGBoost": 0.5},
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

    # 5. Comparative analysis against individual models
    rf_summary = json.loads(Path("reports/individual_models/random_forest/summary_metrics.json").read_text())
    xgb_summary = json.loads(Path("reports/individual_models/xgboost/summary_metrics.json").read_text())

    comparison_data = [
        {
            "Model / Pipeline": "Random Forest Regressor",
            "Mean MAE": round(rf_summary["mean_MAE"], 2),
            "Std MAE": round(rf_summary["std_MAE"], 2),
            "Mean RMSE": round(rf_summary["mean_RMSE"], 2),
            "Std RMSE": round(rf_summary["std_RMSE"], 2),
            "Mean R²": round(rf_summary["mean_R2"], 4),
            "Std R²": round(rf_summary["std_R2"], 4),
        },
        {
            "Model / Pipeline": "XGBoost Regressor",
            "Mean MAE": round(xgb_summary["mean_MAE"], 2),
            "Std MAE": round(xgb_summary["std_MAE"], 2),
            "Mean RMSE": round(xgb_summary["mean_RMSE"], 2),
            "Std RMSE": round(xgb_summary["std_RMSE"], 2),
            "Mean R²": round(xgb_summary["mean_R2"], 4),
            "Std R²": round(xgb_summary["std_R2"], 4),
        },
        {
            "Model / Pipeline": "Simple Averaging Ensemble (RF + XGB)",
            "Mean MAE": round(mean_mae, 2),
            "Std MAE": round(std_mae, 2),
            "Mean RMSE": round(mean_rmse, 2),
            "Std RMSE": round(std_rmse, 2),
            "Mean R²": round(mean_r2, 4),
            "Std R²": round(std_r2, 4),
        },
    ]
    comparison_df = pd.DataFrame(comparison_data)

    if verbose:
        print("\n=== ENSEMBLE VS. INDIVIDUAL MODELS BENCHMARK ===")
        print(comparison_df.to_string(index=False))

    # 6. Save outputs
    ensemble_oof_df.to_csv(rep_path / "oof_predictions.csv", index=False)
    fold_metrics_df.to_csv(rep_path / "fold_metrics.csv", index=False)
    pd.Series(summary_metrics).to_json(rep_path / "summary_metrics.json", indent=2)
    comparison_df.to_csv(rep_path / "ensemble_comparison.csv", index=False)

    # 7. Generate diagnostic plots
    plot_actual_vs_predicted(
        y_true,
        ensemble_preds,
        model_name="Simple Averaging Ensemble (RF + XGB)",
        save_path=rep_path / "actual_vs_predicted.png",
    )

    plot_residuals(
        y_true,
        ensemble_preds,
        model_name="Simple Averaging Ensemble (RF + XGB)",
        save_path=rep_path / "residual_plot.png",
    )

    if verbose:
        print(f"\nOutputs successfully saved to: {rep_path.resolve()}")

    return {
        "fold_metrics_df": fold_metrics_df,
        "summary_metrics": summary_metrics,
        "comparison_df": comparison_df,
        "ensemble_oof_df": ensemble_oof_df,
    }


if __name__ == "__main__":
    results = run_simple_averaging_ensemble(verbose=True)
