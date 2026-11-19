"""
Weighted Averaging Ensemble for Turbofan RUL Prediction.
NASA C-MAPSS FD001 Dataset.

Implements Step B2:
- Systematic evaluation of weight combinations:
    [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
- Strict adherence to zero data leakage (weights selected strictly on training-side CV).
- Multi-metric evaluation (MAE, RMSE, R²).
- Benchmark comparison against individual models and simple averaging.
- Saves optimal weights artifact and diagnostic comparison plots.
- Stores outputs under reports/ensemble/weighted_averaging/.
- Does NOT touch the NASA test dataset.
"""

import sys
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluate import (
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


WEIGHT_CANDIDATES: List[Tuple[float, float]] = [
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
]


def run_weighted_averaging_ensemble(
    rf_oof_path: str = "reports/individual_models/random_forest/oof_predictions.csv",
    xgb_oof_path: str = "reports/individual_models/xgboost/oof_predictions.csv",
    reports_dir: str = "reports/ensemble/weighted_averaging",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates candidate weight combinations for Random Forest and XGBoost predictions.

    Parameters
    ----------
    rf_oof_path : str
        Path to Random Forest OOF predictions CSV.
    xgb_oof_path : str
        Path to XGBoost OOF predictions CSV.
    reports_dir : str
        Output directory.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'weights_grid_df', 'best_weights', 'best_oof_df',
        'comparison_df', and summary metrics.
    """
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    rf_oof = pd.read_csv(rf_oof_path)
    xgb_oof = pd.read_csv(xgb_oof_path)

    y_true = rf_oof["actual_RUL"].values
    rf_pred = rf_oof["predicted_RUL"].values
    xgb_pred = xgb_oof["predicted_RUL"].values
    folds = rf_oof["fold"].values
    engine_ids = rf_oof["engine_id"].values
    n_splits = len(np.unique(folds))

    if verbose:
        print("=" * 80)
        print("           WEIGHTED AVERAGING ENSEMBLE OPTIMIZATION GRID")
        print("=" * 80)

    grid_results = []

    # Evaluate each candidate weight combination across the 5 GroupKFold splits
    for w_rf, w_xgb in WEIGHT_CANDIDATES:
        assert np.isclose(w_rf + w_xgb, 1.0), f"Weights ({w_rf}, {w_xgb}) must sum to 1.0!"

        fold_maes, fold_rmses, fold_r2s = [], [], []

        for f in range(1, n_splits + 1):
            mask = (folds == f)
            y_f_true = y_true[mask]
            y_f_pred = w_rf * rf_pred[mask] + w_xgb * xgb_pred[mask]

            m = compute_regression_metrics(y_f_true, y_f_pred)
            fold_maes.append(m["MAE"])
            fold_rmses.append(m["RMSE"])
            fold_r2s.append(m["R2"])

        mean_mae = float(np.mean(fold_maes))
        std_mae = float(np.std(fold_maes))
        mean_rmse = float(np.mean(fold_rmses))
        std_rmse = float(np.std(fold_rmses))
        mean_r2 = float(np.mean(fold_r2s))
        std_r2 = float(np.std(fold_r2s))

        grid_results.append({
            "w_rf": w_rf,
            "w_xgb": w_xgb,
            "weight_label": f"RF {w_rf:.1f} / XGB {w_xgb:.1f}",
            "mean_MAE": mean_mae,
            "std_MAE": std_mae,
            "mean_RMSE": mean_rmse,
            "std_RMSE": std_rmse,
            "mean_R2": mean_r2,
            "std_R2": std_r2,
        })

        if verbose:
            print(
                f"Weight: RF {w_rf:.1f} / XGB {w_xgb:.1f} | "
                f"MAE: {mean_mae:6.2f} ± {std_mae:4.2f} | "
                f"RMSE: {mean_rmse:6.2f} ± {std_rmse:4.2f} | "
                f"R²: {mean_r2:6.4f} ± {std_r2:4.4f}"
            )

    weights_grid_df = pd.DataFrame(grid_results).sort_values("mean_MAE").reset_index(drop=True)

    # Optimal weights selection based strictly on cross-validation MAE
    best_row = weights_grid_df.iloc[0]
    best_w_rf = float(best_row["w_rf"])
    best_w_xgb = float(best_row["w_xgb"])

    best_weights_summary = {
        "best_w_rf": best_w_rf,
        "best_w_xgb": best_w_xgb,
        "best_weight_label": best_row["weight_label"],
        "mean_MAE": float(best_row["mean_MAE"]),
        "std_MAE": float(best_row["std_MAE"]),
        "mean_RMSE": float(best_row["mean_RMSE"]),
        "std_RMSE": float(best_row["std_RMSE"]),
        "mean_R2": float(best_row["mean_R2"]),
        "std_R2": float(best_row["std_R2"]),
        "selection_criterion": "Lowest cross-validation MAE across 5-fold GroupKFold",
    }

    if verbose:
        print("-" * 80)
        print(f"Optimal Weights: {best_row['weight_label']}")
        print(f"  MAE  : {best_row['mean_MAE']:.2f} ± {best_row['std_MAE']:.2f} cycles")
        print(f"  RMSE : {best_row['mean_RMSE']:.2f} ± {best_row['std_RMSE']:.2f} cycles")
        print(f"  R²   : {best_row['mean_R2']:.4f} ± {best_row['std_R2']:.4f}")
        print("=" * 80)

    # Generate optimal weighted predictions
    best_preds = best_w_rf * rf_pred + best_w_xgb * xgb_pred
    best_oof_df = pd.DataFrame({
        "engine_id": engine_ids,
        "actual_RUL": y_true,
        "rf_pred": rf_pred,
        "xgb_pred": xgb_pred,
        "predicted_RUL": best_preds,
        "absolute_error": np.abs(y_true - best_preds),
        "fold": folds,
    })

    # Benchmark against individual baselines & simple average
    rf_summary = json.loads(Path("reports/individual_models/random_forest/summary_metrics.json").read_text())
    xgb_summary = json.loads(Path("reports/individual_models/xgboost/summary_metrics.json").read_text())
    avg_summary = json.loads(Path("reports/ensemble/simple_averaging/summary_metrics.json").read_text())

    comparison_data = [
        {
            "Pipeline / Model": "Random Forest Regressor",
            "Mean MAE": round(rf_summary["mean_MAE"], 2),
            "Std MAE": round(rf_summary["std_MAE"], 2),
            "Mean RMSE": round(rf_summary["mean_RMSE"], 2),
            "Std RMSE": round(rf_summary["std_RMSE"], 2),
            "Mean R²": round(rf_summary["mean_R2"], 4),
            "Std R²": round(rf_summary["std_R2"], 4),
        },
        {
            "Pipeline / Model": "XGBoost Regressor",
            "Mean MAE": round(xgb_summary["mean_MAE"], 2),
            "Std MAE": round(xgb_summary["std_MAE"], 2),
            "Mean RMSE": round(xgb_summary["mean_RMSE"], 2),
            "Std RMSE": round(xgb_summary["std_RMSE"], 2),
            "Mean R²": round(xgb_summary["mean_R2"], 4),
            "Std R²": round(xgb_summary["std_R2"], 4),
        },
        {
            "Pipeline / Model": "Simple Averaging (50/50)",
            "Mean MAE": round(avg_summary["mean_MAE"], 2),
            "Std MAE": round(avg_summary["std_MAE"], 2),
            "Mean RMSE": round(avg_summary["mean_RMSE"], 2),
            "Std RMSE": round(avg_summary["std_RMSE"], 2),
            "Mean R²": round(avg_summary["mean_R2"], 4),
            "Std R²": round(avg_summary["std_R2"], 4),
        },
        {
            "Pipeline / Model": f"Optimal Weighted ({best_row['weight_label']})",
            "Mean MAE": round(best_row["mean_MAE"], 2),
            "Std MAE": round(best_row["std_MAE"], 2),
            "Mean RMSE": round(best_row["mean_RMSE"], 2),
            "Std RMSE": round(best_row["std_RMSE"], 2),
            "Mean R²": round(best_row["mean_R2"], 4),
            "Std R²": round(best_row["std_R2"], 4),
        },
    ]
    comparison_df = pd.DataFrame(comparison_data)

    if verbose:
        print("\n=== COMPREHENSIVE ENSEMBLE BENCHMARK ===")
        print(comparison_df.to_string(index=False))

    # Save outputs
    weights_grid_df.to_csv(rep_path / "weights_grid_search.csv", index=False)
    best_oof_df.to_csv(rep_path / "oof_predictions.csv", index=False)
    comparison_df.to_csv(rep_path / "ensemble_comparison.csv", index=False)
    pd.Series(best_weights_summary).to_json(rep_path / "best_weights.json", indent=2)

    # Visualizations
    # 1. Weights Grid Curve
    fig, ax1 = plt.subplots(figsize=(9, 5))
    rf_weights = weights_grid_df.sort_values("w_rf")["w_rf"]
    mae_curve = weights_grid_df.sort_values("w_rf")["mean_MAE"]
    rmse_curve = weights_grid_df.sort_values("w_rf")["mean_RMSE"]

    color = "#2166ac"
    ax1.set_xlabel("Random Forest Weight (XGBoost Weight = 1 - w_rf)", fontsize=11)
    ax1.set_ylabel("Mean Absolute Error (Cycles)", color=color, fontsize=11)
    line1 = ax1.plot(rf_weights, mae_curve, color=color, marker="o", linewidth=2.0, label="CV MAE")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#d6604d"
    ax2.set_ylabel("Root Mean Squared Error (Cycles)", color=color, fontsize=11)
    line2 = ax2.plot(rf_weights, rmse_curve, color=color, marker="s", linestyle="--", linewidth=2.0, label="CV RMSE")
    ax2.tick_params(axis="y", labelcolor=color)

    # Highlight optimal weight
    ax1.axvline(best_w_rf, color="green", linestyle=":", linewidth=2, label=f"Optimal w_rf={best_w_rf:.1f}")

    lines = line1 + line2 + [ax1.get_lines()[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", frameon=True)
    plt.title("Weight Optimization Sensitivity Curve (RF vs. XGBoost)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(rep_path / "weight_optimization_curve.png", dpi=300)
    plt.close(fig)

    # 2. Actual vs. Predicted for best weighted model
    plot_actual_vs_predicted(
        y_true,
        best_preds,
        model_name=f"Optimal Weighted Ensemble ({best_row['weight_label']})",
        save_path=rep_path / "actual_vs_predicted.png",
    )

    # 3. Residual diagnostics
    plot_residuals(
        y_true,
        best_preds,
        model_name=f"Optimal Weighted Ensemble ({best_row['weight_label']})",
        save_path=rep_path / "residual_plot.png",
    )

    if verbose:
        print(f"\nOutputs successfully saved to: {rep_path.resolve()}")

    return {
        "weights_grid_df": weights_grid_df,
        "best_weights": best_weights_summary,
        "best_oof_df": best_oof_df,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    results = run_weighted_averaging_ensemble(verbose=True)
