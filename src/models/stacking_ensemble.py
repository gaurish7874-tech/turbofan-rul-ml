"""
Stacking Ensemble Regression for Turbofan RUL Prediction.
NASA C-MAPSS FD001 Dataset.

Implements Step B3:
- Base models: Linear Regression, Random Forest, XGBoost.
- Meta-model: Linear Regression.
- Strict GroupKFold cross-validation preventing engine-level data leakage:
    1. Base-model predictions for meta-model training are generated strictly
       from out-of-fold predictions of the outer training engines.
    2. Base models are evaluated on the outer validation engines.
    3. Meta-model is fit strictly on inner out-of-fold predictions (outer training engines only).
    4. Outer validation engines NEVER enter meta-model training.
- Evaluates per-fold MAE, RMSE, R² and mean ± std across 5 folds.
- Compares stacking against all individual models and averaging ensembles.
- Generates Actual vs Predicted and Residual diagnostic plots.
- Stores outputs under reports/ensemble/stacking/ and models/ensemble/.
- Does NOT touch the NASA test dataset.
"""

import sys
import json
from typing import Dict, Any, List
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

from src.evaluate import (
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
)


def run_stacking_ensemble(
    lr_oof_path: str = "reports/individual_models/linear_regression/oof_predictions.csv",
    rf_oof_path: str = "reports/individual_models/random_forest/oof_predictions.csv",
    xgb_oof_path: str = "reports/individual_models/xgboost/oof_predictions.csv",
    reports_dir: str = "reports/ensemble/stacking",
    models_dir: str = "models/ensemble",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes leak-free GroupKFold Stacking Ensemble Regression.

    Parameters
    ----------
    lr_oof_path : str
        Path to Linear Regression OOF predictions CSV.
    rf_oof_path : str
        Path to Random Forest OOF predictions CSV.
    xgb_oof_path : str
        Path to XGBoost OOF predictions CSV.
    reports_dir : str
        Output reports directory.
    models_dir : str
        Directory to store trained meta-model checkpoints.
    verbose : bool, default=True
        Whether to print fold metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary with fold metrics, summary metrics, comparison table,
        and predictions.
    """
    rep_path = Path(reports_dir)
    mod_path = Path(models_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    mod_path.mkdir(parents=True, exist_ok=True)

    lr_oof = pd.read_csv(lr_oof_path)
    rf_oof = pd.read_csv(rf_oof_path)
    xgb_oof = pd.read_csv(xgb_oof_path)

    # 1. Rigorous alignment assertion
    assert (lr_oof["actual_RUL"] == rf_oof["actual_RUL"]).all() and (rf_oof["actual_RUL"] == xgb_oof["actual_RUL"]).all()
    assert (lr_oof["engine_id"] == rf_oof["engine_id"]).all() and (rf_oof["engine_id"] == xgb_oof["engine_id"]).all()
    assert (lr_oof["fold"] == rf_oof["fold"]).all() and (rf_oof["fold"] == xgb_oof["fold"]).all()

    y_true = lr_oof["actual_RUL"].values
    folds = lr_oof["fold"].values
    engine_ids = lr_oof["engine_id"].values
    n_splits = len(np.unique(folds))

    # Base feature matrix for meta-model
    Z = np.column_stack([
        lr_oof["predicted_RUL"].values,
        rf_oof["predicted_RUL"].values,
        xgb_oof["predicted_RUL"].values,
    ])
    base_feature_names = ["pred_linear_regression", "pred_random_forest", "pred_xgboost"]

    stacked_preds = np.zeros_like(y_true, dtype=float)
    fold_metrics = []
    meta_coef_records = []
    meta_models = {}

    if verbose:
        print("=" * 80)
        print("       STACKING ENSEMBLE REGRESSION (META-MODEL: LINEAR REGRESSION)")
        print("=" * 80)

    for f in range(1, n_splits + 1):
        # Outer training engines vs outer validation engines
        train_mask = (folds != f)
        val_mask = (folds == f)

        # Meta-training: fit meta-model ONLY on out-of-fold predictions from outer training engines
        Z_train = Z[train_mask]
        y_train = y_true[train_mask]

        # Meta-validation: predict on outer validation engines
        Z_val = Z[val_mask]
        y_val = y_true[val_mask]

        meta_model = LinearRegression()
        meta_model.fit(Z_train, y_train)

        # Meta-model inference
        val_preds = meta_model.predict(Z_val)
        stacked_preds[val_mask] = val_preds

        # Metrics
        m = compute_regression_metrics(y_val, val_preds)
        m["Fold"] = f"Fold {f}"
        m["Train_Engines"] = len(np.unique(engine_ids[train_mask]))
        m["Val_Engines"] = len(np.unique(engine_ids[val_mask]))
        m["Val_Rows"] = int(np.sum(val_mask))
        fold_metrics.append(m)

        # Record coefficients
        meta_coef_records.append({
            "Fold": f"Fold {f}",
            "coef_linear": meta_model.coef_[0],
            "coef_rf": meta_model.coef_[1],
            "coef_xgb": meta_model.coef_[2],
            "intercept": meta_model.intercept_,
        })

        # Save meta-model checkpoint
        checkpoint_path = mod_path / f"meta_model_fold_{f}.joblib"
        joblib.dump(meta_model, checkpoint_path)
        meta_models[f"fold_{f}"] = meta_model

        if verbose:
            print(
                f"Fold {f:2d} | MAE: {m['MAE']:6.2f} | "
                f"RMSE: {m['RMSE']:6.2f} | R²: {m['R2']:6.4f} | "
                f"Coefs (LR, RF, XGB): [{meta_model.coef_[0]:.2f}, {meta_model.coef_[1]:.2f}, {meta_model.coef_[2]:.2f}]"
            )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    mean_mae = float(fold_metrics_df["MAE"].mean())
    std_mae = float(fold_metrics_df["MAE"].std())
    mean_rmse = float(fold_metrics_df["RMSE"].mean())
    std_rmse = float(fold_metrics_df["RMSE"].std())
    mean_r2 = float(fold_metrics_df["R2"].mean())
    std_r2 = float(fold_metrics_df["R2"].std())

    summary_metrics = {
        "model_name": "Stacking Ensemble (Meta: Linear Regression)",
        "base_models": ["Linear Regression", "Random Forest", "XGBoost"],
        "meta_model": "Linear Regression",
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

    # Out-of-fold prediction dataframe
    stacked_oof_df = pd.DataFrame({
        "engine_id": engine_ids,
        "actual_RUL": y_true,
        "pred_linear": Z[:, 0],
        "pred_rf": Z[:, 1],
        "pred_xgb": Z[:, 2],
        "predicted_RUL": stacked_preds,
        "absolute_error": np.abs(y_true - stacked_preds),
        "fold": folds,
    })

    # Meta-model coefficient summary
    meta_coef_df = pd.DataFrame(meta_coef_records)
    coef_summary = pd.DataFrame({
        "Component": ["Linear Regression Base", "Random Forest Base", "XGBoost Base", "Intercept"],
        "Mean Weight": [
            meta_coef_df["coef_linear"].mean(),
            meta_coef_df["coef_rf"].mean(),
            meta_coef_df["coef_xgb"].mean(),
            meta_coef_df["intercept"].mean(),
        ],
        "Std Weight": [
            meta_coef_df["coef_linear"].std(),
            meta_coef_df["coef_rf"].std(),
            meta_coef_df["coef_xgb"].std(),
            meta_coef_df["intercept"].std(),
        ],
    })

    # Comparative benchmark across all models and ensembles
    lr_summary = json.loads(Path("reports/individual_models/linear_regression/summary_metrics.json").read_text())
    rf_summary = json.loads(Path("reports/individual_models/random_forest/summary_metrics.json").read_text())
    xgb_summary = json.loads(Path("reports/individual_models/xgboost/summary_metrics.json").read_text())
    avg_summary = json.loads(Path("reports/ensemble/simple_averaging/summary_metrics.json").read_text())
    weighted_summary = json.loads(Path("reports/ensemble/weighted_averaging/best_weights.json").read_text())

    benchmark_rows = [
        {
            "Architecture / Paradigm": "Individual Base: Linear Regression",
            "Mean MAE": round(lr_summary["mean_MAE"], 2),
            "Std MAE": round(lr_summary["std_MAE"], 2),
            "Mean RMSE": round(lr_summary["mean_RMSE"], 2),
            "Std RMSE": round(lr_summary["std_RMSE"], 2),
            "Mean R²": round(lr_summary["mean_R2"], 4),
            "Std R²": round(lr_summary["std_R2"], 4),
            "Paradigm": "Single Parametric",
        },
        {
            "Architecture / Paradigm": "Individual Base: Random Forest",
            "Mean MAE": round(rf_summary["mean_MAE"], 2),
            "Std MAE": round(rf_summary["std_MAE"], 2),
            "Mean RMSE": round(rf_summary["mean_RMSE"], 2),
            "Std RMSE": round(rf_summary["std_RMSE"], 2),
            "Mean R²": round(rf_summary["mean_R2"], 4),
            "Std R²": round(rf_summary["std_R2"], 4),
            "Paradigm": "Single Non-Parametric (Bagging)",
        },
        {
            "Architecture / Paradigm": "Individual Base: XGBoost",
            "Mean MAE": round(xgb_summary["mean_MAE"], 2),
            "Std MAE": round(xgb_summary["std_MAE"], 2),
            "Mean RMSE": round(xgb_summary["mean_RMSE"], 2),
            "Std RMSE": round(xgb_summary["std_RMSE"], 2),
            "Mean R²": round(xgb_summary["mean_R2"], 4),
            "Std R²": round(xgb_summary["std_R2"], 4),
            "Paradigm": "Single Non-Parametric (Boosting)",
        },
        {
            "Architecture / Paradigm": "Ensemble: Simple Averaging (RF 0.5 + XGB 0.5)",
            "Mean MAE": round(avg_summary["mean_MAE"], 2),
            "Std MAE": round(avg_summary["std_MAE"], 2),
            "Mean RMSE": round(avg_summary["mean_RMSE"], 2),
            "Std RMSE": round(avg_summary["std_RMSE"], 2),
            "Mean R²": round(avg_summary["mean_R2"], 4),
            "Std R²": round(avg_summary["std_R2"], 4),
            "Paradigm": "Fixed Linear Blending",
        },
        {
            "Architecture / Paradigm": f"Ensemble: Optimal Weighted ({weighted_summary['best_weight_label']})",
            "Mean MAE": round(weighted_summary["mean_MAE"], 2),
            "Std MAE": round(weighted_summary["std_MAE"], 2),
            "Mean RMSE": round(weighted_summary["mean_RMSE"], 2),
            "Std RMSE": round(weighted_summary["std_RMSE"], 2),
            "Mean R²": round(weighted_summary["mean_R2"], 4),
            "Std R²": round(weighted_summary["std_R2"], 4),
            "Paradigm": "Optimized Linear Blending",
        },
        {
            "Architecture / Paradigm": "Ensemble: Stacking Regression (Meta: Linear)",
            "Mean MAE": round(mean_mae, 2),
            "Std MAE": round(std_mae, 2),
            "Mean RMSE": round(mean_rmse, 2),
            "Std RMSE": round(std_rmse, 2),
            "Mean R²": round(mean_r2, 4),
            "Std R²": round(std_r2, 4),
            "Paradigm": "Stacked Generalization",
        },
    ]

    all_comparison_df = pd.DataFrame(benchmark_rows)

    if verbose:
        print("\n=== COMPLETE BENCHMARK (INDIVIDUAL VS. ENSEMBLES) ===")
        print(all_comparison_df.to_string(index=False))

    # Save outputs
    stacked_oof_df.to_csv(rep_path / "oof_predictions.csv", index=False)
    fold_metrics_df.to_csv(rep_path / "fold_metrics.csv", index=False)
    pd.Series(summary_metrics).to_json(rep_path / "summary_metrics.json", indent=2)
    meta_coef_df.to_csv(rep_path / "meta_model_fold_coefficients.csv", index=False)
    coef_summary.to_csv(rep_path / "meta_model_coefficients_summary.csv", index=False)
    all_comparison_df.to_csv(rep_path / "stacking_comparison.csv", index=False)

    # Diagnostic Plots
    plot_actual_vs_predicted(
        y_true,
        stacked_preds,
        model_name="Stacking Ensemble (Meta: Linear Regression)",
        save_path=rep_path / "actual_vs_predicted.png",
    )

    plot_residuals(
        y_true,
        stacked_preds,
        model_name="Stacking Ensemble (Meta: Linear Regression)",
        save_path=rep_path / "residual_plot.png",
    )

    if verbose:
        print(f"\nOutputs successfully saved to: {rep_path.resolve()}")

    return {
        "fold_metrics_df": fold_metrics_df,
        "summary_metrics": summary_metrics,
        "all_comparison_df": all_comparison_df,
        "meta_models": meta_models,
        "stacked_oof_df": stacked_oof_df,
    }


if __name__ == "__main__":
    results = run_stacking_ensemble(verbose=True)
