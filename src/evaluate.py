"""
Evaluation and Diagnostic Visualization Module for Turbofan RUL Regression.
NASA C-MAPSS FD001 Dataset.

Provides standardized regression metrics calculation (MAE, RMSE, R²) and
publication-quality diagnostic visualization utilities (Actual vs Predicted,
Residual diagnostics, and Feature Importance).
"""

from typing import Dict, Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Computes standard regression performance metrics.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Ground truth continuous target values.
    y_pred : Union[pd.Series, np.ndarray]
        Model prediction values.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'MAE', 'RMSE', and 'R2'.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def plot_actual_vs_predicted(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generates an Actual vs. Predicted RUL scatter plot with an ideal line of identity.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Ground truth RUL values.
    y_pred : Union[pd.Series, np.ndarray]
        Predicted RUL values.
    model_name : str
        Name of the regression model for plot titles.
    save_path : Optional[Union[str, Path]], default=None
        Path to save the generated figure.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_t, y_p, alpha=0.35, color="#1f77b4", edgecolors="none", s=25, label="Out-of-Fold Observations")

    min_val = min(float(y_t.min()), float(y_p.min()))
    max_val = max(float(y_t.max()), float(y_p.max()))
    padding = (max_val - min_val) * 0.05
    plot_min = max(0, min_val - padding)
    plot_max = max_val + padding

    # Reference line of identity y = x
    ax.plot([plot_min, plot_max], [plot_min, plot_max], "r--", linewidth=2.0, label="Ideal Fit (y = x)")

    # Tolerance bands (+- 20 cycles)
    ax.fill_between(
        [plot_min, plot_max],
        [plot_min - 20, plot_max - 20],
        [plot_min + 20, plot_max + 20],
        color="gray",
        alpha=0.15,
        label="±20 Cycles Tolerance Band",
    )

    metrics = compute_regression_metrics(y_t, y_p)
    metric_text = f"MAE:  {metrics['MAE']:.2f}\nRMSE: {metrics['RMSE']:.2f}\nR²:   {metrics['R2']:.4f}"
    ax.text(
        0.05,
        0.95,
        metric_text,
        transform=ax.transAxes,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_xlabel("Actual Remaining Useful Life (Cycles)", fontsize=12)
    ax.set_ylabel("Predicted Remaining Useful Life (Cycles)", fontsize=12)
    ax.set_title(f"{model_name}: Actual vs. Predicted RUL (Out-of-Fold CV)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)

    return fig


def plot_residuals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generates a 2-panel residual diagnostic plot:
    1. Residuals vs. Predicted RUL.
    2. Residual error distribution histogram with KDE.

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Ground truth RUL values.
    y_pred : Union[pd.Series, np.ndarray]
        Predicted RUL values.
    model_name : str
        Model title.
    save_path : Optional[Union[str, Path]], default=None
        Path to save figure.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    residuals = y_t - y_p  # positive: under-prediction (predicted < actual); negative: over-prediction

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Residuals vs Predicted
    axes[0].scatter(y_p, residuals, alpha=0.35, color="#2b5c8f", edgecolors="none", s=25)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.8, label="Zero Error Reference")
    axes[0].set_xlabel("Predicted RUL (Cycles)", fontsize=11)
    axes[0].set_ylabel("Residual Error (Actual - Predicted, Cycles)", fontsize=11)
    axes[0].set_title("Residuals vs. Predicted RUL", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True)

    # Panel 2: Distribution of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1], color="#2b5c8f", bins=40, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.8, label="Zero Bias")
    axes[1].axvline(np.mean(residuals), color="green", linestyle=":", linewidth=2, label=f"Mean Error: {np.mean(residuals):.2f}")
    axes[1].set_xlabel("Residual Error (Cycles)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title(f"Residual Distribution (Std: {np.std(residuals):.2f})", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)

    plt.suptitle(f"{model_name}: Residual Diagnostics (5-Fold Out-of-Fold)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
    top_n: int = 15,
) -> plt.Figure:
    """
    Plots mean feature importance across cross-validation folds with error bars.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with columns 'feature', 'importance_mean', 'importance_std'.
    model_name : str
        Model title.
    save_path : Optional[Union[str, Path]], default=None
        Output file path.
    top_n : int, default=15
        Number of top features to show.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    sorted_df = importance_df.sort_values("importance_mean", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(sorted_df))
    ax.barh(
        y_pos,
        sorted_df["importance_mean"],
        xerr=sorted_df["importance_std"] if "importance_std" in sorted_df.columns else None,
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.85,
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df["feature"], fontsize=11)
    ax.set_xlabel("Mean Feature Importance", fontsize=12)
    ax.set_title(f"{model_name}: Top {top_n} Feature Importance (5-Fold Mean ± Std)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)

    return fig
