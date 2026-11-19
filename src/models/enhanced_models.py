"""Step C2: benchmark unscaled enhanced features on the original FD001 folds."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_loader import load_train_data
from src.evaluate import (
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
)
from src.feature_engineering import PROCESSED_DIR, PROJECT_ROOT, build_feature_dictionary
from src.preprocessing import CANONICAL_15_FEATURES, create_rul_target, separate_features_target
from src.splitting import create_engine_group_splits, get_fold_data


REPORTS_DIR = PROJECT_ROOT / "reports" / "hybrid" / "enhanced_models"
BASELINE_DIR = PROJECT_ROOT / "reports" / "individual_models"
ENHANCED_DATA = PROCESSED_DIR / "train_FD001_enhanced.csv"
MODEL_NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost"}
METRICS = ("MAE", "RMSE", "R2")


def read_aligned_oof(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    """Verify that a saved experiment covers the same observations and folds."""
    oof = pd.read_csv(path)
    keys = ["engine_id", "actual_RUL", "fold"]
    if "time_cycles" in oof:
        keys.append("time_cycles")
    if not oof[keys].equals(expected[keys]) or not np.isfinite(oof["predicted_RUL"]).all():
        raise ValueError(f"OOF observations or fold assignments do not match: {path}")
    return oof


def load_enhanced_cv_data(
    enhanced_data_path: str | Path = ENHANCED_DATA,
    n_splits: int = 5,
):
    """Match the enhanced dataset to the baseline training rows and OOF GroupKFold assignments."""
    if n_splits != 5:
        raise ValueError("The baseline comparison requires the original five folds.")

    data = pd.read_csv(enhanced_data_path, float_precision="round_trip")
    feature_cols = list(build_feature_dictionary())
    expected_columns = {"unit_number", "RUL", *feature_cols}
    if set(data.columns) != expected_columns:
        raise ValueError("Enhanced CSV schema differs from C1. Run python -m src.feature_engineering.")

    baseline = create_rul_target(load_train_data(PROJECT_ROOT / "data" / "raw"))
    baseline_columns = ["unit_number", "RUL", *CANONICAL_15_FEATURES]
    pd.testing.assert_frame_equal(data[baseline_columns], baseline[baseline_columns])

    X, y, units = separate_features_target(data, feature_cols)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("The enhanced feature matrix contains non-finite values.")

    expected = pd.DataFrame({
        "engine_id": units,
        "time_cycles": data["time_cycles"],
        "actual_RUL": y,
        "fold": np.zeros(len(data), dtype=int),
    })
    splits = create_engine_group_splits(X, y, units, n_splits=n_splits, verbose=False)
    for fold, (_, val_idx) in enumerate(splits, start=1):
        expected.loc[val_idx, "fold"] = fold
    baselines = {
        model: read_aligned_oof(BASELINE_DIR / model / "oof_predictions.csv", expected)
        for model in MODEL_NAMES
    }
    return X, y, units, splits, expected, baselines


def make_oof(expected: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    oof = expected.copy()
    oof["predicted_RUL"] = predictions
    oof["absolute_error"] = (oof["actual_RUL"] - oof["predicted_RUL"]).abs()
    return oof


def score_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Use the baseline's unweighted mean and sample std across validation folds."""
    rows = []
    total_engines = oof["engine_id"].nunique()
    for fold, held_out in oof.groupby("fold", sort=True):
        validation_engines = held_out["engine_id"].nunique()
        rows.append({
            "Fold": f"Fold {fold}",
            **compute_regression_metrics(held_out["actual_RUL"], held_out["predicted_RUL"]),
            "Train_Engines": total_engines - validation_engines,
            "Val_Engines": validation_engines,
            "Val_Rows": len(held_out),
        })
    fold_metrics = pd.DataFrame(rows)
    summary = {
        f"{stat}_{metric}": float(getattr(fold_metrics[metric], stat)())
        for metric in METRICS
        for stat in ("mean", "std")
    }
    return fold_metrics, summary


def save_evaluation(
    directory: Path, name: str, oof: pd.DataFrame, folds: pd.DataFrame, summary: dict
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    oof.to_csv(directory / "oof_predictions.csv", index=False)
    folds.to_csv(directory / "fold_metrics.csv", index=False)
    (directory / "summary_metrics.json").write_text(
        json.dumps({"model_name": name, **summary}, indent=2) + "\n", encoding="utf-8"
    )
    actual, predicted = oof["actual_RUL"], oof["predicted_RUL"]
    plt.close(plot_actual_vs_predicted(actual, predicted, name, directory / "actual_vs_predicted.png"))
    plt.close(plot_residuals(actual, predicted, name, directory / "residual_plot.png"))


def comparison_metrics(baseline: dict, enhanced: dict) -> dict[str, float]:
    """Return fold-mean scores and deltas (positive improvement is better)."""
    values = {}
    for metric in METRICS:
        original, advanced = baseline[f"mean_{metric}"], enhanced[f"mean_{metric}"]
        values[f"baseline_{metric}"] = original
        values[f"enhanced_{metric}"] = advanced
        values[f"{metric}_improvement"] = advanced - original if metric == "R2" else original - advanced
    return values


def checkpoint_path(models_dir: Path, model_id: str, fold: int) -> Path:
    return models_dir / model_id / f"fold_{fold}.joblib"


def run_enhanced_models_cv(
    enhanced_data_path: str | Path = ENHANCED_DATA,
    reports_base_dir: str | Path = REPORTS_DIR,
    models_base_dir: str | Path | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict:
    """Fit the baseline model configurations with the 99 enhanced feature columns."""
    X, y, units, splits, expected, baselines = load_enhanced_cv_data(enhanced_data_path, n_splits)
    reports_dir = Path(reports_base_dir)
    models_dir = reports_dir / "models" if models_base_dir is None else Path(models_base_dir)
    # Match the baseline estimators and hyperparameters.
    estimators = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            random_state=random_state, n_jobs=n_jobs,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="reg:squarederror",
            random_state=random_state, n_jobs=n_jobs,
        ),
    }
    summaries = {}
    comparisons = []
    paired_folds = []
    for model_id, estimator in estimators.items():
        name = f"Enhanced {MODEL_NAMES[model_id]}"
        predictions = np.full(len(X), np.nan)
        importances = []
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            split = get_fold_data(X, y, units, train_idx, val_idx)
            estimator.fit(split["X_train"], split["y_train"])
            predictions[val_idx] = estimator.predict(split["X_val"])
            importances.append(estimator.feature_importances_)
            path = checkpoint_path(models_dir, model_id, fold)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(".tmp")
            joblib.dump(estimator, temp_path, compress=3)
            temp_path.replace(path)
            if verbose:
                print(f"{name} | Fold {fold} saved", flush=True)

        oof = make_oof(expected, predictions)
        folds, summary = score_oof(oof)
        parameters = {
            key: value for key, value in estimator.get_params().items()
            if value is not None and not (isinstance(value, float) and np.isnan(value))
        }
        summary.update(n_splits=n_splits, n_features=X.shape[1], parameters=parameters)
        destination = reports_dir / model_id
        save_evaluation(destination, name, oof, folds, summary)

        importance = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": np.mean(importances, axis=0),
            "importance_std": np.std(importances, axis=0),
        }).sort_values("importance_mean", ascending=False)
        importance.to_csv(destination / "feature_importance.csv", index=False)
        plt.close(plot_feature_importance(importance, name, destination / "feature_importance.png", top_n=15))

        baseline_folds, baseline_summary = score_oof(baselines[model_id])
        comparisons.append({
            "model": MODEL_NAMES[model_id],
            **comparison_metrics(baseline_summary, summary),
        })
        paired_folds.append(
            baseline_folds[["Fold", *METRICS]].merge(
                folds[["Fold", *METRICS]], on="Fold", suffixes=("_baseline", "_enhanced"),
                validate="one_to_one",
            ).assign(model=MODEL_NAMES[model_id])
        )
        summaries[model_id] = summary
        if verbose:
            print(pd.DataFrame(comparisons[-1:]).to_string(index=False), flush=True)

    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(reports_dir / "baseline_vs_enhanced_comparison.csv", index=False)
    pd.concat(paired_folds, ignore_index=True).to_csv(reports_dir / "paired_fold_metrics.csv", index=False)
    expected[["engine_id", "time_cycles", "fold"]].to_csv(reports_dir / "fold_assignments.csv", index=False)
    return {"rf_summary": summaries["random_forest"], "xgb_summary": summaries["xgboost"], "comp_df": comparison}


if __name__ == "__main__":
    run_enhanced_models_cv()
