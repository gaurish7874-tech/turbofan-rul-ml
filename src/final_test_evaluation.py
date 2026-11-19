"""Fit frozen FD001 candidates and score the held-out NASA engine endpoints."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data_loader import load_rul_data, load_test_data
from src.evaluate import compute_regression_metrics
from src.feature_engineering import PROJECT_ROOT, generate_advanced_features
from src.models.enhanced_models import load_enhanced_cv_data, read_aligned_oof
from src.preprocessing import CANONICAL_15_FEATURES


RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "final_test_evaluation"
PHM_SOURCE = (
    "https://ntrs.nasa.gov/api/citations/20205001125/downloads/"
    "Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf"
)

# Candidate IDs also identify prediction files. The order is fixed before evaluation.
CANDIDATES = (
    ("linear_regression", "Linear Regression", "Individual", "individual_models/linear_regression/summary_metrics.json"),
    ("random_forest", "Random Forest", "Individual", "individual_models/random_forest/summary_metrics.json"),
    ("xgboost", "XGBoost", "Individual", "individual_models/xgboost/summary_metrics.json"),
    ("simple_averaging", "Simple Averaging", "Ensemble", "ensemble/simple_averaging/summary_metrics.json"),
    ("weighted_averaging", "Weighted Averaging", "Ensemble", "ensemble/weighted_averaging/best_weights.json"),
    ("stacking", "Stacking", "Ensemble", "ensemble/stacking/summary_metrics.json"),
    ("enhanced_random_forest", "Enhanced Random Forest", "Hybrid", "hybrid/enhanced_models/random_forest/summary_metrics.json"),
    ("enhanced_xgboost", "Enhanced XGBoost", "Hybrid", "hybrid/enhanced_models/xgboost/summary_metrics.json"),
    ("final_hybrid_model", "Final Hybrid Model", "Hybrid", "hybrid/final_hybrid_model/summary_metrics.json"),
)
BASE_IDS = ("linear_regression", "random_forest", "xgboost")
CV_NOTES = {
    "weighted_averaging": "Historical weights chosen on the same folds used for reporting.",
    "stacking": "Historical meta-CV used global OOF base predictions. Some meta-training rows depended on outer validation engines.",
    "final_hybrid_model": "Enhanced blend selected and reported on the same folds.",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _frozen_configuration(feature_cols: list[str], n_jobs: int) -> tuple[dict, dict]:
    summaries = {
        key: _read_json(REPORTS_DIR / source) for key, _, _, source in CANDIDATES
    }
    rf = summaries["random_forest"]
    xgb = summaries["xgboost"]
    rf_keys = ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "random_state")
    xgb_keys = ("n_estimators", "max_depth", "learning_rate", "subsample",
                "colsample_bytree", "objective", "random_state")
    rf_params = {key: rf[key] for key in rf_keys}
    if rf_params["max_depth"] == "None":
        rf_params["max_depth"] = None
    xgb_params = {key: xgb[key] for key in xgb_keys}
    enhanced_rf = {key: summaries["enhanced_random_forest"]["parameters"].get(key) for key in rf_keys}
    enhanced_xgb = {key: summaries["enhanced_xgboost"]["parameters"][key] for key in xgb_keys}
    estimators = {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(**rf_params, n_jobs=n_jobs),
        "xgboost": XGBRegressor(**xgb_params, n_jobs=n_jobs),
        "enhanced_random_forest": RandomForestRegressor(**enhanced_rf, n_jobs=n_jobs),
        "enhanced_xgboost": XGBRegressor(**enhanced_xgb, n_jobs=n_jobs),
    }

    simple = summaries["simple_averaging"]["weights"]
    weighted = summaries["weighted_averaging"]
    hybrid = _read_json(REPORTS_DIR / "hybrid" / "final_hybrid_model" / "model_manifest.json")
    if hybrid["feature_columns"] != feature_cols:
        raise ValueError("The frozen hybrid and current enhanced features differ.")
    blends = {
        "simple_averaging": {"random_forest": simple["Random Forest"], "xgboost": simple["XGBoost"]},
        "weighted_averaging": {"random_forest": weighted["best_w_rf"], "xgboost": weighted["best_w_xgb"]},
        "final_hybrid_model": {
            f"enhanced_{component['model']}": component["weight"] for component in hybrid["components"]
        },
    }
    for model, weights in blends.items():
        if not weights.keys() <= estimators.keys() or not np.isclose(sum(weights.values()), 1) or min(weights.values()) < 0:
            raise ValueError(f"Invalid frozen blend: {model}")

    cv_rows = [
        {
            "model_id": key, "Model": name, "Group": group,
            **{metric: summaries[key][f"mean_{metric}"] for metric in ("MAE", "RMSE", "R2")},
            "Source": source, "CV note": CV_NOTES.get(key, ""),
        }
        for key, name, group, source in CANDIDATES
    ]
    return estimators, {"blends": blends, "cv_rows": cv_rows, "hybrid_name": hybrid["model_name"]}


def fit_final_candidates(output_dir: Path = OUTPUT_DIR, n_jobs: int = -1) -> dict:
    """Generate grouped meta-training predictions and fit final bases on all train rows."""
    output_dir = Path(output_dir)
    X, y, units, splits, expected, _ = load_enhanced_cv_data()
    read_aligned_oof(
        REPORTS_DIR / "individual_models/linear_regression/oof_predictions.csv", expected
    )
    estimators, config = _frozen_configuration(list(X.columns), n_jobs)
    baseline = X[CANONICAL_15_FEATURES]

    checkpoints = output_dir / "models"
    checkpoints.mkdir(parents=True, exist_ok=True)
    oof = expected.copy()
    oof[list(BASE_IDS)] = np.nan
    for fold, (train, validation) in enumerate(splits, 1):
        for key in BASE_IDS:
            model = clone(estimators[key])
            model.fit(baseline.iloc[train], y.iloc[train])
            oof.loc[validation, key] = model.predict(baseline.iloc[validation])
            del model
        print(f"Stacking training OOF | fold {fold}/{len(splits)}", flush=True)
    if not np.isfinite(oof[list(BASE_IDS)].to_numpy()).all():
        raise ValueError("Incomplete stacking training predictions.")
    meta = LinearRegression().fit(oof[list(BASE_IDS)], y)
    joblib.dump(meta, checkpoints / "stacking_meta.joblib")
    oof.to_csv(output_dir / "stacking_training_oof.csv", index=False)

    model_specs = {}
    for key, estimator in estimators.items():
        features = list(X.columns) if key.startswith("enhanced_") else CANONICAL_15_FEATURES
        estimator.fit(X[features], y)
        filename = f"{key}.joblib"
        joblib.dump(estimator, checkpoints / filename, compress=3)
        model_specs[key] = {"file": filename, "feature_columns": features}
        print(f"Full FD001 training set | saved {key}", flush=True)

    manifest = {
        "dataset": "FD001", "training_rows": len(y), "training_engines": int(units.nunique()),
        "n_splits": len(splits), "models": model_specs,
        "stacking_meta": {"file": "stacking_meta.joblib", "inputs": list(BASE_IDS)},
        **config,
    }
    _write_json(checkpoints / "manifest.json", manifest)
    pd.DataFrame(config["cv_rows"]).to_csv(output_dir / "cross_validation_metrics.csv", index=False)
    return manifest


def _test_endpoints(test: pd.DataFrame) -> pd.DataFrame:
    original_order = pd.Index(test["unit_number"].unique(), name="unit_number")
    if not np.array_equal(original_order, np.arange(1, 101)):
        raise ValueError("FD001 test engines must first appear in NASA label order (1 through 100).")
    enhanced, _, _ = generate_advanced_features(test)
    last = enhanced.groupby("unit_number", sort=False).tail(1).set_index("unit_number").loc[original_order]
    observed_max = test.groupby("unit_number")["time_cycles"].max().reindex(original_order)
    pd.testing.assert_series_equal(last["time_cycles"], observed_max)
    return last


def _match_nasa_labels(last: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rul = truth["RUL"].to_numpy()
    if (
        not np.array_equal(last.index, np.arange(1, 101))
        or len(rul) != len(last)
        or not np.isfinite(rul).all()
        or (rul < 0).any()
        or not np.equal(rul, np.floor(rul)).all()
    ):
        raise ValueError("Expected 100 nonnegative integer labels in FD001 unit order.")
    labels = pd.DataFrame({"unit_number": np.arange(1, len(rul) + 1), "actual_RUL": rul.astype(int)})
    alignment = (
        last["time_cycles"].rename("last_observed_cycle").reset_index()
        .merge(labels, on="unit_number", sort=False, validate="one_to_one")
    )
    if not np.array_equal(alignment["unit_number"], last.index):
        raise ValueError("FD001 labels changed engine order during matching.")
    return alignment


def nasa_phm_score(actual: pd.Series, predicted: pd.Series) -> float:
    """PHM08 asymmetric summed loss with a larger penalty for RUL overestimates."""
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    scaled = np.abs(error) / np.where(error < 0, 13.0, 10.0)
    return float(np.expm1(scaled).sum())


def _predict_candidates(last: pd.DataFrame, model_dir: Path, manifest: dict) -> dict[str, np.ndarray]:
    predictions = {}
    for key, spec in manifest["models"].items():
        estimator = joblib.load(model_dir / spec["file"])
        predictions[key] = estimator.predict(last[spec["feature_columns"]])
    for key, weights in manifest["blends"].items():
        predictions[key] = sum(weight * predictions[base] for base, weight in weights.items())
    stack = manifest["stacking_meta"]
    meta = joblib.load(model_dir / stack["file"])
    predictions["stacking"] = meta.predict(pd.DataFrame({
        key: predictions[key] for key in stack["inputs"]
    }))
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("A frozen candidate returned a non-finite prediction.")
    return predictions


def _save_test_plots(output_dir: Path, metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    groups = list(predictions.groupby("model_id", sort=False))
    error = predictions["predicted_RUL"] - predictions["actual_RUL"]
    error_limit = np.ceil(error.abs().max() / 10) * 10
    edges = np.linspace(-error_limit, error_limit, 21)
    bounds = predictions[["actual_RUL", "predicted_RUL"]].to_numpy()
    lo, hi = min(0, bounds.min()), bounds.max()
    all_errors, grid = plt.subplots(3, 3, figsize=(12, 9), layout="constrained", sharex=True, sharey=True)
    for ax_summary, (key, values) in zip(grid.flat, groups, strict=True):
        name = values["Model"].iat[0]
        actual, predicted = values["actual_RUL"], values["predicted_RUL"]
        signed_error = predicted - actual
        directory = output_dir / "figures" / key
        directory.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5.7, 5.2), layout="constrained")
        ax.scatter(actual, predicted, color="#176b87", alpha=0.74, s=26)
        ax.plot([lo, hi], [lo, hi], "--", color="0.4")
        ax.set(xlim=(lo - 5, hi + 5), ylim=(lo - 5, hi + 5), title=name,
               xlabel="Actual RUL (cycles)", ylabel="Predicted RUL (cycles)", aspect="equal")
        fig.savefig(directory / "actual_vs_predicted.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4), layout="constrained")
        ax.scatter(predicted, -signed_error, color="#176b87", alpha=0.74, s=26)
        ax.axhline(0, linestyle="--", color="0.4")
        ax.set(title=name, xlabel="Predicted RUL (cycles)", ylabel="Residual (actual − predicted)")
        fig.savefig(directory / "residuals.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4), layout="constrained")
        for panel in (ax, ax_summary):
            panel.hist(signed_error, bins=edges, color="#176b87", alpha=0.85)
            panel.axvline(0, linestyle="--", color="0.35")
            panel.set_title(name)
        ax.set(xlabel="Signed error (predicted − actual)", ylabel="Engines")
        fig.savefig(directory / "error_distribution.png", dpi=160)
        plt.close(fig)
    all_errors.supxlabel("Signed error (predicted − actual RUL, cycles)")
    all_errors.supylabel("Engines")
    all_errors.savefig(output_dir / "error_distributions.png", dpi=160)
    plt.close(all_errors)

    colors = {"Individual": "#186986", "Ensemble": "#3256a4", "Hybrid": "#b26c32"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), layout="constrained", sharey=True)
    plot_rows = metrics.iloc[::-1]
    for ax, metric, label in zip(axes, ("MAE", "RMSE", "R2", "NASA Score"),
                                ("MAE (cycles)", "RMSE (cycles)", "R²", "NASA PHM score"), strict=True):
        ax.barh(plot_rows["Model"], plot_rows[metric], color=plot_rows["Group"].map(colors))
        ax.set_xlabel(label)
        if metric == "NASA Score":
            ax.set_xscale("log")
    fig.savefig(output_dir / "test_metric_comparison.png", dpi=160)
    plt.close(fig)


def evaluate_final_candidates(output_dir: Path = OUTPUT_DIR, data_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Predict using frozen checkpoints, then match labels and score one row per engine."""
    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    manifest = _read_json(model_dir / "manifest.json")
    test = load_test_data(data_dir)
    last = _test_endpoints(test)
    predictions = _predict_candidates(last, model_dir, manifest)
    alignment = _match_nasa_labels(last, load_rul_data(data_dir))

    metrics, details = [], []
    detail_dir = output_dir / "predictions"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for key, name, group, _ in CANDIDATES:
        # Expand float32 predictions before CSV serialization so saved errors are reproducible.
        frame = alignment.assign(predicted_RUL=np.asarray(predictions[key], dtype=float))
        frame["absolute_error"] = (frame["actual_RUL"] - frame["predicted_RUL"]).abs()
        frame.to_csv(detail_dir / f"{key}.csv", index=False)
        details.append(frame.assign(model_id=key, Model=name))
        metrics.append({
            "model_id": key, "Model": name, "Group": group,
            **compute_regression_metrics(frame["actual_RUL"], frame["predicted_RUL"]),
            "NASA Score": nasa_phm_score(frame["actual_RUL"], frame["predicted_RUL"]),
        })
    comparison = pd.DataFrame(metrics)
    all_predictions = pd.concat(details, ignore_index=True)
    comparison.to_csv(output_dir / "test_metrics.csv", index=False)
    all_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    alignment.to_csv(output_dir / "test_label_alignment.csv", index=False)
    _write_json(output_dir / "evaluation_protocol.json", {
        "training_rows": manifest["training_rows"], "training_engines": manifest["training_engines"],
        "test_observed_rows": len(test), "test_engines": len(last),
        "scoring_rows_per_engine": 1, "label_key": "One-based NASA RUL file row equals unit_number.",
        "endpoint_key": "Maximum available time_cycles for each test engine.",
        "predictions": "Raw model outputs. Metrics use the 100 engine endpoints.",
        "nasa_score": "Sum expm1(abs(predicted - actual)/scale). Scale 13 for underestimates and 10 otherwise.",
        "nasa_score_source": PHM_SOURCE,
    })
    _save_test_plots(output_dir, comparison, all_predictions)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "evaluate", "all"), nargs="?", default="all")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    if args.stage in {"fit", "all"}:
        fit_final_candidates(n_jobs=args.jobs)
    if args.stage in {"evaluate", "all"}:
        print(evaluate_final_candidates().to_string(index=False))

