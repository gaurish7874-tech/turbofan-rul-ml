"""Compare FD001 accuracy improvements using training engines only."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.evaluate import compute_regression_metrics
from src.feature_engineering import PROJECT_ROOT
from src.models.enhanced_models import load_enhanced_cv_data
from src.preprocessing import CANONICAL_15_FEATURES
from src.splitting import create_engine_group_splits


OUTPUT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement"
CUTOFFS = (0.35, 0.60, 0.85)
SENSORS = CANONICAL_15_FEATURES[1:]
BASELINE = "Original XGBoost"
REGULARIZED = {
    "n_estimators": 420, "max_depth": 3, "learning_rate": 0.035,
    "min_child_weight": 12, "reg_lambda": 6,
}
CONFIGURATIONS = {
    BASELINE: ("raw", "rul", {}),
    "Regularized XGBoost": ("raw", "rul", REGULARIZED),
    "Robust XGBoost": ("raw", "rul", {
        **REGULARIZED, "objective": "reg:pseudohubererror", "huber_slope": 25,
    }),
    "Sensor-history XGBoost": ("history", "rul", REGULARIZED),
    "Lifetime XGBoost": ("raw", "lifetime", REGULARIZED),
    "Sensor-history lifetime XGBoost": ("history", "lifetime", REGULARIZED),
}
CRITERION = "Highest within 20 cycles, then highest within 10 cycles, then lowest MAE."


def history_features(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Build current, smoothed, initial-relative, and recent-change sensor inputs."""
    if not telemetry.index.is_unique:
        raise ValueError("Telemetry row index must be unique.")
    if telemetry["unit_number"].isna().any() or telemetry.duplicated(["unit_number", "time_cycles"]).any():
        raise ValueError("Telemetry must have distinct engine/cycle observations.")

    ordered = telemetry.sort_values(["unit_number", "time_cycles"])
    units = ordered["unit_number"]
    raw = ordered[CANONICAL_15_FEATURES]
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Cycle and sensor measurements must be finite.")
    readings = raw[SENSORS]
    groups = readings.groupby(units, sort=False)
    average_10 = groups.rolling(10, min_periods=1).mean().droplevel(0).reindex(ordered.index)

    # Estimate each engine's starting reference from its first 10 available readings.
    # At early cycles the reference contains only the readings observed so far.
    counts = groups.cumcount().add(1)
    initial = readings.where(counts.le(10), axis=0)
    initial_sum = initial.groupby(units, sort=False).cumsum().groupby(units, sort=False).ffill()
    initial_mean = initial_sum.div(counts.clip(upper=10), axis=0)
    prior_average = average_10.groupby(units, sort=False).shift(20)
    recent_change = average_10 - prior_average.fillna(average_10.groupby(units, sort=False).transform("first"))

    return pd.concat([
        raw,
        average_10.add_suffix("_mean_10"),
        (average_10 - initial_mean).add_suffix("_change_from_start"),
        recent_change.add_suffix("_change_over_20"),
    ], axis=1).reindex(telemetry.index)


def _endpoints(units: pd.Series) -> pd.DataFrame:
    """Use three fixed observation fractions to simulate censored training histories."""
    rows = []
    for engine, valid in units.groupby(units, sort=True):
        for fraction in CUTOFFS:
            position = max(0, int(np.ceil(len(valid) * fraction)) - 1)
            rows.append({"row": valid.index[position], "engine_id": engine, "observed_fraction": fraction})
    return pd.DataFrame(rows)


def _fixed_settings() -> dict:
    path = PROJECT_ROOT / "reports" / "individual_models" / "xgboost" / "summary_metrics.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    keys = ("n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "objective", "random_state")
    return {key: saved[key] for key in keys}


def _make_model(name: str, settings: dict, jobs: int) -> XGBRegressor:
    return XGBRegressor(**{**settings, **CONFIGURATIONS[name][2], "n_jobs": jobs})


def _predict(model: XGBRegressor, features: pd.DataFrame, target: str) -> np.ndarray:
    predicted = model.predict(features).astype(float)
    if target == "lifetime":
        predicted = np.maximum(0, predicted - features["time_cycles"].to_numpy())
    return predicted


def _fit_predict(name, matrices, y, train, validation, settings, jobs):
    representation, target, _ = CONFIGURATIONS[name]
    features = matrices[representation]
    train_features = features.loc[train]
    target_values = y.loc[train]
    if target == "lifetime":
        target_values = target_values + train_features["time_cycles"]
    model = _make_model(name, settings, jobs)
    model.fit(train_features, target_values)
    return _predict(model, features.loc[validation], target)


def _score(actual: pd.Series, predicted: np.ndarray) -> dict:
    error = np.abs(actual.to_numpy() - predicted)
    return {
        "within_10_pct": float(100 * np.mean(error <= 10)),
        "within_20_pct": float(100 * np.mean(error <= 20)),
        **compute_regression_metrics(actual, predicted),
    }


def _rank_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(
        ["within_20_pct", "within_10_pct", "MAE"],
        ascending=[False, False, True],
        kind="stable",
    )


def _select(matrices, y, folds, endpoints, settings, jobs):
    validation = [(train, endpoints.index[endpoints.index.isin(val)]) for train, val in folds]
    rows = []
    actual = y.loc[np.concatenate([indexes for _, indexes in validation])]
    for name in CONFIGURATIONS:
        predictions = [
            _fit_predict(name, matrices, y, train, indices, settings, jobs)
            for train, indices in validation
        ]
        rows.append({"configuration": name, **_score(actual, np.concatenate(predictions))})
    return _rank_table(rows)


def _fit_final(name, matrices, y, settings, jobs, output_dir):
    representation, target, overrides = CONFIGURATIONS[name]
    features = matrices[representation]
    model = _make_model(name, settings, jobs)
    training_target = y + features["time_cycles"] if target == "lifetime" else y
    model.fit(features, training_target)
    checkpoint = output_dir / "candidate.joblib"
    joblib.dump(model, checkpoint, compress=3)
    manifest = {
        "configuration": name, "representation": representation, "target": target,
        "feature_columns": list(features.columns), "training_rows": len(y),
        "target_conversion": "Subtract the current cycle and floor at zero for lifetime models.",
        "selection_criterion": CRITERION, "parameters": {**settings, **overrides},
        "checkpoint": checkpoint.name,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def predict_candidate(telemetry: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> np.ndarray:
    """Score RUL from an engine history with the training-only candidate."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    if manifest["representation"] == "history":
        features = history_features(telemetry)
    else:
        features = telemetry[CANONICAL_15_FEATURES]
    model = joblib.load(output_dir / manifest["checkpoint"])
    return _predict(model, features[manifest["feature_columns"]], manifest["target"])


def _save_chart(comparison: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.7), layout="constrained")
    bars = comparison.set_index("approach")[["within_10_pct", "within_20_pct"]]
    bars.columns = ["Within 10 cycles", "Within 20 cycles"]
    bars.plot.barh(ax=ax, color=["#197184", "#4564ba"], width=0.65)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=4)
    ax.set(xlim=(0, 100), xlabel="Simulated endpoints within the tolerance (%)",
           ylabel="", title="Training-only comparison on held-out engine histories")
    ax.legend(loc="lower right")
    fig.savefig(output_dir / "training_accuracy_comparison.png", dpi=160)
    plt.close(fig)


def run_accuracy_experiment(output_dir: Path = OUTPUT_DIR, jobs: int = 4) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, units, outer_folds, _, baselines = load_enhanced_cv_data()
    raw = X[CANONICAL_15_FEATURES]
    history = history_features(raw.assign(unit_number=units))
    matrices = {"raw": raw, "history": history}
    settings = _fixed_settings()
    endpoint_info = _endpoints(units).set_index("row")
    endpoint_predictions = []
    inner_reports = []
    for number, (outer_train, outer_validation) in enumerate(outer_folds, 1):
        inner_splits = create_engine_group_splits(
            raw.iloc[outer_train], y.iloc[outer_train], units.iloc[outer_train],
            n_splits=3, verbose=False,
        )
        inner_folds = [(outer_train[tr], outer_train[va]) for tr, va in inner_splits]
        ranking = _select(matrices, y, inner_folds, endpoint_info, settings, jobs)
        selected = ranking["configuration"].iat[0]
        inner_reports.append(ranking.assign(outer_fold=number))
        fold_endpoints = endpoint_info.loc[endpoint_info.index.isin(outer_validation)].copy()
        indices = fold_endpoints.index
        fold_endpoints["fold"] = number
        fold_endpoints["configuration"] = selected
        fold_endpoints["cycle"] = raw.loc[indices, "time_cycles"]
        fold_endpoints["actual_RUL"] = y.loc[indices]
        fold_endpoints["original_prediction"] = baselines["xgboost"].loc[indices, "predicted_RUL"]
        if selected == BASELINE:
            fold_endpoints["candidate_prediction"] = fold_endpoints["original_prediction"]
        else:
            fold_endpoints["candidate_prediction"] = _fit_predict(
                selected, matrices, y, outer_train, indices, settings, jobs
            )
        endpoint_predictions.append(fold_endpoints)
        print(f"Outer fold {number}/{len(outer_folds)} | inner-selected {selected}", flush=True)

    all_endpoints = pd.concat(endpoint_predictions).sort_index()
    if not all_endpoints.index.equals(endpoint_info.sort_index().index):
        raise RuntimeError("Nested cross-validation missed or duplicated simulated endpoints.")
    comparison = pd.DataFrame([
        {"approach": title, **_score(all_endpoints["actual_RUL"], all_endpoints[column].to_numpy())}
        for title, column in ((BASELINE, "original_prediction"), ("Nested-selected candidate", "candidate_prediction"))
    ])
    all_endpoints.to_csv(output_dir / "held_out_training_endpoint_predictions.csv")
    comparison.to_csv(output_dir / "training_accuracy_comparison.csv", index=False)
    pd.concat(inner_reports).to_csv(output_dir / "inner_fold_selection_metrics.csv", index=False)
    _save_chart(comparison, output_dir)
    print(comparison.to_string(index=False), flush=True)

    ranking = _select(matrices, y, outer_folds, endpoint_info, settings, jobs)
    ranking.to_csv(output_dir / "full_training_candidate_selection.csv", index=False)
    winner = ranking["configuration"].iat[0]
    _fit_final(winner, matrices, y, settings, jobs, output_dir)
    print(f"Saved {winner} fitted on all {len(y):,} training observations.", flush=True)

    gain_20 = comparison["within_20_pct"].iat[1] - comparison["within_20_pct"].iat[0]
    gain_10 = comparison["within_10_pct"].iat[1] - comparison["within_10_pct"].iat[0]
    (output_dir / "README.md").write_text(
        f"""# FD001 training-only accuracy improvement

The nested selection procedure changed accuracy within 20 cycles by **{gain_20:+.2f} percentage points** and within 10 cycles by **{gain_10:+.2f} percentage points** on {len(all_endpoints)} simulated endpoints from {units.nunique()} training engines.

Each endpoint follows 35%, 60%, or 85% of a training engine's recorded observations. Endpoints carry uncapped actual RUL. The original five engine folds are the outer assessment; three engine-grouped folds inside each outer training split choose a configuration. {CRITERION} Every score for an outer validation engine comes from a model that excluded that engine from fitting and selection.

The saved candidate is **{winner}**, chosen by the same ranking rule on a separate five-fold run using all training engines, and then fitted to every training row. The nested estimate applies to the selection procedure. The full-training selection table describes the model selection run.

Historical NASA test results stay separate. These simulated training endpoints can differ from the NASA endpoints in RUL distribution and do not provide a new NASA test accuracy estimate.
""", encoding="utf-8"
    )
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    run_accuracy_experiment(jobs=args.jobs)
