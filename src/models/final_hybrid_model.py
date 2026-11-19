"""Step C3: select the strongest eligible enhanced FD001 model on the baseline folds."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import PROJECT_ROOT, generate_advanced_features
from src.models.enhanced_models import (
    BASELINE_DIR,
    ENHANCED_DATA,
    METRICS,
    REPORTS_DIR,
    checkpoint_path,
    comparison_metrics,
    load_enhanced_cv_data,
    make_oof,
    read_aligned_oof,
    save_evaluation,
    score_oof,
)


FINAL_DIR = PROJECT_ROOT / "reports" / "hybrid" / "final_hybrid_model"
ENSEMBLE_DIR = PROJECT_ROOT / "reports" / "ensemble"


def run_final_hybrid_cv(
    enhanced_data_path: str | Path = ENHANCED_DATA,
    enhanced_reports_dir: str | Path = REPORTS_DIR,
    enhanced_models_dir: str | Path | None = None,
    reports_dir: str | Path = FINAL_DIR,
) -> dict:
    """Review prior runs, select a fixed enhanced candidate, and save paired comparisons."""
    X, _, _, splits, expected, baseline = load_enhanced_cv_data(enhanced_data_path)
    enhanced_dir = Path(enhanced_reports_dir)
    models_dir = enhanced_dir / "models" if enhanced_models_dir is None else Path(enhanced_models_dir)
    destination = Path(reports_dir)

    baseline_lr = read_aligned_oof(BASELINE_DIR / "linear_regression" / "oof_predictions.csv", expected)
    simple = read_aligned_oof(ENSEMBLE_DIR / "simple_averaging" / "oof_predictions.csv", expected)
    weighted = read_aligned_oof(ENSEMBLE_DIR / "weighted_averaging" / "oof_predictions.csv", expected)
    stacking = read_aligned_oof(ENSEMBLE_DIR / "stacking" / "oof_predictions.csv", expected)
    blend_settings = json.loads(
        (ENSEMBLE_DIR / "weighted_averaging" / "best_weights.json").read_text(encoding="utf-8")
    )

    enhanced_components = {
        model: read_aligned_oof(enhanced_dir / model / "oof_predictions.csv", expected)
        for model in baseline
    }
    enhanced_average = make_oof(
        expected,
        np.mean([oof["predicted_RUL"].to_numpy() for oof in enhanced_components.values()], axis=0),
    )
    enhanced_candidates = {
        "Enhanced Random Forest": (enhanced_components["random_forest"], {"random_forest": 1.0}),
        "Enhanced XGBoost": (enhanced_components["xgboost"], {"xgboost": 1.0}),
        "Enhanced equal average": (enhanced_average, {"random_forest": 0.5, "xgboost": 0.5}),
    }

    individual = {
        "Baseline Linear Regression": baseline_lr,
        "Baseline Random Forest": baseline["random_forest"],
        "Baseline XGBoost": baseline["xgboost"],
    }
    fixed_baselines = {**individual, "Baseline equal average": simple}
    historical = {
        "Baseline weighted average": (
            weighted, "Blend weights were selected using the same five held-out folds scored here."
        ),
        "Baseline stacking": (
            stacking,
            "The meta-model used global OOF inputs. Some base fits for meta-training rows "
            "included the outer validation engines.",
        ),
    }

    predictions = {
        **fixed_baselines,
        **{name: oof for name, (oof, _) in historical.items()},
        **{name: oof for name, (oof, _) in enhanced_candidates.items()},
    }
    scores = {name: score_oof(oof) for name, oof in predictions.items()}

    # Use the previous weighted experiment's primary criterion for the fixed candidates.
    ranking_key = lambda name: (scores[name][1]["mean_MAE"], scores[name][1]["mean_RMSE"])
    winner = min(enhanced_candidates, key=ranking_key)
    best_clean_baseline = min(fixed_baselines, key=ranking_key)
    best_individual = min(individual, key=ranking_key)
    best_reported_baseline = min([*fixed_baselines, *historical], key=ranking_key)

    ranking = []
    for name in sorted(predictions, key=ranking_key):
        if name in enhanced_candidates:
            status, note = "Enhanced candidate", "Fixed model settings and blend weights."
        elif name in historical:
            status, note = "Historical reference", historical[name][1]
        else:
            status, note = "Eligible baseline", "Fixed model settings and blend weights."
        ranking.append({
            "model": name,
            "features": X.shape[1] if name in enhanced_candidates else 15,
            "evaluation_status": status,
            **scores[name][1],
            "evaluation_note": note,
        })

    best_oof, weights = enhanced_candidates[winner]
    selected_folds, selected_summary = scores[winner]
    components = []
    for model, weight in weights.items():
        paths = [checkpoint_path(models_dir, model, fold) for fold in range(1, len(splits) + 1)]
        if absent := [str(path) for path in paths if not path.is_file()]:
            raise FileNotFoundError(f"Missing C2 checkpoints: {', '.join(absent)}")
        components.append({
            "model": model,
            "weight": weight,
            "fold_checkpoints": [Path(os.path.relpath(path, destination)).as_posix() for path in paths],
        })

    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranking).to_csv(destination / "candidate_comparison.csv", index=False)
    comparison = []
    fold_comparison = []
    for role, name in (
        ("Highest reported baseline", best_reported_baseline),
        ("Strongest eligible baseline", best_clean_baseline),
        ("Strongest baseline individual", best_individual),
    ):
        base_folds, base_summary = scores[name]
        comparison.append({
            "comparison": role,
            "baseline_model": name,
            "hybrid_model": winner,
            **comparison_metrics(base_summary, selected_summary),
        })
        fold_comparison.append(
            base_folds[["Fold", *METRICS]].merge(
                selected_folds[["Fold", *METRICS]], on="Fold",
                suffixes=("_baseline", "_hybrid"), validate="one_to_one",
            ).assign(comparison=role, baseline_model=name)
        )
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(destination / "baseline_vs_hybrid_comparison.csv", index=False)
    pd.concat(fold_comparison, ignore_index=True).to_csv(destination / "paired_fold_metrics.csv", index=False)

    selection = "Lowest mean validation MAE across the five baseline folds, with mean RMSE as tiebreaker."
    save_evaluation(destination, winner, best_oof, selected_folds, {
        **selected_summary,
        "n_splits": len(splits),
        "n_features": X.shape[1],
        "selection_criterion": selection,
        "best_eligible_baseline": best_clean_baseline,
        "best_reported_baseline": best_reported_baseline,
    })
    manifest = {"model_name": winner, "feature_columns": list(X.columns), "components": components}
    (destination / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    best_base = scores[best_clean_baseline][1]
    mae_change = selected_summary["mean_MAE"] - best_base["mean_MAE"]
    if mae_change < 0:
        finding = f"{winner} lowered mean MAE by {abs(mae_change):.2f} cycles against {best_clean_baseline}."
    elif mae_change > 0:
        finding = f"{winner} increased mean MAE by {mae_change:.2f} cycles against {best_clean_baseline}."
    else:
        finding = f"{winner} tied {best_clean_baseline} on mean MAE."

    lines = [
        "# FD001 hybrid cross-validation",
        "",
        finding,
        "",
        "| Model | Mean MAE | Mean RMSE | Mean R² |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in dict.fromkeys([winner, best_clean_baseline, best_individual, best_reported_baseline]):
        summary = scores[name][1]
        lines.append(
            f"| {name} | {summary['mean_MAE']:.3f} | {summary['mean_RMSE']:.3f} | {summary['mean_R2']:.4f} |"
        )
    lines.extend([
        "",
        f"**Selection criterion**  {selection} The three enhanced candidates were Random Forest, "
        "XGBoost, and their fixed equal average. Each used the same 99 unscaled features. "
        "The estimators retained their baseline hyperparameters.",
        "",
        f"The historical weighted average used {blend_settings['best_w_rf']:.0%} Random Forest "
        f"and {blend_settings['best_w_xgb']:.0%} XGBoost. Its weight search used the same held-out "
        "folds used to report performance. The historical stack used global out-of-fold base "
        "predictions, so some meta-training inputs depended on outer validation engines. "
        "Both scores appear as historical references in the candidate comparison.",
        "",
        "Each candidate prediction for an engine came from estimators fitted on other engines. "
        "The same cross-validation results were used for candidate selection and reporting. "
        "Metrics in the tables are unweighted means across folds. Plot annotations use pooled predictions.",
        "",
        "The model manifest references the fitted C2 fold checkpoints. For inference, "
        "supply the observed history for each engine. The predictor averages each component "
        "across its fold models and applies the selected component weights.",
    ])
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"selected_model": winner, "summary_metrics": selected_summary, "comparison": comparison}


def predict_hybrid(telemetry: pd.DataFrame, reports_dir: str | Path = FINAL_DIR) -> pd.DataFrame:
    """Predict from each engine's observed history using the saved hybrid model."""
    destination = Path(reports_dir)
    manifest = json.loads((destination / "model_manifest.json").read_text(encoding="utf-8"))
    enhanced, features, _ = generate_advanced_features(telemetry)
    if features != manifest["feature_columns"]:
        raise ValueError("The saved hybrid and current feature schemas differ.")
    X = enhanced[features]
    predictions = np.zeros(len(X))
    if len(X):
        for component in manifest["components"]:
            checkpoints = component["fold_checkpoints"]
            fold_sum = np.zeros(len(X))
            for path in checkpoints:
                estimator = joblib.load(destination / path)
                fold_sum += estimator.predict(X)
            predictions += component["weight"] * fold_sum / len(checkpoints)
    return enhanced[["unit_number", "time_cycles"]].assign(predicted_RUL=predictions)


if __name__ == "__main__":
    results = run_final_hybrid_cv()
    print(results["comparison"].to_string(index=False))
