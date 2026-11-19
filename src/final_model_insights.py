"""Build the FD001 comparison and diagnostics from saved CV and NASA test results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.final_test_evaluation import OUTPUT_DIR, REPORTS_DIR, nasa_phm_score


INSIGHTS_DIR = REPORTS_DIR / "final_insights"
GROUPS = ("Individual", "Ensemble", "Hybrid")
METRICS = ("MAE", "RMSE", "R2")
IDENTITY = ["model_id", "Model", "Group"]
CONTRASTS = (
    ("Simple Averaging", "XGBoost"),
    ("Weighted Averaging", "XGBoost"),
    ("Stacking", "Simple Averaging"),
    ("Stacking", "Weighted Averaging"),
    ("Enhanced Random Forest", "Random Forest"),
    ("Enhanced XGBoost", "XGBoost"),
    ("Final Hybrid Model", "Simple Averaging"),
)


def build_comparison(test_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    cv = pd.read_csv(test_dir / "cross_validation_metrics.csv", keep_default_na=False)
    test = pd.read_csv(test_dir / "test_metrics.csv")
    comparison = cv.merge(test, on=IDENTITY, suffixes=("_CV", "_test"), validate="one_to_one")
    if len(comparison) != len(cv) or len(comparison) != len(test):
        raise ValueError("Cross-validation and test tables do not cover identical candidates.")
    return comparison[[
        *IDENTITY, *[f"{metric}_CV" for metric in METRICS],
        *[f"{metric}_test" for metric in METRICS], "NASA Score", "Source", "CV note",
    ]]


def _winners(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in (*GROUPS, "Overall"):
        subset = comparison if group == "Overall" else comparison.loc[comparison["Group"].eq(group)]
        for phase, label in (("CV", "Cross-validation"), ("test", "NASA test")):
            first = subset.sort_values([f"MAE_{phase}", f"RMSE_{phase}"]).iloc[0]
            rows.append({
                "Group": group, "Evaluation": label, "Model": first["Model"],
                **{metric: first[f"{metric}_{phase}"] for metric in METRICS},
            })
    return pd.DataFrame(rows)


def _contrasts(comparison: pd.DataFrame) -> pd.DataFrame:
    by_model = comparison.set_index("Model")
    rows = []
    for candidate, reference in CONTRASTS:
        metrics = {
            key: by_model.at[candidate, key] - by_model.at[reference, key]
            for phase in ("CV", "test")
            for key in (f"MAE_{phase}", f"RMSE_{phase}", f"R2_{phase}")
        }
        rows.append({"Candidate": candidate, "Reference": reference, **metrics})
    return pd.DataFrame(rows)


def _error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    bins = pd.cut(
        predictions["actual_RUL"], [-np.inf, 50, 100, np.inf],
        labels=["Below 50", "50 to 99", "100 or more"], right=False,
    )
    error = predictions["predicted_RUL"] - predictions["actual_RUL"]
    return (
        predictions.assign(RUL_range=bins, signed_error=error)
        .groupby(["Model", "RUL_range"], observed=True, sort=False)
        .agg(Engines=("unit_number", "size"), MAE=("absolute_error", "mean"),
             Mean_signed_error=("signed_error", "mean"))
        .reset_index()
    )


def _comparison_plot(comparison: pd.DataFrame, directory: Path) -> None:
    colors = comparison["Group"].map({"Individual": "#186986", "Ensemble": "#3256a4", "Hybrid": "#b26c32"})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained", sharey=True, sharex="col")
    ordered = comparison.iloc[::-1]
    for row, (phase, label) in enumerate((("CV", "Cross-validation"), ("test", "NASA test"))):
        for col, (metric, title) in enumerate(zip(METRICS, ("MAE (cycles)", "RMSE (cycles)", "R²"), strict=True)):
            ax = axes[row, col]
            ax.barh(ordered["Model"], ordered[f"{metric}_{phase}"], color=colors.iloc[::-1])
            ax.set_title(f"{label} · {title}")
            ax.grid(axis="y", visible=False)
    fig.savefig(directory / "cv_and_nasa_test_comparison.png", dpi=160)
    plt.close(fig)


def _error_plot(errors: pd.DataFrame, order: pd.Series, directory: Path) -> None:
    columns = ["Below 50", "50 to 99", "100 or more"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained", sharey=True)
    bias_limit = errors["Mean_signed_error"].abs().max()
    for index, (metric, title) in enumerate((("MAE", "Absolute error"), ("Mean_signed_error", "Signed error"))):
        matrix = errors.pivot(index="Model", columns="RUL_range", values=metric).reindex(index=order, columns=columns)
        options = (
            {"cmap": "Blues", "vmin": 0} if index == 0
            else {"cmap": "vlag", "vmin": -bias_limit, "vmax": bias_limit}
        )
        sns.heatmap(matrix, ax=axes[index], annot=True, fmt=".1f",
                    cbar_kws={"label": "Cycles"}, **options)
        axes[index].set(title=title, xlabel="Actual RUL (cycles)", ylabel="")
        axes[index].tick_params(axis="y", labelrotation=0)
    fig.savefig(directory / "errors_by_actual_rul.png", dpi=160)
    plt.close(fig)


def _describe_delta(delta: float) -> str:
    if np.isclose(delta, 0):
        return "an equal MAE"
    return f"{abs(delta):.2f} cycles {'lower' if delta < 0 else 'higher'} MAE"


def _contrast_text(contrasts: pd.DataFrame, pairs: range) -> str:
    lines = []
    for record in contrasts.iloc[list(pairs)].to_dict("records"):
        lines.append(
            f"- {record['Candidate']} had {_describe_delta(record['MAE_CV'])} in CV and "
            f"{_describe_delta(record['MAE_test'])} on the NASA test set compared with {record['Reference']}."
        )
    return "\n".join(lines)


def build_insights(test_dir: Path = OUTPUT_DIR, output_dir: Path = INSIGHTS_DIR) -> pd.DataFrame:
    test_dir, output_dir = Path(test_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison(test_dir)
    predictions = pd.read_csv(test_dir / "test_predictions.csv")
    errors = _error_summary(predictions)
    contrasts = _contrasts(comparison)
    winners = _winners(comparison)
    comparison.to_csv(output_dir / "master_comparison.csv", index=False)
    winners.to_csv(output_dir / "lowest_mae_by_group.csv", index=False)
    contrasts.to_csv(output_dir / "paired_model_contrasts.csv", index=False)
    errors.to_csv(output_dir / "errors_by_actual_rul.csv", index=False)
    for group in GROUPS:
        comparison.loc[comparison["Group"].eq(group)].to_csv(
            output_dir / f"{group.lower()}_comparison.csv", index=False
        )
    _comparison_plot(comparison, output_dir)
    _error_plot(errors, comparison["Model"], output_dir)

    cv_best = winners.loc[winners["Group"].eq("Overall") & winners["Evaluation"].eq("Cross-validation")].iloc[0]
    test_best = winners.loc[winners["Group"].eq("Overall") & winners["Evaluation"].eq("NASA test")].iloc[0]
    best_rmse = comparison.loc[comparison["RMSE_test"].idxmin()]
    best_r2 = comparison.loc[comparison["R2_test"].idxmax()]
    best_phm = comparison.loc[comparison["NASA Score"].idxmin()]
    focus = errors.loc[errors["Model"].eq(test_best["Model"])].sort_values("MAE", ascending=False).iloc[0]
    n_engines = predictions["unit_number"].nunique()
    highest_phm = comparison.loc[comparison["NASA Score"].idxmax()]
    model_predictions = predictions.loc[predictions["Model"].eq(highest_phm["Model"])]
    signed = model_predictions["predicted_RUL"] - model_predictions["actual_RUL"]
    highest_penalty = model_predictions.loc[(signed.abs() / np.where(signed < 0, 13, 10)).idxmax()]
    penalty_share = nasa_phm_score(
        [highest_penalty["actual_RUL"]], [highest_penalty["predicted_RUL"]]
    ) / highest_phm["NASA Score"]
    text = f"""# FD001 final model comparison

## Evaluation

Cross-validation values are saved means across five engine-grouped folds scored on available validation cycles. The NASA evaluation uses one prediction at the last observed cycle for each of the {n_engines} held-out engines. These metrics summarize different sets of observations and target distributions. The held-out rankings describe generalization on FD001. The training configurations and blend weights were fixed before test scoring.

## Strongest results

The lowest reported CV MAE was **{cv_best['Model']}** at **{cv_best['MAE']:.2f} cycles**. The lowest NASA test MAE was **{test_best['Model']}** at **{test_best['MAE']:.2f} cycles**.

- Lowest test RMSE was {best_rmse['Model']} at {best_rmse['RMSE_test']:.2f} cycles.
- Highest test R² was {best_r2['Model']} at {best_r2['R2_test']:.4f}.
- Lowest NASA PHM score was {best_phm['Model']} at {best_phm['NASA Score']:.2f}. The score penalizes RUL overestimates more heavily. [PHM 2008 scoring reference](https://ntrs.nasa.gov/api/citations/20205001125/downloads/Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf)

See the lowest-MAE table for the individual, ensemble, and hybrid leaders in each evaluation.

## Averaging and stacking

{_contrast_text(contrasts, range(4))}

The historical weighted CV score was reported on the folds used to choose its weights. The historical stacking CV used global out-of-fold base predictions, so some meta-training inputs depended on outer validation engines. The final stack learned its meta-model only from training-side, engine-held-out predictions and scored the separate NASA test engines once.

## Engineered features

{_contrast_text(contrasts, range(4, 7))}

The rolling summaries may smooth useful short-lived changes or add correlated inputs alongside the raw readings. These comparisons measure the combined feature set at the existing model settings.

## Error patterns and limitations

For {test_best['Model']}, the highest endpoint MAE occurred in the **{focus['RUL_range']} cycles** actual-RUL range. Its MAE was **{focus['MAE']:.2f}** over **{focus['Engines']} engines**, with a mean signed error of **{focus['Mean_signed_error']:+.2f} cycles**. Signed error is predicted minus actual RUL. A positive value means the model predicted too much remaining life.

Engine {int(highest_penalty['unit_number'])} accounted for **{penalty_share:.1%}** of the {highest_phm['Model']} PHM score. Its actual RUL was {highest_penalty['actual_RUL']:.0f} cycles and the model predicted {highest_penalty['predicted_RUL']:.1f}. Large endpoint errors can dominate the aggregate PHM score.

FD001 is a simulated benchmark and the test set contains a single censored endpoint per engine. Endpoint accuracy does not measure the consistency of predictions earlier in an engine's life. Similar readings at different latent degradation states and differences in engine life can make RUL difficult to distinguish. These are possible explanations rather than causal findings from the error tables. The reported rankings have sampling uncertainty, particularly when MAE differences are small.

## Technical conclusion

The strongest observed FD001 generalization by MAE came from **{test_best['Model']}**. Cross-validation and held-out performance are presented separately. The paired comparisons show where the ensemble or feature changes helped at the frozen settings, and the signed-error results show where maintenance timing would be most affected by overestimation.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")
    return comparison


if __name__ == "__main__":
    build_insights()
