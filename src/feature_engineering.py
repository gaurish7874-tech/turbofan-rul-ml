"""Causal FD001 time-series features shared by training and inference."""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_train_data
from src.preprocessing import CANONICAL_15_FEATURES, create_rul_target


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INFORMATIVE_SENSORS = tuple(CANONICAL_15_FEATURES[1:])
ROLLING_WINDOWS = (5, 10)
SLOPE_WINDOW = 5

SENSOR_DESCRIPTIONS = {
    "sensor_2": "LPC outlet temperature (degrees Rankine)",
    "sensor_3": "HPC outlet temperature (degrees Rankine)",
    "sensor_4": "LPT outlet temperature (degrees Rankine)",
    "sensor_7": "HPC outlet pressure (psia)",
    "sensor_8": "Physical fan speed (rpm)",
    "sensor_9": "Physical core speed (rpm)",
    "sensor_11": "HPC outlet static pressure (psia)",
    "sensor_12": "Fuel flow to Ps30 ratio (pps/psia)",
    "sensor_13": "Corrected fan speed (rpm)",
    "sensor_14": "Corrected core speed (rpm)",
    "sensor_15": "Bypass ratio",
    "sensor_17": "Bleed enthalpy",
    "sensor_20": "HPT coolant bleed (lbm/s)",
    "sensor_21": "LPT coolant bleed (lbm/s)",
}


def build_feature_dictionary(
    feature_cols: Sequence[str] | None = None,
    *,
    sensors: Sequence[str] = INFORMATIVE_SENSORS,
    windows: Sequence[int] = ROLLING_WINDOWS,
) -> dict[str, str]:
    """Describe each model column in feature-matrix order."""
    features = {"time_cycles": "Operating cycle of the current reading."}
    for sensor in sensors:
        description = SENSOR_DESCRIPTIONS.get(sensor, sensor)
        features[sensor] = f"Current reading of {description}."

    for sensor in sensors:
        description = SENSOR_DESCRIPTIONS.get(sensor, sensor)
        for window in windows:
            features[f"{sensor}_roll_mean_{window}"] = (
                f"Mean of {description} over the last {window} observations of the "
                "same engine, including the current reading. Uses available history at startup."
            )
            features[f"{sensor}_roll_std_{window}"] = (
                f"Sample standard deviation (ddof=1) of {description} over the last "
                f"{window} observations of the same engine, including the current reading. "
                "Uses available history and is zero with one observation."
            )
        features[f"{sensor}_diff_1"] = (
            f"Current minus previous observation of {description} for the same engine. "
            "Zero on the first observation."
        )
        features[f"{sensor}_slope_{SLOPE_WINDOW}"] = (
            f"Least-squares slope of {description} against time_cycles over the last "
            f"{SLOPE_WINDOW} observations of the same engine, including the current reading. "
            "Reported in sensor units per cycle. Uses available history and is zero "
            "with one observation."
        )

    if feature_cols is None:
        return features
    return {key: features[key] for key in feature_cols}


def _rolling_slopes(engine: pd.DataFrame, sensors: list[str]) -> np.ndarray:
    """Fit trailing least-squares slopes for one engine using actual cycle spacing."""
    cycles = engine["time_cycles"].to_numpy(dtype=float)
    readings = engine[sensors].to_numpy(dtype=float)
    # Left padding gives startup rows their available history without adding observations.
    cycle_windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(cycles, (SLOPE_WINDOW - 1, 0), constant_values=np.nan), SLOPE_WINDOW
    )
    sensor_windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(readings, ((SLOPE_WINDOW - 1, 0), (0, 0)), constant_values=np.nan),
        SLOPE_WINDOW,
        axis=0,
    )
    centered_cycles = cycle_windows - np.nanmean(cycle_windows, axis=1, keepdims=True)
    # Center sensor values on the current reading to avoid cancellation at large offsets.
    numerator = np.nansum(
        centered_cycles[:, None, :] * (sensor_windows - readings[:, :, None]), axis=2
    )
    denominator = np.nansum(centered_cycles**2, axis=1)[:, None]
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )


def generate_advanced_features(
    df: pd.DataFrame,
    sensors: Sequence[str] = INFORMATIVE_SENSORS,
    windows: Sequence[int] = ROLLING_WINDOWS,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Append causal features in engine/cycle order and return their model schema.

    Input trajectories require unique engine/cycle pairs and finite sensor readings.
    Any target or other metadata is carried through unchanged and excluded from the
    calculation. Supply each engine's history before selecting a row for prediction.
    """
    sensors = list(sensors)
    windows = tuple(windows)
    if not sensors or len(sensors) != len(set(sensors)) or any(
        not sensor.startswith("sensor_") or not sensor[7:].isdigit() for sensor in sensors
    ):
        raise ValueError("Provide distinct telemetry sensor columns named sensor_<number>.")
    if not windows or len(windows) != len(set(windows)) or any(
        not isinstance(window, (int, np.integer)) or window < 2 for window in windows
    ):
        raise ValueError("Provide distinct integer rolling windows of at least two observations.")

    keys = ["unit_number", "time_cycles"]
    required = [*keys, *sensors]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing telemetry columns: {sorted(missing)}")
    if df["unit_number"].isna().any() or df.duplicated(keys).any():
        raise ValueError("Engine IDs must be present and engine/cycle pairs must be unique.")
    numeric = df[["time_cycles", *sensors]]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in numeric.dtypes) or not (
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).all()
    ):
        raise ValueError("Cycles and sensor readings must be finite numeric values.")

    dictionary = build_feature_dictionary(sensors=sensors, windows=windows)
    feature_cols = list(dictionary)
    ordered = df.sort_values(keys)
    telemetry = ordered[required].reset_index(drop=True)
    grouped = telemetry.groupby("unit_number", sort=False)
    sensor_groups = grouped[sensors]
    first_rows = grouped.cumcount().eq(0)

    engineered = {}
    for window in windows:
        rolling = sensor_groups.rolling(window, min_periods=1)
        means = rolling.mean().droplevel("unit_number")
        deviations = rolling.std(ddof=1).droplevel("unit_number")
        deviations.loc[first_rows] = 0.0
        for sensor in sensors:
            engineered[f"{sensor}_roll_mean_{window}"] = means[sensor]
            engineered[f"{sensor}_roll_std_{window}"] = deviations[sensor]

    differences = sensor_groups.diff()
    differences.loc[first_rows] = 0.0
    slopes = np.zeros((len(telemetry), len(sensors)))
    for _, engine in grouped:
        slopes[engine.index] = _rolling_slopes(engine, sensors)
    for index, sensor in enumerate(sensors):
        engineered[f"{sensor}_diff_1"] = differences[sensor]
        engineered[f"{sensor}_slope_{SLOPE_WINDOW}"] = slopes[:, index]

    derived_cols = feature_cols[len(sensors) + 1:]
    derived = pd.DataFrame(engineered, index=telemetry.index)[derived_cols]
    if not np.isfinite(derived.to_numpy()).all():
        raise ValueError("Feature engineering produced non-finite values.")
    derived.index = ordered.index
    combined = pd.concat([ordered.drop(columns=derived_cols, errors="ignore"), derived], axis=1)
    return combined, feature_cols, dictionary


def save_enhanced_dataset(
    df_engineered: pd.DataFrame,
    output_path: str | Path = PROCESSED_DIR / "train_FD001_enhanced.csv",
) -> Path:
    """Write an enhanced dataset to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_engineered.to_csv(path, index=False)
    return path


def generate_enhanced_training_dataset(
    output_dir: str | Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Generate the C1 training dataset and its feature dictionary."""
    baseline = create_rul_target(load_train_data(PROJECT_ROOT / "data" / "raw"))
    baseline = baseline[["unit_number", *CANONICAL_15_FEATURES, "RUL"]]
    enhanced, features, dictionary = generate_advanced_features(baseline)
    output_dir = Path(output_dir)
    save_enhanced_dataset(
        enhanced[["unit_number", "RUL", *features]], output_dir / "train_FD001_enhanced.csv"
    )
    (output_dir / "FD001_enhanced_feature_dictionary.json").write_text(
        json.dumps(dictionary, indent=2) + "\n", encoding="utf-8"
    )
    return enhanced, features, dictionary


if __name__ == "__main__":
    enhanced, features, _ = generate_enhanced_training_dataset()
    print(f"Saved {len(enhanced):,} rows with {len(features)} features to {PROCESSED_DIR}.")
