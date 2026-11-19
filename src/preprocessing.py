"""
Data Preprocessing and RUL Target Engineering Module.
NASA C-MAPSS Turbofan Degradation (FD001).

This module implements defensible, leak-free preprocessing operations:
1. RUL target engineering from run-to-failure cycle trajectories.
2. Data sanity checks (missing values, duplicate rows, duplicate cycle keys).
3. Physics-informed and variance-based feature selection.
4. Clean separation of features, target, and grouping identifiers (unit_number).
"""

from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


# Human-interpretable justifications for removing non-informative channels in FD001
FEATURE_REMOVAL_JUSTIFICATIONS: Dict[str, str] = {
    "sensor_1": (
        "Fan inlet temperature (T2): Constant at 518.67 °R (variance = 0.000000). "
        "Represents fixed ambient sea-level inlet condition; provides zero degradation signal."
    ),
    "sensor_5": (
        "Fan inlet pressure (P2): Constant at 14.62 psia (variance = 0.000000). "
        "Ambient sea-level pressure condition; completely static."
    ),
    "sensor_6": (
        "Bypass-duct static pressure (P15): Negligible variance (2.0e-6), fluctuating only "
        "between 21.60 and 21.61 psia. Insufficient variation to track engine wear."
    ),
    "sensor_10": (
        "Engine pressure ratio (P50/P2): Constant at 1.30 (variance = 0.000000). "
        "Does not vary across operational cycles in single-operating-condition subset FD001."
    ),
    "sensor_16": (
        "Burner fuel-air ratio (dAmb): Constant at 0.03 (variance = 0.000000). "
        "Static combustion chamber parameter; invariant to component degradation."
    ),
    "sensor_18": (
        "Demanded fan speed (Nf_dmd): Constant at 2388.0 rpm (variance = 0.000000). "
        "Commanded setpoint from digital electronic engine control (FADEC), not an output sensor."
    ),
    "sensor_19": (
        "Demanded corrected fan speed (PCNfR_dmd): Constant at 100.0 rpm (variance = 0.000000). "
        "Nominal control setpoint; completely invariant across all engines."
    ),
    "operational_setting_3": (
        "Flight condition setting 3: Constant at 100.0 (variance = 0.000000). "
        "FD001 simulates a single sea-level operating regime; this setting is stationary."
    ),
}


def create_rul_target(df: pd.DataFrame, validate_zero_end: bool = True) -> pd.DataFrame:
    """
    Computes Remaining Useful Life (RUL) for run-to-failure engine trajectories.

    Formula:
        RUL_t = max(time_cycles for engine) - current_time_cycles

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'unit_number' and 'time_cycles'.
    validate_zero_end : bool, default=True
        Whether to assert that the final recorded cycle of every engine reaches RUL = 0.

    Returns
    -------
    pd.DataFrame
        A new copy of the DataFrame with an additional 'RUL' integer column.
        The original input DataFrame is left completely unmodified.

    Raises
    ------
    KeyError
        If 'unit_number' or 'time_cycles' columns are missing.
    ValueError
        If any engine's final trajectory observation does not equal RUL = 0.
    """
    required_cols = {"unit_number", "time_cycles"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Input DataFrame is missing required columns: {missing}")

    # Create an explicit copy to prevent modifying caller's DataFrame in-place
    df_out = df.copy()

    # Calculate maximum operating cycle for each distinct engine unit
    max_cycles = df_out.groupby("unit_number")["time_cycles"].transform("max")

    # Compute RUL (cycles remaining until failure)
    df_out["RUL"] = (max_cycles - df_out["time_cycles"]).astype(int)

    # Rigorous validation: The final recorded cycle of every training engine must equal 0
    if validate_zero_end:
        last_rul_per_engine = df_out.groupby("unit_number")["RUL"].last()
        invalid_engines = last_rul_per_engine[last_rul_per_engine != 0]
        if not invalid_engines.empty:
            raise ValueError(
                f"RUL validation failed! {len(invalid_engines)} engines do not end at RUL = 0: "
                f"{invalid_engines.to_dict()}"
            )

    return df_out


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Computes count of null / NaN values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to inspect.

    Returns
    -------
    pd.Series
        Missing value counts per column.
    """
    return df.isna().sum()


def check_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> int:
    """
    Computes count of duplicate rows or duplicate keys.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    subset : Optional[List[str]], default=None
        Column subset to check uniqueness for (e.g. ['unit_number', 'time_cycles']).

    Returns
    -------
    int
        Total number of duplicate entries detected.
    """
    return int(df.duplicated(subset=subset).sum())


def identify_low_variance_features(
    df: pd.DataFrame,
    candidate_cols: List[str],
    variance_threshold: float = 1e-5,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Identifies features with zero or near-zero variance.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    candidate_cols : List[str]
        Columns to evaluate.
    variance_threshold : float, default=1e-5
        Variance cut-off below which a feature is categorized as non-informative.

    Returns
    -------
    Tuple[List[str], List[str], pd.DataFrame]
        - low_variance_cols: Features with variance < threshold.
        - active_cols: Features with variance >= threshold.
        - summary_df: DataFrame with variance, std, min, max per feature.
    """
    variances = df[candidate_cols].var()
    stds = df[candidate_cols].std()
    mins = df[candidate_cols].min()
    maxs = df[candidate_cols].max()

    summary_df = pd.DataFrame({
        "feature": candidate_cols,
        "variance": variances.values,
        "std": stds.values,
        "min": mins.values,
        "max": maxs.values,
    }).sort_values("variance").reset_index(drop=True)

    low_variance_cols = summary_df[summary_df["variance"] < variance_threshold]["feature"].tolist()
    active_cols = summary_df[summary_df["variance"] >= variance_threshold]["feature"].tolist()

    return low_variance_cols, active_cols, summary_df


def select_informative_features(
    df: pd.DataFrame,
    variance_threshold: float = 1e-5,
    include_operational_settings: bool = True,
    include_time_cycles: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Defensibly filters out uninformative (zero/low variance) sensors and settings.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw sensor and telemetry channels.
    variance_threshold : float, default=1e-5
        Threshold for near-zero variance detection.
    include_operational_settings : bool, default=True
        Whether to retain operational settings that pass the variance check.
    include_time_cycles : bool, default=True
        Whether to retain 'time_cycles' as an available telemetry feature.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str]]
        - processed_df: DataFrame with uninformative channels dropped.
        - selected_features: List of retained predictive feature names.
        - removed_features: List of removed column names.
    """
    all_sensors = [f"sensor_{i}" for i in range(1, 22)]
    low_var_sensors, active_sensors, _ = identify_low_variance_features(
        df, all_sensors, variance_threshold=variance_threshold
    )

    removed_cols = list(low_var_sensors)
    selected_features = list(active_sensors)

    # Evaluate operational settings
    setting_cols = ["operational_setting_1", "operational_setting_2", "operational_setting_3"]
    low_var_settings, active_settings, _ = identify_low_variance_features(
        df, setting_cols, variance_threshold=variance_threshold
    )
    removed_cols.extend(low_var_settings)

    if include_operational_settings:
        selected_features.extend(active_settings)
    else:
        removed_cols.extend(active_settings)

    if include_time_cycles and "time_cycles" in df.columns:
        selected_features.insert(0, "time_cycles")

    # Columns to retain in the returned DataFrame: unit_number, RUL (if present), plus selected_features
    meta_cols = [col for col in ["unit_number", "RUL"] if col in df.columns]
    keep_cols = meta_cols + selected_features
    processed_df = df[keep_cols].copy()

    return processed_df, selected_features, removed_cols


def separate_features_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RUL",
    id_col: str = "unit_number",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Separates feature matrix X, target series y, and engine unit grouping identifiers.

    Strictly satisfies:
    1. Target column (RUL) is never in the feature set X.
    2. Grouping identifier (unit_number) is never in feature set X, preventing leakage.
    3. Grouping identifier is returned separately for group-aware train/validation splitting.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset containing features, target, and engine identifiers.
    feature_cols : List[str]
        Explicit list of feature column names to extract into X.
    target_col : str, default='RUL'
        Target variable name.
    id_col : str, default='unit_number'
        Engine ID grouping column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.Series]
        - X : pd.DataFrame (predictive features only)
        - y : pd.Series (target RUL)
        - unit_ids : pd.Series (engine unit numbers for grouped cross-validation)
    """
    # Defensive checks
    if target_col in feature_cols:
        raise ValueError(f"Data leakage risk! Target '{target_col}' cannot be an input feature.")
    if id_col in feature_cols:
        raise ValueError(f"Identifier '{id_col}' cannot be an input ML feature.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    unit_ids = df[id_col].copy()

    return X, y, unit_ids


def get_feature_removal_reasons() -> Dict[str, str]:
    """
    Returns the dictionary of engineering and physics-informed justifications
    for all removed uninformative features.
    """
    return FEATURE_REMOVAL_JUSTIFICATIONS.copy()


# Standard canonical ordering of the 15 selected predictive features
CANONICAL_15_FEATURES: List[str] = [
    "time_cycles",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]


def prepare_model_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    model_type: str,
    fold: Optional[int] = None,
    scaler_dir: Optional[Union[str, Path]] = "models/scalers",
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]:
    """
    Executes model-specific feature preprocessing adhering strictly to zero data leakage.

    Preprocessing Rules:
    1. Linear Regression ('linear', 'linear_regression', 'ridge', 'lasso'):
       - Fits StandardScaler strictly on training fold (X_train).
       - Transforms validation fold (X_val) using the fitted scaler.
       - Never fits or calculates statistics on X_val.
       - Optionally serializes the fitted scaler artifact via joblib.
    2. Random Forest ('random_forest', 'rf'):
       - Uses original unscaled numerical features (tree algorithms are invariant
         to monotonic scale transformations).
    3. XGBoost ('xgboost', 'xgb'):
       - Uses original unscaled numerical features.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training fold feature matrix.
    X_val : pd.DataFrame
        Validation fold feature matrix.
    model_type : str
        Algorithm family: 'linear', 'random_forest', or 'xgboost'.
    fold : Optional[int], default=None
        Fold index (1..K) used for naming serialized scaler checkpoints.
    scaler_dir : Optional[Union[str, Path]], default='models/scalers'
        Directory path to persist fitted StandardScaler artifacts via joblib.
        If None, serialization is skipped.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]
        - X_train_prep: Preprocessed training features DataFrame (original index & columns preserved).
        - X_val_prep: Preprocessed validation features DataFrame (original index & columns preserved).
        - fitted_scaler: The fitted StandardScaler instance for linear models, or None for tree models.

    Raises
    ------
    ValueError
        If unrecognized model_type is supplied, or column mismatch occurs.
    """
    # Defensive checks on column consistency
    if list(X_train.columns) != list(X_val.columns):
        raise ValueError(
            f"Feature column mismatch between training and validation folds!\n"
            f"X_train columns: {list(X_train.columns)}\n"
            f"X_val columns:   {list(X_val.columns)}"
        )

    norm_type = model_type.strip().lower()

    # 1. Linear Regression: Requires Standardization
    if norm_type in {"linear", "linear_regression", "linearregression", "ridge", "lasso"}:
        scaler = StandardScaler()
        # FIT STRICTLY ON TRAINING DATA ONLY (Zero Leakage)
        scaler.fit(X_train)

        # Transform both splits using parameters learned strictly from train
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )

        # Store preprocessing artifacts using joblib if directory is configured
        if scaler_dir is not None:
            save_path = Path(scaler_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            fold_suffix = f"_fold_{fold}" if fold is not None else ""
            artifact_file = save_path / f"scaler_{norm_type}{fold_suffix}.joblib"
            joblib.dump(scaler, artifact_file)

        return X_train_scaled, X_val_scaled, scaler

    # 2. Random Forest: Unscaled Numerical Features
    elif norm_type in {"rf", "random_forest", "randomforest", "randomforestregressor"}:
        return X_train.copy(), X_val.copy(), None

    # 3. XGBoost: Unscaled Numerical Features
    elif norm_type in {"xgb", "xgboost", "xgbregressor"}:
        return X_train.copy(), X_val.copy(), None

    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Expected one of: "
            f"'linear', 'random_forest', 'xgboost'."
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_train_data

    print("Testing src/preprocessing.py...")
    raw_train = load_train_data()

    # Test RUL target engineering
    train_with_rul = create_rul_target(raw_train)
    print(f"RUL created! New column present: {'RUL' in train_with_rul.columns}")
    print(f"Original DataFrame unmodified: {'RUL' not in raw_train.columns}")

    # Test feature selection
    proc_df, selected, removed = select_informative_features(train_with_rul)
    print(f"Initial raw columns:      {raw_train.shape[1]}")
    print(f"Removed low-var channels: {len(removed)} -> {removed}")
    print(f"Selected feature count:   {len(selected)} -> {selected}")

    # Test separation
    X, y, units = separate_features_target(proc_df, feature_cols=selected)
    print(f"Feature matrix X shape:   {X.shape}")
    print(f"Target vector y shape:    {y.shape}")
    print(f"Grouping vector units:    {units.nunique()} engines")

    # Test model-specific preprocessing
    dummy_train = X.iloc[:1000]
    dummy_val = X.iloc[1000:1200]
    X_tr_s, X_v_s, scaler = prepare_model_features(dummy_train, dummy_val, model_type="linear", fold=1)
    print(f"Linear prep scaled mean:  {X_tr_s.mean().abs().max():.6f} (expected ~0.0)")
    print(f"Linear prep scaled std:   {X_tr_s.std().mean():.6f} (expected ~1.0)")
    X_tr_rf, X_v_rf, _ = prepare_model_features(dummy_train, dummy_val, model_type="random_forest")
    print(f"RF prep unscaled match:   {np.allclose(X_tr_rf.values, dummy_train.values)}")
    X_tr_xgb, X_v_xgb, _ = prepare_model_features(dummy_train, dummy_val, model_type="xgboost")
    print(f"XGB prep unscaled match:  {np.allclose(X_tr_xgb.values, dummy_train.values)}")
    print("All preprocessing sanity checks passed successfully!")
