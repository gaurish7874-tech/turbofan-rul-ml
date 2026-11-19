"""
Engine-Level GroupKFold Splitting Module for Turbofan RUL Prediction.
NASA C-MAPSS FD001 Dataset.

This module enforces strict zero-data-leakage cross-validation by partitioning
engines via GroupKFold on `unit_number` (never random row-level splitting).
"""

from typing import List, Tuple, Dict, Set, Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def verify_zero_engine_overlap(
    train_units: Union[pd.Series, Set[int], List[int]],
    val_units: Union[pd.Series, Set[int], List[int]],
    fold_num: Optional[int] = None,
) -> bool:
    """
    Verifies that no engine unit ID appears in both training and validation sets.

    Parameters
    ----------
    train_units : Union[pd.Series, Set[int], List[int]]
        Engine unit IDs in the training fold.
    val_units : Union[pd.Series, Set[int], List[int]]
        Engine unit IDs in the validation fold.
    fold_num : Optional[int], default=None
        Fold identifier for error reporting.

    Returns
    -------
    bool
        True if zero overlap is verified.

    Raises
    ------
    ValueError
        If any engine unit ID is shared between train and validation.
    """
    set_train = set(train_units)
    set_val = set(val_units)
    overlap = set_train.intersection(set_val)

    if overlap:
        fold_str = f" in fold {fold_num}" if fold_num is not None else ""
        raise ValueError(
            f"CRITICAL DATA LEAKAGE{fold_str}! Found {len(overlap)} overlapping engine(s): {overlap}"
        )
    return True


def create_engine_group_splits(
    X: pd.DataFrame,
    y: pd.Series,
    unit_ids: pd.Series,
    n_splits: int = 5,
    verbose: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partitions dataset into K folds strictly grouped by engine unit_number.

    Guarantees:
    1. Every engine appears in the validation set of exactly one fold.
    2. No engine is ever shared between train and validation in any fold.
    3. Original DataFrame indices are preserved.
    4. unit_number is used strictly as a grouping variable, never as a model feature.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix (must not contain unit_number or RUL).
    y : pd.Series
        Target RUL series.
    unit_ids : pd.Series
        Engine identifier series (unit_number).
    n_splits : int, default=5
        Number of cross-validation folds.
    verbose : bool, default=True
        Whether to print fold metrics and overlap verifications.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) integer arrays for each fold.
    """
    if len(X) != len(y) or len(X) != len(unit_ids):
        raise ValueError(
            f"Dimension mismatch: X has {len(X)} rows, y has {len(y)}, "
            f"unit_ids has {len(unit_ids)}."
        )

    if "unit_number" in X.columns:
        raise ValueError("Data leakage risk! 'unit_number' must not be in feature matrix X.")

    gkf = GroupKFold(n_splits=n_splits)
    splits = []

    if verbose:
        print("=" * 80)
        print(f"      ENGINE-LEVEL GROUPKFOLD CROSS-VALIDATION (n_splits={n_splits})")
        print("=" * 80)
        print(f"Total Observations: {len(X):,} rows across {unit_ids.nunique()} unique engines")
        print("-" * 80)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=unit_ids), start=1):
        train_engines = set(unit_ids.iloc[train_idx])
        val_engines = set(unit_ids.iloc[val_idx])

        # Assert zero overlap
        verify_zero_engine_overlap(train_engines, val_engines, fold_num=fold)

        splits.append((train_idx, val_idx))

        if verbose:
            print(
                f"Fold {fold}: "
                f"Train Engines={len(train_engines):2d} | Val Engines={len(val_engines):2d} | "
                f"Train Rows={len(train_idx):5,d} | Val Rows={len(val_idx):5,d} | "
                f"Overlap=0 (VERIFIED)"
            )

    if verbose:
        print("-" * 80)
        print("All folds successfully verified for strict zero engine overlap.")
        print("=" * 80)

    return splits


def get_fold_data(
    X: pd.DataFrame,
    y: pd.Series,
    unit_ids: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Subsets feature matrix, target, and engine IDs for a single fold,
    preserving original indices and column names.

    Parameters
    ----------
    X : pd.DataFrame
        Predictive feature matrix.
    y : pd.Series
        Target vector.
    unit_ids : pd.Series
        Engine identifiers.
    train_idx : np.ndarray
        Integer position indices for training fold.
    val_idx : np.ndarray
        Integer position indices for validation fold.

    Returns
    -------
    Dict[str, Union[pd.DataFrame, pd.Series]]
        Dictionary containing 'X_train', 'y_train', 'unit_ids_train',
        'X_val', 'y_val', 'unit_ids_val'.
    """
    return {
        "X_train": X.iloc[train_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "unit_ids_train": unit_ids.iloc[train_idx].copy(),
        "X_val": X.iloc[val_idx].copy(),
        "y_val": y.iloc[val_idx].copy(),
        "unit_ids_val": unit_ids.iloc[val_idx].copy(),
    }


if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader import load_train_data
    from src.preprocessing import (
        create_rul_target,
        select_informative_features,
        separate_features_target,
    )

    print("Testing src/splitting.py...")
    raw_train = load_train_data()
    train_with_rul = create_rul_target(raw_train)
    proc_df, selected_feats, _ = select_informative_features(
        train_with_rul,
        variance_threshold=1e-5,
        include_operational_settings=False,
        include_time_cycles=True,
    )
    X, y, unit_ids = separate_features_target(proc_df, feature_cols=selected_feats)

    splits = create_engine_group_splits(X, y, unit_ids, n_splits=5, verbose=True)
    fold_1_data = get_fold_data(X, y, unit_ids, splits[0][0], splits[0][1])
    print(f"Fold 1 X_train shape: {fold_1_data['X_train'].shape}")
    print(f"Fold 1 X_val shape:   {fold_1_data['X_val'].shape}")
    print("Zero overlap and index preservation successfully confirmed!")
