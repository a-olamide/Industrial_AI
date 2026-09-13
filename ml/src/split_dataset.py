"""Group-aware train / validation / test partitioning.

Windows extracted from the same CWRU recording are highly correlated —
adjacent 2048-sample slices share bearing dynamics, load state and
often the same rotation. Random per-row splitting therefore leaks
information from train into test and inflates apparent accuracy.

We partition by ``recording_id`` (the group) so every window from a
given experiment lands in exactly one split. Concrete split ratios are
left as caller parameters because the 16 initial recordings are unevenly
distributed across four classes and four load levels; stratified
recording-level splitting is only feasible once the actual class/group
counts are known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


GROUP_COLUMN = "recording_id"
LABEL_COLUMN = "fault_class"
LOAD_COLUMN = "motor_load_hp"


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def group_train_val_test_split(
    frame: pd.DataFrame,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    group_column: str = GROUP_COLUMN,
) -> SplitFrames:
    """Split ``frame`` into train/validation/test partitions by group.

    All rows sharing the same value of ``group_column`` stay in one
    split. A small dataset warning is raised when the number of unique
    groups is too low to honor the requested ratios (e.g. only one
    recording per class).
    """
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1 (exclusive)")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be < 1.0")
    if group_column not in frame.columns:
        raise KeyError(f"missing group column {group_column!r} in frame")

    unique_groups = frame[group_column].unique()
    if len(unique_groups) < 3:
        raise ValueError(
            f"need at least 3 distinct groups for a train/val/test split; "
            f"got {len(unique_groups)}"
        )

    holdout_size = validation_size + test_size
    first = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=random_state)
    train_idx, holdout_idx = next(first.split(frame, groups=frame[group_column]))
    train_df = frame.iloc[train_idx].reset_index(drop=True)
    holdout_df = frame.iloc[holdout_idx].reset_index(drop=True)

    relative_test = test_size / holdout_size
    second = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=random_state)
    val_idx, test_idx = next(second.split(holdout_df, groups=holdout_df[group_column]))
    val_df = holdout_df.iloc[val_idx].reset_index(drop=True)
    test_df = holdout_df.iloc[test_idx].reset_index(drop=True)

    return SplitFrames(train=train_df, validation=val_df, test=test_df)


def summarize_split(split: SplitFrames, label_column: str = LABEL_COLUMN) -> pd.DataFrame:
    """Return per-split counts of rows, unique groups, and class balance."""
    rows = []
    for name, frame in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        row: dict[str, object] = {
            "split": name,
            "n_rows": int(len(frame)),
            "n_groups": int(frame[GROUP_COLUMN].nunique()) if len(frame) else 0,
        }
        if label_column in frame.columns and len(frame):
            counts = frame[label_column].value_counts().to_dict()
            for cls, cnt in counts.items():
                row[f"n_{cls}"] = int(cnt)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def load_aware_split(
    frame: pd.DataFrame,
    train_loads: Iterable[float],
    validation_loads: Iterable[float],
    test_loads: Iterable[float],
    load_column: str = LOAD_COLUMN,
    group_column: str = GROUP_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> SplitFrames:
    """Partition rows by explicit motor-load values.

    Every recording belongs to exactly one motor-load level, so slicing
    by ``motor_load_hp`` is equivalent to grouping by ``recording_id``.
    This function additionally asserts that:

    - the three load sets are disjoint and non-empty
    - no ``recording_id`` appears in more than one split
    - every fault class present in the input appears in each split
    """
    train_set = {float(v) for v in train_loads}
    val_set = {float(v) for v in validation_loads}
    test_set = {float(v) for v in test_loads}
    if not train_set or not val_set or not test_set:
        raise ValueError("train/validation/test load sets must each be non-empty")
    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlap:
        raise ValueError(f"load values must be disjoint across splits; overlap={sorted(overlap)}")

    for col in (load_column, group_column, label_column):
        if col not in frame.columns:
            raise KeyError(f"missing required column {col!r}")

    loads = frame[load_column].astype(float)
    train_df = frame[loads.isin(train_set)].reset_index(drop=True)
    val_df = frame[loads.isin(val_set)].reset_index(drop=True)
    test_df = frame[loads.isin(test_set)].reset_index(drop=True)

    train_groups = set(train_df[group_column].unique())
    val_groups = set(val_df[group_column].unique())
    test_groups = set(test_df[group_column].unique())
    group_overlap = (
        (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    )
    assert not group_overlap, f"recording_id leaked across splits: {sorted(group_overlap)}"

    all_classes = set(frame[label_column].unique())
    for name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        classes_in_split = set(df[label_column].unique())
        missing = all_classes - classes_in_split
        assert not missing, f"split {name!r} is missing classes: {sorted(missing)}"

    return SplitFrames(train=train_df, validation=val_df, test=test_df)


__all__ = [
    "GROUP_COLUMN",
    "LABEL_COLUMN",
    "LOAD_COLUMN",
    "SplitFrames",
    "group_train_val_test_split",
    "load_aware_split",
    "summarize_split",
]
