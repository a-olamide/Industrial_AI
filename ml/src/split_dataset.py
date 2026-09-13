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

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


GROUP_COLUMN = "recording_id"
LABEL_COLUMN = "fault_class"


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


__all__ = [
    "GROUP_COLUMN",
    "LABEL_COLUMN",
    "SplitFrames",
    "group_train_val_test_split",
    "summarize_split",
]
