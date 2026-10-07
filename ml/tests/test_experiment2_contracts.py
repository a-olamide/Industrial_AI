"""Contract tests for the Experiment-2 multi-severity split and X/y design.

Run from the project root with the stdlib test runner (no pytest
dependency is required by ``ml/requirements.txt``)::

    python -m unittest discover -s ml/tests -t . -v

Most assertions run against a SYNTHETIC feature frame built from the
real :data:`CWRU_RECORDINGS` metadata, so the suite passes on a clean
checkout without the vendor ``.mat`` downloads. The handful of checks
that need the real expanded CSV skip themselves when it is absent.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.dataset_builder import (
    BASELINE_SPECS,
    CWRU_RECORDINGS,
    FAULT_CLASSES,
    FAULT_SEVERITIES_IN,
)
from ml.src.experiment2_multiseverity import (
    EXCLUDED_FROM_X,
    EXPANDED_CSV,
    TEST_LOADS,
    TRAIN_LOADS,
    VALIDATION_LOADS,
)
from ml.src.split_dataset import (
    GROUP_COLUMN,
    LABEL_COLUMN,
    LOAD_COLUMN,
    SEVERITY_COLUMN,
    WINDOW_COLUMN,
    multiseverity_load_split,
)
from ml.src.train_baseline import FEATURE_COLUMNS, LABEL_COLUMN as TRAIN_LABEL_COLUMN


WINDOWS_PER_RECORDING = 6


def synthetic_frame(specs=CWRU_RECORDINGS, n_windows: int = WINDOWS_PER_RECORDING) -> pd.DataFrame:
    """Build a schema-faithful feature frame without touching raw data."""
    rng = np.random.default_rng(0)
    rows = []
    for spec in specs:
        for window_id in range(n_windows):
            rows.append(
                {
                    "source": "CWRU_12kHz_DE",
                    "asset_id": "cwru_bearing_12kDE",
                    GROUP_COLUMN: spec.recording_id,
                    WINDOW_COLUMN: window_id,
                    "vibration_rms": float(rng.uniform(0.05, 0.8)),
                    "vibration_std": float(rng.uniform(0.05, 0.8)),
                    "vibration_peak": float(rng.uniform(0.2, 7.0)),
                    "vibration_peak_to_peak": float(rng.uniform(0.3, 13.0)),
                    "vibration_kurtosis": float(rng.uniform(-1.0, 33.0)),
                    "vibration_skewness": float(rng.uniform(-0.5, 0.8)),
                    "crest_factor": float(rng.uniform(2.5, 12.0)),
                    "rotational_speed_rpm": float(spec.approx_rpm),
                    LOAD_COLUMN: float(spec.motor_load_hp),
                    LABEL_COLUMN: spec.fault_class,
                    SEVERITY_COLUMN: spec.fault_severity_in,
                    "sampling_rate_hz": int(spec.sampling_rate_hz),
                }
            )
    return pd.DataFrame(rows)


class SplitContractTests(unittest.TestCase):
    """TASK 11 - no leakage, expected loads, class + severity coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = synthetic_frame()
        cls.split, cls.audit = multiseverity_load_split(
            cls.frame,
            train_loads=TRAIN_LOADS,
            validation_loads=VALIDATION_LOADS,
            test_loads=TEST_LOADS,
            expected_severities=FAULT_SEVERITIES_IN,
        )

    def _named(self):
        return (
            ("train", self.split.train),
            ("validation", self.split.validation),
            ("test", self.split.test),
        )

    def test_no_recording_leaks_across_splits(self):
        groups = {name: set(df[GROUP_COLUMN]) for name, df in self._named()}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        # Union must still cover every recording - nothing silently dropped.
        covered = groups["train"] | groups["validation"] | groups["test"]
        self.assertEqual(covered, {s.recording_id for s in CWRU_RECORDINGS})

    def test_no_feature_window_crosses_splits(self):
        keys = {
            name: set(zip(df[GROUP_COLUMN], df[WINDOW_COLUMN]))
            for name, df in self._named()
        }
        self.assertFalse(keys["train"] & keys["validation"])
        self.assertFalse(keys["train"] & keys["test"])
        self.assertFalse(keys["validation"] & keys["test"])

    def test_expected_split_loads(self):
        expected = {
            "train": {float(v) for v in TRAIN_LOADS},
            "validation": {float(v) for v in VALIDATION_LOADS},
            "test": {float(v) for v in TEST_LOADS},
        }
        self.assertEqual(expected["train"], {0.0, 1.0})
        self.assertEqual(expected["validation"], {2.0})
        self.assertEqual(expected["test"], {3.0})
        for name, df in self._named():
            self.assertEqual(set(df[LOAD_COLUMN].astype(float)), expected[name])

    def test_every_split_contains_all_four_classes(self):
        for name, df in self._named():
            self.assertEqual(
                set(df[LABEL_COLUMN]), set(FAULT_CLASSES), msg=f"split={name}"
            )

    def test_every_split_contains_all_three_severities(self):
        for name, df in self._named():
            faults = df[df[LABEL_COLUMN] != "NORMAL"]
            present = {round(float(v), 3) for v in faults[SEVERITY_COLUMN]}
            self.assertEqual(
                present,
                {round(float(s), 3) for s in FAULT_SEVERITIES_IN},
                msg=f"split={name}",
            )

    def test_normal_rows_carry_no_severity(self):
        normal = self.frame[self.frame[LABEL_COLUMN] == "NORMAL"]
        self.assertTrue(normal[SEVERITY_COLUMN].isna().all())

    def test_missing_severity_is_rejected(self):
        """Dropping a severity from one load must fail the split, not pass quietly."""
        crippled = self.frame[
            ~(
                (self.frame[LOAD_COLUMN] == 2.0)
                & (self.frame[SEVERITY_COLUMN] == 0.021)
            )
        ]
        with self.assertRaises(AssertionError):
            multiseverity_load_split(
                crippled,
                train_loads=TRAIN_LOADS,
                validation_loads=VALIDATION_LOADS,
                test_loads=TEST_LOADS,
                expected_severities=FAULT_SEVERITIES_IN,
            )

    def test_overlapping_load_sets_are_rejected(self):
        with self.assertRaises(ValueError):
            multiseverity_load_split(
                self.frame,
                train_loads=(0.0, 1.0),
                validation_loads=(1.0,),
                test_loads=(3.0,),
                expected_severities=FAULT_SEVERITIES_IN,
            )

    def test_audit_counts_match_the_frames(self):
        for name, df in self._named():
            self.assertEqual(self.audit.window_counts[name], len(df))
            self.assertEqual(
                set(self.audit.recordings[name]), set(df[GROUP_COLUMN].unique())
            )


class FeatureContractTests(unittest.TestCase):
    """TASK 11 - fault_severity_in and provenance metadata stay out of X."""

    def test_fault_severity_is_not_a_model_input(self):
        self.assertNotIn(SEVERITY_COLUMN, FEATURE_COLUMNS)
        self.assertIn(SEVERITY_COLUMN, EXCLUDED_FROM_X)

    def test_metadata_columns_are_not_model_inputs(self):
        for column in (
            "recording_id",
            "window_id",
            "source",
            "source_file",
            "asset_id",
            "sampling_rate_hz",
        ):
            self.assertNotIn(column, FEATURE_COLUMNS, msg=column)
            self.assertIn(column, EXCLUDED_FROM_X, msg=column)

    def test_label_is_not_a_model_input(self):
        self.assertNotIn(TRAIN_LABEL_COLUMN, FEATURE_COLUMNS)
        self.assertIn(TRAIN_LABEL_COLUMN, EXCLUDED_FROM_X)

    def test_excluded_and_feature_sets_are_disjoint(self):
        self.assertFalse(set(FEATURE_COLUMNS) & set(EXCLUDED_FROM_X))

    def test_feature_contract_is_the_experiment1_contract(self):
        self.assertEqual(
            tuple(FEATURE_COLUMNS),
            (
                "vibration_rms",
                "vibration_std",
                "vibration_peak",
                "vibration_peak_to_peak",
                "vibration_kurtosis",
                "vibration_skewness",
                "crest_factor",
                "rotational_speed_rpm",
                "motor_load_hp",
            ),
        )

    def test_built_x_contains_only_contract_columns(self):
        frame = synthetic_frame()
        X = frame[list(FEATURE_COLUMNS)]
        self.assertEqual(list(X.columns), list(FEATURE_COLUMNS))
        for column in EXCLUDED_FROM_X:
            self.assertNotIn(column, X.columns)


class BaselineFreezeTests(unittest.TestCase):
    """TASK 11 - Experiment 1 still sees exactly its original 16 recordings."""

    EXPECTED_BASELINE_IDS = {
        "Normal_0", "Normal_1", "Normal_2", "Normal_3",
        "IR007_0", "IR007_1", "IR007_2", "IR007_3",
        "B007_0", "B007_1", "B007_2", "B007_3",
        "OR007@6_0", "OR007@6_1", "OR007@6_2", "OR007@6_3",
    }

    def test_baseline_has_exactly_sixteen_recordings(self):
        self.assertEqual(len(BASELINE_SPECS), 16)

    def test_baseline_recording_ids_are_unchanged(self):
        self.assertEqual(
            {s.recording_id for s in BASELINE_SPECS}, self.EXPECTED_BASELINE_IDS
        )

    def test_baseline_covers_only_normal_and_0007_inch(self):
        severities = {s.fault_severity_in for s in BASELINE_SPECS}
        self.assertEqual(severities, {None, 0.007})

    def test_baseline_is_four_classes_by_four_loads(self):
        counts: dict[str, set] = {}
        for spec in BASELINE_SPECS:
            counts.setdefault(spec.fault_class, set()).add(spec.motor_load_hp)
        self.assertEqual(set(counts), set(FAULT_CLASSES))
        for cls, loads in counts.items():
            self.assertEqual(loads, {0, 1, 2, 3}, msg=cls)

    def test_expanded_table_is_forty_recordings(self):
        self.assertEqual(len(CWRU_RECORDINGS), 40)
        self.assertEqual(len({s.recording_id for s in CWRU_RECORDINGS}), 40)

    def test_baseline_is_a_subset_of_the_expanded_table(self):
        self.assertTrue(set(BASELINE_SPECS).issubset(set(CWRU_RECORDINGS)))


class RealDatasetTests(unittest.TestCase):
    """Checks against the actual expanded CSV; skipped when it is absent."""

    @classmethod
    def setUpClass(cls) -> None:
        if not Path(EXPANDED_CSV).is_file():
            raise unittest.SkipTest(
                f"{EXPANDED_CSV} not present - run "
                "`python -m ml.src.audit_expanded_dataset` first"
            )
        cls.frame = pd.read_csv(EXPANDED_CSV)

    def test_real_split_satisfies_every_experiment2_assertion(self):
        split, audit = multiseverity_load_split(
            self.frame,
            train_loads=TRAIN_LOADS,
            validation_loads=VALIDATION_LOADS,
            test_loads=TEST_LOADS,
            expected_severities=FAULT_SEVERITIES_IN,
        )
        self.assertEqual(len(audit.recordings["train"]), 20)
        self.assertEqual(len(audit.recordings["validation"]), 10)
        self.assertEqual(len(audit.recordings["test"]), 10)
        total = sum(audit.window_counts.values())
        self.assertEqual(total, len(self.frame))
        for name in ("train", "validation", "test"):
            self.assertEqual(set(audit.class_counts[name]), set(FAULT_CLASSES))

    def test_real_frame_exposes_every_contract_column(self):
        for column in FEATURE_COLUMNS:
            self.assertIn(column, self.frame.columns)
        self.assertIn(SEVERITY_COLUMN, self.frame.columns)
        self.assertEqual(self.frame[GROUP_COLUMN].nunique(), 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
