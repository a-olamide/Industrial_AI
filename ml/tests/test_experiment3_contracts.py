"""Contract tests for the Experiment-3 unsupervised anomaly detector.

Run from the project root::

    python -m unittest discover -s ml/tests -t . -v

As with the Experiment-2 suite, most checks run against a SYNTHETIC
feature frame built from the real :data:`CWRU_RECORDINGS` metadata, so
they pass on a clean checkout with no vendor ``.mat`` downloads.

The central guarantee under test is that the anomaly detector never
sees a fault label or a fault window at ``fit()`` time, and that the
frozen Experiment-1 and Experiment-2 contracts are unchanged.
"""

from __future__ import annotations

import contextlib
import io
import unittest

import pandas as pd

from ml.src.dataset_builder import (
    BASELINE_SPECS,
    CWRU_RECORDINGS,
    FAULT_CLASSES,
    FAULT_SEVERITIES_IN,
)
from ml.src.experiment2_multiseverity import (
    EXCLUDED_FROM_X as EXP2_EXCLUDED_FROM_X,
    TEST_LOADS as EXP2_TEST_LOADS,
    TRAIN_LOADS as EXP2_TRAIN_LOADS,
    VALIDATION_LOADS as EXP2_VALIDATION_LOADS,
)
from ml.src.experiment3_anomaly_detection import (
    ANOMALY_FEATURE_COLUMNS,
    DROPPED_OPERATING_POINT_COLUMNS,
    EXCLUDED_FROM_X,
    NORMAL_LABEL,
    TEST_LOADS,
    TRAIN_LOADS,
    VALIDATION_LOADS,
    build_anomaly_splits,
    classify,
    fit_isolation_forest,
    score,
)
from ml.src.split_dataset import (
    GROUP_COLUMN,
    LABEL_COLUMN,
    LOAD_COLUMN,
    SEVERITY_COLUMN,
    WINDOW_COLUMN,
)
from ml.src.train_baseline import FEATURE_COLUMNS as SUPERVISED_FEATURE_COLUMNS

from .test_experiment2_contracts import synthetic_frame


@contextlib.contextmanager
def quiet():
    """Swallow the experiment module's progress output during tests."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class AnomalySplitTests(unittest.TestCase):
    """Split loads, recording disjointness, and NORMAL-only training data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = synthetic_frame()
        with quiet():
            cls.splits = build_anomaly_splits(cls.frame)

    def test_fit_frame_contains_only_normal_windows(self):
        self.assertEqual(
            set(self.splits.train_normal[LABEL_COLUMN]), {NORMAL_LABEL}
        )
        self.assertGreater(len(self.splits.train_normal), 0)

    def test_fault_windows_at_training_loads_are_excluded(self):
        train_loads = {float(v) for v in TRAIN_LOADS}
        all_at_train_loads = self.frame[
            self.frame[LOAD_COLUMN].astype(float).isin(train_loads)
        ]
        n_faults = int((all_at_train_loads[LABEL_COLUMN] != NORMAL_LABEL).sum())
        self.assertEqual(self.splits.unused_train_load_faults, n_faults)
        self.assertGreater(n_faults, 0, "fixture should contain faults at 0/1 HP")

    def test_expected_split_loads(self):
        self.assertEqual({float(v) for v in TRAIN_LOADS}, {0.0, 1.0})
        self.assertEqual({float(v) for v in VALIDATION_LOADS}, {2.0})
        self.assertEqual({float(v) for v in TEST_LOADS}, {3.0})
        self.assertEqual(
            set(self.splits.train_normal[LOAD_COLUMN].astype(float)), {0.0, 1.0}
        )
        self.assertEqual(
            set(self.splits.validation[LOAD_COLUMN].astype(float)), {2.0}
        )
        self.assertEqual(set(self.splits.test[LOAD_COLUMN].astype(float)), {3.0})

    def test_no_recording_crosses_splits(self):
        groups = {
            "train": set(self.splits.train_normal[GROUP_COLUMN]),
            "validation": set(self.splits.validation[GROUP_COLUMN]),
            "test": set(self.splits.test[GROUP_COLUMN]),
        }
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])

    def test_no_window_crosses_splits(self):
        keys = {
            name: set(zip(df[GROUP_COLUMN], df[WINDOW_COLUMN]))
            for name, df in (
                ("train", self.splits.train_normal),
                ("validation", self.splits.validation),
                ("test", self.splits.test),
            )
        }
        self.assertFalse(keys["train"] & keys["validation"])
        self.assertFalse(keys["train"] & keys["test"])
        self.assertFalse(keys["validation"] & keys["test"])

    def test_evaluation_splits_carry_normal_and_every_fault_class(self):
        for name, df in (
            ("validation", self.splits.validation),
            ("test", self.splits.test),
        ):
            self.assertEqual(set(df[LABEL_COLUMN]), set(FAULT_CLASSES), msg=name)

    def test_evaluation_splits_carry_every_severity(self):
        for name, df in (
            ("validation", self.splits.validation),
            ("test", self.splits.test),
        ):
            faults = df[df[LABEL_COLUMN] != NORMAL_LABEL]
            self.assertEqual(
                {round(float(v), 3) for v in faults[SEVERITY_COLUMN]},
                {round(float(s), 3) for s in FAULT_SEVERITIES_IN},
                msg=name,
            )


class FitGuardTests(unittest.TestCase):
    """fit() must refuse anything but NORMAL windows."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = synthetic_frame()
        with quiet():
            cls.splits = build_anomaly_splits(cls.frame)

    def test_fit_accepts_normal_only_frame(self):
        with quiet():
            model = fit_isolation_forest(self.splits.train_normal)
        self.assertTrue(hasattr(model, "score_samples"))

    def test_fit_rejects_a_frame_containing_faults(self):
        faults = self.splits.validation[
            self.splits.validation[LABEL_COLUMN] != NORMAL_LABEL
        ].head(5)
        self.assertEqual(len(faults), 5, "fixture must supply genuine fault rows")
        contaminated = pd.concat(
            [self.splits.train_normal, faults], ignore_index=True
        )
        with self.assertRaises(ValueError), quiet():
            fit_isolation_forest(contaminated)

    def test_fitted_estimator_saw_only_the_vibration_columns(self):
        with quiet():
            model = fit_isolation_forest(self.splits.train_normal)
        self.assertEqual(
            list(model.feature_names_in_), list(ANOMALY_FEATURE_COLUMNS)
        )
        self.assertEqual(
            int(model.named_steps["detector"].n_features_in_),
            len(ANOMALY_FEATURE_COLUMNS),
        )

    def test_scores_are_finite_and_classification_is_binary(self):
        with quiet():
            model = fit_isolation_forest(self.splits.train_normal)
        scores = score(model, self.splits.test)
        self.assertEqual(len(scores), len(self.splits.test))
        self.assertTrue(pd.notna(scores).all())
        flags = set(classify(scores, float(scores.mean())))
        self.assertTrue(flags.issubset({"NORMAL", "ANOMALOUS"}))


class AnomalyFeatureContractTests(unittest.TestCase):
    """Labels and metadata must never be anomaly features."""

    def test_fault_class_is_not_a_feature(self):
        self.assertNotIn(LABEL_COLUMN, ANOMALY_FEATURE_COLUMNS)
        self.assertIn(LABEL_COLUMN, EXCLUDED_FROM_X)

    def test_fault_severity_is_not_a_feature(self):
        self.assertNotIn(SEVERITY_COLUMN, ANOMALY_FEATURE_COLUMNS)
        self.assertIn(SEVERITY_COLUMN, EXCLUDED_FROM_X)

    def test_metadata_columns_are_not_features(self):
        for column in (
            "recording_id",
            "window_id",
            "source",
            "source_file",
            "asset_id",
            "sampling_rate_hz",
        ):
            self.assertNotIn(column, ANOMALY_FEATURE_COLUMNS, msg=column)
            self.assertIn(column, EXCLUDED_FROM_X, msg=column)

    def test_excluded_and_feature_sets_are_disjoint(self):
        self.assertFalse(set(ANOMALY_FEATURE_COLUMNS) & set(EXCLUDED_FROM_X))

    def test_operating_point_columns_are_dropped(self):
        self.assertEqual(
            set(DROPPED_OPERATING_POINT_COLUMNS),
            {"rotational_speed_rpm", "motor_load_hp"},
        )
        for column in DROPPED_OPERATING_POINT_COLUMNS:
            self.assertNotIn(column, ANOMALY_FEATURE_COLUMNS, msg=column)

    def test_anomaly_features_are_the_seven_vibration_statistics(self):
        self.assertEqual(
            tuple(ANOMALY_FEATURE_COLUMNS),
            (
                "vibration_rms",
                "vibration_std",
                "vibration_peak",
                "vibration_peak_to_peak",
                "vibration_kurtosis",
                "vibration_skewness",
                "crest_factor",
            ),
        )

    def test_anomaly_features_are_a_strict_subset_of_the_supervised_contract(self):
        self.assertTrue(
            set(ANOMALY_FEATURE_COLUMNS) < set(SUPERVISED_FEATURE_COLUMNS),
            "anomaly features must be a strict subset of the supervised contract",
        )


class FrozenExperimentContractTests(unittest.TestCase):
    """Experiments 1 and 2 must be unchanged by Experiment 3."""

    def test_experiment1_baseline_is_still_sixteen_recordings(self):
        self.assertEqual(len(BASELINE_SPECS), 16)
        self.assertEqual({s.fault_severity_in for s in BASELINE_SPECS}, {None, 0.007})

    def test_experiment2_feature_contract_is_unchanged(self):
        self.assertEqual(
            tuple(SUPERVISED_FEATURE_COLUMNS),
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

    def test_experiment2_exclusion_list_is_unchanged(self):
        for column in ("fault_severity_in", "recording_id", "window_id", "source"):
            self.assertIn(column, EXP2_EXCLUDED_FROM_X, msg=column)
        self.assertFalse(set(SUPERVISED_FEATURE_COLUMNS) & set(EXP2_EXCLUDED_FROM_X))

    def test_experiment3_reuses_the_experiment2_split_loads(self):
        self.assertEqual(tuple(TRAIN_LOADS), tuple(EXP2_TRAIN_LOADS))
        self.assertEqual(tuple(VALIDATION_LOADS), tuple(EXP2_VALIDATION_LOADS))
        self.assertEqual(tuple(TEST_LOADS), tuple(EXP2_TEST_LOADS))

    def test_expanded_metadata_table_is_unchanged(self):
        self.assertEqual(len(CWRU_RECORDINGS), 40)
        self.assertTrue(set(BASELINE_SPECS).issubset(set(CWRU_RECORDINGS)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
