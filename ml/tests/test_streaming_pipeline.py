"""Contract tests for the online streaming inference pipeline.

Run from the project root::

    python -m unittest discover -s ml/tests -t . -v

Three tiers, each skipping cleanly when its prerequisite is absent so a
fresh checkout still passes:

- pure contract / windowing / feature tests: always run;
- feature-parity against real CWRU windows: needs the vendor ``.mat``
  files;
- model-contract tests: need the gitignored ``.joblib`` artifacts;
- Spark aggregate parity: needs ``pyspark`` and a JVM.

The headline deliverable here is :class:`FeatureParityTests`, which
asserts the streaming and Spark feature implementations agree with the
offline training code used to fit the models.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ml.src.cwru_loader import load_recording
from ml.src.dataset_builder import RECORDINGS_BY_ID, resolve_recording_paths
from ml.src.feature_extraction import compute_window_features, iter_windows
from ml.src.train_baseline import FEATURE_COLUMNS as EXP2_FEATURE_COLUMNS
from ml.streaming.contracts import (
    GROUND_TRUTH_FIELDS,
    OPERATING_CONTEXT_FIELDS,
    WINDOW_SIZE,
    SourceScenario,
    VibrationTelemetryEvent,
    feature_inputs,
)
from ml.streaming.stream_features import (
    FEATURE_NAMES,
    compute_streaming_features,
)
from ml.streaming.windowing import WindowAssembler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "cwru"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

# Documented parity tolerance. Observed worst case across every window of
# four CWRU recordings is ~5e-14 for kurtosis (pure floating-point
# accumulation order); 1e-9 absolute leaves three orders of headroom
# while still catching any genuine definitional divergence such as a
# ddof change or a sample-vs-population moment.
PARITY_TOLERANCE = 1e-9

PARITY_RECORDINGS = ("Normal_3", "OR014@6_3", "B021_3", "IR007_3")


def _recording_path(recording_id: str) -> Path | None:
    spec = RECORDINGS_BY_ID.get(recording_id)
    if spec is None:
        return None
    found, _missing = resolve_recording_paths(RAW_DIR, [spec])
    return found[0].path if found else None


def _load_signal(recording_id: str) -> np.ndarray | None:
    path = _recording_path(recording_id)
    if path is None:
        return None
    spec = RECORDINGS_BY_ID[recording_id]
    return load_recording(path, expected_experiment_number=spec.experiment_number).drive_end_signal


def _event(
    asset: str,
    seq: int,
    value: float,
    load: float = 3.0,
    rpm: float = 1725.0,
    scenario: SourceScenario | None = None,
) -> VibrationTelemetryEvent:
    return VibrationTelemetryEvent(
        assetId=asset,
        timestampUtc="2026-10-07T00:00:00+00:00",
        sequenceNumber=seq,
        vibration=value,
        motorLoadHp=load,
        rotationalSpeedRpm=rpm,
        sourceScenario=scenario,
    )


class TelemetrySchemaTests(unittest.TestCase):
    """Message parsing and the ground-truth / telemetry separation."""

    def test_round_trip_with_ground_truth(self):
        scenario = SourceScenario("OUTER_RACE", 0.014, "OR014@6_3")
        original = _event("MOTOR_001", 42, 0.123, scenario=scenario)
        parsed = VibrationTelemetryEvent.from_json(original.to_json())
        self.assertEqual(parsed.assetId, "MOTOR_001")
        self.assertEqual(parsed.sequenceNumber, 42)
        self.assertAlmostEqual(parsed.vibration, 0.123)
        self.assertEqual(parsed.motorLoadHp, 3.0)
        self.assertEqual(parsed.sourceScenario, scenario)

    def test_round_trip_without_ground_truth(self):
        parsed = VibrationTelemetryEvent.from_json(_event("A", 1, 0.5).to_json())
        self.assertIsNone(parsed.sourceScenario)
        self.assertNotIn("sourceScenario", _event("A", 1, 0.5).to_dict())

    def test_normal_scenario_carries_null_severity(self):
        scenario = SourceScenario("NORMAL", None, "Normal_3")
        parsed = VibrationTelemetryEvent.from_json(
            _event("A", 0, 0.1, scenario=scenario).to_json()
        )
        self.assertIsNone(parsed.sourceScenario.faultSeverityIn)

    def test_window_index_is_derived_from_sequence_number(self):
        self.assertEqual(_event("A", 0, 0.0).windowIndex, 0)
        self.assertEqual(_event("A", 2047, 0.0).windowIndex, 0)
        self.assertEqual(_event("A", 2048, 0.0).windowIndex, 1)
        self.assertEqual(_event("A", 4095, 0.0).windowIndex, 1)

    def test_feature_inputs_exclude_ground_truth(self):
        scenario = SourceScenario("BALL", 0.021, "B021_3")
        values = feature_inputs(_event("A", 0, 0.7, scenario=scenario))
        for forbidden in GROUND_TRUTH_FIELDS:
            self.assertNotIn(forbidden, values)
        for forbidden in ("faultClass", "faultSeverityIn", "recordingId"):
            self.assertNotIn(forbidden, values)
        self.assertEqual(
            set(values), {"vibration", "motor_load_hp", "rotational_speed_rpm"}
        )

    def test_ground_truth_and_operating_context_are_disjoint(self):
        self.assertFalse(set(GROUND_TRUTH_FIELDS) & set(OPERATING_CONTEXT_FIELDS))


class WindowingTests(unittest.TestCase):
    """2048-sample boundaries, non-overlap, asset isolation, ordering."""

    def test_window_emitted_only_when_complete(self):
        assembler = WindowAssembler()
        for seq in range(WINDOW_SIZE - 1):
            self.assertIsNone(assembler.add(_event("A", seq, float(seq))))
        window = assembler.add(_event("A", WINDOW_SIZE - 1, 1.0))
        self.assertIsNotNone(window)
        self.assertEqual(window.sampleCount, WINDOW_SIZE)

    def test_window_boundaries_are_exact(self):
        assembler = WindowAssembler()
        windows = list(
            assembler.add_all(_event("A", s, 0.1) for s in range(3 * WINDOW_SIZE))
        )
        self.assertEqual(len(windows), 3)
        for index, window in enumerate(windows):
            self.assertEqual(window.windowIndex, index)
            self.assertEqual(window.windowStartSequence, index * WINDOW_SIZE)
            self.assertEqual(window.windowEndSequence, (index + 1) * WINDOW_SIZE - 1)
            self.assertEqual(window.sampleCount, WINDOW_SIZE)

    def test_windows_do_not_overlap(self):
        assembler = WindowAssembler()
        windows = list(
            assembler.add_all(_event("A", s, 0.1) for s in range(3 * WINDOW_SIZE))
        )
        seen: set[int] = set()
        for window in windows:
            span = set(range(window.windowStartSequence, window.windowEndSequence + 1))
            self.assertFalse(span & seen, "windows share sequence numbers")
            seen |= span
        self.assertEqual(len(seen), 3 * WINDOW_SIZE)

    def test_partial_trailing_window_is_dropped(self):
        assembler = WindowAssembler()
        windows = list(
            assembler.add_all(
                _event("A", s, 0.1) for s in range(WINDOW_SIZE + 500)
            )
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(assembler.pending_window_count, 1)

    def test_assets_never_share_a_window(self):
        assembler = WindowAssembler()
        interleaved = []
        for seq in range(WINDOW_SIZE):
            interleaved.append(_event("MOTOR_001", seq, 1.0))
            interleaved.append(_event("MOTOR_002", seq, -1.0))
        windows = list(assembler.add_all(interleaved))
        self.assertEqual(len(windows), 2)
        by_asset = {w.assetId: w for w in windows}
        self.assertEqual(set(by_asset), {"MOTOR_001", "MOTOR_002"})
        self.assertTrue(all(v == 1.0 for v in by_asset["MOTOR_001"].samples))
        self.assertTrue(all(v == -1.0 for v in by_asset["MOTOR_002"].samples))

    def test_out_of_order_delivery_yields_the_same_window(self):
        values = [float(i % 17) * 0.01 for i in range(WINDOW_SIZE)]
        in_order = WindowAssembler()
        ordered_window = None
        for seq, value in enumerate(values):
            ordered_window = in_order.add(_event("A", seq, value)) or ordered_window

        shuffled = WindowAssembler()
        indices = list(range(WINDOW_SIZE))
        rng = np.random.default_rng(7)
        rng.shuffle(indices)
        shuffled_window = None
        for seq in indices:
            shuffled_window = (
                shuffled.add(_event("A", seq, values[seq])) or shuffled_window
            )

        self.assertEqual(ordered_window.samples, shuffled_window.samples)
        self.assertEqual(
            ordered_window.windowStartSequence, shuffled_window.windowStartSequence
        )

    def test_duplicate_delivery_does_not_create_a_short_window(self):
        assembler = WindowAssembler()
        events = [_event("A", s, 0.2) for s in range(WINDOW_SIZE)]
        duplicated = events[:100] + events  # at-least-once redelivery
        windows = list(assembler.add_all(duplicated))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].sampleCount, WINDOW_SIZE)

    def test_ground_truth_rides_along_but_stays_separate(self):
        scenario = SourceScenario("BALL", 0.021, "B021_3")
        assembler = WindowAssembler()
        window = None
        for seq in range(WINDOW_SIZE):
            window = assembler.add(_event("A", seq, 0.3, scenario=scenario)) or window
        self.assertEqual(window.sourceScenario, scenario)
        # The feature path never consults it.
        self.assertEqual(len(window.features().as_dict()), 7)


class FeatureParityTests(unittest.TestCase):
    """REQUIRED DELIVERABLE: streaming features == offline training features."""

    def test_parity_on_synthetic_window(self):
        rng = np.random.default_rng(11)
        samples = rng.normal(0.0, 0.25, WINDOW_SIZE)
        offline = compute_window_features(samples, 0, 0).as_dict()
        streaming = compute_streaming_features(samples.tolist()).as_dict()
        for name in FEATURE_NAMES:
            self.assertAlmostEqual(
                offline[name], streaming[name], delta=PARITY_TOLERANCE, msg=name
            )

    def test_parity_on_real_cwru_windows(self):
        checked = 0
        worst = 0.0
        for recording_id in PARITY_RECORDINGS:
            signal = _load_signal(recording_id)
            if signal is None:
                continue
            for wid, start, samples in iter_windows(signal, WINDOW_SIZE):
                offline = compute_window_features(samples, wid, start).as_dict()
                streaming = compute_streaming_features(samples.tolist()).as_dict()
                for name in FEATURE_NAMES:
                    diff = abs(offline[name] - streaming[name])
                    worst = max(worst, diff)
                    self.assertLess(
                        diff,
                        PARITY_TOLERANCE,
                        msg=f"{recording_id} window {wid} feature {name}: {diff:.3e}",
                    )
                checked += 1
                if checked >= 60:
                    break
            if checked >= 60:
                break
        if checked == 0:
            self.skipTest(f"no CWRU .mat files in {RAW_DIR}")
        self.assertGreater(checked, 0)

    def test_zero_signal_edge_cases_match_training_code(self):
        zeros = np.zeros(64)
        offline = compute_window_features(zeros, 0, 0).as_dict()
        streaming = compute_streaming_features(zeros.tolist()).as_dict()
        self.assertEqual(streaming["crest_factor"], 0.0)
        self.assertEqual(streaming["vibration_kurtosis"], 0.0)
        self.assertEqual(streaming["vibration_skewness"], 0.0)
        for name in FEATURE_NAMES:
            self.assertEqual(offline[name], streaming[name], msg=name)

    def test_constant_non_zero_signal_edge_case(self):
        constant = np.full(64, 0.5)
        offline = compute_window_features(constant, 0, 0).as_dict()
        streaming = compute_streaming_features(constant.tolist()).as_dict()
        for name in FEATURE_NAMES:
            self.assertAlmostEqual(
                offline[name], streaming[name], delta=PARITY_TOLERANCE, msg=name
            )

    def test_peak_uses_absolute_value_not_maximum(self):
        # A window whose largest magnitude is negative distinguishes
        # max(|x|) from max(x); getting this wrong is a silent error.
        samples = np.array([-3.0, 1.0, 2.0, -0.5])
        offline = compute_window_features(samples, 0, 0)
        streaming = compute_streaming_features(samples.tolist())
        self.assertEqual(offline.vibration_peak, 3.0)
        self.assertEqual(streaming.vibration_peak, 3.0)
        self.assertEqual(streaming.vibration_peak_to_peak, 5.0)


class SparkFeatureParityTests(unittest.TestCase):
    """Spark SQL aggregates == offline training features."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from pyspark.sql import SparkSession  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("pyspark not installed")
        signal = _load_signal("OR014@6_3")
        if signal is None:
            raise unittest.SkipTest("CWRU .mat files not present")
        cls.signal = signal
        from pyspark.sql import SparkSession

        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("parity-test")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        spark = getattr(cls, "spark", None)
        if spark is not None:
            spark.stop()

    def test_spark_aggregates_match_offline_feature_extraction(self):
        import pandas as pd
        from ml.streaming.spark_inference_job import aggregate_windows, parse_telemetry

        n_windows = 3
        samples = self.signal[: n_windows * WINDOW_SIZE]
        events = [
            _event("MOTOR_001", seq, float(value)).to_json()
            for seq, value in enumerate(samples)
        ]
        raw = self.spark.createDataFrame(pd.DataFrame({"raw": events}))
        rows = {
            int(r["windowIndex"]): r.asDict()
            for r in aggregate_windows(parse_telemetry(raw)).collect()
        }
        self.assertEqual(len(rows), n_windows)

        worst = 0.0
        for wid, start, window_samples in iter_windows(samples, WINDOW_SIZE):
            offline = compute_window_features(window_samples, wid, start).as_dict()
            spark_row = rows[wid]
            self.assertEqual(spark_row["sampleCount"], WINDOW_SIZE)
            self.assertEqual(spark_row["windowStartSequence"], wid * WINDOW_SIZE)
            self.assertEqual(
                spark_row["windowEndSequence"], (wid + 1) * WINDOW_SIZE - 1
            )
            for name in FEATURE_NAMES:
                diff = abs(offline[name] - float(spark_row[name]))
                worst = max(worst, diff)
                self.assertLess(
                    diff, PARITY_TOLERANCE, msg=f"window {wid} feature {name}: {diff:.3e}"
                )
        self.assertLess(worst, PARITY_TOLERANCE)

    def test_spark_drops_incomplete_windows(self):
        import pandas as pd
        from ml.streaming.spark_inference_job import aggregate_windows, parse_telemetry

        # One complete window plus a 500-sample partial tail.
        samples = self.signal[: WINDOW_SIZE + 500]
        events = [
            _event("MOTOR_001", seq, float(value)).to_json()
            for seq, value in enumerate(samples)
        ]
        raw = self.spark.createDataFrame(pd.DataFrame({"raw": events}))
        rows = aggregate_windows(parse_telemetry(raw)).collect()
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["windowIndex"]), 0)

    def test_spark_keeps_assets_in_separate_windows(self):
        import pandas as pd
        from ml.streaming.spark_inference_job import aggregate_windows, parse_telemetry

        events = []
        for seq in range(WINDOW_SIZE):
            events.append(_event("MOTOR_001", seq, 1.0).to_json())
            events.append(_event("MOTOR_002", seq, -2.0).to_json())
        raw = self.spark.createDataFrame(pd.DataFrame({"raw": events}))
        rows = {r["assetId"]: r.asDict() for r in aggregate_windows(parse_telemetry(raw)).collect()}
        self.assertEqual(set(rows), {"MOTOR_001", "MOTOR_002"})
        self.assertAlmostEqual(rows["MOTOR_001"]["vibration_rms"], 1.0, places=12)
        self.assertAlmostEqual(rows["MOTOR_002"]["vibration_rms"], 2.0, places=12)
        self.assertAlmostEqual(rows["MOTOR_002"]["vibration_peak"], 2.0, places=12)


class ModelContractTests(unittest.TestCase):
    """Feature order and model inputs are validated, never guessed."""

    @classmethod
    def setUpClass(cls) -> None:
        from ml.streaming.inference import load_models

        for stem in ("rf_multiseverity_cwru", "isolation_forest_cwru"):
            if not (MODELS_DIR / f"{stem}.joblib").is_file():
                raise unittest.SkipTest(
                    f"{stem}.joblib not present (model binaries are gitignored); "
                    "re-run the experiment to regenerate"
                )
        cls.models = load_models()

    def test_random_forest_uses_the_exact_experiment2_contract(self):
        self.assertEqual(
            list(self.models.classifier_features), list(EXP2_FEATURE_COLUMNS)
        )
        self.assertEqual(len(self.models.classifier_features), 9)

    def test_isolation_forest_receives_exactly_seven_vibration_features(self):
        self.assertEqual(len(self.models.anomaly_features), 7)
        self.assertEqual(list(self.models.anomaly_features), list(FEATURE_NAMES))
        for operating in ("motor_load_hp", "rotational_speed_rpm"):
            self.assertNotIn(operating, self.models.anomaly_features)

    def test_no_ground_truth_column_is_in_either_contract(self):
        forbidden = set(GROUND_TRUTH_FIELDS) | {
            "fault_class",
            "fault_severity_in",
            "recording_id",
            "source_file",
            "sampling_rate_hz",
            "window_id",
        }
        for contract in (self.models.classifier_features, self.models.anomaly_features):
            self.assertFalse(set(contract) & forbidden)

    def test_feature_row_is_built_in_model_order_not_dict_order(self):
        from ml.streaming.inference import build_feature_row

        features = compute_streaming_features([0.1, -0.2, 0.3, -0.4] * 16)
        row = build_feature_row(features, 3.0, 1725.0, self.models.classifier_features)
        self.assertEqual(list(row.columns), list(self.models.classifier_features))
        # Reversing the requested order must reverse the produced columns -
        # proving order comes from the contract, not from insertion order.
        reversed_order = tuple(reversed(self.models.classifier_features))
        reversed_row = build_feature_row(features, 3.0, 1725.0, reversed_order)
        self.assertEqual(list(reversed_row.columns), list(reversed_order))

    def test_contract_validation_rejects_a_permuted_estimator(self):
        from ml.streaming.inference import ModelContractError, _validate_contract
        from ml.streaming.inference import STREAMING_FEATURE_SOURCES

        class FakeEstimator:
            feature_names_in_ = np.array(list(reversed(EXP2_FEATURE_COLUMNS)))

        with self.assertRaises(ModelContractError):
            _validate_contract(
                "fake", FakeEstimator(), EXP2_FEATURE_COLUMNS, STREAMING_FEATURE_SOURCES
            )

    def test_contract_validation_rejects_unsupplyable_feature(self):
        from ml.streaming.inference import ModelContractError, _validate_contract
        from ml.streaming.inference import STREAMING_FEATURE_SOURCES

        class Bare:
            pass

        with self.assertRaises(ModelContractError):
            _validate_contract(
                "fake",
                Bare(),
                list(EXP2_FEATURE_COLUMNS) + ["fault_severity_in"],
                STREAMING_FEATURE_SOURCES,
            )

    def test_anomaly_threshold_comes_from_experiment3_metadata(self):
        expected = float(
            self.models.anomaly_metadata["threshold"]["value"]
        )
        self.assertEqual(self.models.anomaly_threshold, expected)
        self.assertEqual(self.models.anomaly_threshold_name, "train_quantile_0.01")

    def test_inference_result_carries_ground_truth_outside_the_feature_block(self):
        from ml.streaming.inference import infer

        scenario = SourceScenario("OUTER_RACE", 0.014, "OR014@6_3")
        assembler = WindowAssembler()
        window = None
        rng = np.random.default_rng(3)
        for seq in range(WINDOW_SIZE):
            window = (
                assembler.add(
                    _event("A", seq, float(rng.normal(0, 0.3)), scenario=scenario)
                )
                or window
            )
        result = infer(self.models, window)
        self.assertEqual(set(result.features), {
            "vibrationRms", "vibrationStd", "vibrationPeak", "vibrationPeakToPeak",
            "vibrationKurtosis", "vibrationSkewness", "crestFactor",
        })
        self.assertEqual(result.groundTruth["faultClass"], "OUTER_RACE")
        for key in result.features:
            self.assertNotIn("fault", key.lower())
        self.assertIn(result.classification["predictedClass"],
                      set(self.models.classifier_classes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
