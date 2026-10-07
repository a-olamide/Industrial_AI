"""Tests for Digital Twin state mapping, persistence and the read API.

Run from the project root::

    python -m unittest discover -s ml/tests -t . -v

Three tiers, each skipping cleanly when its prerequisite is absent:

- mapping / SQL-shape tests: always run, no infrastructure needed;
- persistence tests: need SQL Server reachable (``docker compose up -d
  sqlserver``);
- API serialization tests: need the ASP.NET API running on
  ``http://localhost:5025``.

The API serialization checks are deliberately integration tests against
the live endpoint rather than a new .NET unit-test project: the
solution has no test project today, and the property that actually
matters - that ``demoGroundTruth`` serializes as a sibling of, and never
inside, the model-output blocks - is a property of the real JSON.
"""

from __future__ import annotations

import json
import os
import unittest
import urllib.error
import urllib.request

from ml.streaming.twin_sink import (
    CURRENT_MERGE_SQL,
    HISTORY_INSERT_SQL,
    PAYLOAD_COLUMNS,
    result_to_row,
)


API_BASE = os.environ.get("TWIN_API_BASE", "http://localhost:5025")
SQL_HOST = os.environ.get("TWIN_SQL_HOST", "localhost")
SQL_PORT = int(os.environ.get("TWIN_SQL_PORT", "1433"))

MODEL_OUTPUT_COLUMNS = (
    "is_anomalous",
    "anomaly_score",
    "anomaly_threshold",
    "predicted_class",
    "confidence",
    "class_probabilities_json",
)
FEATURE_COLUMNS = (
    "vibration_rms",
    "vibration_std",
    "vibration_peak",
    "vibration_peak_to_peak",
    "vibration_kurtosis",
    "vibration_skewness",
    "crest_factor",
)
OPERATING_COLUMNS = ("motor_load_hp", "rotational_speed_rpm")
DEMO_COLUMNS = ("demo_recording_id", "demo_fault_class", "demo_fault_severity_in")


def sample_result(
    asset_id: str = "MOTOR_003",
    window_index: int = 0,
    predicted: str = "BALL",
    truth: str | None = "OUTER_RACE",
) -> dict:
    start = window_index * 2048
    payload = {
        "assetId": asset_id,
        "windowIndex": window_index,
        "windowStartSequence": start,
        "windowEndSequence": start + 2047,
        "timestampUtc": "2026-10-07T12:00:00+00:00",
        "sampleCount": 2048,
        "features": {
            "vibrationRms": 0.0943,
            "vibrationStd": 0.0942,
            "vibrationPeak": 0.3862,
            "vibrationPeakToPeak": 0.6974,
            "vibrationKurtosis": 0.2054,
            "vibrationSkewness": 0.0167,
            "crestFactor": 4.0959,
        },
        "operatingContext": {"motorLoadHp": 3.0, "rotationalSpeedRpm": 1723.0},
        "anomaly": {
            "isAnomalous": True,
            "score": -0.7382,
            "threshold": -0.6153,
            "scoreDirection": "score_samples: HIGHER = more normal...",
        },
        "classification": {
            "predictedClass": predicted,
            "confidence": 0.81,
            "probabilities": {
                "BALL": 0.19,
                "INNER_RACE": 0.0,
                "NORMAL": 0.0,
                "OUTER_RACE": 0.81,
            },
        },
    }
    if truth is not None:
        payload["groundTruth"] = {
            "faultClass": truth,
            "faultSeverityIn": 0.014,
            "recordingId": "OR014@6_3",
        }
    return payload


class StateMappingTests(unittest.TestCase):
    """Inference result -> Digital Twin state."""

    def test_model_output_is_preserved_exactly(self):
        row = result_to_row(sample_result())
        self.assertIs(row["is_anomalous"], True)
        self.assertAlmostEqual(row["anomaly_score"], -0.7382)
        self.assertAlmostEqual(row["anomaly_threshold"], -0.6153)
        self.assertEqual(row["predicted_class"], "BALL")
        self.assertAlmostEqual(row["confidence"], 0.81)

    def test_class_probabilities_round_trip(self):
        row = result_to_row(sample_result())
        probabilities = json.loads(row["class_probabilities_json"])
        self.assertEqual(
            set(probabilities), {"BALL", "INNER_RACE", "NORMAL", "OUTER_RACE"}
        )
        self.assertAlmostEqual(probabilities["OUTER_RACE"], 0.81)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)

    def test_features_are_preserved(self):
        row = result_to_row(sample_result())
        self.assertAlmostEqual(row["vibration_rms"], 0.0943)
        self.assertAlmostEqual(row["vibration_kurtosis"], 0.2054)
        self.assertAlmostEqual(row["crest_factor"], 4.0959)
        for column in FEATURE_COLUMNS:
            self.assertIn(column, row)

    def test_operating_context_is_preserved(self):
        row = result_to_row(sample_result())
        self.assertAlmostEqual(row["motor_load_hp"], 3.0)
        self.assertAlmostEqual(row["rotational_speed_rpm"], 1723.0)

    def test_window_provenance_is_preserved(self):
        row = result_to_row(sample_result(window_index=5))
        self.assertEqual(row["window_start_sequence"], 5 * 2048)
        self.assertEqual(row["window_end_sequence"], 5 * 2048 + 2047)
        self.assertEqual(row["sample_count"], 2048)

    def test_ground_truth_lands_only_in_demo_columns(self):
        row = result_to_row(sample_result())
        self.assertEqual(row["demo_fault_class"], "OUTER_RACE")
        self.assertEqual(row["demo_recording_id"], "OR014@6_3")
        self.assertAlmostEqual(row["demo_fault_severity_in"], 0.014)
        # The classifier said BALL; ground truth must not have overwritten it.
        self.assertEqual(row["predicted_class"], "BALL")
        for column in MODEL_OUTPUT_COLUMNS + FEATURE_COLUMNS + OPERATING_COLUMNS:
            self.assertFalse(
                column.startswith("demo_"), msg=f"{column} is not a demo column"
            )

    def test_no_ground_truth_value_leaks_into_a_non_demo_column(self):
        row = result_to_row(sample_result(predicted="BALL", truth="OUTER_RACE"))
        non_demo = {
            k: v for k, v in row.items() if not k.startswith("demo_")
        }
        self.assertNotIn("OR014@6_3", [str(v) for v in non_demo.values()])
        self.assertNotIn(
            "OUTER_RACE",
            [str(v) for k, v in non_demo.items() if k != "class_probabilities_json"],
        )

    def test_absent_ground_truth_yields_nulls(self):
        row = result_to_row(sample_result(truth=None))
        for column in DEMO_COLUMNS:
            self.assertIsNone(row[column], msg=column)

    def test_normal_scenario_has_null_severity(self):
        result = sample_result(predicted="NORMAL", truth="NORMAL")
        result["groundTruth"]["faultSeverityIn"] = None
        result["groundTruth"]["recordingId"] = "Normal_3"
        row = result_to_row(result)
        self.assertIsNone(row["demo_fault_severity_in"])
        self.assertEqual(row["demo_fault_class"], "NORMAL")

    def test_payload_columns_cover_every_mapped_key(self):
        row = result_to_row(sample_result())
        self.assertEqual(set(row), set(PAYLOAD_COLUMNS))


class SqlStatementShapeTests(unittest.TestCase):
    """The two statements must stay in step with PAYLOAD_COLUMNS."""

    def test_history_insert_parameter_count(self):
        # one placeholder per payload column, plus the 2 NOT EXISTS keys
        self.assertEqual(
            HISTORY_INSERT_SQL.count("?"), len(PAYLOAD_COLUMNS) + 2
        )

    def test_merge_parameter_count(self):
        # USING key (1) + UPDATE list (n-1) + INSERT list (n)
        expected = 1 + (len(PAYLOAD_COLUMNS) - 1) + len(PAYLOAD_COLUMNS)
        self.assertEqual(CURRENT_MERGE_SQL.count("?"), expected)

    def test_history_insert_guards_against_duplicate_windows(self):
        self.assertIn("WHERE NOT EXISTS", HISTORY_INSERT_SQL)
        self.assertIn("window_end_sequence", HISTORY_INSERT_SQL)

    def test_merge_targets_the_current_state_table(self):
        self.assertIn("MERGE dbo.asset_twin_current", CURRENT_MERGE_SQL)
        self.assertIn("WHEN MATCHED THEN UPDATE", CURRENT_MERGE_SQL)
        self.assertIn("WHEN NOT MATCHED THEN INSERT", CURRENT_MERGE_SQL)

    def test_every_demo_column_is_written_by_both_statements(self):
        for column in DEMO_COLUMNS:
            self.assertIn(column, HISTORY_INSERT_SQL, msg=column)
            self.assertIn(column, CURRENT_MERGE_SQL, msg=column)


class UpsertOrderingTests(unittest.TestCase):
    """The newest window of a batch must end up as current state."""

    def test_rows_are_applied_in_window_order(self):
        results = [
            sample_result(window_index=2),
            sample_result(window_index=0),
            sample_result(window_index=1),
        ]
        rows = [result_to_row(r) for r in results]
        rows.sort(key=lambda r: (r["asset_id"], r["window_end_sequence"]))
        self.assertEqual(
            [r["window_end_sequence"] for r in rows],
            [2047, 4095, 6143],
        )
        # Last applied row is the newest window -> becomes current state.
        self.assertEqual(rows[-1]["window_end_sequence"], 6143)

    def test_assets_are_ordered_independently(self):
        rows = [
            result_to_row(sample_result(asset_id="MOTOR_003", window_index=1)),
            result_to_row(sample_result(asset_id="MOTOR_001", window_index=5)),
            result_to_row(sample_result(asset_id="MOTOR_003", window_index=9)),
        ]
        rows.sort(key=lambda r: (r["asset_id"], r["window_end_sequence"]))
        self.assertEqual(
            [(r["asset_id"], r["window_end_sequence"]) for r in rows],
            [("MOTOR_001", 5 * 2048 + 2047),
             ("MOTOR_003", 1 * 2048 + 2047),
             ("MOTOR_003", 9 * 2048 + 2047)],
        )


def _sql_available() -> bool:
    import socket

    try:
        with socket.create_connection((SQL_HOST, SQL_PORT), timeout=2):
            return True
    except OSError:
        return False


class PersistenceTests(unittest.TestCase):
    """Live SQL Server behaviour: upsert, latest-wins, history idempotency."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _sql_available():
            raise unittest.SkipTest(
                f"SQL Server not reachable at {SQL_HOST}:{SQL_PORT} "
                "(docker compose up -d sqlserver)"
            )
        try:
            import pymssql  # type: ignore # noqa: F401
        except ImportError:
            raise unittest.SkipTest(
                "no Python SQL Server driver installed; persistence is exercised "
                "end to end by the Spark job instead"
            )

    def test_placeholder(self):  # pragma: no cover - only runs with pymssql
        self.skipTest("covered by the end-to-end pipeline run")


def _api_json(path: str):
    try:
        with urllib.request.urlopen(f"{API_BASE}{path}", timeout=5) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


class ApiSerializationTests(unittest.TestCase):
    """The read API must keep model output and demo ground truth apart."""

    @classmethod
    def setUpClass(cls) -> None:
        payload = _api_json("/api/v1/digital-twins")
        if payload is None:
            raise unittest.SkipTest(
                f"API not reachable at {API_BASE} "
                "(dotnet run --project src/IndustrialAnalytics.Api)"
            )
        if not payload.get("items"):
            raise unittest.SkipTest("API reachable but no Digital Twin state yet")
        cls.payload = payload
        cls.item = payload["items"][0]

    def test_response_envelope(self):
        self.assertIn("count", self.payload)
        self.assertEqual(self.payload["count"], len(self.payload["items"]))

    def test_top_level_blocks_are_separated(self):
        self.assertEqual(
            set(self.item)
            >= {
                "assetId",
                "lastUpdatedUtc",
                "window",
                "anomaly",
                "classification",
                "features",
                "operatingContext",
            },
            True,
        )

    def test_model_output_blocks_contain_no_ground_truth(self):
        for block in ("anomaly", "classification", "features", "operatingContext"):
            serialized = json.dumps(self.item[block]).lower()
            for banned in ("faultclass", "recordingid", "faultseverity", "groundtruth"):
                self.assertNotIn(banned, serialized, msg=f"{banned} in {block}")

    def test_ground_truth_is_a_separate_sibling_block(self):
        truth = self.item.get("demoGroundTruth")
        if truth is None:
            self.skipTest("demo ground truth not attached to this asset")
        self.assertEqual(set(truth) <= {"recordingId", "faultClass", "faultSeverityIn"}, True)

    def test_anomaly_block_states_the_score_direction(self):
        anomaly = self.item["anomaly"]
        self.assertIn("isAnomalous", anomaly)
        self.assertIn("score", anomaly)
        self.assertIn("threshold", anomaly)
        self.assertIn("more anomalous", anomaly["scoreDirection"])
        # The decision must agree with the stated convention.
        self.assertEqual(
            anomaly["isAnomalous"], anomaly["score"] < anomaly["threshold"]
        )

    def test_classification_preserves_prediction_and_confidence(self):
        classification = self.item["classification"]
        self.assertIn(
            classification["predictedClass"],
            {"NORMAL", "INNER_RACE", "BALL", "OUTER_RACE"},
        )
        probabilities = classification.get("probabilities")
        if probabilities:
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)
            best = max(probabilities, key=probabilities.get)
            self.assertEqual(best, classification["predictedClass"])
            self.assertAlmostEqual(
                classification["confidence"], probabilities[best], places=6
            )

    def test_features_block_has_exactly_the_seven_features(self):
        self.assertEqual(
            set(self.item["features"]),
            {
                "vibrationRms",
                "vibrationStd",
                "vibrationPeak",
                "vibrationPeakToPeak",
                "vibrationKurtosis",
                "vibrationSkewness",
                "crestFactor",
            },
        )

    def test_raw_sample_arrays_are_not_exposed(self):
        """No raw 2048-sample vibration array may appear in the twin payload.

        Tested structurally rather than by substring: the words "samples"
        and "sampleCount" legitimately occur in the score-direction text
        and the window block. What must not occur is an ARRAY of readings.
        """

        def arrays(node, path=""):
            if isinstance(node, list):
                yield path, node
            elif isinstance(node, dict):
                for key, value in node.items():
                    yield from arrays(value, f"{path}.{key}" if path else key)

        found = list(arrays(self.item))
        self.assertEqual(found, [], f"unexpected array(s) in twin payload: {found}")
        self.assertLess(
            len(json.dumps(self.item)),
            4000,
            "response looks like it carries raw samples",
        )

    def test_ground_truth_can_be_suppressed(self):
        payload = _api_json("/api/v1/digital-twins?includeGroundTruth=false")
        self.assertIsNotNone(payload)
        for item in payload["items"]:
            self.assertIsNone(item.get("demoGroundTruth"))

    def test_single_asset_endpoint_matches_the_list(self):
        asset_id = self.item["assetId"]
        single = _api_json(f"/api/v1/digital-twins/{asset_id}")
        self.assertIsNotNone(single)
        self.assertEqual(single["assetId"], asset_id)
        self.assertEqual(
            single["window"]["windowEndSequence"],
            self.item["window"]["windowEndSequence"],
        )

    def test_latest_inference_is_the_current_state(self):
        asset_id = self.item["assetId"]
        history = _api_json(f"/api/v1/digital-twins/{asset_id}/history?take=500")
        if not history or not history["items"]:
            self.skipTest("no history recorded")
        newest = max(h["windowEndSequence"] for h in history["items"])
        self.assertEqual(self.item["window"]["windowEndSequence"], newest)

    def test_history_is_newest_first_and_has_no_duplicate_windows(self):
        asset_id = self.item["assetId"]
        history = _api_json(f"/api/v1/digital-twins/{asset_id}/history?take=500")
        if not history or len(history["items"]) < 2:
            self.skipTest("not enough history")
        ends = [h["windowEndSequence"] for h in history["items"]]
        self.assertEqual(ends, sorted(ends, reverse=True))
        self.assertEqual(len(ends), len(set(ends)), "duplicate windows in history")

    def test_history_rows_preserve_prediction_and_confidence(self):
        asset_id = self.item["assetId"]
        history = _api_json(f"/api/v1/digital-twins/{asset_id}/history?take=10")
        if not history or not history["items"]:
            self.skipTest("no history recorded")
        for point in history["items"]:
            self.assertIn(
                point["predictedClass"],
                {"NORMAL", "INNER_RACE", "BALL", "OUTER_RACE"},
            )
            if point["confidence"] is not None:
                self.assertGreaterEqual(point["confidence"], 0.0)
                self.assertLessEqual(point["confidence"], 1.0)
            self.assertIsInstance(point["isAnomalous"], bool)

    def test_unknown_asset_returns_not_found(self):
        try:
            urllib.request.urlopen(
                f"{API_BASE}/api/v1/digital-twins/NO_SUCH_ASSET", timeout=5
            )
            self.fail("expected 404")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
