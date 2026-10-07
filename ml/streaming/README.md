# Online streaming inference — CWRU vibration → Kafka → Spark → ML

Operationalises the three frozen ML experiments as a live industrial
telemetry pipeline. **No model is retrained, refit or modified here.**

```
CWRU drive-end recording
    └─ ml/streaming/replay_producer.py        telemetry replay simulator
        └─ Kafka  industrial.telemetry.vibration     (keyed by assetId)
            └─ Spark Structured Streaming
                ├─ windowIndex = sequenceNumber ÷ 2048
                ├─ groupBy(assetId, windowIndex)      non-overlapping, per asset
                ├─ 7 vibration features               native Spark SQL aggregates
                └─ foreachBatch
                    ├─ Isolation Forest   → anomaly score + NORMAL/ANOMALOUS
                    └─ Random Forest      → predicted fault class + probabilities
                        └─ structured inference result (JSON)
```

## Module map

| file | role |
|---|---|
| `contracts.py` | telemetry + inference message contracts; **ground-truth separation** |
| `stream_features.py` | streaming implementation of the 7 features (online moments) |
| `windowing.py` | per-asset non-overlapping 2048-sample assembly |
| `inference.py` | loads the frozen artifacts; **validates feature order** |
| `spark_inference_job.py` | the Spark Structured Streaming job |
| `replay_producer.py` | CWRU recording → Kafka telemetry simulator |
| `demo_end_to_end.py` | the three-scenario demonstration |

## Kafka topic and message contract

**Topic:** `industrial.telemetry.vibration` — 3 partitions, **keyed by
`assetId`** so every sample of one asset lands on one partition and
per-asset ordering is preserved, which the 2048-sample windowing
depends on.

This is a *new* topic, deliberately separate from the legacy
`industrial-telemetry`. The two carry different contracts: the legacy
topic is one message per *sensor tag* (`asset_id`/`tag`/`value`,
snake_case, ~35 msg/s); this one is one message per *raw vibration
sample* (camelCase, 12–48 kHz). Forcing them together would have meant
breaking the existing .NET producer and Spark analytics job for no gain.

One message per sample:

```json
{
  "assetId": "MOTOR_001",
  "timestampUtc": "2026-10-07T10:09:12.744447+00:00",
  "sequenceNumber": 12345,
  "vibration": 0.12345,
  "motorLoadHp": 3.0,
  "rotationalSpeedRpm": 1725.0,
  "sourceScenario": {
    "faultClass": "OUTER_RACE",
    "faultSeverityIn": 0.014,
    "recordingId": "OR014@6_3"
  }
}
```

### `sourceScenario` is ground truth, not telemetry

A real machine cannot report its own fault class or defect diameter —
that is exactly what the models are asked to infer. Those three fields
exist only so a demonstration can be scored, and three separate
mechanisms keep them out of the model:

1. **Type separation.** They live in `SourceScenario`, a distinct frozen
   dataclass, never mixed into the sensor fields.
2. **A single sanctioned path.** `contracts.feature_inputs(event)` is the
   only function that turns an event into model inputs, and it never
   touches `sourceScenario`.
3. **Contract validation.** `inference.load_models()` refuses to load a
   model whose saved feature contract contains `fault_class`,
   `fault_severity_in`, `recording_id`, `source_file`, `sampling_rate_hz`
   or `window_id`.

In the Spark job, ground truth is aggregated alongside the window but is
attached to the result **after** both models have run. `--no-ground-truth`
omits the block entirely; predictions are identical either way.

## Windowing behaviour

Window assignment is a pure function of the sample's own sequence
number:

```
windowIndex = sequenceNumber // 2048
```

Window *k* owns sequence numbers `[k*2048, k*2048+2047]` and nothing
else. This single choice delivers:

- **non-overlapping** windows (hop = window size, matching
  `iter_windows(..., hop=None)` used in training);
- **asset isolation** — state is keyed by `(assetId, windowIndex)`, so
  two assets sharing a partition or micro-batch can never share a window;
- **micro-batch independence** — Spark may split, merge, retry or
  reorder batches and the windows are identical, which is what makes
  online features reproduce offline training exactly;
- **partial windows dropped** — the offline extractor discards a short
  trailing window, so the stream does too.

Completeness is asserted without `count(distinct)` (which Spark rejects
inside a streaming aggregation). A window is accepted only when all four
hold: `count == 2048`, `min == k*2048`, `max == k*2048+2047`, and
`sum(sequenceNumber)` equals the arithmetic series for that range.
Together those admit only the exact contiguous set, so an at-least-once
redelivery cannot masquerade as a complete window.

## Feature parity — the formulas actually used

Read off `ml/src/feature_extraction.py` and reproduced exactly:

| feature | definition | trap |
|---|---|---|
| `vibration_rms` | `sqrt(mean(x²))` | |
| `vibration_std` | `np.std(x, ddof=0)` | **population**, not sample |
| `vibration_peak` | `max(abs(x))` | **absolute** peak, not `max(x)` |
| `vibration_peak_to_peak` | `max(x) - min(x)` | |
| `vibration_kurtosis` | `mean(((x-μ)/σ)⁴) - 3` | **excess**, biased moments, σ is ddof=0 |
| `vibration_skewness` | `mean(((x-μ)/σ)³)` | biased (population) |
| `crest_factor` | `max(abs(x)) / rms` | **exactly 0.0 when rms == 0**, not NaN/inf |

Two edge cases are inherited verbatim: when `σ == 0` the training code
returns kurtosis `0.0` and skewness `0.0` (not NaN, not −3.0), and when
`rms == 0` the crest factor is `0.0`.

**On scipy:** the training pipeline does *not* call
`scipy.stats.kurtosis`/`skew` — it hand-rolls both in numpy. The
hand-rolled definitions happen to coincide with scipy's defaults
(`bias=True`, `fisher=True`), but the numpy implementation is the
normative one and is what both streaming implementations reproduce.

Spark's native `stddev_pop`, `kurtosis` and `skewness` use the same
population/biased definitions and the same excess-kurtosis convention,
which is why the job can compute features with plain SQL aggregates.
That equivalence is **measured, not assumed**.

### Parity test results

`ml/tests/test_streaming_pipeline.py` compares three independent
implementations on real CWRU windows:

| pair | max abs difference |
|---|---|
| offline numpy vs streaming online-moments | **5.3e-14** (kurtosis; others ≤ 1.2e-14) |
| offline numpy vs Spark SQL aggregates | **7.6e-15** |

Documented tolerance: **1e-9 absolute** — three orders of magnitude of
headroom over observed floating-point noise, while still catching any
genuine definitional divergence such as a ddof change or a
sample-vs-population moment. Edge cases (all-zero and constant windows)
match *exactly*, not just within tolerance.

## Model loading and feature order

Feature order is never inferred by position. For each model the order is
read from the experiment's saved JSON metadata and cross-checked against
the estimator's own `feature_names_in_`; a mismatch raises
`ModelContractError` rather than silently scoring a permuted vector,
which would yield confident and completely wrong predictions.

```
classifier : rf_multiseverity_cwru.joblib (RandomForestClassifier, 9 features)
             ['vibration_rms','vibration_std','vibration_peak',
              'vibration_peak_to_peak','vibration_kurtosis','vibration_skewness',
              'crest_factor','rotational_speed_rpm','motor_load_hp']
anomaly    : isolation_forest_cwru.joblib (Pipeline, 7 features)
             ['vibration_rms','vibration_std','vibration_peak',
              'vibration_peak_to_peak','vibration_kurtosis','vibration_skewness',
              'crest_factor']
             threshold = -0.615287  (train_quantile_0.01)
```

The Isolation Forest receives **exactly its seven vibration features** —
`motor_load_hp` and `rotational_speed_rpm` are structurally excluded, as
Experiment 3 decided. The Random Forest additionally receives the two
operating-context columns from drive telemetry, which a real plant knows.

The anomaly decision uses the **frozen Experiment-3 threshold**, not
sklearn's default `predict()` boundary, so online decisions match the
offline evaluation exactly.

## Inference result contract

```json
{
  "assetId": "MOTOR_003",
  "windowIndex": 0,
  "windowStartSequence": 0,
  "windowEndSequence": 2047,
  "timestampUtc": "...",
  "sampleCount": 2048,
  "features": {
    "vibrationRms": 0.0943, "vibrationStd": 0.0942,
    "vibrationPeak": 0.3862, "vibrationPeakToPeak": 0.6974,
    "vibrationKurtosis": 0.2054, "vibrationSkewness": 0.0167,
    "crestFactor": 4.0959
  },
  "operatingContext": { "motorLoadHp": 3.0, "rotationalSpeedRpm": 1723.0 },
  "anomaly": {
    "isAnomalous": true,
    "score": -0.7382,
    "threshold": -0.6153,
    "scoreDirection": "score_samples: HIGHER = more normal, LOWER = more anomalous; ANOMALOUS when score < threshold"
  },
  "classification": {
    "predictedClass": "OUTER_RACE",
    "confidence": 0.810,
    "probabilities": { "BALL": 0.19, "INNER_RACE": 0.0, "NORMAL": 0.0, "OUTER_RACE": 0.81 }
  },
  "groundTruth": {
    "faultClass": "OUTER_RACE", "faultSeverityIn": 0.014, "recordingId": "OR014@6_3"
  }
}
```

`scoreDirection` ships inside every message so the sign convention can
never be misread downstream.

## Running the pipeline

### Prerequisites

```bash
source .venv/bin/activate
pip install -r ml/requirements.txt -r ml/requirements-streaming.txt

# Model artifacts are gitignored — generate them once:
python -m ml.src.experiment2_multiseverity
python -m ml.src.experiment3_anomaly_detection
```

### Option A — local, no broker (fastest; what the demo uses)

Spark reads a JSONL file stream instead of Kafka. Every other stage is
identical.

```bash
# 1. all three scenarios, replay + Spark + scoring
python -m ml.streaming.demo_end_to_end

# or drive the stages by hand:
# 2. replay a recording to a file
python -m ml.streaming.replay_producer \
    --recording OR014@6_3 --asset MOTOR_003 --windows 5 \
    --sink file --out /tmp/vib/in/c_hard.jsonl

# 3. run the streaming job
python -m ml.streaming.spark_inference_job \
    --source file --input /tmp/vib/in \
    --checkpoint /tmp/vib/ckpt --output /tmp/vib/out --await-seconds 60

# 4. observe
cat /tmp/vib/out/*.jsonl | python -m json.tool
```

### Option B — full Docker stack with Kafka

```bash
# 1. start infrastructure (Kafka, SQL Server, Grafana, both Spark jobs)
docker compose up -d
docker logs industrial_kafka_init          # expect both topics listed

# 2. the inference job starts automatically; follow it
docker logs -f industrial_spark_vibration
open http://localhost:4041                 # Spark job UI

# 3. replay telemetry into Kafka from the host
python -m ml.streaming.replay_producer \
    --recording OR014@6_3 --asset MOTOR_003 --windows 5 \
    --sink kafka --bootstrap localhost:9092

# 4. observe inference results
tail -f spark/inference-output/*.jsonl

# 5. stop / clean up
docker compose stop spark-vibration        # just this job
docker compose down                        # everything
docker compose down -v                     # everything + volumes
```

`--speed` controls replay pacing: `0` (default) is as-fast-as-possible,
`1` is true 12 kHz acquisition rate. Demonstrations want `0`; a
10-second recording is 120,000 messages.

## Demonstration results

`python -m ml.streaming.demo_end_to_end`, full recordings, 355 windows:

| scenario | recording | windows | anomaly flagged | classification accuracy |
|---|---|---|---|---|
| A healthy | `Normal_3` | 237 | 4 / 237 (1.7%) | 237/237 NORMAL |
| B easy fault | `OR021@6_3` (0.021") | 59 | 59 / 59 (100%) | 59/59 OUTER_RACE |
| C hard fault | `OR014@6_3` (0.014") | 59 | 59 / 59 (100%) | **45/59 (76.3%)** |

**Scenario C's 14 errors are real and are not hidden.** All 14 are
`OUTER_RACE → BALL`, with confidences from 0.550 to 1.000 — the live
pipeline reproduces Experiment 2's offline finding *exactly* (14 of 59
`OR014@6_3` windows misclassified). Scenario A's 4 false alarms likewise
reproduce Experiment 3's offline 4/237 false-positive rate exactly. That
agreement is the strongest available evidence that the online feature
engineering matches the training pipeline.

Cross-model reading, live: of the 118 fault windows streamed, the
Isolation Forest flagged 118 (100%); of the 14 the Random Forest
mistyped, it still flagged 14 (100%). The detector says "something is
wrong" even where the classifier cannot say "what is wrong".

## Known limitations

1. **Kafka and Docker were not exercised in the session that built
   this.** The Docker daemon was unavailable, so the Kafka source path,
   `spark/Dockerfile.ml` and the `spark-vibration` compose service are
   written and reviewed but **not yet run end to end**. Everything else —
   replay, parsing, windowing, features, both models, result contract —
   was executed under a real `SparkSession` via the file source. The
   Kafka path differs only in the `readStream` format block.
2. The dataset limitations of Experiments 1–3 carry over unchanged:
   NORMAL recordings are 48 kHz and fault recordings 12 kHz with no
   resampling, so a 2048-sample window spans a different physical
   duration for each, and health state is confounded with sampling rate.
3. Replay is not a real sensor. Timestamps are synthesised from the
   nominal sampling rate; there is no jitter, dropout, drift or sensor
   noise, and the replay is a clean recording rather than a degrading
   machine.
4. `sequenceNumber` restarts at 0 for every replay run, so re-running a
   scenario re-emits window indices that Spark has already seen. The
   demo wipes its checkpoint each run; a long-lived deployment would key
   windows by a session id as well.
5. State in the streaming aggregation is unbounded — there is no
   watermark, because the window key is a sequence number rather than
   event time. Fine for bounded replay, not for an always-on asset.
6. Output mode is `update`, so a completed group can be re-emitted in a
   later micro-batch; the demo de-duplicates on
   `(assetId, windowIndex)`. A production sink would need idempotent
   upserts.
7. Inference runs in `foreachBatch` on the driver via `collect()`. That
   is fine at this volume (one window per 2048 samples) but would need a
   pandas UDF to distribute at fleet scale.
8. The two models load from local `.joblib` files with no version
   pinning or model registry. `spark/Dockerfile.ml` pins scikit-learn to
   the same range as `ml/requirements.txt` because a minor-version drift
   between pickling and unpickling is the classic cause of silent
   inference breakage.
9. Nothing is persisted to SQL Server yet, and there is no dashboard or
   LLM integration — all explicitly out of scope for this phase.

## Repository hygiene

`.gitignore` now covers `spark/checkpoints/**`, `spark/.ivy2/`,
`spark/metastore/`, `spark/hive-warehouse/` and
`spark/inference-output/`.

**~6,400 Spark runtime files were committed before those rules existed**
(the Ivy jar cache, the Derby metastore, Hive Parquet files and
checkpoint metadata) and a `.gitignore` rule does not untrack an
existing file. They are still tracked. Untracking them is a separate,
deliberately un-bundled cleanup:

```bash
git rm -r --cached spark/.ivy2 spark/metastore spark/hive-warehouse \
                   'spark/checkpoints/raw' 'spark/checkpoints/analytics'
git commit -m "chore(spark): untrack generated runtime state"
```

That removes them from the index only — working-tree files are kept and
Spark regenerates them on next boot.
