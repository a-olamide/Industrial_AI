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

## Verified Kafka run

Executed against a live broker, not the file source:

```
$ docker exec industrial_kafka kafka-get-offsets.sh --topic industrial.telemetry.vibration
industrial.telemetry.vibration:0:121991      # MOTOR_003
industrial.telemetry.vibration:1:0
industrial.telemetry.vibration:2:491787      # MOTOR_001 + MOTOR_002
```

Messages are keyed by `assetId`, so each asset's samples land on a single
partition and per-asset ordering is preserved.

Spark's checkpoint confirms it consumed from the broker rather than a
file:

```
$ cat <checkpoint>/offsets/0
{"industrial.telemetry.vibration":{"2":6144,"1":0,"0":0}}
```

The containerized `spark-vibration` service produced identical scores to
the earlier file-source run (−0.7897 / −0.7773 / −0.7782 for
`OR021@6_3` windows 0–2), which is what makes the two paths
interchangeable.

### Python version note

`apache/spark:3.5.1` ships Python 3.8.10, but the models were pickled
under Python 3.9 with scikit-learn 1.6.1, and scikit-learn ≥ 1.6
requires Python ≥ 3.9 — so the artifacts simply cannot be unpickled in
the stock image. `spark/Dockerfile.ml` installs Python 3.9 from
deadsnakes and points `PYSPARK_PYTHON` at it, pinning numpy, pandas,
scipy, scikit-learn and joblib to exactly the versions that produced the
pickles.

The build deliberately does **not** `apt-get purge`/`autoremove` the
install tooling afterwards: doing so strips shared libraries the numpy
and pandas C extensions link against, and the failure surfaces much
later as a confusing `partially initialized module 'pandas'` ABI error.

## Digital Twin persistence

Each completed window is written to SQL Server by the same Spark job:

| table | shape |
|---|---|
| `dbo.asset_twin_current` | one row per asset, upserted (MERGE) |
| `dbo.asset_twin_inference_history` | append-only, one row per window |

**Why direct JDBC.** `spark/jobs/industrial_streaming_analytics.py`
already writes six tables this way and already runs a py4j MERGE for
`asset_risk_current`, so this follows an established path: no new
broker, database, service or HTTP hop, and the current-state/history
pair mirrors the existing `asset_risk_current`/`asset_risk_minute`
shape. A REST ingestion endpoint was the alternative and was rejected —
it adds a network hop and a second deployable for data Spark can already
write. Volume makes it safe: one row per 2048 samples, ~6 rows/second
per asset even at full 12 kHz replay.

**Column groups** are load-bearing. Model output (`is_anomalous`,
`anomaly_*`, `predicted_class`, `confidence`), inputs (`vibration_*`,
`motor_load_hp`, `rotational_speed_rpm`) and demo ground truth are
separated, and every ground-truth column carries a `demo_` prefix so it
cannot be mistaken for a model output in a query, a DTO or on screen.

**Idempotency.** History inserts are guarded by
`WHERE NOT EXISTS (asset_id, window_end_sequence)` behind a unique
index, because Spark's `update` output mode can re-emit a completed
group and Kafka delivery is at-least-once. Verified: replaying
`OR021@6_3` a second time took MOTOR_002 from 3 to 59 history rows, not
62.

**Current state** is last-write-wins, with rows applied in
`(asset_id, window_end_sequence)` order within a batch so the newest
window of a batch survives.

**One JDBC gotcha, documented in code:** `--packages` puts mssql-jdbc on
Spark's *context* classloader, but `java.sql.DriverManager` only
consults drivers registered with the *system* classloader and answers
"No suitable driver found" even though the jar is demonstrably loaded.
`twin_sink.py` instantiates the driver and calls `connect` directly
instead.

## Digital Twin API and UI

| endpoint | returns |
|---|---|
| `GET /api/v1/digital-twins` | current state for every asset |
| `GET /api/v1/digital-twins/{assetId}` | current state for one asset |
| `GET /api/v1/digital-twins/{assetId}/history?take=N` | recent inferences, newest first |

All accept `?includeGroundTruth=false` to omit the demo block. Responses
separate `anomaly` and `classification` (model output) from `features`
and `operatingContext` (inputs) and from `demoGroundTruth`. Raw
2048-sample arrays are never exposed — only the window's sequence range.

Blazor page: **`/digital-twin`** (nav: "ML Digital Twin"). Shows health,
anomaly score, predicted condition, confidence with the full class
distribution, operating context, the four headline features and last
updated; ground truth sits in a visually distinct bordered panel labelled
"Demo scenario / ground truth — not a model output and never a model
input", with an explicit MATCHES/DIFFERS verdict. The history table
highlights disagreements in red.

A future explanation service has a reserved seam in
`HealthExplanationContextDto` / `HealthExplanationContextFactory`. It
carries model output, features, operating context and window
provenance — and deliberately **not** ground truth. Nothing calls an LLM
in this phase.

## Full end-to-end demo

```bash
# 1. infrastructure
docker compose up -d kafka sqlserver
docker compose up kafka-init sqlserver-init          # topics + schema
docker compose up -d --build spark-vibration         # Kafka -> ML -> SQL

# 2. API + UI (net9.0 projects on a .NET 10 runtime)
DOTNET_ROLL_FORWARD=Major dotnet run \
    --project src/IndustrialAnalytics.Api --urls http://localhost:5025 &
DOTNET_ROLL_FORWARD=Major dotnet run \
    --project src/IndustrialAnalytics.Ui  --urls http://localhost:5080 &

# 3. replay the three demo scenarios
python -m ml.streaming.replay_producer --recording Normal_3  --asset MOTOR_001 \
    --sink kafka --bootstrap localhost:9092
python -m ml.streaming.replay_producer --recording OR021@6_3 --asset MOTOR_002 \
    --sink kafka --bootstrap localhost:9092
python -m ml.streaming.replay_producer --recording OR014@6_3 --asset MOTOR_003 \
    --sink kafka --bootstrap localhost:9092

# 4. observe
docker logs -f industrial_spark_vibration            # live inference
curl -s http://localhost:5025/api/v1/digital-twins | python -m json.tool
open http://localhost:5080/digital-twin              # the UI

# 5. stop / clean up
docker compose down          # stop everything
docker compose down -v       # also drop Kafka + SQL volumes
```

### Demo results through the real pipeline

| asset | recording | windows | anomaly flagged | classifier correct |
|---|---|---|---|---|
| MOTOR_001 | `Normal_3` | 237 | 4 (1.7% false alarms) | 237/237 NORMAL |
| MOTOR_002 | `OR021@6_3` 0.021" | 59 | 59/59 | 59/59 OUTER_RACE |
| MOTOR_003 | `OR014@6_3` 0.014" | 59 | 59/59 | **45/59** |

MOTOR_003's 14 `OUTER_RACE → BALL` errors and MOTOR_001's 4 false alarms
match the offline Experiment-2 and Experiment-3 numbers exactly, now
through Kafka, Spark, SQL Server and the API. The UI shows the
disagreements rather than hiding them.

## Claude maintenance explanations — "ML predicts; Claude explains"

The final layer turns the Digital Twin's numbers into prose. **Claude is
not the detector and not the classifier.** The Isolation Forest and the
Random Forest remain the sole authorities on whether something is
anomalous and which fault class it is; Claude only translates their
output into a maintenance explanation, and the API response echoes the
ML verdict alongside the prose so the two can always be compared.

```
Digital Twin state (SQL)  ──►  narrow evidence bundle  ──►  Claude  ──►  structured explanation
   authoritative ML output        no ground truth              Haiku 4.5      summary / condition /
                                                                              evidence / actions /
                                                                              confidence note
```

### Configuration

| setting | where | default |
|---|---|---|
| API key | **`ANTHROPIC_API_KEY` environment variable** (or user secrets) | — |
| `Claude:Model` | `appsettings.json` | `claude-haiku-4-5` |
| `Claude:MaxTokens` | `appsettings.json` | `1024` |
| `Claude:TimeoutSeconds` | `appsettings.json` | `30` |
| `Claude:TrendWindowCount` | `appsettings.json` | `60` |
| `Claude:CacheEnabled` / `CacheMinutes` | `appsettings.json` | `true` / `30` |

```bash
export ANTHROPIC_API_KEY="sk-ant-..."        # never committed, never in appsettings.json
# or, for local dev:
dotnet user-secrets set "Claude:ApiKey" "sk-ant-..." \
    --project src/IndustrialAnalytics.Api
```

The key is read once at service construction and held server-side only.
It is never logged (the startup line says whether a key was *found*,
never any part of it), never placed in a DTO, and never reachable from
Blazor or the browser — the client posts an asset id and nothing else.

### What Claude receives — and what it never receives

Sent (`ClaudeExplanationRequestDto`, ~750 characters):

- `assetId`, `observedAtUtc`
- `anomaly`: `isAnomalous`, `score`, `threshold`, score-direction note
- `classification`: `predictedClass`, `confidence`, full `probabilities`
- `features`: the seven engineered vibration features
- `operatingContext`: `motorLoadHp`, `rotationalSpeedRpm`
- `recentTrend`: counts only — windows considered, windows flagged,
  predicted-class counts

**Never sent, and asserted by tests:** raw 2048-sample vibration arrays,
the CWRU filename, `recordingId`, `demoFaultClass`, `demoFaultSeverityIn`
— any demo ground truth.

This is enforced at three levels, not by convention:

1. The service loads twin state with `includeGroundTruth: false`, so the
   `demo_*` columns never leave the database on this path.
2. The request DTO copies named fields rather than spreading the state
   object, so adding a demo column later cannot silently widen the prompt.
3. Tests assert the serialized payload contains no `demoGroundTruth`,
   `recordingId`, `faultSeverityIn`, `OR014`, or `0.014`.

Why it matters: ground truth is a classroom evaluation aid. If it reached
the model, the explanation would be a restatement of the answer key
rather than a reading of the evidence — and the MOTOR_003 demonstration
below would prove nothing.

### Prompt constraints

The system prompt tells Claude it is not the detector or classifier, and
forbids: changing the anomaly result, changing `predictedClass`, claiming
certainty beyond the supplied confidence, inventing sensor readings or
maintenance history, and proposing unsupported causes. It requires
distinguishing a model *prediction* from a confirmed diagnosis,
recommending inspection rather than asserting that costly or destructive
work is necessary, stating uncertainty explicitly when confidence is low
or probabilities are close, explaining detector/classifier disagreement
rather than hiding it, and *not* inventing a fault when both models say
normal.

### Structured output

The response is constrained by a JSON schema (`output_config.format`), so
the five UI fields — `summary`, `likelyCondition`, `evidence`,
`recommendedActions`, `confidenceNote` — are guaranteed rather than
parsed hopefully. Output is still validated afterwards: an unparseable or
empty-summary response raises `ClaudeMalformedResponseException` and the
page shows an error, and a partial-but-valid response is filled with safe
defaults. **Malformed AI output can never break the Digital Twin page.**

### Endpoint

```
POST /api/v1/digital-twins/{assetId}/explanation[?refresh=true]
```

The caller supplies only an asset id. The server loads the authoritative
state itself, so a client cannot submit fabricated ML values and have the
server narrate them as real.

| condition | status |
|---|---|
| unknown asset | `404` |
| `ANTHROPIC_API_KEY` not configured | `503` (server is fine; the integration is off) |
| Claude timeout / upstream failure / refusal | `502` |
| response fails schema validation | `502` |

### UI

`/digital-twin` gains an **AI Maintenance Insight** section per asset,
behind an explicit **Generate AI Insight** button — explanations are not
produced for every page refresh or every streaming window. It shows
Assessment, Likely Condition, Evidence, Recommended Actions and a
Confidence Note, labelled *"an AI-generated explanation of the ML results
above, not an independent diagnosis"*, with loading and error states and
a footer restating the ML verdict being explained.

### The MOTOR_003 case

`OR014@6_3` is the demonstration that matters. The Random Forest gets
45/59 windows right and calls 14 of them `BALL`, so the current twin
state often reads `BALL` while the demo ground truth says `OUTER_RACE`.

Claude never sees `OUTER_RACE` as a ground-truth label. It sees
`predictedClass: "BALL"` with `probabilities: {BALL: 0.68, OUTER_RACE:
0.32, ...}` — the classifier's own distribution — and the prompt requires
it to explain what the classifier said and to state the uncertainty when
the top two probabilities are close. It cannot produce `OUTER_RACE` from
demo metadata, because the metadata is not in the request. A test asserts
this directly.

### Cost control

Measured request size for a MOTOR_003 explanation:

| part | chars | ≈ tokens |
|---|---|---|
| system prompt | 2,788 | 697 |
| response schema | 1,440 | 360 |
| evidence bundle | 748 | 187 |
| **total input** | **4,976** | **~1,244** |

At Claude Haiku 4.5 ($1/$5 per MTok) with a ~350-token answer, that is
roughly **$0.003 per explanation**; the same call on `claude-opus-5`
would be ~$0.015. Haiku 4.5 is the default because this is a short,
schema-constrained explanation of nine numbers for a dashboard card —
switch via `Claude:Model` if a deployment values explanation depth over
cost.

Four controls keep spend bounded: explanations are **on-demand only**
(never per 2048-sample window), `MaxTokens` is capped at 1024, the
evidence bundle carries counts rather than history rows, and results are
**cached against `(assetId, windowEndSequence)`** so repeated requests for
an unchanged twin state cost nothing. `?refresh=true` bypasses the cache.

### Live verification

**No live Claude call was made while building this.** `ANTHROPIC_API_KEY`
was not set in the build environment and no `ant` CLI profile existed, so
the integration is verified by 41 mocked tests plus the live
unconfigured-path checks below. To verify against the real API yourself:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
DOTNET_ROLL_FORWARD=Major dotnet run --project src/IndustrialAnalytics.Api \
    --urls http://localhost:5025 &

curl -s -X POST http://localhost:5025/api/v1/digital-twins/MOTOR_003/explanation \
  | python3 -m json.tool
```

Verified live without a key (real API, real HTTP):

```
POST /api/v1/digital-twins/MOTOR_003/explanation     -> 503
  {"title":"AI explanation unavailable",
   "detail":"ANTHROPIC_API_KEY is not configured on the server."}
POST /api/v1/digital-twins/NO_SUCH_ASSET/explanation -> 404
  {"error":"No Digital Twin state for asset 'NO_SUCH_ASSET'."}
```

Zero key-like strings appear in the API log.

### Security boundaries

- Claude calls originate **server-side only**; the Blazor client posts an
  asset id and receives rendered text.
- The API key never enters a DTO, a log line, or an HTTP response.
- The client cannot supply ML values for narration.
- Demo ground truth is excluded at the query, the DTO and the test level.
- Claude's output is validated before it reaches the UI.

### Limitations

- Explanation quality is unverified against a live model — the shape is
  guaranteed by the schema, but no human has read a real answer yet.
- The cache is in-process (`IMemoryCache`); a multi-instance deployment
  would re-bill once per instance.
- There is no rate limit on the endpoint beyond the cache; a determined
  caller can spend money by passing `refresh=true` repeatedly.
- The explanation is not persisted — it is not part of the Digital Twin
  state and is lost on restart.
- Trend counts are the only history given to Claude; it cannot see when a
  prediction changed, only how often each class occurred.

## Known limitations

1. ~~Kafka and Docker were not exercised.~~ **Now verified.** The full
   path — replay → real Kafka broker → containerized Spark → both models
   → SQL Server — has been executed end to end. See "Verified Kafka run"
   below.
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
