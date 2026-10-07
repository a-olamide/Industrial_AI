# Industrial Digital Twin - Machine Learning Workspace

This directory is the Python machine-learning workspace for the
Industrial Digital Twin / Predictive Maintenance project. It is
separate from the .NET services in `src/` and the Spark / Kafka
infrastructure in `spark/` and `docker-compose.yml`.

The first public ML dataset used here is the **Case Western Reserve
University (CWRU) Bearing Data Center** dataset. The initial fault
taxonomy is:

- `NORMAL`
- `INNER_RACE`
- `BALL`
- `OUTER_RACE`

## Directory layout

```
ml/
├── README.md                 <- this file
├── requirements.txt
├── data/
│   ├── raw/cwru/             <- drop downloaded CWRU .mat files here (gitignored)
│   └── processed/            <- feature parquet output (gitignored)
├── notebooks/
│   ├── 01_cwru_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_baseline_classification.ipynb
├── src/
│   ├── cwru_loader.py        <- .mat file loader
│   ├── feature_extraction.py <- 2048-sample time-domain windows
│   ├── dataset_builder.py    <- CWRU metadata + canonical feature frame
│   ├── split_dataset.py      <- group-aware train/val/test split
│   ├── train_baseline.py     <- Random Forest baseline
│   ├── evaluate.py           <- classification report + confusion matrix
│   ├── run_experiment.py     <- Experiment 1 (FROZEN 0.007" baseline)
│   ├── audit_expanded_dataset.py  <- 40-recording dataset audit
│   └── experiment2_multiseverity.py <- Experiment 2 (multi-severity)
├── tests/                    <- stdlib unittest contract checks
├── models/                   <- trained artifacts (gitignored)
└── reports/figures/          <- generated plots
```

## Environment setup (macOS, Python 3)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r ml/requirements.txt
```

The virtual environment (`.venv/`) is intentionally not committed. The
same applies to any downloaded raw datasets or generated artifacts.

To run the notebooks:

```bash
source .venv/bin/activate
jupyter lab ml/notebooks
```

## Data placement

CWRU `.mat` files must be downloaded manually from
<https://engineering.case.edu/bearingdatacenter> and placed in
`ml/data/raw/cwru/`. The initial 16 recordings for this milestone are
listed in `ml/src/dataset_builder.py::CWRU_RECORDINGS` together with
their vendor-documented metadata (fault class, motor load, approximate
RPM, fault diameter, outer-race position where applicable).

The dataset builder accepts either the friendly filenames
(`Normal_0.mat`, `IR007_0.mat`, ...) or the raw numeric CWRU filenames
(`97.mat`, `105.mat`, ...). If neither is present the recording is
skipped and `resolve_recording_paths` will report it.

## Canonical ML feature contract

Every window emitted by the offline pipeline and (later) by the online
Spark Structured Streaming pipeline conforms to this schema:

| column                       | type   | notes |
|------------------------------|--------|-------|
| `source`                     | string | provenance tag, e.g. `CWRU_12kHz_DE` or `KAFKA_ASSET_STREAM` |
| `asset_id`                   | string | physical asset identifier |
| `recording_id`               | string | offline: CWRU recording tag; online: session/stream id |
| `window_id`                  | int    | monotonically increasing per recording/stream |
| `vibration_rms`              | float  | RMS of window samples |
| `vibration_std`              | float  | population std |
| `vibration_peak`             | float  | absolute peak |
| `vibration_peak_to_peak`     | float  | max - min |
| `vibration_kurtosis`         | float  | excess kurtosis |
| `vibration_skewness`         | float  |       |
| `crest_factor`               | float  | peak / RMS, defined as 0 when RMS = 0 |
| `rotational_speed_rpm`       | float  | measured RPM if available, else vendor-approximate |
| `motor_load_hp`              | float  | 0-3 for CWRU |
| `fault_class`                | string | one of `NORMAL`, `INNER_RACE`, `BALL`, `OUTER_RACE` |

This is an **ML feature contract**. It is deliberately separate from
the operational telemetry SQL schema used by the .NET services. No SQL
migration is added by this branch.

## Offline training pipeline

```
CWRU 12 kHz Drive End .mat
    -> scipy.io.loadmat -> flatten *_DE_time signal
        -> 2048-sample non-overlapping windows
            -> time-domain feature extraction
                -> canonical MachineFeatureVector (pandas DataFrame)
                    -> group-aware train / validation / test split
                        (by recording_id, so windows from the same
                         experiment never span two splits)
                        -> Random Forest baseline
                            -> classification report + confusion matrix
                                -> later: model comparison (SVM, GBM, MLP)
```

## Online pipeline (future integration)

```
Industrial simulator
    -> Kafka                                 (real-time event transport)
        -> Spark Structured Streaming        (windowing + feature engineering)
            -> canonical MachineFeatureVector
                -> trained ML model          (fault / anomaly prediction)
                    -> SQL Server            (prediction persistence)
                        -> Digital Twin UI
                            -> Claude API    (evidence-grounded explanation)
```

Role of each component:

- **Kafka** - real-time event transport between the simulator and the
  streaming feature engineering job.
- **Spark Structured Streaming** - windowing and feature engineering on
  live telemetry; must emit the same MachineFeatureVector as the
  offline pipeline for the trained model to apply.
- **Machine Learning** - fault / anomaly learning offline and
  prediction online.
- **SQL Server** - operational and prediction persistence for the
  Digital Twin UI.
- **Claude API** - evidence-grounded explanation and maintenance
  guidance. Claude is **not** the ML predictor; it consumes the model's
  output plus operational context to produce human-readable narrative.

Hive is **not** part of the ML architecture in this branch.

## Train / validation / test policy

Multiple 2048-sample windows sliced from the same CWRU recording are
strongly correlated. Random per-row splitting would place adjacent
windows on both sides of the train/test boundary and produce
optimistic-but-invalid accuracy numbers.

For this initial 16-recording baseline we use an **explicit
recording-and-load-aware split** rather than random group shuffling:

| split      | motor loads | recordings | recordings/class |
|------------|-------------|-----------|-------------------|
| train      | 0 HP, 1 HP  | 8         | 2                 |
| validation | 2 HP        | 4         | 1                 |
| test       | 3 HP        | 4         | 1                 |

That is a **50 / 25 / 25 recording-level split**, not 70/15/15. It is
selected because only four independent operating-load recordings exist
per fault class at this milestone; two loads for train and one load
each for validation and test is the finest recording-level partition
that keeps every fault class in every split without leaking windows
between splits. `ml/src/split_dataset.py::load_aware_split` asserts
both properties at run time.

The generic `group_train_val_test_split` helper (`GroupShuffleSplit`
under the hood) remains available for future experiments that add
additional fault diameters (0.014", 0.021"), the other outer-race
positions (3 o'clock, 12 o'clock), and the fan-end fault set — at
which point broader group-aware cross-validation becomes feasible.

## Running the baseline end-to-end

Once the 16 `.mat` files are in `ml/data/raw/cwru/`:

```bash
source .venv/bin/activate
python -m ml.src.run_experiment
```

That single command produces:

- `ml/data/processed/cwru_features.csv` (gitignored)
- `ml/models/rf_baseline_cwru.joblib` + `.json` metadata (gitignored)
- All figures under `ml/reports/figures/` (committed)

## Dataset expansion (pre-Experiment-2)

The `CWRU_RECORDINGS` metadata table in `ml/src/dataset_builder.py`
now covers **40 recordings** across three fault severities:

| fault_class  | severities         | motor loads   | recordings |
|--------------|--------------------|---------------|------------|
| NORMAL       | -                  | 0, 1, 2, 3 HP | 4          |
| INNER_RACE   | 0.007", 0.014", 0.021" | 0, 1, 2, 3 HP | 12         |
| BALL         | 0.007", 0.014", 0.021" | 0, 1, 2, 3 HP | 12         |
| OUTER_RACE@6 | 0.007", 0.014", 0.021" | 0, 1, 2, 3 HP | 12         |

The 16-recording 0.007" subset used by the frozen Experiment-1
baseline is available as `BASELINE_SPECS` and remains the sole input
to `python -m ml.src.run_experiment`. Nothing about the baseline
model, metrics, figures or CSV is regenerated by the expansion.

Inspect / build features / plot severity-conditioned distributions
for the full 40-recording set with:

```bash
python -m ml.src.audit_expanded_dataset
```

That script writes `ml/data/processed/cwru_features_expanded.csv`
(gitignored) and three exploratory figures with the `expanded_`
prefix under `ml/reports/figures/`. It does **not** split, train,
or persist any model - Experiment 2 will be designed after reviewing
the audit output.

## Experiment 2 - multi-severity fault classification

**Research question:** can a supervised classifier trained across
*multiple* bearing-fault severities (0.007", 0.014", 0.021")
generalize to an unseen motor load?

```bash
python -m ml.src.experiment2_multiseverity
```

Experiment 1 is frozen and is NOT re-run, re-fit or overwritten by
this command. Experiment 2 writes only `experiment2_*` figures and the
`rf_multiseverity_cwru.*` artifact pair.

### Split design

Strictly by complete recording / motor load - never by random rows:

| split      | motor load | recordings | windows | severities present |
|------------|-----------|------------|---------|--------------------|
| train      | 0 + 1 HP  | 20         | 1,417   | 0.007, 0.014, 0.021 |
| validation | 2 HP      | 10         | 767     | 0.007, 0.014, 0.021 |
| test       | 3 HP      | 10         | 769     | 0.007, 0.014, 0.021 |

`ml/src/split_dataset.py::multiseverity_load_split` asserts at run time
that (1) no `recording_id` appears in more than one split, (2) all four
target classes appear in every split, (3) no individual feature window
crosses a split boundary, and (4) all three fault severities appear in
the fault recordings of every split.

### Feature contract and the severity-leakage rule

`X` is exactly `ml/src/train_baseline.py::FEATURE_COLUMNS` - the same
nine time-domain columns Experiment 1 uses. `y` is `fault_class`.

`fault_severity_in` is **experimental metadata and is never a model
input.** The physical defect diameter of an unknown machine is not
observable at diagnosis time, so feeding it to the classifier would
leak information production never has. `recording_id`, `window_id`,
`source`, `asset_id` and `sampling_rate_hz` are excluded for the same
family of reasons - `sampling_rate_hz` in particular is 48 kHz for
every NORMAL recording and 12 kHz for every fault recording, so it
would act as a direct NORMAL-vs-FAULT label.

### Measured results

**Validation (2 HP, 767 windows)**

| model               | validation_accuracy | validation_macro_f1 |
|---------------------|--------------------|---------------------|
| random_forest       | 0.9804             | 0.9789              |
| gradient_boosting   | 0.9505             | 0.9470              |
| logistic_regression | 0.8618             | 0.8514              |

`random_forest` is selected on validation macro-F1 alone; the gap to
the runner-up (0.0319) is far outside the documented 0.005 tie
tolerance, so the "prefer the simpler model on a tie" rule does not
apply. Test data played no part in the choice.

Class-imbalance handling is **not** symmetric across the three
candidates and this is deliberate: Logistic Regression and Random
Forest use `class_weight="balanced"`; sklearn's
`GradientBoostingClassifier` has no `class_weight` parameter at all.
The only equivalent is `sample_weight` at `fit()` time, which this
experiment does not pass, so GB trains under the raw class priors.

**Test (3 HP held-out, 769 windows, Random Forest)**

- accuracy 0.9649, macro-precision 0.9649, macro-recall 0.9619,
  macro-F1 0.9625.
- 27 / 769 windows misclassified. **Every single error is a 0.014"
  recording.**

| severity | windows | accuracy | macro-F1 | IR recall | BALL recall | OR recall |
|----------|---------|----------|----------|-----------|-------------|-----------|
| 0.007"   | 178     | 1.0000   | 1.0000   | 1.000     | 1.000       | 1.000     |
| 0.014"   | 177     | 0.8475   | 0.8534   | 0.864     | 0.915       | 0.763     |
| 0.021"   | 177     | 1.0000   | 1.0000   | 1.000     | 1.000       | 1.000     |

NORMAL is not a severity bucket (recall 1.0000 on 237 windows).

Confusion patterns, all inside 0.014": `OUTER_RACE -> BALL` (14
windows, OR014@6_3), `INNER_RACE -> BALL` (8, IR014_3),
`BALL -> INNER_RACE` (4, B014_3), `BALL -> NORMAL` (1, B014_3).

### Why 0.014" is the hard severity

The separability probe in the same run explains it. On the test split
the 0.014" recordings collapse the amplitude ordering that separates
the classes at the other two severities:

| class       | 0.007" mean RMS | 0.014" mean RMS | 0.021" mean RMS |
|-------------|-----------------|-----------------|-----------------|
| BALL        | 0.154           | 0.130           | 0.118           |
| INNER_RACE  | 0.314           | 0.181           | 0.448           |
| OUTER_RACE  | 0.580           | 0.094           | 0.555           |

At 0.014" the OUTER_RACE signal is *quieter* than BALL, inverting the
relationship the model learned from the other severities. NORMAL stays
cleanly separated everywhere (RMS 0.060-0.071 on test, no overlap with
any fault class), which is why NORMAL recall is 1.000 throughout.

This is also the answer to the Experiment-1 "suspiciously easy"
concern. Single-feature depth-3 decision trees now reach only
0.86-0.89 test accuracy (`vibration_rms` 0.8648, `vibration_std`
0.8648, `vibration_peak` 0.8934) versus the full model's 0.9649, and
the test RMS intervals of INNER_RACE, BALL and OUTER_RACE all mutually
overlap. Adding 0.014" and 0.021" turned the trivially-separable
Experiment-1 task into one where the classifier is doing real work -
though NORMAL-vs-FAULT remains trivial.

### Random Forest feature importance (selected model)

1. `vibration_peak_to_peak` 0.255
2. `vibration_std` 0.252
3. `vibration_rms` 0.217
4. `vibration_peak` 0.147
5. `vibration_kurtosis` 0.054
6. `vibration_skewness` 0.027
7. `crest_factor` 0.024
8. `rotational_speed_rpm` 0.021
9. `motor_load_hp` 0.003

The top four are strongly correlated amplitude statistics, so they
*share* importance - a low score does not prove a feature is
uninformative. Importance is model-specific (Gradient Boosting puts
0.537 on `vibration_peak_to_peak` alone) and is not physical
causation.

### Figures

All prefixed `experiment2_` so no baseline or audit figure is
overwritten:

- `experiment2_validation_model_comparison.png`
- `experiment2_validation_confusion_logistic_regression.png`
- `experiment2_validation_confusion_random_forest.png`
- `experiment2_validation_confusion_gradient_boosting.png`
- `experiment2_test_confusion_matrix.png`
- `experiment2_selected_model_interpretability.png`
- `experiment2_test_performance_by_severity.png`
- `experiment2_rms_separability_by_class.png`

### Artifacts

- `ml/models/rf_multiseverity_cwru.joblib` (gitignored)
- `ml/models/rf_multiseverity_cwru.json` (gitignored) - experiment
  name, research question, split design, loads, severities, feature
  columns, selected model, validation + test metrics, per-severity
  breakdown, misclassification list, interpretability, separability
  probe, class labels and known limitations.

`rf_baseline_cwru.joblib` / `.json` are never written by this module.

### Experiment-2 limitations

1. CWRU is a controlled laboratory bearing dataset on a test rig, not
   an in-service industrial fleet.
2. Faults are **seeded** (machined defects of known diameter), not
   naturally occurring progressive degradation.
3. NORMAL recordings are published at 48 kHz while the selected fault
   recordings are 12 kHz Drive End. **No resampling is performed.**
4. A fixed 2048-sample window therefore spans ~42.7 ms for NORMAL and
   ~170.7 ms for fault recordings - different physical durations for
   the same nominal window size.
5. All features are time-domain window statistics. No frequency-domain
   features are used.
6. These numbers are **not** production-level industrial performance.
7. The experiment tests generalization to an unseen **motor load**
   only. All three severities are present during training, so it is
   NOT a test of generalization to an unseen defect size.
8. Windows from one recording are strongly correlated, which is why
   splitting is by complete recording rather than by random rows.
9. `rotational_speed_rpm` is nearly a deterministic function of motor
   load (1797/1772/1750/1730 RPM for 0/1/2/3 HP) and `motor_load_hp`
   is constant within a recording. Both are retained for parity with
   the inherited feature contract, but at test time both take values
   never seen in training, so neither can contribute usable signal to
   load generalization. Their near-zero Random Forest importance is
   consistent with that.
10. `fault_severity_in` is excluded from `X` by design.

## Tests

Lightweight contract checks live in `ml/tests/` and use the standard
library `unittest` runner, so no extra dependency is needed:

```bash
python -m unittest discover -s ml/tests -t . -v
```

They cover recording leakage, window leakage, expected split loads,
class coverage, severity coverage, `fault_severity_in` exclusion from
`X`, metadata exclusion from `X`, and the invariant that
`BASELINE_SPECS` still describes exactly the original 16 Experiment-1
recordings. Most tests run against a synthetic frame built from the
real metadata table, so they pass without the vendor `.mat` downloads;
the few that need `cwru_features_expanded.csv` skip themselves when it
is absent.

## Current Experimental Status (Experiment 1 - FROZEN)

Numbers below are the actual measured output from
`python -m ml.src.run_experiment` on the 16-recording CWRU set with
the 50/25/25 load-aware split described above. They are regenerated
whenever the script runs; nothing here is hard-coded.

**Dataset**
- 16 recordings loaded (4 classes × 4 motor loads).
- Total 2048-sample non-overlapping windows: **1,537**.
- Windows per class: NORMAL 828, INNER_RACE 237, BALL 236, OUTER_RACE 236.
  Class imbalance is a consequence of Normal recordings being ~4×
  longer (48 kHz for ~19 s) than the fault recordings (12 kHz for
  ~10 s); `class_weight="balanced"` compensates during training.

**Validation (motor load = 2 HP, 413 windows across 4 recordings)**
- accuracy = 1.0000, macro-precision = 1.0000, macro-recall = 1.0000, macro-F1 = 1.0000.
- Confusion matrix: perfectly diagonal, 236/59/59/59 per class.

**Test (motor load = 3 HP, 415 windows across 4 recordings)**
- accuracy = 1.0000, macro-precision = 1.0000, macro-recall = 1.0000, macro-F1 = 1.0000.
- Confusion matrix: perfectly diagonal, 237/60/59/59 per class.

**Random Forest feature importance (mean decrease in Gini impurity)**
1. `vibration_std` ~ 0.234
2. `vibration_peak` ~ 0.227
3. `vibration_rms` ~ 0.212
4. `vibration_peak_to_peak` ~ 0.178
5. `vibration_kurtosis` ~ 0.105
6. `crest_factor` ~ 0.023
7. `vibration_skewness` ~ 0.020
8. `rotational_speed_rpm` ~ 0.003
9. `motor_load_hp` ~ 4e-16 (effectively unused, as expected — this
   feature is constant *within* every recording, so the forest cannot
   use it to separate classes)

**Caveats / interpretation**

- 100% is real for this split but should NOT be presented as a
  production quality signal. The 0.007" fault severity produces
  dramatically different vibration amplitudes across the four classes
  (RMS ranges: NORMAL 0.06–0.08, BALL 0.13–0.16, INNER_RACE 0.28–0.33,
  OUTER_RACE 0.54–0.71). At this severity the classes are trivially
  separable by any single amplitude statistic; the RF is not doing
  much heavy lifting. Adding 0.014" and 0.021" fault severities and/or
  additional outer-race positions will make the task materially
  harder.
- Feature importance does not imply causality. It only reports which
  features the trees split on.
- The `Normal_2.mat` download from CWRU contains **both** `X098_*`
  and `X099_*` variables; `ml/src/cwru_loader.py` uses the
  `experiment_number` field on each `RecordingSpec` to pick the
  correct series (`X099_DE_time` for Normal_2 at 2 HP). Without this
  disambiguation, Normal_2 silently duplicates Normal_1's signal.
- Sampling-rate is NOT stored in the `.mat` files. The Normal set is
  published at 48 kHz while the 0.007" fault set (12 kHz Drive End)
  is at 12 kHz. Every `RecordingSpec` records the vendor sampling
  rate; the current time-domain baseline does not consume it, but any
  future frequency-domain feature must respect it.

## Bridge to the online pipeline

```
CWRU .mat        --------- offline ---------->  trained rf_baseline_cwru.joblib
    ^                                                  |
    |                                                  v
    +----- same MachineFeatureVector contract ---------+
    |                                                  |
Kafka simulator ---------> Spark Structured Streaming -+---> live prediction
                          (equivalent 2048-sample                     |
                           time-domain feature engineering)           v
                                                              SQL Server + UI
                                                                      |
                                                                      v
                                                            Claude API narration
```

The offline model is portable to the online pipeline **iff** the
Spark job emits rows that match `FEATURE_COLUMNS` in
`ml/src/train_baseline.py`. That is the contract.

## Non-goals for this branch

- No frequency-domain features (FFT bands, envelope spectrum, bearing
  characteristic frequencies).
- No SQL migrations for the ML feature schema.
- No online integration with Kafka/Spark.
- No hyperparameter search and no model registry. (Experiment 2 does
  compare three candidate classifiers on validation, but with fixed
  baseline configurations only.)
- No anomaly detection and no Claude/LLM integration yet.
- No automated download of the CWRU dataset.
