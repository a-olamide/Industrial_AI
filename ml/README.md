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
│   └── evaluate.py           <- classification report + confusion matrix
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

## Current Experimental Status

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
- No model comparison, hyperparameter search, or model registry.
- No automated download of the CWRU dataset.
