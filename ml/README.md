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

Instead, we partition **by `recording_id`** using
`sklearn.model_selection.GroupShuffleSplit`. Every window from a given
recording lands in exactly one split. Final split ratios (70/15/15,
60/20/20, ...) will be tuned once all 16 recordings are downloaded and
the class-by-load matrix is visible, because stratifying 16 groups
across four classes and four load levels is a hard constraint.

## Non-goals for this branch

- No frequency-domain features (FFT bands, envelope spectrum, bearing
  characteristic frequencies).
- No SQL migrations for the ML feature schema.
- No online integration with Kafka/Spark.
- No model comparison, hyperparameter search, or model registry.
- No automated download of the CWRU dataset.
