"""Online streaming inference: CWRU replay -> Kafka -> Spark -> ML.

Operationalises the frozen Experiment-2 (Random Forest fault
classifier) and Experiment-3 (Isolation Forest anomaly detector)
artifacts against live vibration telemetry. Nothing in this package
trains, refits or modifies a model.

See ``ml/streaming/README.md`` for the message contracts, the windowing
rule, the feature-parity results and how to run the pipeline.
"""
