"""spark-submit entry point for the vibration inference stream.

Thin launcher. All logic lives in ``ml/streaming/spark_inference_job.py``
so the same code runs under spark-submit in Docker and under
``python -m`` locally, and so the unit tests exercise the real thing.

The ``ml`` package is mounted at ``/opt/ml`` and reached via
``PYTHONPATH=/opt`` (see docker-compose.yml).
"""

import sys

from ml.streaming.spark_inference_job import main

if __name__ == "__main__":
    sys.exit(main())
