# Spark image with the Python ML runtime needed for online inference.
#
# Why this image exists at all: the stock apache/spark image has no
# scikit-learn, so the Isolation Forest and Random Forest artifacts
# cannot be unpickled inside it. Nothing is trained here.
#
# Why Python 3.9 is installed: apache/spark:3.5.1 ships Python 3.8.10
# (Ubuntu 20.04), but the models were pickled under Python 3.9 with
# scikit-learn 1.6.1, and scikit-learn >= 1.6 requires Python >= 3.9.
# Loading them under 3.8 is impossible, so a 3.9 interpreter is added
# from deadsnakes and PySpark is pointed at it. PySpark 3.5 supports
# Python 3.8-3.11, so this is a supported combination.
#
# Versions are pinned to exactly what produced the pickles. A
# scikit-learn or numpy drift between pickling and unpickling is the
# classic cause of silent inference breakage, so this is deliberate
# rather than cautious.
FROM apache/spark:3.5.1

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates curl; \
    add-apt-repository -y ppa:deadsnakes/ppa; \
    apt-get update; \
    apt-get install -y --no-install-recommends python3.9 python3.9-distutils; \
    curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py -o /tmp/get-pip.py; \
    python3.9 /tmp/get-pip.py; \
    rm -f /tmp/get-pip.py; \
    python3.9 -m pip install --no-cache-dir \
        numpy==2.0.2 \
        pandas==2.3.3 \
        scipy==1.13.1 \
        scikit-learn==1.6.1 \
        joblib==1.5.3; \
    rm -rf /var/lib/apt/lists/*; \
    python3.9 -c "import sklearn, numpy, pandas, scipy, joblib; print('ml runtime ok', sklearn.__version__)"

# NOTE: the build deliberately does NOT `apt-get purge`/`autoremove` the
# tooling afterwards. Doing so strips shared libraries that the numpy and
# pandas C extensions link against, and the failure surfaces much later
# as a confusing "partially initialized module 'pandas'" ABI error.

# Both the driver and the executors must use the 3.9 interpreter, or the
# model unpickles on one side and fails on the other.
ENV PYSPARK_PYTHON=/usr/bin/python3.9 \
    PYSPARK_DRIVER_PYTHON=/usr/bin/python3.9

USER spark
