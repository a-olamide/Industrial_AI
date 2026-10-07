# Spark image with the Python ML runtime needed for online inference.
#
# The stock apache/spark image has no scikit-learn, so the Isolation
# Forest and Random Forest artifacts cannot be unpickled inside it. This
# adds exactly the packages the two frozen models need to load and score
# - nothing is trained here.
#
# Versions are pinned to match ml/requirements.txt so that a model
# pickled offline unpickles cleanly in the container. A scikit-learn
# minor-version drift between the two is the classic cause of silent
# inference breakage.
FROM apache/spark:3.5.1

USER root

RUN set -eux; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir \
        "numpy>=1.26,<3" \
        "pandas>=2.1,<3" \
        "scipy>=1.11,<2" \
        "scikit-learn>=1.4,<2" \
        "joblib>=1.3,<2"

USER spark
