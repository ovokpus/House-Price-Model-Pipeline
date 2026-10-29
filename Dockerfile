# Image that trains the notebook pipeline (on first boot) and serves /predict.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ARTIFACT_DIR=/app/artifacts \
    PORT=8000

WORKDIR /app

# scikit-learn's wheels link against OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY 01_research-and-development/datasets/housing/housing.tgz \
     /app/01_research-and-development/datasets/housing/housing.tgz

RUN pip install --no-cache-dir .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && mkdir -p /app/artifacts

# The process stays root so the named volume in docker-compose.yml is writable
# on first boot. Running as a non-root user (and an fsGroup on the volume) is
# a production follow-up; see PRODUCTIONIZING.md.
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
