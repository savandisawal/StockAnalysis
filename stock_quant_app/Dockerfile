# Multi-stage build: locked dependencies compiled in a builder layer,
# slim non-root runtime with only what serving needs.

FROM python:3.13-slim AS builder

WORKDIR /build

# Build tools only exist in this stage
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.lock .
RUN pip install --prefix=/install --no-cache-dir -r requirements-runtime.lock

COPY pyproject.toml .
COPY app ./app
COPY data ./data
COPY features ./features
COPY model ./model
COPY services ./services
COPY scripts ./scripts
COPY ui_streamlit ./ui_streamlit
COPY utils ./utils
RUN pip install --prefix=/install --no-cache-dir --no-deps .


FROM python:3.13-slim

# libgomp1: required by LightGBM; curl: container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home appuser

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser features ./features
COPY --chown=appuser:appuser model ./model
COPY --chown=appuser:appuser services ./services
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser ui_streamlit ./ui_streamlit
COPY --chown=appuser:appuser utils ./utils
COPY --chown=appuser:appuser .streamlit ./.streamlit

# All mutable state (SQLite DBs, logs, model bundles) lives on this volume
ENV DATA_DIR=/data
RUN mkdir -p /data && chown appuser:appuser /data
VOLUME /data

USER appuser
EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
