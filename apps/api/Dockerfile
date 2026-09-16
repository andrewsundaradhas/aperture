# Aperture API. Multi-stage so the runtime image carries no build toolchain.
FROM python:3.11-slim AS build

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency metadata first, so edits to source do not invalidate the dependency layer.
COPY pyproject.toml ./
COPY aperture/__init__.py ./aperture/__init__.py
# `datasets` brings the real LeRobot v3 reader (parquet + mp4). Drop it for a smaller image if
# you only ingest RLDS .tfrecord or the legacy JSON, both of which need nothing extra.
RUN pip install --no-cache-dir ".[datasets]"

COPY . .
RUN pip install --no-cache-dir --no-deps .


FROM python:3.11-slim

# libgl/libglib: Pillow and av need them at runtime and slim omits them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Never run as root: an ingestion bug that writes outside its sandbox should not own the host.
RUN useradd --create-home --uid 10001 aperture
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1

WORKDIR /app
COPY --chown=aperture:aperture alembic.ini ./
COPY --chown=aperture:aperture alembic ./alembic
USER aperture

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "aperture.main:app", "--host", "0.0.0.0", "--port", "8000"]
