FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIDEKICK=0.0.0 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

RUN groupadd --gid 10001 sidekick \
    && useradd --uid 10001 --gid sidekick --create-home --home-dir /home/sidekick sidekick \
    && mkdir -p /sidekick-data \
    && chown sidekick:sidekick /sidekick-data

WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src

RUN chmod -R a+rX /app \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

USER sidekick
