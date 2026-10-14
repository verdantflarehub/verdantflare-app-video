# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

WORKDIR /opt/h3-api
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir . && python -m compileall -q src

RUN useradd --system --uid 10001 --no-create-home h3api
USER 10001:10001
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
LABEL org.opencontainers.image.title="VerdantFlare MiniMax H3 5090 API" \
      org.opencontainers.image.version="0.5.0" \
      org.opencontainers.image.licenses="Apache-2.0"
ENTRYPOINT ["python", "-m", "h3_api.app"]
