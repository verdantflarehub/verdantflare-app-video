FROM docker.io/library/python:3.11.10-slim

ARG PYPI_INDEX_URL=https://pypi.org/simple
LABEL org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-video"
LABEL org.opencontainers.image.title="VerdantFlare Video MCP Server"
LABEL org.opencontainers.image.version="0.8.1"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" -r requirements.txt && pip check
COPY verdantflare_video_mcp ./verdantflare_video_mcp
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
CMD ["python3", "-m", "uvicorn", "verdantflare_video_mcp.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
