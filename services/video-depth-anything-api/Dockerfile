FROM docker.io/library/python:3.11.10-slim
LABEL org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-video"
LABEL org.opencontainers.image.title="VerdantFlare Video Depth Anything API"
LABEL org.opencontainers.image.version="0.1.0"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
