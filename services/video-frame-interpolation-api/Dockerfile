# Build context: services/video-frame-interpolation-api/.
FROM docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
LABEL org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-video"
LABEL org.opencontainers.image.title="VerdantFlare video-frame-interpolation-api"
LABEL org.opencontainers.image.version="0.1.0"
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/hzwer/ECCV2022-RIFE.git /opt/rife && git -C /opt/rife checkout 5d8adbdd40e12c2c8f91930eff838aebe561c086
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip check
COPY video_interpolation /app/video_interpolation
COPY main.py engine.py backend-lock.json /app/
ENV PYTHONPATH=/app VIDEO_UPSTREAM_ROOT=/opt/rife
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
