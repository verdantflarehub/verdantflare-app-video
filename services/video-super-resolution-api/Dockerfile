# Build context: services/video-super-resolution-api/.
FROM docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
LABEL org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-video"
LABEL org.opencontainers.image.title="VerdantFlare video-super-resolution-api"
LABEL org.opencontainers.image.version="0.1.1"
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
# Conditioning tensors are supplied in the external model bundle, not the image.
RUN git clone --filter=blob:none --no-checkout https://github.com/ByteDance-Seed/SeedVR.git /opt/seedvr \
    && git -C /opt/seedvr sparse-checkout set --no-cone '/*' '!/pos_emb.pt' '!/neg_emb.pt' '!/assets/' \
    && git -C /opt/seedvr checkout e4de8c24441a67e1b7df56abea10645059bb1185
# Prefer the distribution FFmpeg with libx264 over Conda's minimal binary.
RUN mkdir -p /opt/video-bin && ln -s /usr/bin/ffmpeg /opt/video-bin/ffmpeg && ln -s /usr/bin/ffprobe /opt/video-bin/ffprobe
ENV PATH=/opt/video-bin:${PATH}
RUN ffmpeg -v error -f lavfi -i color=size=16x16:rate=1 -frames:v 1 -c:v libx264 -preset fast -f null -
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn==2.5.9.post1 --no-build-isolation
RUN pip check
COPY video_sr /app/video_sr
COPY main.py engine.py seedvr_memory.py calibrate.py download_model.py backend-lock.json /app/
ENV PYTHONPATH=/app VIDEO_UPSTREAM_ROOT=/opt/seedvr
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
