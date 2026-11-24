# syntax=docker/dockerfile:1.7
FROM registry.cn-qingdao.aliyuncs.com/wod/verdantflare-app:video-minimax-h3-fasth3-v0.1.2

USER root
RUN apt-get update && apt-get install -y --no-install-recommends tini git ffmpeg && rm -rf /var/lib/apt/lists/*

# Install LatentSync 1.6 inference dependencies while preserving native PyTorch 2.13+cu129 (RTX 5090 SM120)
RUN python3 -m pip install --no-cache-dir \
      diffusers==0.32.2 \
      transformers==4.48.0 \
      decord==0.6.0 \
      accelerate==0.26.1 \
      einops \
      omegaconf==2.3.0 \
      opencv-python-headless \
      mediapipe \
      python_speech_features==0.6 \
      librosa \
      ffmpeg-python==0.2.0 \
      imageio \
      imageio-ffmpeg \
      face-alignment \
      kornia \
      insightface \
      onnxruntime \
      DeepCache==0.1.1 \
      huggingface-hub==0.30.2 \
      av \
      soundfile

ARG LATENTSYNC_COMMIT=a229c3948406bc2cf6eaf4873e662e70c6a04746
RUN git init /opt/latentsync \
    && cd /opt/latentsync \
    && git remote add origin https://github.com/bytedance/LatentSync.git \
    && git sparse-checkout set configs latentsync scripts eval \
    && git fetch --depth 1 --filter=blob:none origin "${LATENTSYNC_COMMIT}" \
    && git checkout --detach FETCH_HEAD

WORKDIR /opt/video-lipsync
COPY pyproject.toml ./
COPY src ./src
RUN python3 -m pip install --no-deps --no-cache-dir . && python3 -m compileall -q /opt/video-lipsync/src

LABEL org.opencontainers.image.title="VerdantFlare Video LipSync (LatentSync 1.6)" \
      org.opencontainers.image.version="0.1.4" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONPATH=/opt/video-lipsync/src:/opt/latentsync \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LATENTSYNC_ROOT=/opt/latentsync \
    LIPSYNC_MODEL_ROOT=/models/LatentSync-1.6 \
    LIPSYNC_ARTIFACT_ROOT=/data/projects/video \
    LIPSYNC_TASK_ROOT=/data/projects/lipsync/tasks \
    LIPSYNC_MIN_FREE_VRAM_MIB=28672 \
    LIPSYNC_ENABLE_CPU_OFFLOAD=1

EXPOSE 8000
ENTRYPOINT ["tini", "-g", "--", "python3", "-m", "video_lipsync.api"]
