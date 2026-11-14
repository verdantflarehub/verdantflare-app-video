# syntax=docker/dockerfile:1.7
FROM registry.cn-qingdao.aliyuncs.com/wod/verdantflare-app:video-h3-latent-upscaler-api-v0.1.1@sha256:444a8bfcc8b4511817af91061e62b1c1716bb334cfb866c4c1ec77d91dd1cac8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends tini && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/h3-fasth3
COPY pyproject.toml ./
COPY src ./src
RUN python3 -m pip install --no-deps --no-cache-dir . && python3 -m compileall -q /opt/h3-fasth3/src

LABEL org.opencontainers.image.title="VerdantFlare MiniMax H3 FastH3 NVFP4 Ref2VA" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT AND LicenseRef-MiniMax-H3-Community"

ENV PYTHONPATH=/opt/h3-fasth3/src:/opt/comfy \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    FASTH3_MODEL_ROOT=/models/MiniMax-H3 \
    FASTH3_COMPONENT_MODEL_ROOT=/models/MiniMax-H3/singularity \
    FASTH3_TASK_ROOT=/data/projects/fasth3/tasks \
    FASTH3_MIN_FREE_VRAM_MIB=28672

EXPOSE 8000
ENTRYPOINT ["tini", "-g", "--", "python3", "-m", "h3_fasth3.api"]
