# syntax=docker/dockerfile:1.7
FROM registry.cn-qingdao.aliyuncs.com/wod/verdantflare-app:video-h3-latent-upscaler-api-v0.1.1@sha256:444a8bfcc8b4511817af91061e62b1c1716bb334cfb866c4c1ec77d91dd1cac8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends tini && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/h3-regenerate-2k
COPY pyproject.toml ./
COPY src ./src
RUN python3 -m pip install --no-deps --no-cache-dir . && python3 -m compileall -q /opt/h3-regenerate-2k/src

LABEL org.opencontainers.image.title="VerdantFlare MiniMax H3 Regenerate-2K IC" \
      org.opencontainers.image.version="0.1.3" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT AND LicenseRef-MiniMax-H3-Community"

ENV PYTHONPATH=/opt/h3-regenerate-2k/src:/opt/comfy \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    REGENERATE_2K_MODEL_ROOT=/models/MiniMax-H3 \
    REGENERATE_2K_ARTIFACT_ROOT=/source/video \
    REGENERATE_2K_TASK_ROOT=/data/projects/regenerate-2k/tasks \
    REGENERATE_2K_MIN_FREE_VRAM_MIB=28672

EXPOSE 8000
ENTRYPOINT ["tini", "-g", "--", "python3", "-m", "h3_regenerate_2k.api"]
