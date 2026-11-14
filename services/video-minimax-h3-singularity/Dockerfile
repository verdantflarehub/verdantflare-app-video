# syntax=docker/dockerfile:1.7
FROM registry.cn-qingdao.aliyuncs.com/wod/verdantflare-app:video-h3-latent-upscaler-api-v0.1.1@sha256:444a8bfcc8b4511817af91061e62b1c1716bb334cfb866c4c1ec77d91dd1cac8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends tini && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/h3-singularity
COPY pyproject.toml ./
COPY src ./src
RUN python3 -m pip install --no-deps . && python3 -m compileall -q /opt/h3-singularity/src

LABEL org.opencontainers.image.title="VerdantFlare MiniMax H3 Singularity Ref2VA & 2K IC" \
      org.opencontainers.image.version="0.1.39" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT AND LicenseRef-MiniMax-H3-Community"

ENV PYTHONPATH=/opt/h3-singularity/src:/opt/comfy \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SINGULARITY_MODEL_ROOT=/models/singularity \
    H3_BASE_MODEL_ROOT=/models/MiniMax-H3 \
    H3_COMPONENT_MODEL_ROOT=/models/singularity \
    SINGULARITY_TASK_ROOT=/data/projects/singularity/tasks \
    SINGULARITY_ARTIFACT_ROOT=/source/video \
    SINGULARITY_MIN_FREE_VRAM_MIB=28672

EXPOSE 8000
ENTRYPOINT ["tini", "-g", "--", "python3", "-m", "h3_singularity.api"]
