# Immutable CUDA 13.0.2 / torch 2.10.0 runtime built by the v0.2.0 workflow.
# Its dependencies and upstream revision are unchanged by this guard-only patch.
FROM registry.cn-qingdao.aliyuncs.com/wod/verdantflare-app:video-depth-anything-api-v0.2.0@sha256:c024ea884c4f9793ec2585a045f5b0e7db8b12238875a2d7ec37d008b81000ae
LABEL org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-video"
LABEL org.opencontainers.image.title="VerdantFlare Video Depth Anything API"
LABEL org.opencontainers.image.version="0.2.1"
WORKDIR /app
COPY main.py model.py media.py download_model.py gpu_guard.py model-lock.json ./
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
