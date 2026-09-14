# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c
LABEL org.opencontainers.image.title="VerdantFlare MiniMax H3 VDN" \
      org.opencontainers.image.version="0.3.0" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-MiniMax-H3-Community"
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv git ffmpeg ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /opt/verdantflare-vdn
COPY pyproject.toml ./
COPY scripts/prepare-source.py scripts/prepare-source.py
COPY patches/0002-diffusers-vdn-layout.patch patches/0002-diffusers-vdn-layout.patch
RUN python scripts/prepare-source.py --destination /opt/upstream
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu129 . /opt/upstream/diffusers \
    && pip check
COPY . .
RUN python scripts/verify-vendor.py --upstream /opt/upstream/vdn \
    && python scripts/check-ref2va-layout.py \
    && python -m unittest discover -s tests -v \
    && python -c "from diffusers import ModularPipeline; from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline; from diffusers.hooks import apply_group_offloading; from accelerate import cpu_offload_with_hook; from diffusers.utils.export_utils import encode_video; import torch; import av; assert torch.version.cuda == '12.9'" \
    && python -m compileall -q . \
    && pip freeze > installed-packages.txt
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python", "/opt/verdantflare-vdn/src/run-inference.py"]
CMD ["--help"]
