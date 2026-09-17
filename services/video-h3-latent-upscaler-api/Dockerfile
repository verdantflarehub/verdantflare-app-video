# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c
LABEL org.opencontainers.image.title="VerdantFlare H3 Latent Upscaler" org.opencontainers.image.version="0.1.1"
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv git ffmpeg ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY backend-lock.json pyproject.toml ./
COPY scripts/prepare-source.py scripts/prepare-source.py
RUN python scripts/prepare-source.py
RUN pip install torch==2.13.0+cu129 torchvision==0.28.0+cu129 torchaudio==2.11.0+cu129 --extra-index-url https://download.pytorch.org/whl/cu129
COPY . .
RUN pip install -r /opt/comfy/requirements.txt . httpx==0.28.1 && pip check
ENV PYTHONPATH=/app/src:/opt/comfy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN python scripts/check-backend-imports.py && python scripts/verify-vendor.py /opt/h3-upscale-upstream && python -m unittest discover -s tests -v && python -m compileall -q src && pip freeze > installed-packages.txt
EXPOSE 8000
CMD ["python", "-m", "h3_latent_upscaler.api"]
