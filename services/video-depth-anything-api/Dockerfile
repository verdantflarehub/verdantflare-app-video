FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c
LABEL org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-video"
LABEL org.opencontainers.image.title="VerdantFlare Video Depth Anything API"
LABEL org.opencontainers.image.version="0.1.0"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 torch==2.10.0 torchvision && pip3 install --break-system-packages --no-cache-dir -r requirements.txt
COPY main.py .
EXPOSE 8000
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
