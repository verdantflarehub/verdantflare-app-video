# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c

LABEL org.opencontainers.image.title="VerdantFlare MiniMax H3 Sol" \
      org.opencontainers.image.version="0.2.13" \
      org.opencontainers.image.source="https://github.com/verdantflarehub/verdantflare-app-video" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-MiniMax-H3-Community"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git patch ffmpeg build-essential python3.12 python3.12-dev python3.12-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV UV_NO_CACHE=1 PATH=/opt/venv/bin:$PATH CUDNN_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn
WORKDIR /opt/verdantflare-sol
COPY pyproject.toml requirements.lock requirements.build.lock ./
RUN python -m pip install --no-cache-dir uv==0.8.22 \
    && uv pip install --python /opt/venv/bin/python -r requirements.build.lock \
    && python -c "from pathlib import Path; p=Path('requirements.lock'); Path('/tmp/runtime.lock').write_text(''.join(l for l in p.read_text().splitlines(True) if not l.startswith('nvidia-cudnn-frontend ')))" \
    && uv pip install --python /opt/venv/bin/python --no-deps --index-strategy unsafe-best-match -r /tmp/runtime.lock
RUN CMAKE_BUILD_PARALLEL_LEVEL=2 CMAKE_PREFIX_PATH=/opt/venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11 uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation --index-strategy unsafe-best-match -r requirements.lock \
    && uv pip install --python /opt/venv/bin/python --no-deps . \
    && python -m pip check

RUN git init /opt/sana \
    && git -C /opt/sana remote add origin https://github.com/NVlabs/Sana.git \
    && git -C /opt/sana sparse-checkout set models/minimax_h3/Sol-H3 \
    && git -C /opt/sana fetch --depth=1 --filter=blob:none origin 2936c47637380842aaa4a4488fac5006cc542b70 \
    && git -C /opt/sana checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/sana rev-parse HEAD)" = 2936c47637380842aaa4a4488fac5006cc542b70 \
    && cp -a /opt/sana/models/minimax_h3/Sol-H3 /opt/sol-h3 \
    && rm -rf /opt/sana
COPY . ./
RUN mkdir -p /usr/share/licenses/video-minimax-h3-sol \
    && cp LICENSE.upstream NOTICE.upstream.md /usr/share/licenses/video-minimax-h3-sol/
RUN python verify-source.py /opt/sol-h3 \
    && python prepare-patched-source.py --source /opt/sol-h3 --output /opt/sol-h3-experimental \
    && python -m unittest discover -s tests -v \
    && PYTHONPATH=/opt/sol-h3 python -c "from h3_runtime import MiniMaxH3Inference; from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference; from cudnn import BSA" \
    && python -m pip freeze > /opt/verdantflare-sol/installed-packages.txt
RUN PYTHONPATH=/opt/sol-h3-experimental python -c "from h3_runtime import MiniMaxH3Inference; import inspect; p=inspect.signature(MiniMaxH3Inference).parameters; assert all(k in p for k in ('cpu_offload', 'output_width', 'output_height'))" \
    && python -m compileall -q /opt/verdantflare-sol /opt/sol-h3-experimental
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/opt/sol-h3:/opt/verdantflare-sol
ENTRYPOINT ["python", "/opt/verdantflare-sol/run-inference.py"]
CMD ["--help"]
