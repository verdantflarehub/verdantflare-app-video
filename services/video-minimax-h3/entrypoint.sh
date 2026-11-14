#!/usr/bin/env bash
set -euo pipefail

readonly model_repository="MiniMaxAI/MiniMax-H3"
readonly model_revision="42ed227ee7df40d41602854ae760620d6eb651fe"
readonly model_path="${MODEL_PATH:-/models/MiniMax-H3}"
readonly revision_file="${model_path}/.verdantflare-revision"
readonly transformer_weights_path="${model_path}/serialized-int8/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"
readonly output_path="${H3_OUTPUT_PATH:-/data/projects/h3/tasks}"
readonly visible_gpu_count="${H3_VISIBLE_GPU_COUNT:-1}"
readonly num_gpus="${H3_NUM_GPUS:-${visible_gpu_count}}"
readonly tp_size="${H3_TP_SIZE:-1}"
readonly ulysses_degree="${H3_ULYSSES_DEGREE:-1}"
readonly layerwise_offload_components="${H3_LAYERWISE_OFFLOAD_COMPONENTS:-dit,text_encoder}"
readonly dit_layerwise_resident_layers="${H3_DIT_LAYERWISE_RESIDENT_LAYERS:-0}"

for numeric_value in \
    "${visible_gpu_count}" \
    "${num_gpus}" \
    "${tp_size}" \
    "${ulysses_degree}" \
    "${dit_layerwise_resident_layers}"; do
    if [[ ! "${numeric_value}" =~ ^[0-9]+$ ]]; then
        echo "MiniMax H3 GPU topology values must be non-negative integers." >&2
        exit 1
    fi
done

if (( visible_gpu_count < 1 || num_gpus < 1 || tp_size < 1 || ulysses_degree < 1 )); then
    echo "MiniMax H3 GPU counts and parallel degrees must be positive." >&2
    exit 1
fi

if (( num_gpus != visible_gpu_count )); then
    echo "H3_NUM_GPUS must equal H3_VISIBLE_GPU_COUNT." >&2
    exit 1
fi

if (( tp_size * ulysses_degree != num_gpus )); then
    echo "H3_TP_SIZE multiplied by H3_ULYSSES_DEGREE must equal H3_NUM_GPUS." >&2
    exit 1
fi

mapfile -t gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed '/^[[:space:]]*$/d')
if (( ${#gpu_uuids[@]} != visible_gpu_count )); then
    echo "MiniMax H3 Full INT8 requires exactly ${visible_gpu_count} visible CUDA GPU; found ${#gpu_uuids[@]}." >&2
    exit 1
fi

unique_gpu_count="$(printf '%s\n' "${gpu_uuids[@]}" | sort -u | wc -l | tr -d ' ')"
readonly unique_gpu_count
if (( unique_gpu_count != visible_gpu_count )); then
    echo "MiniMax H3 requires ${visible_gpu_count} distinct CUDA GPU UUIDs; found ${unique_gpu_count}." >&2
    exit 1
fi

if [[ ! -f "${revision_file}" ]] || [[ "$(<"${revision_file}")" != "${model_revision}" ]]; then
    mkdir -p "${model_path}"
    MODEL_REPOSITORY="${model_repository}" \
    MODEL_REVISION="${model_revision}" \
    MODEL_PATH="${model_path}" \
    python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPOSITORY"],
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_PATH"],
    allow_patterns=["LICENSE", "README.md", "model_index.json", "Ref2VA/*"],
)
PY
    printf '%s\n' "${model_revision}" >"${revision_file}"
fi

if [[ ! -f "${transformer_weights_path}" ]]; then
    echo "MiniMax H3 Full INT8 weights do not exist: ${transformer_weights_path}" >&2
    exit 1
fi

mkdir -p "${output_path}"

exec /opt/nvidia/nvidia_entrypoint.sh sglang serve \
    --model-type diffusion \
    --model-path "${model_path}" \
    --model-variant ref2va \
    --transformer-weights-path "${transformer_weights_path}" \
    --num-gpus "${num_gpus}" \
    --tp-size "${tp_size}" \
    --ulysses-degree "${ulysses_degree}" \
    --encoder-parallel auto \
    --attention-backend fa \
    --performance-mode memory \
    --layerwise-offload-components "${layerwise_offload_components}" \
    --dit-offload-prefetch-size 1 \
    --dit-layerwise-resident-layers "${dit_layerwise_resident_layers}" \
    --enable-torch-compile false \
    --warmup-mode off \
    --output-path "${output_path}" \
    --host 0.0.0.0 \
    --port 8000 \
    "$@"
