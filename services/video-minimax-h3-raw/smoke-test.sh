#!/usr/bin/env bash
set -euo pipefail

readonly base_url="${H3_BASE_URL:-http://127.0.0.1:8000}"
readonly output_file="${H3_SMOKE_OUTPUT:-/tmp/minimax-h3-ref2va-smoke.mp4}"
readonly reference_video="${H3_SMOKE_VIDEO_URL:-https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4}"

curl --fail --silent --show-error "${base_url}/health" >/dev/null

video_id="$(
    curl --fail-with-body --silent --show-error \
        --request POST \
        --url "${base_url}/v1/videos" \
        --header 'Content-Type: application/json' \
        --data-binary @- <<JSON | jq -er '.id'
{
  "model": "MiniMaxAI/MiniMax-H3",
  "task": "ref2va",
  "prompt": "Use <Video 1> as the subject and motion reference. Preserve the person and pastoral setting with subtle natural movement and coherent ambient sound.",
  "seconds": 4,
  "conditions": [
    {
      "type": "video",
      "uri": "${reference_video}",
      "role": "reference"
    }
  ],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "9:16",
    "duration_seconds": 4.0
  },
  "num_outputs_per_prompt": 1,
  "num_inference_steps": 21,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 7
}
JSON
)"

for _ in $(seq 1 720); do
    status="$(curl --fail --silent --show-error "${base_url}/v1/videos/${video_id}" | jq -er '.status')"
    case "${status}" in
        completed)
            break
            ;;
        failed|failure)
            echo "MiniMax H3 smoke generation failed." >&2
            exit 1
            ;;
        queued|in_progress)
            sleep 10
            ;;
        *)
            echo "MiniMax H3 returned unknown status: ${status}" >&2
            exit 1
            ;;
    esac
done

if [[ "${status}" != "completed" ]]; then
    echo "MiniMax H3 smoke generation did not finish before the timeout." >&2
    exit 1
fi

temp_file="${output_file}.part"
curl --fail --silent --show-error \
    "${base_url}/v1/videos/${video_id}/content" \
    --output "${temp_file}"
ffprobe -v error \
    -show_entries stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels \
    -of json "${temp_file}" | jq -e '
        ([.streams[] | select(.codec_type == "video" and .codec_name == "h264" and .r_frame_rate == "24/1")] | length) == 1 and
        ([.streams[] | select(.codec_type == "audio" and .codec_name == "aac" and .sample_rate == "32000" and .channels == 2)] | length) == 1
    ' >/dev/null
mv "${temp_file}" "${output_file}"
printf 'MiniMax H3 smoke output: %s\n' "${output_file}"
