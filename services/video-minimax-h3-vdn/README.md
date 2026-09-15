# video-minimax-h3-vdn

## 项目结构

| 文件 | 用途 |
| --- | --- |
| `src/vendor/infer_diffusers.py` | 推理脚本 |
| `patches/` | 源码补丁 |
| `src/run-inference.py` | 命令行入口与运行记录 |
| `src/resident_worker.py`、`src/resident_api.py` | 常驻服务与 HTTP 接口 |
| `src/task_store.py` | 持久化任务队列 |
| `src/vdn_io.py` | 输入、文件及媒体检查 |
| `src/lock-model.py` | 生成本地模型文件校验清单 |
| `scripts/prepare-source.py`、`scripts/verify-vendor.py` | 准备与校验构建源码 |
| `pyproject.toml` | 项目元数据、依赖和源码版本 |
| `Dockerfile` | 镜像构建 |
| `tests/` | 测试 |
| `LICENSE.upstream`、`LICENSE.model` | 许可证 |

以下命令在本目录执行。推理需要 CUDA GPU、完整的本地模型文件和只读输入目录。

## 构建与检查

```bash
python3 -m unittest discover -s tests -v
podman build -t localhost/video-minimax-h3-vdn:review .
# 或使用 Docker：
docker build -t video-minimax-h3-vdn:review .
```

本地测试需要 Python 3.12、ffmpeg 和 ffprobe。镜像内包含运行依赖。查看入口参数：

```bash
podman run --rm localhost/video-minimax-h3-vdn:review --help
```

## 准备模型

模型目录包含 `modular_model_index.json`（或 `model_index.json`）、各组件配置及权重。组件按索引中的 subfolder 存放，无 subfolder 时使用组件名子目录。8-step 使用 `stage-dmd-step-250/diffusers`，50-step 使用 `stage-b-step-2000/diffusers`。

在已安装依赖的环境中，为模型目录生成校验清单；清单须放在模型目录外：

```bash
python src/lock-model.py --models /models/vdn-8step \
  --revision MODEL_REPOSITORY_FULL_COMMIT --steps 8 \
  --output /records/model-lock.json
```

将 `MODEL_REPOSITORY_FULL_COMMIT` 替换为实际的 40 位模型仓库 commit。程序不会下载模型。

## 输入格式

`/inputs/request.json`：

```json
{
  "schema_version": 1,
  "task": "t2va",
  "prompt": "A river at dawn",
  "seed": 7,
  "frames": 124,
  "steps": 8,
  "input_artifacts": []
}
```

`frames` 为 124–362 范围内满足 `17n+5` 的整数，帧率为 24；`steps` 为 8 或 50，须与模型清单一致。

I2VA 使用 `first`，L2VA 使用 `last`，FL2VA 同时使用两者。例如 I2VA 的 `input_artifacts`：

```json
[{"role": "first", "path": "first.png", "sha256": "实际文件SHA-256"}]
```

文件路径相对 `/inputs`，输入文件和模型文件的哈希必须与清单一致。

## 启动

在已安装依赖的环境中执行：

```bash
python src/run-inference.py --manifest /inputs/request.json --inputs /inputs \
  --models /models/vdn-8step --model-lock /records/model-lock.json \
  --output /outputs/new-run --dry-run
```

`--dry-run` 只检查文件与参数；移除后执行推理。输出目录必须尚不存在。

Docker 启动示例：

```bash
docker run --rm --gpus all \
  -v /models:/models:ro -v /inputs:/inputs:ro \
  -v /records:/records:ro -v /outputs:/outputs \
  video-minimax-h3-vdn:review \
  --manifest /inputs/request.json --inputs /inputs \
  --models /models/vdn-8step --model-lock /records/model-lock.json \
  --output /outputs/new-run
```

运行状态保存在 `record.json`，成功产物为 `video.mp4`；失败返回非零退出码并保留运行记录。镜像默认命令为 `--help`。

## 常驻服务

将访问 Token 放入 `VDN_RUNTIME_TOKEN` 环境变量，然后启动：

```bash
export VDN_MODELS=/models/VDN-H3-Ref2VA
export VDN_MODEL_LOCK=/models/VDN-H3-Ref2VA.lock.json
export VDN_TASK_ROOT=/projects/h3-vdn/tasks
export VDN_ARTIFACT_SOURCE=http://video-mcp-server:8000
export VDN_GPU_UUIDS=GPU_FIRST_UUID,GPU_SECOND_UUID
python src/resident_worker.py
```

`VDN_FP8` 默认为 `0`（BF16）；设置 `VDN_FP8=1` 启用 Transformer FP8 E4M3。健康接口和任务结果记录精度与 FP8 层数。

容器使用 `--entrypoint python`，命令为 `/opt/verdantflare-vdn/src/resident_worker.py`。监听端口 8000；`GET /live` 检查进程，`GET /health` 在模型加载完成后返回 200。

任务接口需要 `Authorization: Bearer <Token>`：

- `POST /v1/videos`：提交 Ref2VA 任务。
- `GET /v1/videos/{id}`：查询任务。
- `GET /v1/videos/{id}/content`：下载成功产物。
- `DELETE /v1/videos/{id}`：取消排队任务。

请求示例：

```json
{
  "model": "MiniMaxAI/MiniMax-H3",
  "task": "ref2va",
  "prompt": "The character walks toward the camera",
  "seconds": 5,
  "conditions": [{"type": "image", "role": "reference", "uri": "http://video-mcp-server:8000/runtime-artifacts/art_REPLACE_WITH_ID/content", "sha256": "REPLACE_WITH_SHA256", "size": 12345}],
  "target": {"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 5.0},
  "num_outputs_per_prompt": 1,
  "num_inference_steps": 8,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 7,
  "idempotency_key": "new-attempt"
}
```

替换示例中的 Artifact ID、哈希和大小。支持图片、视频、音频参考；至少包含一张图片或一段视频，最多 9 张图片、3 段视频、3 段音频，总计不超过 12 个。时长档 5/10/15 秒对应 124/243/345 帧，实际时长写入结果。画布为 768×1344，24 FPS。

每个任务目录只由一个常驻进程使用；重启后未完成任务标记失败，原幂等键不会重新生成。
