# MiniMax H3 5090 API

`v0.4.0` 是运行在 5090 dev 的内部 API 层，固定转发到现有 Singularity Runtime。它不加载模型、不申请 GPU，也不是 MiniMax 官方完整 API。原 SGLang 版本保存在相邻的 `video-minimax-h3-raw/`。

## 接口

- `GET /live`：API 进程存活。
- `GET /health`：Singularity 已就绪。
- `GET /capabilities`：当前实际支持能力，需要 Bearer。
- `POST /v1/videos`、`GET /v1/videos/{id}`、`GET /v1/videos/by-idempotency/{key}`、`GET /v1/videos/{id}/content`：保留 Video MCP 内部契约；生成、幂等与队列由 Singularity 处理。

`/v1` 仅支持已登记参考素材的 `ref2va`、4–15 秒、768 短边、4 NFE。当前没有 Huobao 使用的 `/v2/video_generation`、纯文生视频、首尾帧或 2K。不要把本服务的 Token 发给 Huobao。

## 配置

- `H3_API_TOKEN`：API 入站 Bearer；从受控 Secret 注入。
- `SINGULARITY_RUNTIME_TOKEN`：Singularity 出站 Bearer；从受控 Secret 注入。
- `H3_API_UPSTREAM_URL`：固定内部 HTTP origin，默认 `http://video-minimax-h3-singularity:8000`。
- `PORT`：监听端口，默认 `8000`。

只在 `verdantflare-video` 命名空间内提供 ClusterIP。5090 环境清单和实际版本以根工作区 `deploys/k8s.dev.verdantflarehub.com/verdantflare-video/` 为准；镜像由 GitHub Actions 生成，不进行本机 Docker 构建。

轻量测试：在安装 `fastapi`、`httpx` 和 `pytest` 的 Python 环境执行 `PYTHONPATH=src python -m pytest tests/test_api.py`。
