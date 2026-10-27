# Huobao H3 兼容层

轻量镜像只提供 Huobao `/v2` 兼容入口，不加载模型、不申请 GPU。5090 的 `/v1` 由相邻 `video-minimax-h3-singularity/` 提供，Video MCP 直接调用该服务。原 SGLang 版本保存在相邻 `video-minimax-h3-raw/`。产品契约和现场配置以设计工作区为准，不把本包当成 MiniMax 官方完整 API。

## 1. Huobao 兼容进程

默认启动 `python -m h3_api.huobao`，提供 `/v2/video_generation`、`/v2/query/video_generation/{task_id}` 及短时签名下载。仅通过 Studio 调用项目权限、素材导入和 Video 工具；不接收调用者传入的用户、组织或项目，不转交 Runtime 凭据。

必需配置：`STUDIO_ORIGIN`、`STUDIO_BEARER_TOKEN`、`HUOBAO_API_KEY`、`HUOBAO_SIGNING_KEY`、`HUOBAO_PROJECT_ID`、`HUOBAO_OWNER_USER_ID`、`HUOBAO_OWNER_ORG_ID`。凭据与真实归属从受控 Secret 注入。

`HUOBAO_BRIDGE_DATA` 默认 `/data`，必须持久化，且只运行单进程单副本。`HUOBAO_BRIDGE_ORIGIN` 是 Huobao 后端可达的内部下载 origin；`HUOBAO_MEDIA_ORIGIN` 是允许导入的 Huobao origin；`HUOBAO_MEDIA_INTERNAL_ORIGIN` 为同服务的内部读取地址。全部 origin 不含路径、查询或账号密码。

支持参考生成、4–15 秒、768P；不支持纯文生、首尾帧和 2K。单素材 32 MiB、总素材 96 MiB、JSON 请求 48 MiB；图片 9、视频 3、音频 3，总计最多 12。下游还会执行格式、时长、像素和完整解码校验。调用方传 `Idempotency-Key`；无此头时同请求 24 小时内复用原任务。生产接口不暴露原始上游异常，诊断按固定任务 ID 查受控记录。

## 2. 检查与部署依据

本地轻量测试：`PYTHONPATH=src python -m pytest tests`。镜像仅由 release GitHub Actions 构建。

- [Huobao 接入设计](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/studio/details/06-studio-project-design.external-workbench-h3-api.draft.md)
- [Aliyun 历史试用部署（已清退）](https://github.com/verdantflarehub/verdantflare-design/blob/22ed06c4/deploys/k8s.ali.wodcloud.com/verdantflare-workbench-trial/README.md)
- [5090 Video 部署](https://github.com/verdantflarehub/verdantflare-design/blob/dev/deploys/k8s.dev.verdantflarehub.com/verdantflare-video/README.md)
