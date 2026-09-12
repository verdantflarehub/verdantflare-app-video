# VerdantFlare App Video

视频应用的代码与镜像构建仓库。统一分开使用 **H3**、**H3-Sol**、**MCP** 三个名称。

| 名称 | 工程 | 当前交付 |
| --- | --- | --- |
| H3 | `services/video-minimax-h3-api` | 原版推理服务，已部署 `video-minimax-h3-api-v0.3.0` |
| H3-Sol | `services/video-minimax-h3-sol` | Sol 常驻双 rank 推理服务，版本 `video-minimax-h3-sol-v0.2.0`；上线结果见中央验收记录 |
| MCP | `services/video-mcp-server` | 统一调用入口及 Dashboard，版本 `video-mcp-server-v0.4.0`（上线证据以中央发布记录为准） |
| Depth | `services/video-depth-anything-api` | Video MCP 内部深度转换推理服务，版本 `video-depth-anything-api-v0.1.0` |

H3-Sol 的既有双卡实验出片与人工质量限制，以中央设计记录为准。实验镜像可追溯发布不等于热启动服务、生产接入或质量验收完成。H3、H3-Sol 的后续模型服务不使用 Kubernetes Job，分别复用各自模型进程；MCP 不加载模型。

## 目录与事实源

```text
.github/workflows/
services/
  video-minimax-h3-api/
  video-minimax-h3-sol/
  video-mcp-server/
  video-depth-anything-api/
```

设计、接口与部署清单统一维护在 [verdantflare-design](https://github.com/verdantflarehub/verdantflare-design/tree/dev)：

- [Dashboard 设计、接口与发布验收](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/app/video/dashboard.md)
- [H3-Sol 设计与实验历史](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/app/video/minimax-h3/minimax-h3-sol.md)
- [H3 / H3-Sol 热启动需求](https://github.com/verdantflarehub/verdantflare-design/blob/dev/changes/change_20260910_h3_sol_warm_start_optimization.md)
- [中央部署清单](https://github.com/verdantflarehub/verdantflare-design/tree/dev/deploys/k8s.cn-chengdu.bc-cloud.com/verdantflare-video)

应用仓库不维护私有 `docs/`、`deploy/` 或 `deploys/`。旧 SGLang / LightX2V benchmark 源码和 workflow 从活动目录移除，保留在 [整理前的 Git 历史](https://github.com/verdantflarehub/verdantflare-app-video/tree/45299cb7da82ab98c283c59ed69590b2d3d011d5)。不依赖已经不存在的 `archive/` 目录；源码移除不删除旧镜像、共享模型或用户产物。

## 本地检查与发布

```bash
python3 -m unittest discover -s services/video-minimax-h3-sol/tests -v
python3 -m pip install -r services/video-mcp-server/requirements.txt
PYTHONPATH=services/video-mcp-server python3 -m unittest discover -s services/video-mcp-server/tests -v
```

日常提交至 `dev`，测试通过后只以 `--ff-only` 合并至 `release` 并推送，触发服务自己的镜像流水线，然后切回 `dev`。镜像使用不可覆盖的语义化版本；本地构建和临时挂载不能作为发布证据。H3-Sol 的 CI 执行 CPU 契约、分配校验、官方来源及补丁校验，构建中再检查引擎导入，只有 `release` 推送才发布镜像。

## MCP 与工作台

MCP 保持 `artifact.import`、`video.generate`、`video.status`、`video.result` 契约。Dashboard 分为 MCP 服务状态、模型服务、任务状态。H3 / H3-Sol 展示就绪、当前、期望实例数，支持模型 → 实例 → GPU 指标与短期历史。未上报的任务执行实例显示未知。使用真实任务数据，支持素材导入、提交、筛选、缩略图、视频回放和下载；H3-Sol 显式选择，支持 5 / 10 / 15 秒；历史任务固定沿原路线查询，不静默回退。

以中央部署公布的地址访问 `/video/dashboard`。数据接口及产物下载使用现有 Bearer Token；浏览器仅在页面内存保存 Token。内部 `/runtime-artifacts/` 不公开。工具不暴露内部 Runtime Task ID、服务地址或宿主路径。

H3-Sol 部署、真实冷 / 热推理对比和人工质量审核仍按中央变更记录执行，未完成前不推进 `main`。
