# VerdantFlare App Video

视频应用的代码与镜像构建仓库。各子服务统一采用无冗余 `-api` 后缀的工程命名：

| 服务目录                                  | 职责说明                                               |
| ----------------------------------------- | ------------------------------------------------------ |
| `services/video-mcp-server`               | 统一视频 MCP 调用入口、2K 串联编排与 Dashboard         |
| `services/video-minimax-h3`               | 原版 MiniMax-H3 SGLang Full-INT8 `/v1/videos` 推理服务 |
| `services/video-minimax-h3-fasth3`        | 5090 FastH3 NVFP4 在线基础生成服务                     |
| `services/video-minimax-h3-regenerate-2k` | 5090 Open-H3 2K IC 重生成服务                          |
| `services/video-minimax-h3-singularity`   | H3 Singularity 推理服务                                |
| `services/video-minimax-h3-sol`           | H3-Sol 双 rank 推理服务                                |
| `services/video-minimax-h3-vdn`           | H3-VDN 推理服务                                        |
| `services/video-frame-interpolation`      | RIFE 视频补帧推理子服务                                |
| `services/video-super-resolution`         | SeedVR2 视频超分辨率推理子服务                         |
| `services/video-h3-latent-upscaler`       | H3 潜空间放大后处理子服务                              |
| `services/video-depth-anything`           | Video Depth Anything 深度提取子服务                    |

H3-Sol 的既有双卡实验出片与人工质量限制，以中央设计记录为准。实验镜像可追溯发布不等于热启动服务、生产接入或质量验收完成。H3、H3-Sol 的后续模型服务不使用 Kubernetes Job，分别复用各自模型进程；MCP 不加载模型。

## 目录与事实源

```text
.github/workflows/
services/
  video-depth-anything/
  video-frame-interpolation/
  video-h3-latent-upscaler/
  video-mcp-server/
  video-minimax-h3/
  video-minimax-h3-fasth3/
  video-minimax-h3-regenerate-2k/
  video-minimax-h3-singularity/
  video-minimax-h3-sol/
  video-minimax-h3-vdn/
  video-super-resolution/
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
python3 -m pip install ./services/video-mcp-server
PYTHONPATH=services/video-mcp-server python3 -m unittest discover -s services/video-mcp-server/tests -v
```

日常提交至 `dev`，测试通过后只以 `--ff-only` 合并至 `release` 并推送，触发服务自己的镜像流水线，然后切回 `dev`。镜像使用不可覆盖的语义化版本；本地构建和临时挂载不能作为发布证据。H3-Sol 的 CI 执行 CPU 契约、分配校验、官方来源及补丁校验，构建中再检查引擎导入，只有 `release` 推送才发布镜像。

## MCP 与工作台

MCP 保持 `artifact.import`、`video.generate`、`video.status`、`video.result` 契约。Dashboard 分为 MCP 服务状态、模型服务、任务状态。H3 / H3-Sol 展示就绪、当前、期望实例数，支持模型 → 实例 → GPU 指标与短期历史。未上报的任务执行实例显示未知。使用真实任务数据，支持素材导入、提交、筛选、缩略图、视频回放和下载；H3-Sol 显式选择，支持 5 / 10 / 15 秒；历史任务固定沿原路线查询，不静默回退。

以中央部署公布的地址访问 `/video/dashboard`。数据接口及产物下载使用现有 Bearer Token；浏览器仅在页面内存保存 Token。内部 `/runtime-artifacts/` 不公开。工具不暴露内部 Runtime Task ID、服务地址或宿主路径。

H3-Sol 部署、真实冷 / 热推理对比和人工质量审核仍按中央变更记录执行，未完成前不推进 `main`。

## Python 依赖管理

每个 `services/<service>/pyproject.toml` 独立声明项目元数据和依赖；仓库根不合并不同服务的 CUDA / PyTorch 环境。常规安装使用 `python -m pip install ./services/<service>`，容器仍按 Dockerfile 复制并运行源码，项目 wheel 仅承载依赖元数据。

轻量 CPU 测试依赖放在 `[dependency-groups].test`，使用 pip 25.3：

```bash
python -m pip install --upgrade pip==25.3
# 仅适用于声明了 test 组的服务；不安装 GPU 运行时。
python -m pip install --group services/<service>/pyproject.toml:test
```

GPU 索引、预装运行时与需要关闭构建隔离的编译步骤仍由 Dockerfile 控制，普通本地安装不能替代 GPU 镜像验证。

H3-Sol 的直接依赖源为 `pyproject.toml`；`requirements.lock` 与 `requirements.build.lock` 保留已验证的完整运行时和构建锁定结果，构建继续分阶段安装并执行 `pip check`。更新直接依赖时须同步重新解析锁文件。H3 API 的空依赖清单表示运行时完全继承已锁定基础镜像；Depth Anything 也继续继承固定镜像中的依赖。

Depth Anything 的测试扩展使用 `python -m pip install "./services/video-depth-anything[test]"`；Video MCP 的浏览器扩展使用 `python -m pip install "./services/video-mcp-server[browser]"`。超分引擎测试的 `cpu-torch` 组使用 PyTorch CPU 索引安装，再安装 `engine-test` 组。

H3-Sol 的清单版本对应现行 resident 服务；`Dockerfile.resident` 复制并安装该清单的元数据，运行依赖仍继承固定基础镜像。

## 独立渠道仓库

Higgsfield 渠道服务与页面迁至 [verdantflare-app-higgsfield](https://github.com/verdantflarehub/verdantflare-app-higgsfield)，fal.ai 渠道服务与页面迁至 [verdantflare-app-fal](https://github.com/verdantflarehub/verdantflare-app-fal)。Video 仅保留渠道消费者与任务／Artifact 集成；渠道独立发布，沿用现行内部接口。
