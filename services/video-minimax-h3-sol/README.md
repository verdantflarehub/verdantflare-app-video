# H3-Sol 常驻服务与实验工程

当前常驻镜像版本为 **`video-minimax-h3-sol-v0.2.0`**，入口 `resident_worker.py` 通过一个 torchrun 启动两个长期存活的 rank，串行处理持久化队列并复用 engine。构建使用 `Dockerfile.resident`，继承 v0.1.1 的固定 digest；默认源代码 Dockerfile 保留用于重建基础实验包，不覆盖历史版本。

内部健康与任务接口、部署资源、MCP 路由和验收口径见[中央常驻服务方案](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/app/video/minimax-h3/minimax-h3-sol-service.md)。模型加载完成才就绪；首次真实输入仍需预热。只有相同 Prompt、参考内容及 duration 才复用预热命中；其他请求保守重新预热，但不重新加载模型。运行中取消不支持，内部 DELETE 只接受排队任务。

`resident_api.py` 提供受保护的 HTTP 接口、SQLite 持久队列和恢复语义；`resident_worker.py` 负责双卡核验、模型哈希、分布式常驻推理与产物校验。模型和产物由中央 PVC 提供，不打进镜像。依赖与封存源码来自以下历史基础包说明。许可证 FAQ 的既有辅助缺项仍显式记录，真实创作质量仍待人工验收。

## v0.1.1 历史基础包

本次源码与镜像版本为 **`video-minimax-h3-sol-v0.1.1`**，由 `.github/workflows/video-minimax-h3-sol.yml` 在 `release` 分支构建发布。已有 `v0.1.0` 是历史实验镜像，不覆盖。

封装 NVIDIA 官方 Sol-H3、锁定依赖、四项 CPU offload / 竖屏补丁，以及后续实验脚本。历史双 RTX 4090 已跑通 B01 视频，但人工质量门未通过；历史正式 runner 的 4090 profile 仍为 blocked；v0.2.0 的独立常驻入口显式使用封存实验适配，不修改历史 runner 门禁。完整历史与缺项见[中央方案](https://github.com/verdantflarehub/verdantflare-design/blob/dev/docs/design/app/video/minimax-h3/minimax-h3-sol.md)。

## 镜像内的可追溯源码

| 路径 | 内容 |
| --- | --- |
| `/opt/sol-h3` | 官方 revision `2936c47637380842aaa4a4488fac5006cc542b70`，72 个文件逐一校验 |
| `/opt/sol-h3-experimental` | 构建时以零 fuzz 应用四项封存补丁，核验全部 before / after 哈希，并保存 `patches.manifest.json` |
| `/opt/verdantflare-sol` | 契约 runner、输入 / 模型检查、实验入口、CPU 权重保留适配与测试源码 |
| `/opt/verdantflare-sol/installed-packages.txt` | 镜像中实际安装的依赖清单 |

默认 `PYTHONPATH` 使用官方源码，默认入口 `run-inference.py --help` 不运行模型。实验源码保持独立，不能用实验补丁绕过正式 runner 的来源与 4090 门禁。镜像现在直接携带实验代码，不需要本地代码挂载。镜像不包含模型、凭据或媒体文件。

## 本地与 CI 验证

```bash
python3 -m unittest discover -s services/video-minimax-h3-sol/tests -v
python3 services/video-minimax-h3-sol/verify-source.py /path/to/sealed/Sol-H3
python3 services/video-minimax-h3-sol/prepare-patched-source.py \
  --source /path/to/sealed/Sol-H3 --output /path/to/new/experimental-Sol-H3
```

需要系统 `patch`。CPU 检查不需要 CUDA 或模型。CI 在 `dev` / Pull Request 运行检查，在 `release` 检查成功后构建并推送版本镜像；构建中检查官方与实验引擎导入、补丁参数和依赖一致性。既有完整依赖锁和 CUDA 基础镜像 digest 保持固定。APT 包未逐位锁定，不承诺镜像重建字节一致。

## GPU 实验边界

现有 `experimental-sample.py`、`integrated-load-probe.py` 和 `gpu_tests/` 仅保留已开展实验的代码。运行这些脚本必须通过中央资源核验，在同一分配内将实时批准的两张物理卡 UUID 注入 `SOL_EXPECTED_GPU_UUIDS`（逗号分隔）；缺失、重复或与可见设备不符时拒绝运行，不再使用写死的历史 GPU UUID。

直接运行 `gpu_tests/` 时，Python 搜索路径需包含本工程目录；镜像默认已包含。实验引擎另需显式使用 `/opt/sol-h3-experimental`。这不是自动执行授权，不创建 Job；后续服务化按中央热启动设计与部署清单交付。

冻结输入继续要求 `schema_version=1`、`task=ref2va`、`status=frozen`、可追溯批准 ID、原样 Prompt、5–15 秒整数时长、seed 及带 SHA-256 的受控参考资产。至少一张图片或一段视频，不接受音频独占输入、越界路径或改变的哈希。模型完整性核验保持启用；历史实验允许的唯一辅助 FAQ 缺项必须显式记录，不能宣称模型包完整。

正式 runner 与历史实验入口仍为单次执行；受控保温、预热复用、取消及异常恢复的服务化改造尚未完成。媒体技术通过不代表创作质量通过，不能把本次镜像发布描述为 H3-Sol 已上线。
