# fal.ai 视频渠道网关

## 1. 运行边界

此服务把 fal 组织级 API Key 与 Video MCP 任务分离。管理监听 `8000` 只供 Studio 受控 AppDashboard 代理调用；执行监听 `8001` 只供 Video MCP 调用。两个监听各有独立 Bearer Token，均不开放公网。应用发行、权限和部署事实以设计仓库的 `docs/design/app/fal/README.md` 与对应清单为准。

## 2. 必需配置

- `FAL_MANAGEMENT_TOKEN`：Studio 独占的内部管理凭据，至少 32 字符。
- `FAL_EXECUTION_TOKEN`：Video MCP 独占的执行凭据，至少 32 字符。
- `FAL_ENCRYPTION_KEY`：Fernet 主密钥，由 Kubernetes Secret 注入；与数据库分开保管。
- `FAL_STATE_DB`：持久数据库路径，默认 `/data/fal/state.sqlite3`。

管理请求须携带 Studio 验证后的 `X-User-Id` 与 `X-Organization-Id`。Studio 注册规则对每个管理路径设置 `requireManage=true`。执行请求只信任 Video MCP 内部凭据及其组织声明。浏览器不直接接触上述内部 Token；提交的 fal Key 不返回、不写日志或客户端持久存储。

## 3. 授权与执行

管理接口提供概览、授权保存／只读检查／撤销、用量和每日提交上限。Key 验证使用 fal 官方需要 `models:requests:read` 的只读请求查询；成功仅证明此权限，不证明余额或生成资格。管理员须再明确开启新任务并设置正数上限。首期执行白名单只有 `minimax/h3/reference-to-video` 的现有 480P 适配。

执行提交先持久化幂等记录，再请求固定的 fal Queue 地址。结果不确定时保留 `submission_unconfirmed`，不自动重提。历史 Video fal 任务仍按其原适配器版本恢复；新任务按 `video-fal-gateway-v0.1.0` 走网关。媒体下载与项目 Artifact 仍由 Video MCP 完成。

## 4. 轻量验证

在服务目录运行 `python -m unittest discover -s tests -v`；测试用 MockTransport，不持有真实 Key、不触发付费生成。Dashboard 构建可运行 `python -m fal_gateway.dashboard --source dashboard --output <空目录>`。镜像由 GitHub Actions 的 `video-fal-gateway.yml` 构建。
