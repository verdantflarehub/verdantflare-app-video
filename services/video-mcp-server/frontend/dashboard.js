"use strict";
const $ = (id) => document.getElementById(id);
const base = location.pathname.split("/dashboard")[0];
const detailTaskId =
  location.pathname.match(/\/dashboard\/tasks\/([^/]+)$/)?.[1] || null;
const tokenStorageKey = `verdantflare.video.dashboard.token:${base}`;
function storeToken(value) {
  try {
    if (value) localStorage.setItem(tokenStorageKey, value);
    else localStorage.removeItem(tokenStorageKey);
  } catch {
    // Storage-disabled browsers still support the current in-memory session.
  }
}
let grokCapability = null;
const names = { grok: "Grok", h3: "h3", "h3-sol": "h3-sol", "h3-vdn": "h3-vdn", fal: "fal", mcp: "MCP" };
const labels = {
  queued: "排队中",
  running: "渲染中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};
const gpuQueueBlockingLabels = {
  human_protected: "视口编辑免杀保护中",
  draining_evictable_peer: "等待空闲热缓存清场释放",
  draining_peer: "等待空闲热缓存清场释放",
  card_committed_saturation: "物理卡承诺预算已满",
  insufficient_vram: "可用显存不足（等待在途任务释放）",
  head_of_line_reservation: "队首独占预约屏障等待中",
  telemetry_stale_fail_closed: "节点遥测过期（安全关闭新准入）",
  node_agent_unhealthy: "节点 GPU 代理未就绪（安全阻断）",
  quarantined_holder_blocking: "前序异常进程隔离冻结中",
  quota_slot_held_by_protected: "GPU 副本配额被受保护实例占用",
  k8s_slot_exhausted: "GPU 副本配额已满",
  engine_slot: "等待同引擎执行槽位释放",
  gpu_stage_slot: "等待本地执行槽位释放",
  stage_slot_busy: "本地执行槽位繁忙",
  scheduler_unavailable: "GPU 调度服务暂不可达",
};
const gpuQueueIsDraining = (gq) =>
  Boolean(
    gq &&
      (gq.lease_state === "draining" ||
        gq.waiting_reason === "draining_peer" ||
        gq.blocking_reason === "draining_evictable_peer"),
  );
const gpuQueueRankText = (gq) => {
  const pos = Number(gq?.queue_position);
  return Number.isInteger(pos) && pos > 0 ? `#${pos}` : "";
};
const sanitizeBlockingHolder = (holder) => {
  if (!holder || typeof holder !== "string") return "";
  const trimmed = holder.trim();
  if (!trimmed) return "";
  const appPrefix = trimmed.split("/")[0];
  const appNames = {
    blender: "Blender 常驻实例",
    comfyui: "ComfyUI 常驻实例",
    "video-mcp": "本地视频任务",
    "music-mcp": "本地音乐任务",
    "speech-mcp": "本地语音任务",
    "image-mcp": "本地图像任务",
  };
  return appNames[appPrefix] || "前序 GPU 实例";
};
const gpuQueueReasonText = (task) => {
  const gq = task?.gpu_queue;
  if (!gq || task?.status !== "queued") return "";
  const reasonKey =
    gq.blocking_reason && gq.blocking_reason !== "none"
      ? gq.blocking_reason
      : gq.waiting_reason || "";
  const baseLabel =
    gpuQueueBlockingLabels[reasonKey] ||
    (gpuQueueIsDraining(gq) ? "等待空闲热缓存清场释放" : "等待可用 GPU 显存准入");
  const holderLabel = sanitizeBlockingHolder(gq.blocking_holder_id);
  return holderLabel ? `${baseLabel}（当前占用：${holderLabel}）` : baseLabel;
};
const taskStatusLabel = (task) => {
  if (task.status === "running" && task.runtime_stage === "submission_unknown") {
    return "提交待确认";
  }
  if (
    task.service === "higgsfield" &&
    task.status === "running" &&
    (task.runtime_stage === "collecting" || task.timing?.generation_completed_at)
  ) {
    return "视频已生成";
  }
  if (task.status === "queued" && task.gpu_queue) {
    const stage = gpuQueueIsDraining(task.gpu_queue) ? "清场衔接中" : "排队中";
    const rank = gpuQueueRankText(task.gpu_queue);
    return rank ? `${stage} · ${rank}` : stage;
  }
  return labels[task.status] || task.status;
};
const generationTimeHint = "按服务端确认 Higgsfield 完成的时间计算；官方 CLI 未提供精确完成时间。";
// Display metadata is shared by the list and detail API; raw IDs stay in task records.
const generationServices = new Set(["h3", "h3-sol", "h3-vdn", "h3-fasth3", "h3-singularity", "regenerate_2k", "character_swap", "lipsync", "fal", "grok", "google", "google-flow", "higgsfield"]);
const localProcessingServices = new Set(["h3-latent-upscale", "sr", "interpolate", "depth"]);
const serviceDisplayNames = {
  "h3-latent-upscale": "H3 Latent Upscaler", sr: "Video Super Resolution",
  interpolate: "Frame Interpolation", regenerate_2k: "H3 Regenerate 2K", character_swap: "H3-Character-Swap", lipsync: "LatentSync 1.6", depth: "Depth Anything",
};
const taskModel = (task) => generationServices.has(task.service) ? task.presentation?.model_label || "未记录模型" : "";
const taskRoute = (task) =>
  task.presentation?.channel_label ||
  (localProcessingServices.has(task.service)
    ? "本地"
    : generationServices.has(task.service)
      ? "未记录渠道"
      : "");
const higgsfieldLogo = "<svg width=\"14\" height=\"14\" class=\"channel-logo\" aria-hidden=\"true\" focusable=\"false\" viewBox=\"0 0 1355 1355\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\">\n<rect width=\"1355\" height=\"1355\" rx=\"363\" fill=\"#D1FE17\"/>\n<path d=\"M1088.73 670.726L1087.95 662.156C1080.52 579.906 1026.99 425.16 878.52 425.16C768.341 425.16 685.111 533.914 611.658 629.796C553.046 706.582 502.264 772.452 446.385 772.452C431.533 770.898 412.391 763.49 400.676 746.73C390.125 731.522 387.392 712.035 392.858 688.649C401.448 651.619 450.686 617.315 502.644 580.677C531.161 561.19 560.472 540.529 580.79 520.651C639.402 464.134 669.094 423.205 669.094 357.336C669.094 291.467 632.76 258.716 602.283 244.682C541.331 216.625 451.862 232.994 394.818 282.114C386.227 289.913 377.625 297.31 369.807 304.328C312.371 355.392 273.696 390.086 185 363.571V470.369C302.605 522.216 401.46 423.205 438.97 377.605C467.878 347.592 498.355 330.049 521.024 330.049H522.2C532.359 330.441 540.95 334.34 547.211 341.357C557.37 353.056 561.279 366.699 559.319 381.896C555.018 413.864 521.808 451.273 460.856 492.203C389.352 540.149 269.799 620.444 260.413 721.399C253.379 793.905 290.889 866.4 349.501 894.458C486.247 959.165 569.477 847.684 657.77 729.969C725.365 639.149 789.443 553.01 878.531 553.01C958.625 553.01 988.318 619.271 988.318 660.982V669.173L980.5 670.726C786.318 705.029 680.429 886.67 680.429 970.473C680.429 1054.28 751.541 1126 839.061 1126C941.433 1126 1068.02 1038.69 1088.34 793.123L1089.12 784.162H1170V670.737H1088.73V670.726ZM982.841 796.62C967.216 943.566 891.804 1012.17 846.095 1012.17C825.385 1012.17 796.477 995.021 796.477 963.064C796.477 927.208 850.003 818.454 970.341 786.106L984.409 782.597L982.841 796.632V796.62Z\" fill=\"black\"/>\n</svg>";
const localChannelLogo = '<svg class="channel-logo" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true" focusable="false"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4"/></svg>';
const falChannelLogo = '<svg class="channel-logo" width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false"><rect width="24" height="24" rx="6" fill="#6C3AED"/><path d="M8.5 7H15.5M8.5 12H13.5M8.5 7V17" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="15.5" cy="15.5" r="1.8" fill="#4CE39A"/></svg>';
const channelLogo = (task) => {
  const icon = task.presentation?.channel_icon;
  if (icon === "google-flow" || task.service === "google-flow") return '<svg class="channel-logo" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true" fill="none"><rect width="24" height="24" rx="6" fill="#1a73e8"/><path d="M5 15c3-7 5-7 7 0s4 6 7-1" stroke="white" stroke-width="2" stroke-linecap="round"/></svg>';
  if (icon === "higgsfield") return higgsfieldLogo;
  if (icon === "local" || localProcessingServices.has(task.service)) return localChannelLogo;
  if (icon === "fal" || task.presentation?.channel_label === "fal.ai" || task.route === "fal" || task.service === "fal") return falChannelLogo;
  return "";
};
const generationBadge = (task) => {
  const code = task.presentation?.generation_code;
  const label = task.presentation?.generation_label;
  return code && label ? `<span class="generation-type" tabindex="0" title="${escapeHTML(label)}" aria-label="${escapeHTML(code)} · ${escapeHTML(label)}" data-tooltip="${escapeHTML(label)}">${escapeHTML(code)}</span>` : "";
};
const taskOperation = (task) =>
  ({
    h3: "视频生成",
    "google-flow": "视频生成",
    higgsfield: "视频生成",
    grok: "视频生成",
    "h3-fasth3": "视频生成",
    "h3-singularity": "视频生成",
    "h3-sol": "视频生成",
    "h3-vdn": "视频生成",
    fal: "视频生成",
    "h3-latent-upscale": "H3 潜空间超分",
    regenerate_2k: "2K 潜空间重生成",
    character_swap: "视频角色替换",
    lipsync: "视频口型同步",
    sr: "视频超分",
    interpolate: "视频补帧",
    depth: "深度视频",
  })[task.service] || "视频处理";
const taskService = (task) => serviceDisplayNames[task.service] || "";
const taskIdentity = (task) => {
  if (generationServices.has(task.service)) {
    return `<div class="task-identity generation-identity" title="${escapeHTML(taskModel(task))} × ${escapeHTML(taskRoute(task))}"><strong class="identity-model">${escapeHTML(taskModel(task))}</strong><span class="identity-separator" aria-hidden="true">×</span><strong class="identity-channel">${channelLogo(task)}${escapeHTML(taskRoute(task))}</strong></div>`;
  }
  const route = taskRoute(task) || "本地";
  return `<div class="task-identity generation-identity" title="${escapeHTML(taskOperation(task))} × ${escapeHTML(route)}"><strong class="identity-model">${escapeHTML(taskOperation(task))}</strong><span class="identity-separator" aria-hidden="true">×</span><strong class="identity-channel">${channelLogo(task)}${escapeHTML(route)}</strong></div>`;
};
const taskPrompt = (task) =>
  generationServices.has(task.service)
    ? task.prompt || "未提供提示词"
    : taskOperation(task);
const taskResolution = (task) => {
  if (task.resolution) return task.resolution;
  const h = Number(task.media?.height || 0), w = Number(task.media?.width || 0);
  const shortSide = h && w ? Math.min(h, w) : 0;
  const longSide = h && w ? Math.max(h, w) : 0;
  if (shortSide >= 2160) return "4K";
  if (shortSide >= 1408 || longSide >= 2464) return "2K";
  if (shortSide >= 1080) return "1080p";
  if (shortSide >= 760) return "768p";
  if (shortSide >= 720) return "720p";
  if (shortSide >= 480) return "480p";
  const route = task.route || task.service;
  if (task.service === "regenerate_2k") return "2K";
  if (["h3", "h3-sol", "h3-vdn", "h3-fasth3", "h3-singularity"].includes(route) || ["h3", "h3-sol", "h3-vdn", "h3-fasth3", "h3-singularity"].includes(task.service)) return "768p";
  if (route === "fal" || task.service === "fal") return "480p";
  if (task.service === "higgsfield" || task.service === "grok") return "720p";
  return "";
};
const parseFrameRateNumber = (value) => {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (/^\d+\/\d+$/.test(trimmed)) {
      const [num, den] = trimmed.split("/").map(Number);
      if (den > 0) return num / den;
    }
    const parsed = Number(trimmed);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  return null;
};
const taskDurationSecondsValue = (t) => {
  if (t?.duration_seconds != null && Number.isFinite(Number(t.duration_seconds))) {
    return Number(t.duration_seconds);
  }
  const mediaDur = Number(t?.media?.duration_seconds);
  if (t?.media?.duration_seconds != null && Number.isFinite(mediaDur) && mediaDur > 0) {
    return Math.round(mediaDur * 10) / 10;
  }
  const frames = Number(t?.media?.frames);
  const fps = parseFrameRateNumber(t?.media?.frame_rate ?? t?.media?.fps);
  if (Number.isFinite(frames) && frames > 0 && fps) {
    return Math.round((frames / fps) * 10) / 10;
  }
  return null;
};
const taskAspectOrSizeText = (t) => {
  if (t?.aspect_ratio) return String(t.aspect_ratio);
  if (t?.media?.width && t?.media?.height) return `${t.media.width}×${t.media.height}`;
  return "";
};
const taskDurationAspect = (t) => {
  const dur = taskDurationSecondsValue(t);
  const aspect = taskAspectOrSizeText(t);
  const parts = [];
  if (dur != null) parts.push(`${dur}s`);
  if (aspect) parts.push(aspect);
  return parts.length ? parts.join(" · ") : "—";
};
const taskFps = (task) => `${task.media?.frame_rate || 24}FPS`;
const taskTags = (task) => {
  if (generationServices.has(task.service)) {
    const res = taskResolution(task);
    const fps = taskFps(task);
    const dur = taskDurationSecondsValue(task);
    return `${generationBadge(task)}${res ? `<span>${escapeHTML(res)}</span>` : ""}${task.aspect_ratio ? `<span>${escapeHTML(task.aspect_ratio)}</span>` : ""}${dur != null ? `<span>${escapeHTML(dur)}S</span>` : ""}${fps ? `<span>${escapeHTML(fps)}</span>` : ""}`;
  }
  const media = task.media || {};
  const size =
    media.width && media.height
      ? `<span>${escapeHTML(media.width)}×${escapeHTML(media.height)}</span>`
      : "";
  const fps = media.frame_rate
    ? `<span>${escapeHTML(media.frame_rate)}FPS</span>`
    : "";
  return `${size}${fps}`;
};
const taskTableIdentity = (task) =>
  generationServices.has(task.service)
    ? `${escapeHTML(taskModel(task))} × ${escapeHTML(taskRoute(task))}<div class="model-tags">${generationBadge(task)}</div>`
    : `${escapeHTML(taskOperation(task))} × ${escapeHTML(taskRoute(task) || "本地")}`;
let token = "",
  authorized = false,
  page = 1,
  status = "all",
  total = 0,
  busy = false,
  loadingMore = false,
  loadMoreError = "",
  loadedTasks = [],
  sentinelIntersecting = false,
  scrollObserver = null,
  timer,
  searchTimer,
  listReady = false;
const thumbnails = new Map();
const thumbnailRequests = new Map();
const thumbnailFailures = new Map();
let modalTask = null,
  modalUrls = [],
  restoreFocus = null,
  openMenuTaskId = null,
  revision = 0;
const escapeHTML = (v) =>
  String(v ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const date = (v) => (v ? new Date(v).toLocaleString() : "—");
const seconds = (v) => {
  if (v == null || !Number.isFinite(v)) return "—";
  const n = Math.max(0, Math.floor(v));
  return n < 60 ? `${n}s` : `${Math.floor(n / 60)}m${n % 60}s`;
};
const preciseSeconds = (v) => {
  const num = Number(v);
  if (v == null || !Number.isFinite(num)) return "—";
  const clamped = Math.max(0, num);
  const exact = `${clamped.toFixed(2)}s`;
  return clamped >= 60 ? `${exact}（${seconds(clamped)}）` : exact;
};
function elapsed(t) {
  if (t.service === "higgsfield") {
    if (t.timing?.processing_seconds != null) return `运行约 ${seconds(t.timing.processing_seconds)}`;
    if (t.status === "succeeded" || t.runtime_stage === "collecting") return "运行时间未记录";
  }
  const chainActive = t.delivery?.timing?.chain_active_seconds;
  if (chainActive != null && Number.isFinite(Number(chainActive)) && Number(chainActive) > 0 && t.status === "succeeded") {
    const q = t.delivery?.timing?.queue_seconds ?? t.timing?.queue_seconds;
    return q != null ? `排队 ${seconds(q)} / 链路活跃 ${seconds(Number(chainActive))}` : `链路活跃 ${seconds(Number(chainActive))}`;
  }
  if (t.timing) {
    const q = t.timing.queue_seconds;
    const r =
      t.timing.processing_seconds ?? t.timing.processing_elapsed_seconds;
    if (q != null && r != null)
      return `排队 ${seconds(q)} / ${t.timing.processing_elapsed_seconds != null ? "运行中" : "运行"} ${seconds(r)}`;
    if (t.status === "queued" && t.timing.queue_elapsed_seconds != null)
      return `排队中 ${seconds(t.timing.queue_elapsed_seconds)}`;
    if (q != null) return `排队 ${seconds(q)} / 运行 —`;
    if (t.timing.total_seconds != null)
      return `总耗时 ${seconds(t.timing.total_seconds)}（未记录分段）`;
  }
  return seconds(
    Math.max(
      0,
      ((["queued", "running"].includes(t.status)
        ? Date.now()
        : Date.parse(t.completed_at || t.updated_at)) -
        Date.parse(t.created_at)) /
        1000,
    ),
  );
}
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}
function clearMedia() {
  window.studioReleaseMedia?.();
  modalUrls.forEach(URL.revokeObjectURL);
  modalUrls = [];
}
function openModal(id) {
  restoreFocus = document.activeElement;
  $(id).hidden = false;
  const panel = $(id).querySelector(".modal-window");
  if (window.studioEmbedded) panel?.scrollIntoView({ block: "start" });
  $(id).querySelector("input,button,textarea")?.focus({ preventScroll: true });
}
function closeModal(id) {
  $(id).hidden = true;
  restoreFocus?.focus();
}
function invalidate() {
  listReady = false;
  loadingMore = false;
  loadMoreError = "";
  loadedTasks = [];
  thumbnails.forEach((v) => URL.revokeObjectURL(v.url));
  thumbnails.clear();
  thumbnailRequests.clear();
  thumbnailFailures.clear();
  authorized = false;
  clearTimeout(timer);
  revision++;
  modalTask = null;
  clearMedia();
  clearTaskDetail();
  $("galleryContainer").replaceChildren();
  $("taskRows").replaceChildren();
  $("emptyState").hidden = false;
  $("emptyState").textContent = "请连接 MCP 以查看任务。";
  $("pageLabel").textContent = "0 个任务";
  if ($("loadMoreBtn")) $("loadMoreBtn").hidden = true;
  if ($("retryLoadMoreBtn")) $("retryLoadMoreBtn").hidden = true;
  [
    "metricQueued",
    "metricRunning",
    "metricCompleted",
    "metricFailed",
    "metricLatency",
  ].forEach((id) => ($(id).textContent = "—"));
  clearBusiness();
}
function serviceFailure(body) {
  const legacy = {
    authentication_not_configured: "服务尚未配置访问 Token。",
    permission_denied: "没有此项目的访问权限。",
    not_found: "任务或素材不存在。",
    title_conflict: "名称已被修改，请核对最新名称后重试。",
    idempotency_conflict: "幂等键与已有请求冲突，请核对原任务。",
    invalid_request: "请检查参数、参考素材类型及项目归属。",
    request_too_large: "请求内容过大。",
    service_not_connected: "所选渠道尚未接入。",
    preview_unavailable: "结果预览尚不可用。",
    preview_extraction_failed: "视频首帧提取失败，请稍后重试。",
    upstream_operation_failed: "上游操作失败；提交结果不确定时请查询原任务。",
    operation_failed: "操作失败，请稍后查询原任务。",
  };
  const candidate = body?.error && typeof body.error === "object" ? body.error : body?.diagnostic;
  if (!candidate || typeof candidate !== "object") {
    const code = body?.error;
    return typeof code === "string" && Object.hasOwn(legacy, code)
      ? { code, message: legacy[code], source: "video" } : null;
  }
  if (!/^[A-Za-z0-9_.-]{1,100}$/.test(candidate.code || "") ||
      !["video", "studio", "higgsfield", "fal", "grok", "google", "google-flow"].includes(candidate.source) ||
      typeof candidate.message !== "string") return null;
  const message = candidate.message
    .replace(/https?:\/\/\S+/gi, "[已隐藏链接]")
    .replace(/(["']?(?:api[_-]?key|access_token|refresh_token|token|password|secret|authorization)["']?\s*[:=]\s*)["']?[^\s,}\]]+/gi, "$1[已隐藏]")
    .replace(/bearer\s+\S+/gi, "Bearer [已隐藏]")
    .replace(/(?<!\w)\/(?:[\w.-]+\/)+[\w./-]*/g, "[已隐藏路径]")
    .replace(/\b[a-z]:[\\/][^\s,;]+/gi, "[已隐藏路径]")
    .replace(/\b[A-Za-z0-9_-]{40,}\b/g, "[已隐藏]")
    .replace(/[\u0000-\u001f\u007f]/g, " ").slice(0, 512);
  const value = { code: candidate.code, message, source: candidate.source, retryable: candidate.retryable === true };
  if (candidate.accepted === null || typeof candidate.accepted === "boolean") value.accepted = candidate.accepted;
  if (Number.isInteger(candidate.http_status) && candidate.http_status >= 100 && candidate.http_status <= 599) value.http_status = candidate.http_status;
  if (/^video_task_[0-9a-f]{32}$/.test(candidate.task_id || "")) value.task_id = candidate.task_id;
  return value;
}
function failureMessage(failure) {
  let message = `${failure.message}（${failure.code} · ${failure.source}）`;
  if (Object.hasOwn(failure, "accepted")) {
    message += failure.accepted === false ? " 请求未受理。"
      : failure.accepted === true ? " 原请求已受理，请查询原任务，不要重复提交。"
      : " 是否受理尚未确认，请查询原任务，不要重复提交。";
  }
  return message;
}
async function api(path, options = {}) {
  const headers = { Authorization: `Bearer ${token}`, ...options.headers };
  if (options.body) headers["Content-Type"] = "application/json";
  const response = window.studioEmbedded
    ? await window.studioRequest(path, options)
    : await fetch(`${base}${path}`, {
        ...options,
        headers,
        cache: options.blob ? "default" : "no-store",
      });
  if (!response.ok) {
    let failure = null;
    try {
      const raw = await response.text();
      if (raw.length <= 32768) failure = serviceFailure(JSON.parse(raw));
    } catch { /* Non-JSON proxy failures use the safe HTTP fallback below. */ }
    if (response.status === 401 && (!failure || ["video", "studio"].includes(failure.source))) {
      token = "";
      storeToken("");
      invalidate();
      $("pollState").textContent = "认证失败";
      $("tokenButton").textContent = "Token：请重新设置";
    }
    const messages = {
      401: window.studioEmbedded
        ? "Studio 会话已过期，请重新登录。"
        : "Token 无效或已过期，请重新设置。",
      400: path.endsWith("/title") ? "名称须为 1–64 字的单行纯文本。" : "请检查参数、参考素材类型及项目归属。",
      403: "没有此项目的编辑权限。",
      404: "任务或素材不存在。",
      409: path.endsWith("/title") ? "名称已被修改，请核对最新名称后重试。" : "幂等键冲突或所选服务尚未接入。",
      413: "请求内容过大。",
      422: "视频首帧提取失败，请稍后重试。",
      429: "请求受到限流，请稍后查询原任务。",
      502: "推理或素材服务暂时不可用。提交结果不确定时请先查询原任务，不要更换幂等键重发。",
      503: "服务暂时不可用；提交结果不确定时请先查询原任务，不要更换幂等键重发。",
    };
    const message = failure ? failureMessage(failure)
      : messages[response.status] || `操作失败（HTTP ${response.status}），请稍后查询原任务。`;
    const error = new Error(message);
    if (failure) {
      const { message: _message, ...diagnostic } = failure;
      Object.assign(error, diagnostic);
    }
    error.status = response.status;
    throw error;
  }
  return options.blob ? response.blob() : response.json();
}
function query(targetPage = page) {
  return new URLSearchParams({
    page: targetPage,
    page_size: 24,
    status,
    channel: $("filterChannel").value,
    project_id: $("filterProject").value,
    q: $("searchInput").value.trim(),
  });
}
const hasResultPreview = (task) =>
  task?.status === "succeeded" &&
  Boolean(
    task.artifact_id ||
      (["higgsfield", "google-flow"].includes(task.service) && task.media?.sha256),
  );
const previewKey = (task) =>
  `${task.video_task_id}:${task.status}:${task.artifact_id || task.media?.sha256 || ""}`;
const thumbPlaceholderText = (task) => {
  if (hasResultPreview(task)) return "正在加载首帧…";
  if (task.status === "succeeded") return "结果视频待取回";
  if (task.status === "running") return "生成中 · 完成后提取首帧";
  if (task.status === "queued") return "排队中 · 待生成结果";
  if (task.status === "failed") return "任务失败 · 无结果视频";
  if (task.status === "cancelled") return "任务已取消 · 无结果视频";
  return "暂无结果首帧";
};
const thumbPlaceholderHTML = (task) => {
  const key = previewKey(task);
  if (hasResultPreview(task) && thumbnailFailures.has(key)) {
    return `<span class="thumb-placeholder thumb-error" data-thumb-status="error"><span>首帧提取失败</span><button type="button" class="button thumb-retry-btn" data-thumb-retry="${escapeHTML(task.video_task_id)}">重试首帧</button></span>`;
  }
  return `<span class="thumb-placeholder" data-thumb-status="${hasResultPreview(task) ? "loading" : escapeHTML(task.status)}">${escapeHTML(thumbPlaceholderText(task))}</span>`;
};
function applyCardThumbnail(task) {
  const gallery = $("galleryContainer");
  if (!gallery?.children) return;
  const card = [...gallery.children].find(
    (el) => el.dataset?.task === task.video_task_id,
  );
  const key = previewKey(task);
  if (!card || card.dataset.previewKey !== key) return;
  const cached = thumbnails.get(task.video_task_id);
  if (cached && cached.key === key) {
    let img = card.querySelector(".card-thumb-wrap > img");
    if (!img) {
      img = document.createElement("img");
      img.className = "card-thumb-img";
      const slot = card.querySelector(".card-thumb-wrap > span");
      if (slot) slot.replaceWith(img);
      else card.querySelector(".card-thumb-wrap")?.prepend(img);
    }
    if (img.getAttribute("src") !== cached.url) img.src = cached.url;
    img.alt = "生成视频首帧";
    return;
  }
  const wrap = card.querySelector(".card-thumb-wrap");
  if (!wrap) return;
  const template = document.createElement("template");
  template.innerHTML = thumbPlaceholderHTML(task);
  const nextSpan = template.content.firstElementChild;
  const currentMedia = wrap.querySelector(":scope > img, :scope > span");
  if (currentMedia && nextSpan) {
    if (currentMedia.tagName === "IMG" || currentMedia.outerHTML !== nextSpan.outerHTML)
      currentMedia.replaceWith(nextSpan);
  } else if (nextSpan) {
    wrap.prepend(nextSpan);
  }
}
const THUMBNAIL_CONCURRENCY = 4;
async function previews(tasks) {
  const active = new Set(tasks.map((t) => t.video_task_id));
  for (const [id, item] of thumbnails)
    if (!active.has(id)) {
      URL.revokeObjectURL(item.url);
      thumbnails.delete(id);
    }
  const version = revision;
  const pendingTasks = [];
  for (const task of tasks) {
    if (version !== revision || !authorized) return;
    if (!hasResultPreview(task)) {
      applyCardThumbnail(task);
      continue;
    }
    const key = previewKey(task);
    const cached = thumbnails.get(task.video_task_id);
    if (cached && cached.key === key) {
      applyCardThumbnail(task);
      continue;
    }
    if (thumbnailFailures.has(key)) {
      applyCardThumbnail(task);
      continue;
    }
    pendingTasks.push(task);
  }
  if (!pendingTasks.length || document.hidden || !authorized) return;
  let cursor = 0;
  async function worker() {
    while (cursor < pendingTasks.length) {
      if (version !== revision || !authorized || document.hidden) return;
      const task = pendingTasks[cursor++];
      const key = previewKey(task);
      if (!thumbnailRequests.has(key)) {
        const request = api(`/api/tasks/${encodeURIComponent(task.video_task_id)}/thumbnail`, {
          blob: true,
        })
          .then((blob) => {
            if (version !== revision || !authorized || !blob || !blob.size)
              return null;
            const old = thumbnails.get(task.video_task_id);
            const item = {
              url: URL.createObjectURL(blob),
              key,
              artifact: task.artifact_id || task.media?.sha256 || "",
            };
            thumbnails.set(task.video_task_id, item);
            thumbnailFailures.delete(key);
            if (old) URL.revokeObjectURL(old.url);
            return item;
          })
          .catch((err) => {
            if (version === revision && authorized) {
              thumbnailFailures.set(key, {
                failedAt: Date.now(),
                status: err?.status || 0,
              });
            }
            return null;
          })
          .finally(() => thumbnailRequests.delete(key));
        thumbnailRequests.set(key, request);
      }
      await thumbnailRequests.get(key);
      if (version === revision && authorized) applyCardThumbnail(task);
    }
  }
  await Promise.all(
    Array.from(
      { length: Math.min(THUMBNAIL_CONCURRENCY, pendingTasks.length) },
      () => worker(),
    ),
  );
}
function retryThumbnail(taskId) {
  const task = loadedTasks.find((t) => t.video_task_id === taskId);
  if (!task) return Promise.resolve();
  thumbnailFailures.delete(previewKey(task));
  applyCardThumbnail(task);
  return previews([task]);
}
function mergeTaskBatches(batches) {
  const seen = new Set();
  const merged = [];
  for (const list of batches) {
    for (const item of list || []) {
      if (!item?.video_task_id || seen.has(item.video_task_id)) continue;
      seen.add(item.video_task_id);
      merged.push(item);
    }
  }
  return merged;
}
const hasMoreTasks = () =>
  total > 0 && loadedTasks.length < total && page * 24 < total;
function updatePagerUI() {
  const loadedCount = loadedTasks.length;
  const canLoadMore = hasMoreTasks();
  if ($("prevPage")) {
    $("prevPage").hidden = true;
    $("prevPage").disabled = page <= 1;
  }
  if ($("nextPage")) {
    $("nextPage").hidden = true;
    $("nextPage").disabled = !canLoadMore;
  }
  if ($("loadMoreBtn")) {
    $("loadMoreBtn").hidden = !canLoadMore || loadingMore || Boolean(loadMoreError);
    $("loadMoreBtn").disabled = loadingMore || !canLoadMore;
  }
  if ($("retryLoadMoreBtn")) {
    $("retryLoadMoreBtn").hidden = !loadMoreError || loadingMore;
    $("retryLoadMoreBtn").disabled = loadingMore;
  }
  if (!$("pageLabel")) return;
  if (loadingMore) {
    $("pageLabel").textContent = `正在加载更多任务…（已加载 ${loadedCount} / 共 ${total} 个任务）`;
  } else if (loadMoreError) {
    $("pageLabel").textContent = `加载更多失败：${loadMoreError}（已保留 ${loadedCount} / 共 ${total} 个任务）`;
  } else if (canLoadMore) {
    $("pageLabel").textContent = `已加载 ${loadedCount} / 共 ${total} 个任务 · 向下滚动自动加载`;
  } else if (total > 0) {
    $("pageLabel").textContent = `已加载全部 ${total} 个任务`;
  } else {
    $("pageLabel").textContent = "0 个任务";
  }
}
// Keep task and image nodes across polling; update only changed metadata.
function reconcileGallery(html, tasks) {
  const gallery = $("galleryContainer"),
    template = document.createElement("template");
  template.innerHTML = html;
  const existing = new Map(
    [...gallery.children].map((el) => [el.dataset.task, el]),
  );
  const taskMap = new Map(tasks.map((t) => [t.video_task_id, t]));
  [...template.content.children].forEach((next, index) => {
    const id = next.dataset.task,
      task = taskMap.get(id);
    let card = existing.get(id);
    if (card) {
      existing.delete(id);
      card.setAttribute("aria-label", next.getAttribute("aria-label"));
      card.classList.toggle("menu-open", openMenuTaskId === id);
      for (const selector of [".task-identity", ".card-badges", ".card-body"]) {
        const current = card.querySelector(selector),
          replacement = next.querySelector(selector);
        if (current.innerHTML !== replacement.innerHTML)
          current.innerHTML = replacement.innerHTML;
      }
      const times = card.querySelectorAll(".card-times span"),
        nextTimes = next.querySelectorAll(".card-times span");
      times.forEach((el, i) => {
        if (el.textContent !== nextTimes[i].textContent)
          el.textContent = nextTimes[i].textContent;
      });
      const placeholder = card.querySelector(".card-thumb-wrap > span");
      const nextPlaceholder = next.querySelector(".card-thumb-wrap > span");
      if (placeholder && nextPlaceholder && placeholder.outerHTML !== nextPlaceholder.outerHTML)
        placeholder.replaceWith(nextPlaceholder.cloneNode(true));
    } else card = next;
    card.dataset.previewKey = previewKey(task);
    if (gallery.children[index] !== card)
      gallery.insertBefore(card, gallery.children[index] || null);
  });
  existing.forEach((card) => card.remove());
}
function render(data) {
  total = data.total;
  loadedTasks = data.tasks || [];
  try {
    grokCapability = data.grok_capability;
    const option = $("dispatchForm").elements.route.querySelector('option[value="grok"]');
    option.disabled = !(grokCapability?.enabled && grokCapability?.configured);
    option.textContent = option.disabled ? "Grok · 未配置" : `Grok · ${grokCapability.model}`;
  } catch { /* Existing channels remain available when discovery fails. */ }
  const falOption = $("dispatchForm").elements.route.querySelector('option[value="fal"]');
  const falConnected = data.services?.fal === "configured";
  falOption.disabled = !falConnected;
  falOption.textContent = falConnected ? "fal" : "fal · 未配置 FAL_KEY";
  $("metricQueued").textContent = data.counts.queued;
  $("metricRunning").textContent = data.counts.running;
  $("metricFailed").textContent = data.counts.failed;
  $("metricCompleted").textContent = data.counts.succeeded;
  const unknownCount = Number(data.counts?.unknown ?? 0);
  const failedCount = Number(data.counts?.failed ?? 0);
  latestCounts = {
    queued: Number(data.counts?.queued ?? 0),
    running: Number(data.counts?.running ?? 0),
    succeeded: Number(data.counts?.succeeded ?? 0),
    failed: failedCount,
    cancelled: Number(data.counts?.cancelled ?? 0),
    unknown: unknownCount,
  };
  if ($("metricUnknown")) $("metricUnknown").textContent = unknownCount;
  if ($("chipUnknown")) $("chipUnknown").classList.toggle("amber-chip", unknownCount > 0);
  if ($("btnCleanupTasks")) {
    $("btnCleanupTasks").disabled = unknownCount === 0 && failedCount === 0;
    $("btnCleanupTasks").classList.toggle("amber-cleanup", unknownCount > 0);
  }
  $("metricLatency").textContent = seconds(data.average_elapsed_seconds);
  const channel = $("filterChannel").value;
  const channels = data.channels || [];
  $("filterChannel").innerHTML = '<option value="all">全部渠道</option>' +
    channels.map((item) => `<option value="${escapeHTML(item.id)}">${escapeHTML(item.label)}</option>`).join("");
  if (channel !== "all" && !channels.some((item) => item.id === channel))
    $("filterChannel").add(new Option(channel, channel));
  $("filterChannel").value = channel;
  const project = $("filterProject").value;
  const projectOptions = data.project_options || (data.projects || []).map((id, i) => ({id, label: `未分类 · 历史项目 ${i + 1}（名称不可用）`}));
  $("filterProject").innerHTML =
    '<option value="all">全部项目</option>' +
    projectOptions
      .map((v) => `<option value="${escapeHTML(v.id)}">${escapeHTML(v.label)}</option>`)
      .join("");
  if (project !== "all" && !projectOptions.some((item) => item.id === project))
    $("filterProject").add(new Option("原选项目（名称不可用）", project));
  $("filterProject").value = project;
  $("emptyState").hidden = loadedTasks.length > 0;
  $("emptyState").textContent = data.total
    ? "此页没有任务，请返回上一页。"
    : "暂无符合条件的任务。可调整筛选，或新建运镜任务。";
  const badge = (t) =>
    `<span class="badge-status ${t.status === "succeeded" ? "completed" : t.status === "queued" && gpuQueueIsDraining(t.gpu_queue) ? "queued draining" : escapeHTML(t.status)}">${t.runtime_stage === "submission_unknown" ? "提交待确认" : t.runtime_stage === "collect_retry" ? "回收重试中" : escapeHTML(taskStatusLabel(t))}</span>`;
  const queueReasonMarkup = (t, cls) => {
    const reason = gpuQueueReasonText(t);
    return reason ? `<p class="${cls}">阻塞原因：${escapeHTML(reason)}</p>` : "";
  };
  const nsfwBadge = (t) =>
    t.content_rating === "nsfw"
      ? `<span class="badge-tag badge-nsfw" data-rating="nsfw">18+</span>`
      : "";
  const cardActionMenu = (t) => {
    const id = escapeHTML(t.video_task_id);
    const isNsfw = t.content_rating === "nsfw";
    const toggleLabel = isNsfw ? "移除 18+ 标记" : "标记为 18+ (NSFW)";
    const isOpen = openMenuTaskId === t.video_task_id;
    return `<div class="card-more-wrap"><button type="button" class="card-more-btn" data-menu-toggle="${id}" aria-label="更多操作" aria-haspopup="menu" aria-expanded="${isOpen ? "true" : "false"}">···</button><div class="card-action-menu${isOpen ? " open" : ""}" role="menu" data-menu-for="${id}"${isOpen ? "" : " hidden"}><button type="button" class="card-action-item" role="menuitem" data-task-action="edit" data-task-id="${id}">编辑</button><button type="button" class="card-action-item" role="menuitem" data-task-action="toggle-nsfw" data-task-id="${id}" data-rating-action="${isNsfw ? "clear" : "mark"}">${toggleLabel}</button><button type="button" class="card-action-item danger" role="menuitem" data-task-action="delete" data-task-id="${id}">删除</button></div></div>`;
  };
  const galleryHTML = loadedTasks
    .map(
      (t) =>
        `<article class="model-card${openMenuTaskId === t.video_task_id ? " menu-open" : ""}" tabindex="0" role="button" data-task="${escapeHTML(t.video_task_id)}" aria-label="查看 ${escapeHTML(taskDisplayName(t))}"><div class="card-thumb-wrap">${thumbPlaceholderHTML(t)}${taskIdentity(t)}<div class="card-badges">${nsfwBadge(t)}${badge(t)}</div><div class="card-times"><span title="开始时间">${date(t.created_at)}</span><span title="${t.service === "higgsfield" ? generationTimeHint : "排队 / 运行"}">${elapsed(t)}</span></div></div><div class="card-body"><div><div class="card-title-row"><span title="${escapeHTML(taskDisplayName(t))}">${escapeHTML(taskDisplayName(t))}</span></div><p class="card-prompt">${escapeHTML(taskPrompt(t))}</p>${queueReasonMarkup(t, "card-queue-reason")}</div><div class="card-footer"><div class="model-tags">${taskTags(t)}</div>${cardActionMenu(t)}</div></div></article>`,
    )
    .join("");
  reconcileGallery(galleryHTML, loadedTasks);
  const tableHTML = loadedTasks
    .map(
      (t) =>
        `<tr data-task="${escapeHTML(t.video_task_id)}"><td title="${escapeHTML(taskDisplayName(t))}">${escapeHTML(taskDisplayName(t))}</td><td>${taskTableIdentity(t)}</td><td>${escapeHTML((t.prompt || "").slice(0, 100))}</td><td>${escapeHTML(taskDurationAspect(t))}</td><td>${elapsed(t)}</td><td>${nsfwBadge(t)}${badge(t)}${queueReasonMarkup(t, "table-queue-reason")}</td><td>${cardActionMenu(t)}</td></tr>`,
    )
    .join("");
  if ($("taskRows").innerHTML !== tableHTML)
    $("taskRows").innerHTML = tableHTML;
  previews(loadedTasks);
  listReady = true;
  setupScrollAppend();
  updatePagerUI();
  if (window.studioEmbedded) window.studioNavigate("/dashboard" + (location.hash || "#tasks"));
  notice(
    data.sync_errors
      ? `${data.sync_errors} 个任务暂未同步，保留最后已知状态。最后同步：${date(data.synced_at)}`
      : `任务状态同步：${date(data.synced_at)} · 数据按当前筛选统计。`,
    !!data.sync_errors,
  );
}
async function refresh() {
  clearTimeout(timer);
  if (/^#(?:create|higgsfield(?:-task-[a-zA-Z0-9_-]+)?)$/.test(location.hash)) return;
  if (!authorized || document.hidden) return;
  if (busy || loadingMore) {
    timer = setTimeout(refresh, 500);
    return;
  }
  busy = true;
  refreshBusiness();
  const version = revision;
  try {
    if (detailTaskId) {
      await refreshTaskDetail(decodeURIComponent(detailTaskId), version);
    } else {
      const firstPage = await api(`/api/dashboard?${query(1)}`);
      if (version !== revision) return;
      const maxPage = Math.max(1, Math.ceil((firstPage.total || 0) / 24));
      const targetPages = Math.min(Math.max(1, page), maxPage);
      const batches = [firstPage.tasks || []];
      for (let p = 2; p <= targetPages; p++) {
        const nextData = await api(`/api/dashboard?${query(p)}`);
        if (version !== revision) return;
        batches.push(nextData.tasks || []);
      }
      page = targetPages;
      const merged = mergeTaskBatches(batches);
      render({ ...firstPage, tasks: merged });
    }
    $("pollState").textContent = "2s 自动刷新";
  } catch (error) {
    if (error.status === 403 && !detailTaskId && window.studioEmbedded) {
      location.hash = '#higgsfield';
      return;
    }
    if (version === revision) {
      notice(error.message, true);
      $("pollState").textContent = "连接异常 · 任务为最后快照";
    }
  } finally {
    busy = false;
    if (authorized && !document.hidden) timer = setTimeout(refresh, 2000);
  }
}
async function loadMore() {
  if (
    !authorized ||
    document.hidden ||
    detailTaskId ||
    !listReady ||
    loadingMore ||
    !hasMoreTasks()
  )
    return;
  if (busy) {
    setTimeout(loadMore, 120);
    return;
  }
  clearTimeout(timer);
  loadingMore = true;
  loadMoreError = "";
  updatePagerUI();
  const version = revision;
  const nextPageNum = page + 1;
  try {
    const data = await api(`/api/dashboard?${query(nextPageNum)}`);
    if (version !== revision || !authorized) return;
    page = nextPageNum;
    total = data.total;
    loadedTasks = mergeTaskBatches([loadedTasks, data.tasks || []]);
    loadingMore = false;
    loadMoreError = "";
    render({ ...data, tasks: loadedTasks });
    if (scrollObserver && $("scrollSentinel") && hasMoreTasks()) {
      scrollObserver.unobserve($("scrollSentinel"));
      scrollObserver.observe($("scrollSentinel"));
    }
  } catch (error) {
    if (version === revision && authorized) {
      loadingMore = false;
      loadMoreError = error.message || "加载更多任务失败，请重试。";
      updatePagerUI();
    }
  } finally {
    if (version === revision) loadingMore = false;
    if (authorized && !document.hidden) timer = setTimeout(refresh, 2000);
  }
}
function setupScrollAppend() {
  if (scrollObserver || typeof IntersectionObserver !== "function") return;
  const sentinel = $("scrollSentinel");
  const footer = $("taskListFooter");
  if (!sentinel && !footer) return;
  scrollObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.target === sentinel || entry.target === footer) {
          sentinelIntersecting = entry.isIntersecting;
        }
      }
      if (sentinelIntersecting && !loadMoreError) {
        loadMore();
      }
    },
    { root: null, rootMargin: "240px 0px", threshold: 0 },
  );
  if (sentinel) scrollObserver.observe(sentinel);
  if (footer) scrollObserver.observe(footer);
}
function filter() {
  page = 1;
  revision++;
  loadingMore = false;
  loadMoreError = "";
  loadedTasks = [];
  $("galleryContainer")?.replaceChildren();
  $("taskRows")?.replaceChildren();
  updatePagerUI();
  refresh();
}
async function mediaURL(id, taskId) {
  const blob = await api(`/artifacts/${encodeURIComponent(id)}/content`, {
    blob: true,
  });
  if (modalTask !== taskId) throw new Error("详情已关闭");
  const url = URL.createObjectURL(blob);
  window.studioRememberMedia?.(url, blob);
  modalUrls.push(url);
  return url;
}
function inspect(id) {
  if (window.studioEmbedded)
    window.studioNavigate(`/dashboard/tasks/${encodeURIComponent(id)}`);
  else location.href = `${base}/dashboard/tasks/${encodeURIComponent(id)}`;
}
function dispatch() {
  if (!authorized) {
    openModal("tokenModal");
    return;
  }
  if (!$("attemptInput").value)
    $("attemptInput").value = `shot/${crypto.randomUUID()}`;
  openModal("dispatchModal");
}
function closeAllActionMenus() {
  openMenuTaskId = null;
  document.querySelectorAll(".card-action-menu").forEach((menu) => {
    menu.hidden = true;
    menu.classList.remove("open");
  });
  document.querySelectorAll("button[data-menu-toggle]").forEach((btn) => {
    btn.setAttribute("aria-expanded", "false");
  });
  document.querySelectorAll(".model-card.menu-open, tr.menu-open").forEach((el) => {
    el.classList.remove("menu-open");
  });
}
let latestCounts = { queued: 0, running: 0, succeeded: 0, failed: 0, cancelled: 0, unknown: 0 };
let pendingDeleteTask = null;
let pendingEditTask = null;

function syncCleanupModalState() {
  const unknown = Number(latestCounts.unknown || 0);
  const failed = Number(latestCounts.failed || 0);
  const chkUnknown = $("chkCleanUnknown");
  const chkFailed = $("chkCleanFailed");
  if (!chkUnknown || !chkFailed) return;
  if ($("badgeUnknownCount")) $("badgeUnknownCount").textContent = `${unknown} 项`;
  if ($("badgeFailedCount")) $("badgeFailedCount").textContent = `${failed} 项`;
  if ($("cardCheckUnknown")) $("cardCheckUnknown").classList.toggle("checked-amber", chkUnknown.checked);
  if ($("cardCheckFailed")) $("cardCheckFailed").classList.toggle("checked-danger", chkFailed.checked);
  let types = 0;
  let totalSelected = 0;
  if (chkUnknown.checked) { types += 1; totalSelected += unknown; }
  if (chkFailed.checked) { types += 1; totalSelected += failed; }
  if ($("cleanupSelectedSummary")) {
    $("cleanupSelectedSummary").textContent =
      types > 0 ? `已选 ${types} 类 · 共 ${totalSelected} 项待清理` : "请至少勾选一类待清理任务";
  }
  if ($("btnConfirmCleanup")) $("btnConfirmCleanup").disabled = totalSelected === 0;
}

function openCleanupModal() {
  const unknown = Number(latestCounts.unknown || 0);
  const failed = Number(latestCounts.failed || 0);
  if ($("chkCleanUnknown")) $("chkCleanUnknown").checked = unknown > 0;
  if ($("chkCleanFailed")) $("chkCleanFailed").checked = unknown === 0 && failed > 0;
  syncCleanupModalState();
  openModal("cleanupTasksModal");
}

async function executeCleanupTasks() {
  const chkUnknown = Boolean($("chkCleanUnknown")?.checked);
  const chkFailed = Boolean($("chkCleanFailed")?.checked);
  if (!chkUnknown && !chkFailed) return;
  const btn = $("btnConfirmCleanup");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "清理中…";
  }
  try {
    const result = await api("/api/tasks/_cleanup/delete", {
      method: "POST",
      body: JSON.stringify({ clean_unknown: chkUnknown, clean_failed: chkFailed }),
    });
    if (result?.ok !== true) throw new Error("未取得有效清理回执");
    closeModal("cleanupTasksModal");
    const removedIds = new Set([
      ...(Array.isArray(result.cleaned_unknown_task_ids) ? result.cleaned_unknown_task_ids : []),
      ...(Array.isArray(result.cleaned_failed_task_ids) ? result.cleaned_failed_task_ids : []),
      ...(Array.isArray(result.cleaned_cancelled_task_ids) ? result.cleaned_cancelled_task_ids : []),
    ]);
    if (removedIds.size > 0) {
      loadedTasks = loadedTasks.filter((t) => !removedIds.has(t.video_task_id));
    }
    const parts = [];
    if (chkUnknown) {
      const rec = Number(result.recovered_count || 0);
      const cln = Number(result.cleaned_unknown_count || 0);
      parts.push(rec > 0 ? `追回 ${rec} 项、清理 ${cln} 项待核对/超时任务` : `${cln} 项待核对/超时任务`);
    }
    if (chkFailed) {
      parts.push(`${Number(result.cleaned_failed_count || 0)} 项已失败任务`);
    }
    if (Number(result.skipped_count || 0) > 0) {
      parts.push(`跳过 ${Number(result.skipped_count)} 项取消确认中任务`);
    }
    notice(`已统一清理：${parts.join(" + ")}`);
    await refresh();
  } catch (error) {
    notice(`清理失败：${error.message || "请稍后重试。"}`, true);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "确认清理";
    }
  }
}

async function confirmDeleteTask() {
  if (!pendingDeleteTask) return;
  const { id, name } = pendingDeleteTask;
  const btn = $("btnConfirmDeleteTask");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "删除中…";
  }
  try {
    await api(`/api/tasks/${encodeURIComponent(id)}/delete`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    closeModal("deleteTaskModal");
    pendingDeleteTask = null;
    loadedTasks = loadedTasks.filter((t) => t.video_task_id !== id);
    notice(`已删除任务：${name}`);
    await refresh();
  } catch (error) {
    closeModal("deleteTaskModal");
    pendingDeleteTask = null;
    notice(error.message || "删除失败，请稍后重试。", true);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "确认删除";
    }
  }
}

async function confirmEditTaskTitle(event) {
  event.preventDefault();
  if (!pendingEditTask) return;
  const { id, current, revision } = pendingEditTask;
  const trimmed = ($("editTitleInput")?.value || "").trim();
  if (!trimmed || trimmed === current) {
    closeModal("editTitleModal");
    pendingEditTask = null;
    return;
  }
  const btn = $("btnConfirmEditTitle");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "保存中…";
  }
  try {
    await api(`/api/tasks/${encodeURIComponent(id)}/title`, {
      method: "POST",
      body: JSON.stringify({
        title: trimmed,
        expected_title_revision: revision,
      }),
    });
    closeModal("editTitleModal");
    pendingEditTask = null;
    notice(`已更新任务名称：${trimmed}`);
    await refresh();
  } catch (error) {
    notice(error.message || "更新名称失败，请稍后重试。", true);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "保存名称";
    }
  }
}

async function handleTaskAction(action, id, ratingAction) {
  if (!id) return;
  const task = loadedTasks.find((t) => t.video_task_id === id);
  try {
    if (action === "edit") {
      const current = task ? taskDisplayName(task) : "";
      pendingEditTask = {
        id,
        current,
        revision: Number(task?.title_revision ?? 0),
      };
      if ($("editTitleInput")) $("editTitleInput").value = current;
      openModal("editTitleModal");
      $("editTitleInput")?.focus?.();
      $("editTitleInput")?.select?.();
    } else if (action === "toggle-nsfw") {
      const nextAction =
        ratingAction || (task?.content_rating === "nsfw" ? "clear" : "mark");
      const reason =
        nextAction === "mark" ? "manual_mark_nsfw" : "manual_clear_nsfw";
      await api(`/api/tasks/${encodeURIComponent(id)}/content-rating`, {
        method: "POST",
        body: JSON.stringify({ action: nextAction, reason }),
      });
      notice(
        nextAction === "mark" ? "已标记为 18+ (NSFW)。" : "已移除 18+ 标记。",
      );
      await refresh();
    } else if (action === "delete") {
      const name = task ? taskDisplayName(task) : id;
      pendingDeleteTask = { id, name };
      if ($("deleteTaskDesc")) {
        $("deleteTaskDesc").textContent = `确认删除任务「${name}」吗？此操作不可恢复。`;
      }
      openModal("deleteTaskModal");
    }
  } catch (error) {
    notice(error.message || "操作失败，请稍后重试。", true);
  }
}
$("btnCleanupTasks")?.addEventListener?.("click", openCleanupModal);
$("chkCleanUnknown")?.addEventListener?.("change", syncCleanupModalState);
$("chkCleanFailed")?.addEventListener?.("change", syncCleanupModalState);
$("btnConfirmCleanup")?.addEventListener?.("click", executeCleanupTasks);
$("btnConfirmDeleteTask")?.addEventListener?.("click", confirmDeleteTask);
$("editTitleForm")?.addEventListener?.("submit", confirmEditTaskTitle);
document.addEventListener("click", (event) => {
  const retryBtn = event.target?.closest?.("button[data-thumb-retry]");
  if (retryBtn) {
    event.preventDefault();
    event.stopPropagation();
    retryThumbnail(retryBtn.dataset.thumbRetry);
    return;
  }
  const menuToggle = event.target?.closest?.("button[data-menu-toggle]");
  if (menuToggle) {
    event.preventDefault();
    event.stopPropagation();
    const id = menuToggle.dataset.menuToggle;
    const wrap = menuToggle.closest(".card-more-wrap");
    const menu = wrap?.querySelector(".card-action-menu");
    const willOpen = menu ? menu.hidden : openMenuTaskId !== id;
    closeAllActionMenus();
    if (willOpen && menu) {
      openMenuTaskId = id;
      menu.hidden = false;
      menu.classList.add("open");
      menuToggle.setAttribute("aria-expanded", "true");
      wrap.closest(".model-card, tr")?.classList.add("menu-open");
    }
    return;
  }
  const actionBtn = event.target?.closest?.("button[data-task-action]");
  if (actionBtn) {
    event.preventDefault();
    event.stopPropagation();
    const action = actionBtn.dataset.taskAction;
    const id = actionBtn.dataset.taskId;
    const ratingAction = actionBtn.dataset.ratingAction;
    closeAllActionMenus();
    handleTaskAction(action, id, ratingAction);
    return;
  }
  if (openMenuTaskId) closeAllActionMenus();
  const target = event.target?.closest?.("button,a,article,tr[data-task]");
  if (target?.dataset.action === "token") openModal("tokenModal");
  if (target?.dataset.action === "dispatch") dispatch();
  if (target?.dataset.action === "refresh") refresh();
  if (target?.dataset.close) closeModal(target.dataset.close);
  if (target?.dataset.task) {
    $("resourceDrawer").close();
    inspect(target.dataset.task);
  }
  if (target?.dataset.view) {
    const gallery = target.dataset.view === "gallery";
    $("galleryContainer").classList.toggle("hidden", !gallery);
    $("tableContainer").classList.toggle("hidden", gallery);
    $("btnViewGallery").classList.toggle("active", gallery);
    $("btnViewTable").classList.toggle("active", !gallery);
  }
  if (target?.dataset.status) {
    status = target.dataset.status;
    document
      .querySelectorAll("button[data-status]")
      .forEach((b) =>
        b.classList.toggle("active", b.dataset.status === status),
      );
    filter();
  }
  if (event.target?.classList?.contains("modal-backdrop"))
    closeModal(event.target.id);
});
document.addEventListener("keydown", (event) => {
  const dialogs = [
    ...document.querySelectorAll(".modal-backdrop:not([hidden])"),
  ];
  if ($("resourceDrawer").open) return;
  const dialog = dialogs.at(-1);
  if (event.key === "Escape" && dialog) closeModal(dialog.id);
  if (event.key === "Tab" && dialog) {
    const elements = [
      ...dialog.querySelectorAll(
        "button:not(:disabled),input,textarea,select,a[href]",
      ),
    ];
    const first = elements[0],
      last = elements.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first?.focus();
    }
  }
  if (
    ["Enter", " "].includes(event.key) &&
    event.target.matches("article[data-task]")
  ) {
    event.preventDefault();
    inspect(event.target.dataset.task);
  }
});
$("tokenForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  invalidate();
  token = $("tokenInput").value.trim();
  storeToken(token);
  $("tokenInput").value = "";
  authorized = true;
  $("tokenButton").textContent = "Token：已设置";
  closeModal("tokenModal");
  await refresh();
});
$("clearToken").addEventListener("click", () => {
  token = "";
  storeToken("");
  invalidate();
  $("tokenInput").value = "";
  $("tokenButton").textContent = "Token：未配置";
  $("pollState").textContent = "已断开";
  notice("Token 已清除。");
  closeModal("tokenModal");
});
$("searchInput").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(filter, 250);
});
["filterChannel", "filterProject"].forEach((id) =>
  $(id).addEventListener("change", filter),
);
$("prevPage").addEventListener("click", () => {
  if (page > 1) {
    page--;
    revision++;
    refresh();
  }
});
$("nextPage").addEventListener("click", () => {
  loadMore();
});
$("loadMoreBtn")?.addEventListener("click", () => {
  loadMore();
});
$("retryLoadMoreBtn")?.addEventListener("click", () => {
  loadMoreError = "";
  loadMore();
});
window.addEventListener("scroll", () => {
  if (
    !authorized ||
    document.hidden ||
    detailTaskId ||
    loadingMore ||
    loadMoreError ||
    !hasMoreTasks()
  )
    return;
  if (
    window.innerHeight + window.scrollY >=
    document.documentElement.scrollHeight - 320
  )
    loadMore();
});
window.addEventListener("pagehide", () => {
  clearTimeout(timer);
  clearTimeout(searchTimer);
  scrollObserver?.disconnect();
  scrollObserver = null;
  thumbnails.forEach((v) => URL.revokeObjectURL(v.url));
  thumbnails.clear();
  thumbnailRequests.clear();
  clearMedia();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
  else clearTimeout(timer);
});
$("openImport").addEventListener("click", () => {
  $("importForm").elements.project_id.value =
    $("dispatchForm").elements.project_id.value;
  openModal("importModal");
});
$("importForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.target.querySelector("button[type=submit]");
  button.disabled = true;
  $("importMessage").textContent = "正在导入并校验…";
  try {
    const artifact = await api("/api/artifacts/import", {
      method: "POST",
      body: JSON.stringify(Object.fromEntries(new FormData(event.target))),
    });
    const kind = artifact.media_type.split("/")[0];
    if (artifact.project_id === $("dispatchForm").elements.project_id.value) {
      $("referenceInput").value +=
        `${$("referenceInput").value ? "\n" : ""}${kind} | ${artifact.artifact_id} | reference`;
    }
    $("importMessage").textContent =
      `导入成功：${artifact.artifact_id} · SHA-256 ${artifact.sha256}`;
  } catch (error) {
    $("importMessage").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
$("dispatchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("submitTask");
  button.disabled = true;
  $("dispatchMessage").textContent = "正在提交，请勿重复操作…";
  try {
    const body = Object.fromEntries(new FormData(event.target));
    body.duration_seconds = Number(body.duration_seconds);
    const selectedFalModel = body.fal_model || $("falModelSelect")?.value || "minimax-h3-ref2va";
    delete body.fal_model;
    body.model = body.route === "grok" ? grokCapability?.model : body.route === "fal" ? selectedFalModel : "minimax-h3-ref2va";
    if (body.route === "fal" && body.model === "bytedance-seedance-2.0-reference-to-video") {
      body.mode = "reference_to_video";
    }
    const references = { images: [], videos: [], audios: [] };
    const kinds = { image: "images", video: "videos", audio: "audios" };
    for (const line of body.references.trim().split("\n")) {
      if (!line.trim()) continue;
      const parts = line.split("|").map((s) => s.trim());
      if (parts.length !== 3 || !kinds[parts[0]] || !parts[1] || !parts[2])
        throw new Error(
          "每行请填写：image / video / audio | Artifact ID | 用途",
        );
      references[kinds[parts[0]]].push({
        artifact_id: parts[1],
        purpose: parts[2],
      });
    }
    const isFalSeedance = body.route === "fal" && body.model === "bytedance-seedance-2.0-reference-to-video";
    const missingReference = body.route === "fal" && !isFalSeedance
      ? !references.images.length && !references.videos.length && !references.audios.length
      : !references.images.length && !references.videos.length;
    if (missingReference && body.route !== "grok")
      throw new Error(body.route === "fal" && !isFalSeedance ? "fal 至少需要一个参考素材。" : "至少需要一张图片或一段视频参考。");
    if (body.route === "grok") {
      if (references.videos.length || references.audios.length || references.images.length > 1 || references.images.some(r => r.purpose !== "first_frame")) throw new Error("Grok 仅接受一张 first_frame 图片，或不使用参考。");
      body.mode = references.images.length ? "image_to_video" : "text_to_video";
      if (references.images.length) delete body.aspect_ratio;
    }
    body.references = references;
    const postprocess = buildDispatchPostprocess(body.route);
    if (postprocess) body.postprocess = postprocess;
    const task = await api("/api/tasks", {
      method: "POST",
      body: JSON.stringify(body),
    });
    $("dispatchMessage").textContent = `已提交：${taskDisplayName(task)}`;
    closeModal("dispatchModal");
    $("attemptInput").value = "";
    page = 1;
    revision++;
    await refresh();
    await inspect(task.video_task_id);
  } catch (error) {
    $("dispatchMessage").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

function buildDispatchPostprocess(route) {
  if (!["h3", "h3-fasth3", "h3-singularity"].includes(route)) return null;
  const useRegen2k = Boolean($("postprocessRegen2k")?.checked);
  const useInterpolate = Boolean($("postprocessInterpolate")?.checked);
  if (!useRegen2k && !useInterpolate) return null;
  const postprocess = {};
  if (useRegen2k) {
    const gridMode = $("postprocessGridMode")?.value === "exact_2x" ? "exact_2x" : "budget_1440p";
    postprocess.regenerate_2k = { profile: "open_h3_2k_ic", grid_mode: gridMode };
  }
  if (useInterpolate) {
    postprocess.interpolate = { profile: "rife_2x" };
  }
  return postprocess;
}

const syncDispatchRouteControls = () => {
  const duration = $("dispatchForm").elements.duration_seconds;
  const route = $("dispatchForm").elements.route.value;
  const grok = route === "grok";
  const fal = route === "fal";
  const falModelSelect = $("falModelSelect");
  if ($("falModelField")) $("falModelField").hidden = !fal;
  if (falModelSelect) falModelSelect.disabled = !fal;
  const falSeedance = fal && falModelSelect?.value === "bytedance-seedance-2.0-reference-to-video";
  const hasResolution = grok || falSeedance;
  $("referenceInput").required = !grok;
  $("grokResolution").hidden = !hasResolution;
  $("dispatchForm").elements.resolution.disabled = !hasResolution;
  const localModernH3 = ["h3", "h3-fasth3", "h3-singularity"].includes(route);
  if ($("postprocessGroup")) $("postprocessGroup").hidden = !localModernH3;
  if ($("postprocessGridMode")) {
    $("postprocessGridMode").disabled = !localModernH3 || !Boolean($("postprocessRegen2k")?.checked);
  }
  const legacy916Only = route.startsWith("h3-sol") || route === "h3-vdn";
  const minFive = legacy916Only || (fal && !falSeedance);
  const ratio = $("dispatchForm").elements.aspect_ratio;
  duration.min = grok ? "1" : minFive ? "5" : "4";
  duration.step = "1";
  const minDuration = Number(duration.min);
  if (!Number.isInteger(Number(duration.value)) || Number(duration.value) < minDuration || Number(duration.value) > 15)
    duration.value = String(Math.max(5, minDuration));
  [...ratio.options].forEach((option) => {
    option.hidden = grok
      ? !["16:9", "9:16", "1:1", "4:3", "3:4"].includes(option.value)
      : falSeedance
        ? option.value === "adaptive"
        : legacy916Only && option.value !== "9:16";
  });
  if (grok) ratio.value = "16:9";
  else if (fal && !falSeedance) ratio.value = "adaptive";
  else if (falSeedance && ratio.value === "adaptive") ratio.value = "16:9";
  else if (legacy916Only) ratio.value = "9:16";
};
$("dispatchForm").elements.route.addEventListener("change", syncDispatchRouteControls);
$("falModelSelect")?.addEventListener?.("change", syncDispatchRouteControls);
$("postprocessRegen2k")?.addEventListener?.("change", syncDispatchRouteControls);

window.addEventListener("DOMContentLoaded", () => {
  if (window.studioEmbedded) {
    notice("正在读取 Video 工作区…");
    return;
  }
  try {
    token = localStorage.getItem(tokenStorageKey) || "";
  } catch {}
  if (token) {
    authorized = true;
    $("tokenButton").textContent = "Token：已设置";
    refresh();
  }
});

if (window.studioEmbedded) {
  window.studioViewState = () =>
    detailTaskId
      ? null
      : {
          project: $("filterProject").value,
          channel: $("filterChannel").value,
          q: $("searchInput").value,
          status,
          page,
          table: $("btnViewTable").classList.contains("active"),
        };
  const saveView = () => {
    if (!detailTaskId && authorized && listReady) window.studioNavigate("/dashboard" + (location.hash || "#tasks"));
  };
  for (const eventName of ["click", "change", "input"]) {
    document.addEventListener(eventName, () => queueMicrotask(saveView));
  }
  window.addEventListener("studio-restore", (event) => {
    if (detailTaskId) {
      authorized = true;
      refresh();
      return;
    }
    const state = event.detail;
    if (state) {
      if (typeof state.project === "string" && state.project.length <= 64) {
        const option = new Option(state.project, state.project);
        $("filterProject").add(option);
        $("filterProject").value = state.project;
      }
      if (typeof state.q === "string")
        $("searchInput").value = state.q.slice(0, 512);
      const savedChannel = state.channel ?? state.engine;
      const channel = ["h3", "h3-sol", "h3-vdn", "h3-fasth3", "h3-singularity"].includes(savedChannel) ? "local" : savedChannel;
      if (typeof channel === "string" && channel.length <= 128) {
        if (![...$("filterChannel").options].some((option) => option.value === channel))
          $("filterChannel").add(new Option(channel, channel));
        $("filterChannel").value = channel;
      }
      if (
        [
          "all",
          "queued",
          "running",
          "succeeded",
          "failed",
          "cancelled",
        ].includes(state.status)
      )
        status = state.status;
      if (Number.isInteger(state.page) && state.page > 0 && state.page < 100000)
        page = state.page;
      document
        .querySelectorAll("button[data-status]")
        .forEach((b) =>
          b.classList.toggle("active", b.dataset.status === status),
        );
      if (state.table) $("btnViewTable").click();
    }
    authorized = true;
    refresh();
  });
}
