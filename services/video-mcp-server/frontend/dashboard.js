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
const taskStatusLabel = (task) => task.service === "higgsfield" && task.status === "running" && (task.runtime_stage === "collecting" || task.timing?.generation_completed_at) ? "视频已生成" : labels[task.status] || task.status;
const generationTimeHint = "按服务端确认 Higgsfield 完成的时间计算；官方 CLI 未提供精确完成时间。";
// Display metadata is shared by the list and detail API; raw IDs stay in task records.
const generationServices = new Set(["h3", "h3-sol", "h3-vdn", "h3-singularity", "fal", "grok", "google", "higgsfield"]);
const serviceDisplayNames = {
  "h3-latent-upscale": "H3 Latent Upscaler", sr: "Video Super Resolution",
  interpolate: "Frame Interpolation", depth: "Depth Anything",
};
const taskModel = (task) => generationServices.has(task.service) ? task.presentation?.model_label || "未记录模型" : "";
const taskRoute = (task) => generationServices.has(task.service) ? task.presentation?.channel_label || "未记录渠道" : "";
const higgsfieldLogo = "<svg width=\"14\" height=\"14\" class=\"channel-logo\" aria-hidden=\"true\" focusable=\"false\" viewBox=\"0 0 1355 1355\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\">\n<rect width=\"1355\" height=\"1355\" rx=\"363\" fill=\"#D1FE17\"/>\n<path d=\"M1088.73 670.726L1087.95 662.156C1080.52 579.906 1026.99 425.16 878.52 425.16C768.341 425.16 685.111 533.914 611.658 629.796C553.046 706.582 502.264 772.452 446.385 772.452C431.533 770.898 412.391 763.49 400.676 746.73C390.125 731.522 387.392 712.035 392.858 688.649C401.448 651.619 450.686 617.315 502.644 580.677C531.161 561.19 560.472 540.529 580.79 520.651C639.402 464.134 669.094 423.205 669.094 357.336C669.094 291.467 632.76 258.716 602.283 244.682C541.331 216.625 451.862 232.994 394.818 282.114C386.227 289.913 377.625 297.31 369.807 304.328C312.371 355.392 273.696 390.086 185 363.571V470.369C302.605 522.216 401.46 423.205 438.97 377.605C467.878 347.592 498.355 330.049 521.024 330.049H522.2C532.359 330.441 540.95 334.34 547.211 341.357C557.37 353.056 561.279 366.699 559.319 381.896C555.018 413.864 521.808 451.273 460.856 492.203C389.352 540.149 269.799 620.444 260.413 721.399C253.379 793.905 290.889 866.4 349.501 894.458C486.247 959.165 569.477 847.684 657.77 729.969C725.365 639.149 789.443 553.01 878.531 553.01C958.625 553.01 988.318 619.271 988.318 660.982V669.173L980.5 670.726C786.318 705.029 680.429 886.67 680.429 970.473C680.429 1054.28 751.541 1126 839.061 1126C941.433 1126 1068.02 1038.69 1088.34 793.123L1089.12 784.162H1170V670.737H1088.73V670.726ZM982.841 796.62C967.216 943.566 891.804 1012.17 846.095 1012.17C825.385 1012.17 796.477 995.021 796.477 963.064C796.477 927.208 850.003 818.454 970.341 786.106L984.409 782.597L982.841 796.632V796.62Z\" fill=\"black\"/>\n</svg>";
const localChannelLogo = '<svg class="channel-logo" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true" focusable="false"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4"/></svg>';
const falChannelLogo = '<svg class="channel-logo" width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false"><rect width="24" height="24" rx="6" fill="#6C3AED"/><path d="M8.5 7H15.5M8.5 12H13.5M8.5 7V17" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="15.5" cy="15.5" r="1.8" fill="#4CE39A"/></svg>';
const channelLogo = (task) => {
  const icon = task.presentation?.channel_icon;
  if (icon === "higgsfield") return higgsfieldLogo;
  if (icon === "local") return localChannelLogo;
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
    higgsfield: "视频生成",
    grok: "视频生成",
    "h3-singularity": "视频生成",
    "h3-sol": "视频生成",
    "h3-vdn": "视频生成",
    fal: "视频生成",
    "h3-latent-upscale": "H3 潜空间超分",
    sr: "视频超分",
    interpolate: "视频补帧",
    depth: "深度视频",
  })[task.service] || "视频处理";
const taskService = (task) => serviceDisplayNames[task.service] || "";
const taskIdentity = (task) => {
  if (generationServices.has(task.service)) {
    return `<div class="task-identity generation-identity" title="${escapeHTML(taskModel(task))} × ${escapeHTML(taskRoute(task))}"><strong class="identity-model">${escapeHTML(taskModel(task))}</strong><span class="identity-separator" aria-hidden="true">×</span><strong class="identity-channel">${channelLogo(task)}${escapeHTML(taskRoute(task))}</strong></div>`;
  }
  return `<div class="task-identity"><div><span>操作：</span><strong>${escapeHTML(taskOperation(task))}</strong></div><div><span>服务：</span><strong>${escapeHTML(taskService(task) || task.service || "未指定")}</strong></div></div>`;
};
const taskPrompt = (task) =>
  generationServices.has(task.service)
    ? task.prompt || "未提供提示词"
    : taskOperation(task);
const taskResolution = (task) => {
  if (task.resolution) return task.resolution;
  const h = Number(task.media?.height || 0), w = Number(task.media?.width || 0);
  const shortSide = h && w ? Math.min(h, w) : 0;
  if (shortSide >= 2160) return "4K";
  if (shortSide >= 1440) return "2K";
  if (shortSide >= 1080) return "1080p";
  if (shortSide >= 760) return "768p";
  if (shortSide >= 720) return "720p";
  if (shortSide >= 480) return "480p";
  const route = task.route || task.service;
  if (["h3", "h3-sol", "h3-vdn", "h3-singularity"].includes(route) || ["h3", "h3-sol", "h3-vdn", "h3-singularity"].includes(task.service)) return "768p";
  if (route === "fal" || task.service === "fal") return "480p";
  if (task.service === "higgsfield" || task.service === "grok") return "720p";
  return "";
};
const taskFps = (task) => `${task.media?.frame_rate || 24}FPS`;
const taskTags = (task) => {
  if (generationServices.has(task.service)) {
    const res = taskResolution(task);
    const fps = taskFps(task);
    return `${generationBadge(task)}${res ? `<span>${escapeHTML(res)}</span>` : ""}${task.aspect_ratio ? `<span>${escapeHTML(task.aspect_ratio)}</span>` : ""}${task.duration_seconds != null ? `<span>${escapeHTML(task.duration_seconds)}S</span>` : ""}${fps ? `<span>${escapeHTML(fps)}</span>` : ""}`;
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
    : `${escapeHTML(taskOperation(task))}<br>${escapeHTML(taskService(task) || task.service || "未指定服务")}`;
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
function elapsed(t) {
  if (t.service === "higgsfield") {
    if (t.timing?.processing_seconds != null) return `运行约 ${seconds(t.timing.processing_seconds)}`;
    if (t.status === "succeeded" || t.runtime_stage === "collecting") return "运行时间未记录";
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
async function api(path, options = {}) {
  const headers = { Authorization: `Bearer ${token}`, ...options.headers };
  if (options.body) headers["Content-Type"] = "application/json";
  const response = window.studioEmbedded
    ? await window.studioRequest(path, options)
    : await fetch(`${base}${path}`, {
        ...options,
        headers,
        cache: "no-store",
      });
  if (!response.ok) {
    if (response.status === 401) {
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
      502: "推理或素材服务暂时不可用。提交结果不确定时请先查询原任务，不要更换幂等键重发。",
      503: "服务尚未配置访问 Token。",
    };
    const error = new Error(messages[response.status] || "操作失败，请稍后重试。");
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
      (task.service === "higgsfield" && task.media?.sha256),
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
    `<span class="badge-status ${t.status === "succeeded" ? "completed" : escapeHTML(t.status)}">${t.runtime_stage === "submission_unknown" ? "提交待确认" : t.runtime_stage === "collect_retry" ? "回收重试中" : escapeHTML(taskStatusLabel(t))}</span>`;
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
        `<article class="model-card${openMenuTaskId === t.video_task_id ? " menu-open" : ""}" tabindex="0" role="button" data-task="${escapeHTML(t.video_task_id)}" aria-label="查看 ${escapeHTML(taskDisplayName(t))}"><div class="card-thumb-wrap">${thumbPlaceholderHTML(t)}${taskIdentity(t)}<div class="card-badges">${nsfwBadge(t)}${badge(t)}</div><div class="card-times"><span title="开始时间">${date(t.created_at)}</span><span title="${t.service === "higgsfield" ? generationTimeHint : "排队 / 运行"}">${elapsed(t)}</span></div></div><div class="card-body"><div><div class="card-title-row"><span title="${escapeHTML(taskDisplayName(t))}">${escapeHTML(taskDisplayName(t))}</span></div><p class="card-prompt">${escapeHTML(taskPrompt(t))}</p></div><div class="card-footer"><div class="model-tags">${taskTags(t)}</div>${cardActionMenu(t)}</div></div></article>`,
    )
    .join("");
  reconcileGallery(galleryHTML, loadedTasks);
  const tableHTML = loadedTasks
    .map(
      (t) =>
        `<tr data-task="${escapeHTML(t.video_task_id)}"><td title="${escapeHTML(taskDisplayName(t))}">${escapeHTML(taskDisplayName(t))}</td><td>${taskTableIdentity(t)}</td><td>${escapeHTML((t.prompt || "").slice(0, 100))}</td><td>${escapeHTML(t.duration_seconds)}s · ${escapeHTML(t.aspect_ratio)}</td><td>${elapsed(t)}</td><td>${nsfwBadge(t)}${badge(t)}</td><td>${cardActionMenu(t)}</td></tr>`,
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
async function handleTaskAction(action, id, ratingAction) {
  if (!id) return;
  const task = loadedTasks.find((t) => t.video_task_id === id);
  try {
    if (action === "edit") {
      const current = task ? taskDisplayName(task) : "";
      const nextTitle = window.prompt("编辑任务名称（1–64 字）：", current);
      if (nextTitle == null) return;
      const trimmed = nextTitle.trim();
      if (!trimmed || trimmed === current) return;
      await api(`/api/tasks/${encodeURIComponent(id)}/title`, {
        method: "POST",
        body: JSON.stringify({
          title: trimmed,
          expected_title_revision: Number(task?.title_revision ?? 0),
        }),
      });
      notice(`已更新任务名称：${trimmed}`);
      await refresh();
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
      if (!window.confirm(`确认删除任务「${name}」吗？此操作不可恢复。`)) return;
      await api(`/api/tasks/${encodeURIComponent(id)}/delete`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      loadedTasks = loadedTasks.filter((t) => t.video_task_id !== id);
      notice(`已删除任务：${name}`);
      await refresh();
    }
  } catch (error) {
    notice(error.message || "操作失败，请稍后重试。", true);
  }
}
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
      const channel = ["h3", "h3-sol", "h3-vdn", "h3-singularity"].includes(savedChannel) ? "local" : savedChannel;
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
