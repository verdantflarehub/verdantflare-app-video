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
const names = { h3: "h3", "h3-sol": "h3-sol", "h3-vdn": "h3-vdn", mcp: "MCP" };
const labels = {
  queued: "排队中",
  running: "渲染中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};
const modelDisplayNames = {
  seedvr2: "SeedVR2",
  rife: "RIFE",
  "video-depth-anything": "Video-Depth-Anything",
  "h3-latent-upscaler": "H3 Latent Upscaler",
};
const generationServices = new Set(["h3", "h3-sol", "h3-vdn"]);
const knownChannels = new Set(["h3", "h3-sol", "h3-vdn"]);
const serviceDisplayNames = {
  "h3-latent-upscale": "H3 Latent Upscaler",
  sr: "Video Super Resolution",
  interpolate: "Frame Interpolation",
  depth: "Depth Anything",
};
const taskModel = (task) => {
  if (!generationServices.has(task.service)) return "";
  const model = task.model || "minimax-h3-ref2va";
  return modelDisplayNames[model] || model;
};
const taskRoute = (task) => {
  if (!generationServices.has(task.service)) return "";
  const route = task.route || task.service;
  return knownChannels.has(route) ? route : "";
};
const taskOperation = (task) =>
  ({
    h3: "视频生成",
    "h3-sol": "视频生成",
    "h3-vdn": "视频生成",
    "h3-latent-upscale": "H3 潜空间超分",
    sr: "视频超分",
    interpolate: "视频补帧",
    depth: "深度视频",
  })[task.service] || "视频处理";
const taskService = (task) => serviceDisplayNames[task.service] || "";
const taskIdentity = (task) => {
  if (generationServices.has(task.service)) {
    return `<div class="task-identity"><div><span>模型：</span><strong>${escapeHTML(taskModel(task))}</strong></div>${taskRoute(task) ? `<div class="task-channel"><span>渠道：</span><strong>${escapeHTML(taskRoute(task))}</strong></div>` : ""}</div>`;
  }
  return `<div class="task-identity"><div><span>操作：</span><strong>${escapeHTML(taskOperation(task))}</strong></div><div><span>服务：</span><strong>${escapeHTML(taskService(task) || task.service || "未指定")}</strong></div></div>`;
};
const taskPrompt = (task) =>
  generationServices.has(task.service)
    ? task.prompt || "未提供提示词"
    : taskOperation(task);
const taskTags = (task) => {
  if (generationServices.has(task.service)) {
    return `<span>REF2VA</span>${task.aspect_ratio ? `<span>${escapeHTML(task.aspect_ratio)}</span>` : ""}${task.duration_seconds != null ? `<span>${escapeHTML(task.duration_seconds)}s</span>` : ""}`;
  }
  const media = task.media || {};
  const size =
    media.width && media.height
      ? `<span>${escapeHTML(media.width)}×${escapeHTML(media.height)}</span>`
      : "";
  const fps = media.frame_rate
    ? `<span>${escapeHTML(media.frame_rate)} FPS</span>`
    : "";
  return `${size}${fps}`;
};
const taskTableIdentity = (task) =>
  generationServices.has(task.service)
    ? `${escapeHTML(taskModel(task))}<br>${escapeHTML(taskRoute(task) || "未指定渠道")}`
    : `${escapeHTML(taskOperation(task))}<br>${escapeHTML(taskService(task) || task.service || "未指定服务")}`;
let token = "",
  authorized = false,
  page = 1,
  status = "all",
  total = 0,
  busy = false,
  timer,
  searchTimer;
const thumbnails = new Map();
let modalTask = null,
  modalUrls = [],
  restoreFocus = null,
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
  modalUrls.forEach(URL.revokeObjectURL);
  modalUrls = [];
}
function openModal(id) {
  restoreFocus = document.activeElement;
  $(id).hidden = false;
  $(id).querySelector("input,button,textarea")?.focus();
}
function closeModal(id) {
  $(id).hidden = true;
  restoreFocus?.focus();
}
function invalidate() {
  thumbnails.forEach((v) => URL.revokeObjectURL(v.url));
  thumbnails.clear();
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
      400: "请检查参数、参考素材类型及项目归属。",
      404: "任务或素材不存在。",
      409: "幂等键冲突或所选服务尚未接入。",
      413: "请求内容过大。",
      502: "推理或素材服务暂时不可用。提交结果不确定时请先查询原任务，不要更换幂等键重发。",
      503: "服务尚未配置访问 Token。",
    };
    throw new Error(messages[response.status] || "操作失败，请稍后重试。");
  }
  return options.blob ? response.blob() : response.json();
}
function query() {
  return new URLSearchParams({
    page,
    page_size: 24,
    status,
    service: $("filterEngine").value,
    project_id: $("filterProject").value,
    q: $("searchInput").value.trim(),
  });
}
const thumbnailRequests = new Map();
const previewKey = (task) => `${task.video_task_id}:${task.artifact_id || ""}`;
async function previews(tasks) {
  const active = new Set(tasks.map((t) => t.video_task_id));
  for (const [id, item] of thumbnails)
    if (!active.has(id)) {
      URL.revokeObjectURL(item.url);
      thumbnails.delete(id);
    }
  const version = revision;
  for (const task of tasks) {
    if (version !== revision || !authorized) return;
    const key = previewKey(task);
    let cached = thumbnails.get(task.video_task_id);
    if (!cached || cached.artifact !== task.artifact_id) {
      if (!thumbnailRequests.has(key)) {
        const request = api(`/api/tasks/${task.video_task_id}/thumbnail`, {
          blob: true,
        })
          .then((blob) => {
            const card = [...$("galleryContainer").children].find(
              (el) => el.dataset.task === task.video_task_id,
            );
            if (
              version !== revision ||
              !authorized ||
              card?.dataset.previewKey !== key
            )
              return null;
            const old = thumbnails.get(task.video_task_id);
            const item = {
              url: URL.createObjectURL(blob),
              artifact: task.artifact_id,
            };
            thumbnails.set(task.video_task_id, item);
            if (old) URL.revokeObjectURL(old.url);
            return item;
          })
          .catch(() => null)
          .finally(() => thumbnailRequests.delete(key));
        thumbnailRequests.set(key, request);
      }
      cached = await thumbnailRequests.get(key);
    }
    if (!cached || version !== revision || !authorized) continue;
    const card = [...$("galleryContainer").children].find(
      (el) => el.dataset.task === task.video_task_id,
    );
    if (!card || card.dataset.previewKey !== key) continue;
    let img = card.querySelector(".card-thumb-wrap > img");
    if (!img) {
      img = document.createElement("img");
      card.querySelector(".card-thumb-wrap > span").replaceWith(img);
    }
    if (img.getAttribute("src") !== cached.url) img.src = cached.url;
    img.alt = task.artifact_id ? "生成视频首帧" : "参考素材预览";
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
      if (placeholder)
        placeholder.textContent = next.querySelector(
          ".card-thumb-wrap > span",
        ).textContent;
    } else card = next;
    card.dataset.previewKey = previewKey(task);
    if (gallery.children[index] !== card)
      gallery.insertBefore(card, gallery.children[index] || null);
  });
  existing.forEach((card) => card.remove());
}
function render(data) {
  total = data.total;
  $("metricQueued").textContent = data.counts.queued;
  $("metricRunning").textContent = data.counts.running;
  $("metricFailed").textContent = data.counts.failed;
  $("metricCompleted").textContent = data.counts.succeeded;
  $("metricLatency").textContent = seconds(data.average_elapsed_seconds);
  const project = $("filterProject").value;
  $("filterProject").innerHTML =
    '<option value="all">全部项目</option>' +
    data.projects
      .map((v) => `<option value="${escapeHTML(v)}">${escapeHTML(v)}</option>`)
      .join("");
  $("filterProject").value = data.projects.includes(project) ? project : "all";
  $("emptyState").hidden = data.tasks.length > 0;
  $("emptyState").textContent = data.total
    ? "此页没有任务，请返回上一页。"
    : "暂无符合条件的任务。可调整筛选，或新建运镜任务。";
  const badge = (t) =>
    `<span class="badge-status ${t.status === "succeeded" ? "completed" : escapeHTML(t.status)}">${labels[t.status] || escapeHTML(t.status)}</span>`;
  const galleryHTML = data.tasks
    .map(
      (t) =>
        `<article class="model-card" tabindex="0" role="button" data-task="${escapeHTML(t.video_task_id)}" aria-label="查看 ${escapeHTML(t.idempotency_key)}"><div class="card-thumb-wrap"><span>${t.status === "succeeded" ? "▶" : t.status === "running" ? "◌" : "◇"}</span>${taskIdentity(t)}<div class="card-badges">${badge(t)}</div><div class="card-times"><span title="开始时间">${date(t.created_at)}</span><span title="排队 / 运行">${elapsed(t)}</span></div></div><div class="card-body"><div><div class="card-title-row"><span title="${escapeHTML(t.video_task_id)} · ${escapeHTML(t.project_id)} / ${escapeHTML(t.idempotency_key)}">${escapeHTML(t.project_id)} / ${escapeHTML(t.idempotency_key)}</span></div><p class="card-prompt">${escapeHTML(taskPrompt(t))}</p></div><div class="card-footer"><div class="model-tags">${taskTags(t)}</div><span class="card-footer-action">查看详情 →</span></div></div></article>`,
    )
    .join("");
  reconcileGallery(galleryHTML, data.tasks);
  const tableHTML = data.tasks
    .map(
      (t) =>
        `<tr><td>${escapeHTML(t.project_id)}<br>${escapeHTML(t.idempotency_key)}</td><td>${escapeHTML(t.video_task_id)}</td><td>${taskTableIdentity(t)}</td><td>${escapeHTML(t.execution_instance_id || (t.status === "queued" ? "尚未分配" : "未知"))}</td><td>${escapeHTML(t.prompt.slice(0, 100))}</td><td>${escapeHTML(t.duration_seconds)}s · ${escapeHTML(t.aspect_ratio)}</td><td>${elapsed(t)}</td><td>${badge(t)}</td><td><button class="button" data-task="${escapeHTML(t.video_task_id)}">${t.status === "succeeded" ? "回放" : "详情"}</button></td></tr>`,
    )
    .join("");
  if ($("taskRows").innerHTML !== tableHTML)
    $("taskRows").innerHTML = tableHTML;
  previews(data.tasks);
  $("pageLabel").textContent =
    `第 ${page} / ${Math.max(1, Math.ceil(total / 24))} 页 · ${total} 个任务`;
  $("prevPage").disabled = page <= 1;
  $("nextPage").disabled = page * 24 >= total;
  notice(
    data.sync_errors
      ? `${data.sync_errors} 个任务暂未同步，保留最后已知状态。最后同步：${date(data.synced_at)}`
      : `任务状态同步：${date(data.synced_at)} · 数据按当前筛选统计。`,
    !!data.sync_errors,
  );
}
async function refresh() {
  clearTimeout(timer);
  if (!authorized || document.hidden) return;
  if (busy) {
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
      const data = await api(`/api/dashboard?${query()}`);
      if (version !== revision) return;
      render(data);
    }
    $("pollState").textContent = "2s 自动刷新";
  } catch (error) {
    if (version === revision) {
      notice(error.message, true);
      $("pollState").textContent = "连接异常 · 任务为最后快照";
    }
  } finally {
    busy = false;
    if (authorized) timer = setTimeout(refresh, 2000);
  }
}
function filter() {
  page = 1;
  revision++;
  refresh();
}
async function mediaURL(id, taskId) {
  const blob = await api(`/artifacts/${encodeURIComponent(id)}/content`, {
    blob: true,
  });
  if (modalTask !== taskId) throw new Error("详情已关闭");
  const url = URL.createObjectURL(blob);
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
document.addEventListener("click", (event) => {
  const target = event.target.closest("button,a,article");
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
  if (event.target.classList.contains("modal-backdrop"))
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
["filterEngine", "filterProject"].forEach((id) =>
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
  if (page * 24 < total) {
    page++;
    revision++;
    refresh();
  }
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
    body.model = "minimax-h3-ref2va";
    const references = { images: [], videos: [], audios: [] };
    const kinds = { image: "images", video: "videos", audio: "audios" };
    for (const line of body.references.trim().split("\n")) {
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
    if (!references.images.length && !references.videos.length)
      throw new Error("至少需要一张图片或一段视频参考。");
    body.references = references;
    const task = await api("/api/tasks", {
      method: "POST",
      body: JSON.stringify(body),
    });
    $("dispatchMessage").textContent = `已提交：${task.video_task_id}`;
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

$("dispatchForm").elements.route.addEventListener("change", () => {
  const duration = $("dispatchForm").elements.duration_seconds;
  const route = $("dispatchForm").elements.route.value;
  const sol = route.startsWith("h3-sol") || route === "h3-vdn";
  duration.min = sol ? "5" : "4";
  duration.step = sol ? "5" : "1";
  if (sol && ![5, 10, 15].includes(Number(duration.value)))
    duration.value = "5";
});

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
          engine: $("filterEngine").value,
          q: $("searchInput").value,
          status,
          page,
          table: $("btnViewTable").classList.contains("active"),
        };
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
      if (["all", "h3", "h3-sol", "h3-vdn"].includes(state.engine))
        $("filterEngine").value = state.engine;
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
