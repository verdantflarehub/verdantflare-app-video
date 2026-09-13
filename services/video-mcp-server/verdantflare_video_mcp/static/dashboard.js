"use strict";
const $ = (id) => document.getElementById(id);
const base = location.pathname.replace(/\/dashboard\/?$/, "");
const tokenStorageKey = `verdantflare.video.dashboard.token:${base}`;
function storeToken(value) {
  try {
    if (value) localStorage.setItem(tokenStorageKey, value);
    else localStorage.removeItem(tokenStorageKey);
  } catch {
    // Storage-disabled browsers still support the current in-memory session.
  }
}
const names = { h3: "h3", "h3-sol": "h3-sol", "h3-sol-4090": "h3-sol-4090", mcp: "MCP" };
const labels = {
  queued: "排队中",
  running: "渲染中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};
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
const seconds = (v) =>
  v == null
    ? "—"
    : v < 60
      ? `${v.toFixed(1)}s`
      : `${Math.floor(v / 60)}m ${Math.round(v % 60)}s`;
function elapsed(t) {
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
  if (id === "inspectorModal") {
    modalTask = null;
    clearMedia();
    $("inspectorBody").replaceChildren();
  }
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
  $("inspectorBody").replaceChildren();
  $("inspectorModal").hidden = true;
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
  const response = await fetch(`${base}${path}`, {
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
      401: "Token 无效或已过期，请重新设置。",
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
    let cached = thumbnails.get(task.video_task_id);
    if (cached && cached.artifact !== task.artifact_id) {
      URL.revokeObjectURL(cached.url);
      thumbnails.delete(task.video_task_id);
      cached = null;
    }
    if (!cached) {
      try {
        const blob = await api(`/api/tasks/${task.video_task_id}/thumbnail`, {
          blob: true,
        });
        if (version !== revision || !authorized) return;
        cached = { url: URL.createObjectURL(blob), artifact: task.artifact_id };
        thumbnails.set(task.video_task_id, cached);
      } catch {
        continue;
      }
    }
    const card = [...$("galleryContainer").children].find(
      (el) => el.dataset.task === task.video_task_id,
    );
    if (card && !card.querySelector("img")) {
      const img = document.createElement("img");
      img.src = cached.url;
      img.alt = task.artifact_id ? "生成视频首帧" : "参考素材预览";
      card.querySelector(".card-thumb-wrap > span").replaceWith(img);
    }
  }
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
  $("galleryContainer").innerHTML = data.tasks
    .map(
      (t) =>
        `<article class="model-card" tabindex="0" role="button" data-task="${escapeHTML(t.video_task_id)}" aria-label="查看 ${escapeHTML(t.idempotency_key)}"><div class="card-thumb-wrap"><span>${t.status === "succeeded" ? "▶" : t.status === "running" ? "◌" : "◇"}</span><div class="card-badges"><span class="badge-tag engine-h3">${names[t.service] || escapeHTML(t.service)}</span>${badge(t)}</div></div><div class="card-body"><div><div class="card-title-row"><span>${escapeHTML(t.project_id)} / ${escapeHTML(t.idempotency_key)}</span><span>${elapsed(t)}</span></div><p class="card-prompt">${escapeHTML(t.prompt)}</p></div><div class="model-tags"><span>REF2VA</span><span>${escapeHTML(t.aspect_ratio)}</span><span>${escapeHTML(t.duration_seconds)}s</span><span>${t.media ? escapeHTML(t.media.frame_rate || "24") + " FPS" : "待验收"}</span></div><div class="card-footer"><span>${date(t.created_at)}</span><span class="card-footer-action">${t.status === "succeeded" ? "运镜回放" : "查看详情"} →</span></div></div></article>`,
    )
    .join("");
  $("taskRows").innerHTML = data.tasks
    .map(
      (t) =>
        `<tr><td>${escapeHTML(t.project_id)}<br>${escapeHTML(t.idempotency_key)}</td><td>${escapeHTML(t.video_task_id)}</td><td>${names[t.service] || escapeHTML(t.service)}</td><td>${escapeHTML(t.execution_instance_id || (t.status === "queued" ? "尚未分配" : "未知"))}</td><td>${escapeHTML(t.prompt.slice(0, 100))}</td><td>${escapeHTML(t.duration_seconds)}s · ${escapeHTML(t.aspect_ratio)}</td><td>${elapsed(t)}</td><td>${badge(t)}</td><td><button class="button" data-task="${escapeHTML(t.video_task_id)}">${t.status === "succeeded" ? "回放" : "详情"}</button></td></tr>`,
    )
    .join("");
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
    const data = await api(`/api/dashboard?${query()}`);
    if (version !== revision) return;
    render(data);
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
async function inspect(id) {
  clearMedia();
  modalTask = id;
  $("inspectorBody").textContent = "正在读取任务…";
  openModal("inspectorModal");
  try {
    const task = await api(`/api/tasks/${encodeURIComponent(id)}`);
    if (modalTask !== id) return;
    $("modalShotTitle").textContent =
      `${task.project_id} / ${task.idempotency_key}`;
    $("inspectorBody").innerHTML =
      `<p>${names[task.service] || escapeHTML(task.service)} · ${labels[task.status] || escapeHTML(task.status)} · ${date(task.created_at)}</p><p>执行模型：<button class="button" data-resource="model" data-model="${escapeHTML(task.service)}">${names[task.service] || escapeHTML(task.service)} →</button> · 执行实例：${task.execution_instance_id ? `<button class="button" data-resource="instance" data-model="${escapeHTML(task.service)}" data-instance="${escapeHTML(task.execution_instance_id)}">${escapeHTML(task.execution_instance_id)} →</button>` : task.status === "queued" ? "尚未分配" : "未知（未上报）"}</p><div id="videoArea"></div><div class="actions" id="resultActions"></div><p id="resultMessage" role="status"></p><h3>动态运镜 Prompt</h3><p>${escapeHTML(task.prompt)}</p><div class="reference-grid" id="referenceGrid"></div><pre>${escapeHTML(JSON.stringify({ video_task_id: task.video_task_id, seed: task.seed, duration_seconds: task.duration_seconds, aspect_ratio: task.aspect_ratio, runtime_version: task.runtime_version, input_digest: task.input_digest, media: task.media, error: task.error }, null, 2))}</pre><p>技术完成后仍需人工检查构图、连续性和动态运镜。参考素材不代表已锁定首尾帧。</p>`;
    const loadResult = async () => {
      $("resultMessage").textContent = "正在获取并校验视频…";
      try {
        const complete = task.artifact
          ? task
          : await api(`/api/tasks/${encodeURIComponent(id)}/result`, {
              method: "POST",
            });
        if (modalTask !== id) return;
        const url = await mediaURL(complete.artifact.artifact_id, id);
        if (modalTask !== id) return;
        const video = document.createElement("video");
        video.controls = true;
        video.preload = "metadata";
        video.src = url;
        $("videoArea").replaceChildren(video);
        const link = document.createElement("a");
        link.className = "button primary";
        link.href = url;
        link.download = complete.artifact.filename;
        link.textContent = "下载视频";
        $("resultActions").replaceChildren(link);
        $("resultMessage").textContent =
          `SHA-256: ${complete.artifact.sha256} · ${complete.artifact.size} bytes · 创作质量待人工审核`;
      } catch (error) {
        if (modalTask === id) $("resultMessage").textContent = error.message;
      }
    };
    if (task.status === "succeeded") {
      const button = document.createElement("button");
      button.className = "button primary";
      button.textContent = "加载视频回放";
      button.addEventListener("click", async () => {
        button.disabled = true;
        await loadResult();
        button.disabled = false;
      });
      $("resultActions").append(button);
    }
    for (const reference of task.references) {
      if (modalTask !== id) break;
      const box = document.createElement("div");
      const caption = document.createElement("p");
      caption.textContent = `${reference.kind} · ${reference.purpose}`;
      box.append(caption);
      $("referenceGrid").append(box);
      if (reference.unavailable) {
        box.append("参考素材不可用");
        continue;
      }
      try {
        const url = await mediaURL(reference.artifact_id, id);
        if (modalTask !== id) break;
        const element = document.createElement(
          reference.kind === "images"
            ? "img"
            : reference.kind === "videos"
              ? "video"
              : "audio",
        );
        element.src = url;
        element.alt = reference.purpose;
        if (element.tagName !== "IMG") element.controls = true;
        box.prepend(element);
      } catch {
        box.append("参考素材加载失败");
      }
    }
  } catch (error) {
    if (modalTask === id) $("inspectorBody").textContent = error.message;
  }
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
  if (target?.dataset.task) { $("resourceDrawer").close(); inspect(target.dataset.task); }
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
      .forEach((b) => b.classList.toggle("active", b.dataset.status === status));
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
    if (body.service === "h3-sol-4090") { body.route = "h3-sol-4090"; body.service = "h3-sol"; }
    else if (body.service === "h3-sol") body.route = "h3-sol";
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

$("dispatchForm").elements.service.addEventListener("change",()=>{
  const duration=$("dispatchForm").elements.duration_seconds;
  const sol=$("dispatchForm").elements.service.value.startsWith('h3-sol');
  duration.min=sol?'5':'4';duration.step=sol?'5':'1';
  if(sol&&![5,10,15].includes(Number(duration.value)))duration.value='5';
});

window.addEventListener("DOMContentLoaded", () => {
  try { token = localStorage.getItem(tokenStorageKey) || ""; } catch {}
  if (token) {
    authorized = true;
    $("tokenButton").textContent = "Token：已设置";
    refresh();
  }
});
