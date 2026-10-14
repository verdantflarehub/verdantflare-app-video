"use strict";
let detailRendered = false, resultArtifact = null, resultLoading = false;
function clearTaskDetail() {
  setTaskName("任务不可用");
  detailRendered = false;
  resultArtifact = null;
  resultLoading = false;
  $("detailContent").hidden = true;
  $("detailSubtitle").textContent = "";
  $("detailStatus").textContent = "";
  $("taskPrompt").replaceChildren();
  $("taskParameters").replaceChildren();
  $("referenceGroups").replaceChildren();
  $("videoArea").replaceChildren();
  $("resultActions").replaceChildren();
  $("resultMessage").textContent = "";
  $("detailMessage").textContent = "请连接 MCP 以查看任务。";
  $("imagePreview").close();
  $("imagePreviewImage").removeAttribute("src");
}
function taskDisplayName(task) {
  const key = typeof task.idempotency_key === "string" ? task.idempotency_key.trim() : "";
  const project = typeof task.project_id === "string" ? task.project_id.trim() : "";
  return key ? (project ? `${project} / ${key}` : key) : "未命名任务";
}
function setTaskName(name) {
  $("detailTitle").textContent = name;
  $("detailTitle").title = name;
  const current = $("videoBreadcrumb").querySelector('[aria-current="page"]');
  if (current && detailTaskId) { current.textContent = name; current.title = name; }
}
function businessView() {
  const selected = detailTaskId ? "tasks" : ["tasks", "models", "mcp"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "tasks";
  document.querySelectorAll("[data-business-view]").forEach(el => { el.hidden = !!detailTaskId || el.dataset.businessView !== selected; });
  document.querySelectorAll("[data-nav]").forEach(el => {
    el.href = `${base}/dashboard#${el.dataset.nav}`;
    el.classList.toggle("active", el.dataset.nav === selected);
    if (el.dataset.nav === selected) el.setAttribute("aria-current", "page"); else el.removeAttribute("aria-current");
  });
  $("taskDetail").hidden = !detailTaskId;
  if (detailTaskId) document.querySelector(".workspace-nav")?.remove();
  const crumb = $("videoBreadcrumb");
  const app = document.createElement("a"); app.href = `${base}/dashboard#tasks`; app.textContent = "Video MCP";
  const sep = () => { const span = document.createElement("span"); span.textContent = " / "; span.setAttribute("aria-hidden", "true"); return span; };
  const current = document.createElement("span"); current.setAttribute("aria-current", "page");
  current.textContent = detailTaskId ? $("detailTitle").textContent : { tasks: "任务", models: "模型", mcp: "MCP" }[selected];
  crumb.replaceChildren(app, sep());
  if (detailTaskId) { const tasks = document.createElement("a"); tasks.href = app.href; tasks.textContent = "任务"; crumb.append(tasks, sep()); }
  crumb.append(current);
}
function promptReferences(prompt) {
  const fragment = document.createDocumentFragment();
  const pattern = /<(Picture|Image|Video|Audio)\s+(\d+)>/gi;
  let last = 0;
  for (const match of prompt.matchAll(pattern)) {
    fragment.append(document.createTextNode(prompt.slice(last, match.index)));
    const kind = { picture: "images", image: "images", video: "videos", audio: "audios" }[match[1].toLowerCase()];
    const id = `reference-${kind}-${Number(match[2])}`;
    if (document.getElementById(id)) {
      const button = document.createElement("button");
      button.className = "reference-link";
      button.textContent = match[0];
      button.onclick = () => {
        document.querySelectorAll(".reference-focus").forEach(el => el.classList.remove("reference-focus"));
        const target = document.getElementById(id);
        target.classList.add("reference-focus");
        target.scrollIntoView({ behavior: "smooth", block: "nearest" });
        target.focus({ preventScroll: true });
      };
      fragment.append(button);
    } else fragment.append(document.createTextNode(match[0]));
    last = match.index + match[0].length;
  }
  fragment.append(document.createTextNode(prompt.slice(last)));
  $("taskPrompt").replaceChildren(fragment);
}
async function loadReference(reference, card, taskId, version) {
  const preview = card.querySelector(".reference-preview");
  if (reference.unavailable) { preview.textContent = "素材不可用"; return; }
  try {
    const url = await mediaURL(reference.artifact_id, taskId);
    if (version !== revision || !authorized || !card.isConnected) return;
    const kind = reference.kind;
    const media = document.createElement(kind === "images" ? "img" : kind === "videos" ? "video" : "audio");
    media.src = url;
    if (kind === "images") {
      media.alt = reference.purpose || reference.filename || "参考图片";
      const button = document.createElement("button");
      button.className = "image-open";
      button.setAttribute("aria-label", `放大 ${card.dataset.label}`);
      button.append(media);
      button.onclick = () => {
        $("imagePreviewTitle").textContent = `${card.dataset.label} · ${reference.filename || "图片预览"}`;
        $("imagePreviewImage").src = url;
        $("imagePreview").showModal();
      };
      preview.replaceChildren(button);
    } else {
      media.controls = true;
      media.preload = "metadata";
      media.setAttribute("aria-label", card.dataset.label);
      if (kind === "videos") media.playsInline = true;
      preview.replaceChildren(media);
    }
    media.addEventListener("error", () => { preview.textContent = "预览不可用：文件格式或编码不受浏览器支持"; });
  } catch (error) {
    if (version === revision && card.isConnected) preview.textContent = error.message;
  }
}
function renderReferences(task, version) {
  const definitions = [["images", "图片", "Picture", 9], ["videos", "视频", "Video", 3], ["audios", "音频", "Audio", 3]];
  const pending = [];
  $("referenceGroups").replaceChildren();
  for (const [kind, label, prefix, limit] of definitions) {
    const references = (task.references || []).filter(ref => ref.kind === kind);
    const group = document.createElement("section");
    group.className = "reference-group";
    group.innerHTML = `<h3>${label} <small>${references.length} / ${limit}</small></h3><div class="reference-cards"></div>`;
    const cards = group.querySelector(".reference-cards");
    if (!references.length) cards.textContent = `未使用${label}参考`;
    references.forEach((reference, index) => {
      const card = document.createElement("article");
      card.className = `reference-card ${kind}`;
      card.id = `reference-${kind}-${index + 1}`;
      card.tabIndex = -1;
      card.dataset.label = `${prefix} ${index + 1}`;
      card.innerHTML = `<div class="reference-preview">加载预览…</div><div class="reference-caption"><strong>${prefix} ${index + 1}</strong><span title="${escapeHTML(reference.filename || "")}">${escapeHTML(reference.filename || "素材不可用")}</span><p>${escapeHTML(reference.purpose)}</p></div>`;
      cards.append(card);
      pending.push([reference, card]);
    });
    $("referenceGroups").append(group);
  }
  // Keep every slot visible while limiting parallel downloads of large media.
  let next = 0;
  async function worker() { while (next < pending.length && version === revision && authorized) { const [reference, card] = pending[next++]; await loadReference(reference, card, task.video_task_id, version); } }
  Promise.allSettled([worker(), worker(), worker()]);
}
async function loadTaskResult(task, version, retrieve = false) {
  if (resultLoading) return;
  resultLoading = true;
  $("resultMessage").textContent = "正在加载视频…";
  try {
    if (retrieve) task = await api(`/api/tasks/${encodeURIComponent(task.video_task_id)}/result`, { method: "POST" });
    if (version !== revision || !authorized) return;
    let url;
    if (task.content_ref) {
      const blob = await api(`/api/tasks/${encodeURIComponent(task.video_task_id)}/content`, { blob: true });
      if (modalTask !== task.video_task_id) throw new Error("详情已关闭");
      url = URL.createObjectURL(blob); modalUrls.push(url);
    } else url = await mediaURL(task.artifact.artifact_id, task.video_task_id);
    if (version !== revision || !authorized) return;
    const video = document.createElement("video");
    video.controls = true; video.playsInline = true; video.preload = "metadata"; video.src = url;
    video.setAttribute("aria-label", "生成结果");
    video.addEventListener("error", () => { $("resultMessage").textContent = "浏览器无法播放该编码，可下载原视频。"; });
    $("videoArea").replaceChildren(video);
    const link = document.createElement("a"); link.className = "button primary"; link.href = url;
    link.download = task.artifact?.filename || `${task.video_task_id}.mp4`; link.textContent = "下载视频 ↓";
    $("resultActions").replaceChildren(link);
    $("resultMessage").textContent = `${task.media?.width || "—"} × ${task.media?.height || "—"} · ${task.media?.frame_rate || "—"} FPS`;
    resultArtifact = task.content_ref?.version_id || task.artifact.artifact_id;
  } catch (error) {
    if (version === revision && authorized) {
      $("resultMessage").textContent = error.message;
      const retry = document.createElement("button"); retry.className = "button"; retry.textContent = "重试加载";
      retry.onclick = () => loadTaskResult(task, version, retrieve); $("resultActions").replaceChildren(retry);
      // Avoid repeatedly downloading a failing asset on every poll.
      resultArtifact = task.artifact?.artifact_id || "unavailable";
    }
  } finally { resultLoading = false; }
}
async function refreshTaskDetail(id, version) {
  let task;
  try { task = await api(`/api/tasks/${encodeURIComponent(id)}`); }
  catch (error) { if (version === revision) { clearTaskDetail(); setTaskName(error.status === 404 ? "未找到任务" : "任务不可用"); $("detailMessage").textContent = error.message; } return; }
  if (version !== revision || !authorized) return;
  setTaskName(taskDisplayName(task));
  modalTask = id;
  $("detailMessage").textContent = "";
  $("detailContent").hidden = false;
  $("detailSubtitle").textContent = `${task.project_id} / ${task.idempotency_key}`;
  $("detailStatus").className = `detail-status ${task.status}`;
  $("detailStatus").textContent = labels[task.status] || task.status;
  $("outputCount").textContent = task.artifact || task.content_ref ? "1 个视频" : "0 / 1 个视频";
  const rows = [["任务 ID", task.video_task_id], ["模型", taskModel(task)], ["渠道", taskRoute(task)], ["生成类型", task.presentation?.generation_code ? `${task.presentation.generation_code} · ${task.presentation.generation_label}` : "未记录"], ["执行后端", task.route || task.service || "未记录"], ["执行实例", task.execution_instance_id || "未上报"], ["开始时间", date(task.created_at)], ["排队时间", seconds(task.timing?.queue_seconds)], ["运行时间", seconds(task.timing?.processing_seconds ?? task.timing?.processing_elapsed_seconds)], ["总耗时", seconds(task.timing?.total_seconds)], ["时长 / 画幅", `${task.duration_seconds}s · ${task.aspect_ratio}`], ["Seed", task.seed], ["Runtime 版本", task.runtime_version], ["输入摘要", task.input_digest]];
  if (task.runtime_stage) rows.push(["任务阶段", task.runtime_stage === "submission_unknown" ? "提交待确认，请勿重复生成" : task.runtime_stage]);
  if (task.error) rows.push(["异常信息", typeof task.error === "string" ? task.error : JSON.stringify(task.error)]);
  $("taskParameters").innerHTML = rows.map(([label,value]) => `<dt>${escapeHTML(label)}</dt><dd>${escapeHTML(value ?? "—")}</dd>`).join("");
  if (!detailRendered) {
    renderReferences(task, version);
    promptReferences(task.prompt || "");
    detailRendered = true;
  }
  if (task.status === "succeeded") {
    if ((task.artifact || task.content_ref) && !resultArtifact) await loadTaskResult(task, version);
    else if (!task.artifact && !task.content_ref && !resultLoading && !$("resultActions").children.length) {
      $("videoArea").textContent = "任务已完成，等待获取视频结果。";
      const button = document.createElement("button"); button.className = "button primary"; button.textContent = "获取视频结果";
      button.onclick = async () => { button.disabled = true; await loadTaskResult(task, version, true); button.disabled = false; };
      $("resultActions").replaceChildren(button);
    }
  } else {
    $("videoArea").textContent = task.status === "failed" ? `生成失败：${task.error?.message || task.error?.code || "请查看任务信息"}` : task.status === "cancelled" ? "任务已取消" : `${labels[task.status] || task.status}，结果就绪后将在这里显示。`;
  }
}
window.addEventListener("hashchange", businessView);
window.addEventListener("pagehide", clearMedia);
window.addEventListener("DOMContentLoaded", () => {
  businessView();
  if (detailTaskId && !authorized) $("detailMessage").textContent = "请连接 MCP 以查看任务。";
  $("closeImagePreview").onclick = () => $("imagePreview").close();
  $("imagePreview").addEventListener("close", () => $("imagePreviewImage").removeAttribute("src"));
});
