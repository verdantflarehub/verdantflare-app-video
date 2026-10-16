"use strict";
let detailRendered = false, resultArtifact = null, resultLoading = false;
let currentTitleTask = null, titleEditRevision = 0;
let zoomScale = 1, zoomTx = 0, zoomTy = 0, zoomDragging = false, zoomStartX = 0, zoomStartY = 0;
function syncImageZoom() {
  const img = $("imagePreviewImage");
  if (img) img.style.transform = `translate3d(${zoomTx}px, ${zoomTy}px, 0) scale(${zoomScale})`;
  if ($("zoomReadout")) $("zoomReadout").textContent = `${Math.round(zoomScale * 100)}%`;
}
function resetImageZoom() {
  zoomScale = 1; zoomTx = 0; zoomTy = 0; zoomDragging = false;
  $("imagePreviewStage")?.classList.remove("dragging");
  syncImageZoom();
}
function openImagePreview(url, label) {
  if (window.studioEmbedded && window.studioPreviewImage) {
    window.studioPreviewImage(url, label).catch(error => { $("detailMessage").textContent = error.message; });
    return;
  }
  resetImageZoom();
  $("imagePreviewTitle").textContent = `${label} · 游标仿射缩放 (0.5x – 5.0x)`;
  $("imagePreviewImage").src = url;
  if (!$("imagePreview").open) $("imagePreview").showModal();
}
function clearTaskDetail() {
  window.studioClosePreview?.();
  currentTitleTask = null;
  if ($("titleForm")) $("titleForm").hidden = true;
  if ($("editTitle")) $("editTitle").hidden = true;
  setTaskName("任务不可用");
  detailRendered = false;
  resultArtifact = null;
  resultLoading = false;
  $("detailContent").hidden = true;
  $("detailSubtitle").textContent = "";
  $("detailStatus").textContent = "";
  if ($("detailIdentityBar")) $("detailIdentityBar").replaceChildren();
  if ($("detailOutputMeta")) $("detailOutputMeta").replaceChildren();
  $("taskPrompt").replaceChildren();
  $("taskParameters").replaceChildren();
  $("referenceGroups").replaceChildren();
  $("videoArea").replaceChildren();
  $("resultActions").replaceChildren();
  $("resultMessage").textContent = "";
  $("detailMessage").textContent = "请连接 MCP 以查看任务。";
  resetImageZoom();
  $("imagePreview").close();
  $("imagePreviewImage").removeAttribute("src");
}
function taskDisplayName(task) {
  return typeof task.title === "string" && task.title.trim() ? task.title.trim() : "视频任务";
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
  if (!prompt || !prompt.trim()) {
    const empty = document.createElement("div");
    empty.className = "empty-factor-box ref-empty-box";
    empty.textContent = "未传入提示词（0 / 1）";
    $("taskPrompt").replaceChildren(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  const pattern = /<(Picture|Image|Video|Audio)\s+(\d+)>/gi;
  let last = 0;
  for (const match of prompt.matchAll(pattern)) {
    fragment.append(document.createTextNode(prompt.slice(last, match.index)));
    const kind = { picture: "images", image: "images", video: "videos", audio: "audios" }[match[1].toLowerCase()];
    const id = `reference-${kind}-${Number(match[2])}`;
    if (document.getElementById(id)) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "reference-link prompt-ref-chip";
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
      button.type = "button";
      button.className = "image-open";
      button.setAttribute("aria-label", `放大 ${card.dataset.label}`);
      button.append(media);
      button.onclick = () => openImagePreview(url, `${card.dataset.label} · ${reference.filename || "图片预览"}`);
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
  const definitions = [
    ["images", "2. 参考图", "参考图", "Picture", 9, "张", "全局最大 9 张", "ref-image-grid"],
    ["videos", "3. 参考视频", "参考视频", "Video", 3, "个", "全局最大 3 个", "ref-video-grid"],
    ["audios", "4. 参考音频", "参考音频", "Audio", 1, "个", "全局最大 1 个", "ref-audio-list"],
  ];
  const pending = [];
  $("referenceGroups").replaceChildren();
  for (const [kind, heading, emptyNoun, prefix, limit, unit, quotaLabel, gridClass] of definitions) {
    const references = (task.references || []).filter(ref => ref.kind === kind);
    const group = document.createElement("section");
    group.className = "reference-group input-factor-section";
    group.dataset.factorKind = kind;
    group.innerHTML = `<div class="factor-header pane-title"><h3>${heading} <small>（${quotaLabel}，不同模型上限不同）</small></h3><span class="factor-quota ${references.length === 0 ? "empty" : ""}">${references.length} / ${limit} ${unit}</span></div>`;
    if (!references.length) {
      const empty = document.createElement("div");
      empty.className = "empty-factor-box ref-empty-box";
      empty.innerHTML = `<span>未传入${emptyNoun}（0 / ${limit} ${unit}）</span><small>${quotaLabel} · 本次任务未传入该类参考素材</small>`;
      group.append(empty);
    } else {
      const cards = document.createElement("div");
      cards.className = `reference-cards ${gridClass}`;
      references.forEach((reference, index) => {
        const card = document.createElement("article");
        card.className = `reference-card ${kind === "images" ? "ref-image-card " : ""}${kind}`;
        card.id = `reference-${kind}-${index + 1}`;
        card.tabIndex = -1;
        card.dataset.label = `${prefix} ${index + 1}`;
        card.innerHTML = `<span class="ref-slot-tag">${prefix} ${index + 1}</span><div class="reference-preview ${kind === "images" ? "ref-image-thumb" : kind === "videos" ? "ref-video-stage" : ""}">加载预览…</div><div class="reference-caption"><strong>${prefix} ${index + 1}</strong><span title="${escapeHTML(reference.filename || "")}">${escapeHTML(reference.filename || "素材不可用")}</span><p>${escapeHTML(reference.purpose || "")}</p></div>`;
        cards.append(card);
        pending.push([reference, card]);
      });
      group.append(cards);
    }
    $("referenceGroups").append(group);
  }
  // Keep every slot visible while limiting parallel downloads of large media.
  let next = 0;
  async function worker() { while (next < pending.length && version === revision && authorized) { const [reference, card] = pending[next++]; await loadReference(reference, card, task.video_task_id, version); } }
  Promise.allSettled([worker(), worker(), worker()]);
}
function showResultLoader(task, version, retrieve, retry = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "result-load-target";
  button.textContent = retry ? "加载失败，点击重试" : "点击加载视频";
  button.setAttribute("aria-label", retry ? "重新加载视频" : "加载视频结果");
  button.onclick = () => loadTaskResult(task, version, retrieve);
  $("videoArea").replaceChildren(button);
  $("resultActions").replaceChildren();
}
async function loadTaskResult(task, version, retrieve = false) {
  if (resultLoading) return;
  resultLoading = true;
  $("resultMessage").textContent = "正在加载视频…";
  const trigger = $("videoArea").querySelector(".result-load-target");
  if (trigger) { trigger.disabled = true; trigger.textContent = "正在加载视频…"; }
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
    $("outputCount").textContent = "1 / 1 个视频产物";
  } catch (error) {
    if (version === revision && authorized) {
      $("resultMessage").textContent = error.message;
      showResultLoader(task, version, retrieve, true);
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
  currentTitleTask = task;
  $("editTitle").hidden = false;
  setTaskName(taskDisplayName(task));
  modalTask = id;
  $("detailMessage").textContent = "";
  $("detailContent").hidden = false;
  const res = taskResolution(task);
  const specSummary = [res, task.aspect_ratio, task.duration_seconds != null ? `${task.duration_seconds}S` : null, taskFps(task)].filter(Boolean).join(" / ");
  $("detailSubtitle").textContent = `${date(task.created_at)} · ${taskOperation(task)}${specSummary ? ` · 规格：${specSummary}` : ""}`;
  $("detailStatus").className = `detail-status ${task.status}`;
  $("detailStatus").textContent = taskStatusLabel(task);
  $("outputCount").textContent = task.artifact || task.content_ref ? "1 / 1 个视频产物" : "0 / 1 个视频";
  if ($("detailIdentityBar")) {
    if (generationServices.has(task.service)) {
      $("detailIdentityBar").innerHTML =
        `<span class="identity-channel detail-identity-pill"><strong class="identity-model">${escapeHTML(taskModel(task))}</strong> <span class="identity-separator" aria-hidden="true">×</span> ${channelLogo(task)}<span>${escapeHTML(taskRoute(task))}</span></span>`;
    } else {
      $("detailIdentityBar").innerHTML =
        `<span class="identity-channel detail-identity-pill"><strong>${escapeHTML(taskOperation(task))}</strong> <span class="identity-separator" aria-hidden="true">·</span> <span>${escapeHTML(taskService(task) || task.service || "视频处理")}</span></span>`;
    }
  }
  if ($("detailOutputMeta")) {
    const pills = [];
    if (generationServices.has(task.service)) {
      pills.push(`<span>${escapeHTML(taskModel(task))}</span>`);
      if (task.presentation?.generation_code) {
        pills.push(`<span>${escapeHTML(task.presentation.generation_code)}</span>`);
      }
      if (res) pills.push(`<span>${escapeHTML(res)}</span>`);
      if (task.aspect_ratio) pills.push(`<span>${escapeHTML(task.aspect_ratio)}</span>`);
      if (task.duration_seconds != null) pills.push(`<span>${escapeHTML(task.duration_seconds)}S</span>`);
      const fps = taskFps(task);
      if (fps) pills.push(`<span>${escapeHTML(fps)}</span>`);
    } else {
      pills.push(`<span>${escapeHTML(taskOperation(task))}</span>`);
      if (task.media?.width && task.media?.height) pills.push(`<span>${escapeHTML(task.media.width)}×${escapeHTML(task.media.height)}</span>`);
      if (task.media?.frame_rate) pills.push(`<span>${escapeHTML(task.media.frame_rate)}FPS</span>`);
    }
    $("detailOutputMeta").innerHTML = pills.join("");
  }
  const rows = [["任务 ID", task.video_task_id], ["项目 ID", task.project_id], ["幂等键", task.idempotency_key], ["模型", taskModel(task)], ["渠道", taskRoute(task)], ["生成类型", task.presentation?.generation_code ? `${task.presentation.generation_code} · ${task.presentation.generation_label}` : "未记录"], ["执行后端", task.route || task.service || "未记录"], ["执行实例", task.execution_instance_id || "未上报"], ["开始时间", date(task.created_at)], ["排队时间", seconds(task.timing?.queue_seconds)], ["运行时间", seconds(task.timing?.processing_seconds ?? task.timing?.processing_elapsed_seconds)], ["总耗时", seconds(task.timing?.total_seconds)], ["时长 / 画幅", `${task.duration_seconds}s · ${task.aspect_ratio}`], ["Seed", task.seed], ["Runtime 版本", task.runtime_version], ["输入摘要", task.input_digest]];
  if (task.service === "higgsfield") {
    rows.splice(rows.findIndex(([label]) => label === "总耗时"), 1);
    const runtime = rows.find(row => row[0] === "运行时间");
    runtime[1] = elapsed(task);
    rows.push(["渠道确认完成", date(task.timing?.generation_completed_at)], ["计时说明", generationTimeHint]);
  }
  if (task.runtime_stage) rows.push(["任务阶段", task.runtime_stage === "submission_unknown" ? "提交待确认，请勿重复生成" : task.service === "higgsfield" && task.runtime_stage === "collecting" ? "视频已生成" : task.runtime_stage]);
  if (task.error) rows.push(["异常信息", typeof task.error === "string" ? task.error : JSON.stringify(task.error)]);
  $("taskParameters").innerHTML = rows.map(([label,value]) => `<dt>${escapeHTML(label)}</dt><dd>${escapeHTML(value ?? "—")}</dd>`).join("");
  const copy = document.createElement("button"); copy.className = "button"; copy.textContent = "复制任务 ID";
  copy.onclick = async () => {
    try {
      if (navigator.clipboard && !window.studioEmbedded) await navigator.clipboard.writeText(task.video_task_id);
      else { const input = document.createElement('textarea'); input.value = task.video_task_id; document.body.append(input); input.select(); const copied = document.execCommand('copy'); input.remove(); if (!copied) throw new Error('copy'); }
      copy.textContent = "已复制";
    } catch { copy.textContent = "复制不可用，请选中任务 ID"; }
  };
  $("taskParameters").querySelector('dd')?.append(copy);
  if (!detailRendered) {
    renderReferences(task, version);
    promptReferences(task.prompt || "");
    detailRendered = true;
  }
  if (task.status === "succeeded") {
    if ((task.artifact || task.content_ref) && !resultArtifact) await loadTaskResult(task, version);
    else if (!task.artifact && !task.content_ref && !resultLoading && !resultArtifact && !$("videoArea").children.length) {
      showResultLoader(task, version, true);
    }
  } else {
    $("videoArea").textContent = task.status === "failed" ? `生成失败：${task.error?.message || task.error?.code || "请查看任务信息"}` : task.status === "cancelled" ? "任务已取消" : `${taskStatusLabel(task)}，结果就绪后将在这里显示。`;
    $("resultMessage").textContent = task.status === "failed" ? "本次任务在推理阶段中断，未产生可下载视频" : task.status === "cancelled" ? "本次任务已取消" : "引擎推理中，完成后可在线预览与下载";
  }
}
window.addEventListener("hashchange", businessView);
window.addEventListener("pagehide", clearMedia);
window.addEventListener("DOMContentLoaded", () => {
  businessView();
  if (detailTaskId && !authorized) $("detailMessage").textContent = "请连接 MCP 以查看任务。";
  $("editTitle").onclick = () => {
    if (!currentTitleTask) return;
    titleEditRevision = currentTitleTask.title_revision;
    $("titleInput").value = currentTitleTask.title;
    $("titleMessage").textContent = ""; $("titleForm").hidden = false;
    $("titleInput").focus();
  };
  $("cancelTitle").onclick = () => { $("titleForm").hidden = true; $("editTitle").focus(); };
  // The Studio iframe deliberately omits allow-forms. Keep writes on its
  // controlled request bridge, including keyboard activation.
  $("titleForm").onsubmit = (event) => event.preventDefault();
  $("titleInput").onkeydown = (event) => {
    if (event.key === "Enter" && !event.isComposing) {
      event.preventDefault(); $("saveTitle").click();
    }
  };
  $("saveTitle").onclick = async (event) => {
    event.preventDefault(); const version = revision, id = currentTitleTask?.video_task_id;
    if (!id || !authorized) return;
    const name = $("titleInput").value.trim();
    if (![...name].length || [...name].length > 64 || /[<>\p{C}\u2028\u2029]/u.test(name)) {
      $("titleMessage").textContent = "名称须为 1–64 字的单行纯文本。"; return;
    }
    $("saveTitle").disabled = true;
    try {
      await api(`/api/tasks/${encodeURIComponent(id)}/title`, { method: 'POST', body: JSON.stringify({title: name, expected_title_revision: titleEditRevision}) });
      if (version !== revision || !authorized || currentTitleTask?.video_task_id !== id) return;
      await refreshTaskDetail(id, version); $("titleForm").hidden = true; $("editTitle").focus();
    } catch (error) {
      if (version !== revision || !authorized) return;
      if (error.status === 409) { await refreshTaskDetail(id, version); titleEditRevision = currentTitleTask?.title_revision; }
      $("titleMessage").textContent = error.message;
    } finally { $("saveTitle").disabled = false; }
  };
  const stage = $("imagePreviewStage");
  if (stage) {
    stage.addEventListener("wheel", (event) => {
      event.preventDefault();
      const rect = stage.getBoundingClientRect();
      const cx = event.clientX - rect.left - rect.width / 2;
      const cy = event.clientY - rect.top - rect.height / 2;
      const next = Math.max(0.5, Math.min(5, zoomScale * (event.deltaY < 0 ? 1.15 : 1 / 1.15)));
      const r = next / zoomScale;
      zoomTx = cx - (cx - zoomTx) * r;
      zoomTy = cy - (cy - zoomTy) * r;
      zoomScale = next;
      syncImageZoom();
    }, { passive: false });
    stage.addEventListener("pointerdown", (event) => {
      zoomDragging = true;
      zoomStartX = event.clientX - zoomTx;
      zoomStartY = event.clientY - zoomTy;
      stage.classList.add("dragging");
      stage.setPointerCapture?.(event.pointerId);
    });
    stage.addEventListener("pointermove", (event) => {
      if (!zoomDragging) return;
      zoomTx = event.clientX - zoomStartX;
      zoomTy = event.clientY - zoomStartY;
      syncImageZoom();
    });
    const stopDrag = () => { zoomDragging = false; stage.classList.remove("dragging"); };
    stage.addEventListener("pointerup", stopDrag);
    stage.addEventListener("pointercancel", stopDrag);
  }
  $("imagePreview")?.addEventListener("click", (event) => {
    const stepBtn = event.target.closest("[data-zstep]");
    if (stepBtn) {
      zoomScale = Math.max(0.5, Math.min(5, zoomScale + Number(stepBtn.dataset.zstep)));
      syncImageZoom();
      return;
    }
    if (event.target.closest("#zoomReset")) {
      resetImageZoom();
    }
  });
  $("closeImagePreview").onclick = () => $("imagePreview").close();
  $("imagePreview").addEventListener("close", () => {
    resetImageZoom();
    $("imagePreviewImage").removeAttribute("src");
  });
});
