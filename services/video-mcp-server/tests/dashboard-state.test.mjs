import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Execute the real event handler with a minimal DOM, without starting a service.
const script = readFileSync(new URL('../frontend/dashboard.js', import.meta.url), 'utf8');
function setup() {
  const listeners = new Map(), microtasks = [], messages = [], revoked = [];
  const node = () => ({ value: '', textContent: '', innerHTML: '', children: [], options: [], add(option) { this.options.push(option) }, replaceChildren() { this.children = []; this.innerHTML = '' }, hidden: true, disabled: false, classList: { contains: () => false, toggle() {}, add() {}, remove() {} },
    elements: { route: { addEventListener() {}, querySelector: () => ({ disabled: false, textContent: '' }) } }, addEventListener(type, fn) { listeners.set('el:' + type, fn) } });
  const nodes = new Map();
  const urlGlobal = Object.assign(function URL(...args) { return new globalThis.URL(...args) }, {
    createObjectURL: () => 'blob:preview-' + Math.random().toString(36).slice(2, 8),
    revokeObjectURL: (u) => revoked.push(u),
  });
  const context = vm.createContext({
    window: { studioEmbedded: true, studioNavigate: path => messages.push(path), addEventListener: (type, fn) => listeners.set('window:' + type, fn), innerHeight: 900, scrollY: 0 },
    Option: function(text, value) { this.text = text; this.value = value },
    document: { hidden: false, documentElement: { scrollHeight: 2000 }, createElement: () => ({ innerHTML: '', content: { children: [], firstElementChild: null } }), querySelectorAll: () => [], getElementById: id => { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id) }, addEventListener: (type, fn) => listeners.set(type, fn) },
    location: { pathname: '/dashboard', hash: '#tasks', search: '' }, URLSearchParams, URL: urlGlobal,
    queueMicrotask: fn => microtasks.push(fn), setTimeout: () => 1, clearTimeout() {},
    refreshBusiness() {}, clearBusiness() {}, clearTaskDetail() {}, taskDisplayName: t => t.title || t.video_task_id,
  });
  vm.runInContext(script, context);
  const flush = () => { while (microtasks.length) microtasks.shift()() };
  return { listeners, messages, revoked, nodes, flush, run: code => vm.runInContext(code, context) };
}

test('HTTP 503 keeps distinct channel reasons and original acceptance metadata', async () => {
  const app = setup();
  for (const code of ['authorization_required', 'limits_unavailable', 'account_unavailable']) {
    const failure = await app.run(`(async () => {
      window.studioRequest = async () => ({ ok: false, status: 503,
        text: async () => JSON.stringify({error: {code: '${code}', message: 'Channel reason ${code}',
          source: 'higgsfield', retryable: true, accepted: null, http_status: 503,
          task_id: 'video_task_' + 'a'.repeat(32)}}) });
      try { await api('/api/tasks'); } catch (error) {
        return {message: error.message, code: error.code, source: error.source,
          accepted: error.accepted, retryable: error.retryable, http_status: error.http_status, task_id: error.task_id};
      }
    })()`);
    assert.equal(failure.code, code);
    assert.equal(failure.source, 'higgsfield');
    assert.equal(failure.accepted, null);
    assert.equal(failure.http_status, 503);
    assert.equal(failure.retryable, true);
    assert.equal(failure.task_id, 'video_task_' + 'a'.repeat(32));
    assert.match(failure.message, /不要重复提交/);
    assert.ok(failure.message.includes(code));
    assert.doesNotMatch(failure.message, /未配置访问 Token/);
  }
});

test('generic proxy 503 does not invent missing credentials or display response bodies', async () => {
  const app = setup();
  const message = await app.run(`(async () => {
    window.studioRequest = async () => ({ok: false, status: 503, text: async () => '<html>private upstream body</html>'});
    try { await api('/api/dashboard'); } catch (error) { return error.message; }
  })()`);
  assert.match(message, /服务暂时不可用/);
  assert.doesNotMatch(message, /Token|private|html/);
  const configured = app.run(`serviceFailure({error:'authentication_not_configured'})`);
  assert.match(configured.message, /未配置访问 Token/);
  assert.equal(app.run(`serviceFailure({error:'private-raw-exception'})`), null);
});

test('safe structured reasons redact credentials, URLs and local paths', () => {
  const app = setup();
  const failure = app.run(`serviceFailure({error:{code:'account_unavailable',source:'google',retryable:false,
    message:'Account unavailable; token=private-value HTTPS://private.invalid/signed?secret=private /internal/config/settings',
    accepted:false}})`);
  assert.match(failure.message, /Account unavailable/);
  assert.doesNotMatch(failure.message, /private-value|private\.invalid|\/internal\/config/);
  assert.equal(failure.accepted, false);
});

test('channel authorization failure preserves the Studio session while real session 401 invalidates it', async () => {
  const app = setup();
  await app.run(`(async () => {
    authorized = true; token = 'existing-session';
    window.studioRequest = async () => ({ok:false,status:401,text:async()=>JSON.stringify({error:{
      code:'authorization_required',message:'Channel authorization expired',source:'higgsfield',retryable:false,accepted:false}})});
    try { await api('/api/tasks'); } catch (error) { if (error.accepted !== false) throw error; }
    if (!authorized || token !== 'existing-session') throw Error('Channel error invalidated Studio session');
    window.studioRequest = async () => ({ok:false,status:401,text:async()=>''});
    try { await api('/api/tasks'); } catch {}
    if (authorized || token) throw Error('Session failure did not invalidate Studio session');
  })()`);
});

test('Higgsfield member request preserves the shared safe channel error and legacy quote errors', async () => {
  const app = setup();
  app.run(`document.querySelector = () => ({append() {}}); globalThis.crypto = {randomUUID: () => 'test-attempt'};`);
  const member = readFileSync(new URL('../frontend/higgsfield-member.js', import.meta.url), 'utf8');
  app.run(member.replace(/\}\)\(\);\s*$/, 'window.testMemberRequest = request; })();'));
  const error = await app.run(`(async () => {
    window.studioRequest=async()=>({ok:false,status:503,json:async()=>({error:{
      code:'limits_unavailable',source:'higgsfield',message:'Membership limits unavailable',accepted:true,retryable:true,http_status:503}})});
    try { await window.testMemberRequest('tasks'); } catch(e) { return {code:e.code,message:e.message,accepted:e.accepted,http_status:e.http_status}; }
  })()`);
  assert.equal(error.code, 'limits_unavailable');
  assert.equal(error.accepted, true);
  assert.equal(error.http_status, 503);
  assert.match(error.message, /Membership limits unavailable/);
  assert.match(error.message, /不要重复提交/);
  const legacy = await app.run(`(async () => {
    window.studioRequest=async()=>({ok:false,status:409,json:async()=>({error:'quote_changed'})});
    try { await window.testMemberRequest('preflight',{}); } catch(e) { return {code:e.code,message:e.message}; }
  })()`);
  assert.equal(legacy.code, 'quote_changed');
  assert.match(legacy.message, /重新报价/);
});

test('restoring table view cannot signal list readiness before the first render', () => {
  const app = setup();
  app.run('authorized = true;');
  app.listeners.get('click')(); app.flush();
  assert.equal(app.messages.length, 0);
  app.run('listReady = true;');
  for (const type of ['click', 'change', 'input']) { app.listeners.get(type)(); app.flush(); }
  assert.deepEqual(app.messages, ['/dashboard#tasks', '/dashboard#tasks', '/dashboard#tasks']);
  app.run('authorized = false;');
  app.listeners.get('input')(); app.flush();
  assert.equal(app.messages.length, 3);
});

test('task cards render Model x [Logo] Channel and 5 output tags without total frame count', () => {
  const app = setup();
  const identityHtml = app.run(`taskIdentity({
    service: 'fal',
    route: 'fal',
    model: 'minimax-h3-ref2va',
    presentation: { is_generation: true, model_label: 'MiniMax H3', channel_label: 'fal.ai', channel_icon: null, generation_code: 'R2V', generation_label: '参考生视频' }
  })`);
  assert.match(identityHtml, /MiniMax H3/);
  assert.match(identityHtml, /channel-logo/);
  assert.match(identityHtml, /fal\.ai/);

  const tagsHtml = app.run(`taskTags({
    service: 'higgsfield',
    route: 'higgsfield',
    resolution: '720p',
    aspect_ratio: '9:16',
    duration_seconds: 8,
    media: { frame_rate: 24, frames: 192 },
    presentation: { is_generation: true, generation_code: 'R2V', generation_label: '参考生视频' }
  })`);
  assert.match(tagsHtml, />R2V<\/span><span>720p<\/span><span>9:16<\/span><span>8S<\/span><span>24FPS<\/span>/);
  assert.doesNotMatch(tagsHtml, /192/);
});

test('task detail HTML and CSS match prototype v1.3 two-pane inspector, 380px stage, 3-col video grid, and affine zoom lightbox', () => {
  const detailHtml = readFileSync(new URL('../frontend/task-detail.html', import.meta.url), 'utf8');
  const dashboardHtml = readFileSync(new URL('../frontend/dashboard.html', import.meta.url), 'utf8');
  const css = readFileSync(new URL('../frontend/dashboard.css', import.meta.url), 'utf8');
  for (const html of [detailHtml, dashboardHtml]) {
    assert.match(html, /id="taskDetail"[^>]*class="detail-shell"|class="detail-shell"[^>]*id="taskDetail"/);
    assert.match(html, /class="inspect-layout"/);
    assert.match(html, /id="detailIdentityBar"/);
    assert.doesNotMatch(html, /id="backToTasksLink"/);
    assert.doesNotMatch(html, /id="detailTriSteps"|令牌直通/);
    assert.match(html, /id="detailOutputMeta"/);
    assert.match(html, /id="imagePreviewStage"/);
    assert.match(html, /id="zoomReadout"/);
    assert.match(html, /id="zoomReset"/);
    assert.match(html, /<section class="output-pane">[\s\S]*id="detailOutputMeta"[\s\S]*id="taskPrompt"[\s\S]*<\/section><div class="input-pane"><section class="input-factors-container"><div id="referenceGroups">/);
  }
  assert.match(dashboardHtml, /id="loadMoreBtn"/);
  assert.match(dashboardHtml, /id="retryLoadMoreBtn"/);
  assert.match(dashboardHtml, /id="scrollSentinel"/);
  assert.doesNotMatch(css, /\.result-stage\{[^}]*width:640px/);
  assert.match(css, /\.result-stage\{[^}]*width:100%;height:380px/);
  assert.match(css, /\.result-stage video\{[^}]*max-height:348px[^}]*object-fit:contain[^}]*aspect-ratio:auto/);
  assert.match(css, /\.inspect-layout\{display:grid;grid-template-columns:minmax\(0,0\.82fr\) minmax\(0,1\.18fr\);align-items:start\}/);
  assert.match(css, /\.ref-video-grid\{display:grid!important;grid-template-columns:repeat\(3,minmax\(0,1fr\)\);gap:12px;width:100%\}/);
});


test('channel and project query and saved state use the selected values', () => {
  const app = setup();
  app.run('$("filterChannel").value = "higgsfield"; $("filterProject").value = "project-two"; $("searchInput").value = " needle "; page = 3; status = "failed";');
  const params = new URLSearchParams(app.run('query().toString()'));
  assert.equal(params.get('channel'), 'higgsfield');
  assert.equal(params.get('project_id'), 'project-two');
  assert.equal(params.get('q'), 'needle');
  assert.equal(params.get('page'), '3');
  assert.equal(params.has('service'), false);
  const state = app.run('window.studioViewState()');
  assert.equal(state.channel, 'higgsfield');
  assert.equal(state.project, 'project-two');
  app.run('refresh = () => {}; filter()');
  assert.equal(app.run('page'), 1);
});


test('return from details restores channels, projects and legacy local engines', () => {
  for (const saved of [{ channel: 'higgsfield' }, { engine: 'h3-vdn' }, { channel: 'future-channel' }]) {
    const app = setup();
    app.run('refresh = () => {};');
    app.listeners.get('window:studio-restore')({ detail: { ...saved, project: 'project-two', page: 2, status: 'failed' } });
    const state = app.run('window.studioViewState()');
    assert.equal(state.channel, saved.channel || 'local');
    assert.equal(state.project, 'project-two');
    assert.equal(state.page, 2);
    assert.equal(state.status, 'failed');
  }
});

test('thumbnails only request succeeded result videos, use explicit status placeholders, and apply failure backoff with retry', async () => {
  const app = setup();
  const queuedHtml = app.run(`thumbPlaceholderHTML({ video_task_id: 't1', status: 'queued', service: 'h3' })`);
  const runningHtml = app.run(`thumbPlaceholderHTML({ video_task_id: 't2', status: 'running', service: 'h3' })`);
  const failedHtml = app.run(`thumbPlaceholderHTML({ video_task_id: 't3', status: 'failed', service: 'h3' })`);
  const okHtml = app.run(`thumbPlaceholderHTML({ video_task_id: 't4', status: 'succeeded', service: 'h3', artifact_id: 'art_1' })`);
  assert.match(queuedHtml, /排队中 · 待生成结果/);
  assert.match(runningHtml, /生成中 · 完成后提取首帧/);
  assert.match(failedHtml, /任务失败 · 无结果视频/);
  assert.match(okHtml, /正在加载首帧/);
  for (const html of [queuedHtml, runningHtml, failedHtml, okHtml]) {
    assert.doesNotMatch(html, /▶/);
  }

  await app.run(`(async () => {
    authorized = true;
    const calls = [];
    let shouldFail = true;
    api = async (path) => {
      calls.push(path);
      if (path.includes('t_fail') && shouldFail) {
        const err = new Error('extract failed');
        err.status = 422;
        throw err;
      }
      return { size: 128 };
    };
    const tasks = [
      { video_task_id: 't_queued', status: 'queued', service: 'h3' },
      { video_task_id: 't_failed', status: 'failed', service: 'fal' },
      { video_task_id: 't_ok', status: 'succeeded', service: 'h3-singularity', artifact_id: 'art_ok' },
      { video_task_id: 't_hf', status: 'succeeded', service: 'higgsfield', media: { sha256: 'abc' } },
      { video_task_id: 't_fail', status: 'succeeded', service: 'fal', artifact_id: 'art_bad' },
    ];
    loadedTasks = tasks;
    await previews(tasks);
    if (calls.length !== 3) throw new Error('Expected 3 thumbnail calls, got ' + calls.length);
    // Second poll must hit cache for t_ok/t_hf and respect failure backoff for t_fail (0 new calls).
    await previews(tasks);
    if (calls.length !== 3) throw new Error('Expected 0 additional calls on poll, got ' + calls.length);
    const errMarkup = thumbPlaceholderHTML(tasks[4]);
    if (!errMarkup.includes('data-thumb-retry="t_fail"')) throw new Error('Missing retry button markup');
    // Explicit retry clears backoff and fetches again.
    shouldFail = false;
    await retryThumbnail('t_fail');
    if (calls.length !== 4) throw new Error('Expected retry call, got ' + calls.length);
  })()`);
});

test('continuous scroll loadMore appends without duplicates, preserves items on error, and resets on filter', async () => {
  const app = setup();
  await app.run(`(async () => {
    authorized = true;
    listReady = true;
    page = 1;
    total = 50;
    loadedTasks = Array.from({ length: 24 }, (_, i) => ({
      video_task_id: 'task_' + (i + 1),
      status: 'failed',
      service: 'h3',
      prompt: 'p',
      created_at: '2026-10-17T00:00:00Z',
    }));
    let failNext = true;
    api = async (path) => {
      if (failNext) throw new Error('network error');
      const p = new URLSearchParams(path.split('?')[1]).get('page');
      return {
        total: 50,
        counts: { queued: 0, running: 0, succeeded: 0, failed: 50, cancelled: 0 },
        channels: [],
        projects: [],
        tasks: p === '1'
          ? Array.from({ length: 24 }, (_, i) => ({
              video_task_id: 'task_' + (i + 1),
              status: 'failed',
              service: 'h3',
              prompt: 'p1',
              created_at: '2026-10-17T00:00:00Z',
            }))
          : [
              { video_task_id: 'task_24', status: 'failed', service: 'h3', prompt: 'dup', created_at: '2026-10-17T00:00:00Z' },
              ...Array.from({ length: 24 }, (_, i) => ({
                video_task_id: 'task_' + (i + 25),
                status: 'failed',
                service: 'h3',
                prompt: 'p2',
                created_at: '2026-10-17T00:00:00Z',
              })),
            ],
      };
    };
    await loadMore();
    if (page !== 1 || loadedTasks.length !== 24 || !loadMoreError) {
      throw new Error('Failed loadMore must preserve page=1 and 24 existing tasks');
    }
    failNext = false;
    await loadMore();
    if (page !== 2 || loadedTasks.length !== 48 || loadMoreError) {
      throw new Error('Expected page=2 and 48 deduplicated tasks, got ' + page + '/' + loadedTasks.length);
    }
    // Polling refresh when page === 2 must preserve all 48 loaded tasks instead of truncating to 24.
    await refresh();
    if (page !== 2 || loadedTasks.length !== 48) {
      throw new Error('Polling refresh must preserve 2 loaded pages (48 tasks), got ' + page + '/' + loadedTasks.length);
    }
  })()`);

  // Pagehide revokes cached thumbnail URLs.
  app.run(`thumbnails.set('t1', { url: 'blob:test-1', key: 'k1' });`);
  app.listeners.get('window:pagehide')();
  assert.ok(app.revoked.includes('blob:test-1'));
});

test('regenerate_2k and 2K proportional grid dimensions render VSR2VA badge and 2K resolution label', () => {
  const app = setup();
  for (const [width, height] of [[2464, 1408], [1408, 2464], [2560, 1120], [3072, 1344], [1920, 1440], [1440, 1440]]) {
    assert.equal(app.run(`taskResolution({ service: 'regenerate_2k', media: { width: ${width}, height: ${height} } })`), '2K');
    assert.equal(app.run(`taskResolution({ service: 'h3-singularity', media: { width: ${width}, height: ${height} } })`), '2K');
  }
  assert.equal(app.run(`taskResolution({ service: 'regenerate_2k' })`), '2K');
  assert.equal(app.run(`taskResolution({ service: 'h3-singularity', media: { width: 1344, height: 768 } })`), '768p');

  const identityHtml = app.run(`taskIdentity({
    service: 'regenerate_2k',
    route: 'h3-singularity',
    model: 'open-h3-regenerate-2k-ic',
    presentation: { is_generation: true, model_label: 'H3-Regenerate-2K', channel_label: '本地', channel_icon: 'local', generation_code: 'VSR2VA', generation_label: '超分音视频' }
  })`);
  assert.match(identityHtml, /H3-Regenerate-2K/);
  assert.match(identityHtml, /本地/);

  const tagsHtml = app.run(`taskTags({
    service: 'regenerate_2k',
    route: 'h3-singularity',
    aspect_ratio: '16:9',
    duration_seconds: 5,
    media: { width: 2464, height: 1408, frame_rate: 24, frames: 121 },
    presentation: { is_generation: true, generation_code: 'VSR2VA', generation_label: '超分音视频' }
  })`);
  assert.match(tagsHtml, />VSR2VA<\/span><span>2K<\/span><span>16:9<\/span><span>5S<\/span><span>24FPS<\/span>/);
  assert.doesNotMatch(tagsHtml, /1080p|121/);
});

test('stage timing breakdown, standalone regenerate_2k duration/size fallback, and optional postprocess switches', async () => {
  const app = setup();
  const detailScript = readFileSync(new URL('../frontend/task-detail.js', import.meta.url), 'utf8');
  app.run(detailScript);

  // 1. Standalone regenerate_2k without duration_seconds/aspect_ratio must never render "nulls · null".
  assert.equal(app.run(`taskDurationAspect({ service: 'regenerate_2k', duration_seconds: null, aspect_ratio: null, media: { width: 1408, height: 2464, frames: 214, frame_rate: '48' } })`), '4.5s · 1408×2464');
  assert.equal(app.run(`taskDurationAspect({ service: 'regenerate_2k', duration_seconds: null, aspect_ratio: null })`), '—');

  // 2. Optional postprocess switches default to off and emit normalized payloads only when enabled on local H3 routes.
  for (const html of [
    readFileSync(new URL('../frontend/dashboard.html', import.meta.url), 'utf8'),
    readFileSync(new URL('../frontend/task-detail.html', import.meta.url), 'utf8'),
  ]) {
    assert.match(html, /id="postprocessGroup"/);
    assert.match(html, /id="postprocessRegen2k"/);
    assert.match(html, /id="postprocessGridMode"/);
    assert.match(html, /id="postprocessInterpolate"/);
    assert.match(html, /id="detailStageTiming"/);
  }
  assert.equal(app.run(`buildDispatchPostprocess('h3')`), null);
  app.run(`$("postprocessRegen2k").checked = true; $("postprocessGridMode").value = "budget_1440p"; $("postprocessInterpolate").checked = true;`);
  assert.deepEqual(JSON.parse(app.run(`JSON.stringify(buildDispatchPostprocess('h3'))`)), {
    regenerate_2k: { profile: 'open_h3_2k_ic', grid_mode: 'budget_1440p' },
    interpolate: { profile: 'rife_2x' },
  });
  assert.equal(app.run(`buildDispatchPostprocess('fal')`), null);

  // 3. Task detail renders explicit P1 / P2 / P3 stage timing in both #detailStageTiming and #taskParameters.
  await app.run(`(async () => {
    authorized = true;
    revision = 7;
    document.createDocumentFragment = () => ({ append() {} });
    document.createTextNode = text => ({ text });
    const origGet = document.getElementById;
    document.getElementById = id => id.startsWith('reference-') ? null : origGet(id);
    $("videoBreadcrumb").querySelector = () => null;
    $("imagePreview").close = () => {};
    $("imagePreviewImage").removeAttribute = () => {};
    $("taskParameters").querySelector = () => ({ append() {} });
    renderReferences = () => {};
    loadTaskResult = async () => {};
    api = async () => ({
      video_task_id: 'video_task_e2e2',
      project_id: 'demo',
      idempotency_key: 'k2',
      service: 'regenerate_2k',
      route: 'h3-regenerate-2k',
      status: 'succeeded',
      created_at: '2026-11-14T14:40:00Z',
      duration_seconds: null,
      aspect_ratio: null,
      artifact: { artifact_id: 'art_m3' },
      grid_mode: 'budget_1440p',
      media: { width: 1408, height: 2464, frames: 214, frame_rate: '48' },
      presentation: { is_generation: true, model_label: 'Open-H3-Regenerate-2K-IC', channel_label: '本地', channel_icon: 'local', generation_code: 'VSR2VA', generation_label: '超分音视频' },
      timing: { queue_seconds: 0, processing_seconds: 389.94, total_seconds: 412.1 },
      delivery: {
        stage: 'done',
        status: 'succeeded',
        profile: 'rife_2x',
        profiles: ['rife_2x'],
        grid_mode: 'budget_1440p',
        source_artifact_id: 'art_src',
        regenerate_2k_artifact_id: 'art_m2',
        final_artifact_id: 'art_m3',
        total_seconds: 412.1,
        timing: {
          queue_seconds: 0,
          generation_active_seconds: 0,
          regenerate_2k_active_seconds: 389.94,
          interpolation_active_seconds: 22.05,
          chain_active_seconds: 411.99,
        },
      },
    });
    await refreshTaskDetail('video_task_e2e2', 7);
  })()`);
  const stageTimingHtml = app.nodes.get('detailStageTiming').innerHTML;
  const paramsHtml = app.nodes.get('taskParameters').innerHTML;
  assert.equal(app.nodes.get('detailStageTiming').hidden, false);
  assert.match(stageTimingHtml, /P2 2K 重生成.*389\.94s/);
  assert.match(stageTimingHtml, /P3 RIFE 补帧.*22\.05s/);
  assert.match(stageTimingHtml, /链路活跃合计.*411\.99s/);
  assert.doesNotMatch(stageTimingHtml, /P1 生成/);
  assert.match(paramsHtml, /P2 2K 重生成活跃耗时.*389\.94s/);
  assert.match(paramsHtml, /P3 RIFE 补帧活跃耗时.*22\.05s/);
  assert.match(paramsHtml, /全链路活跃总耗时.*411\.99s/);
  assert.match(paramsHtml, /4\.5s · 1408×2464/);
  assert.doesNotMatch(paramsHtml, /nulls|null/);

  // 4. Full P1 -> P2 -> P3 chain task renders P1, P2, P3, and chain active total.
  await app.run(`(async () => {
    api = async () => ({
      video_task_id: 'video_task_e2e1',
      project_id: 'demo',
      idempotency_key: 'k1',
      service: 'h3-fasth3',
      route: 'h3-fasth3',
      status: 'succeeded',
      created_at: '2026-11-14T14:20:00Z',
      duration_seconds: 5,
      aspect_ratio: '16:9',
      artifact: { artifact_id: 'art_m3_16x9' },
      grid_mode: 'budget_1440p',
      media: { width: 2464, height: 1408, frames: 248, frame_rate: '48' },
      presentation: { is_generation: true, model_label: 'Open-H3-FastH3-Ref2VA', channel_label: '本地', channel_icon: 'local', generation_code: 'R2V', generation_label: '参考生视频' },
      timing: { queue_seconds: 0, processing_seconds: 61.08, total_seconds: 567.2 },
      delivery: {
        stage: 'done',
        status: 'succeeded',
        profile: 'rife_2x',
        profiles: ['open_h3_2k_ic', 'rife_2x'],
        grid_mode: 'budget_1440p',
        source_artifact_id: 'art_m1',
        regenerate_2k_artifact_id: 'art_m2',
        final_artifact_id: 'art_m3_16x9',
        total_seconds: 567.2,
        timing: {
          queue_seconds: 0,
          generation_active_seconds: 61.08,
          regenerate_2k_active_seconds: 483.0,
          interpolation_active_seconds: 22.36,
          chain_active_seconds: 566.44,
        },
      },
    });
    await refreshTaskDetail('video_task_e2e1', 7);
  })()`);
  assert.match(app.nodes.get('detailStageTiming').innerHTML, /P1 生成.*61\.08s/);
  assert.match(app.nodes.get('detailStageTiming').innerHTML, /P2 2K 重生成.*483\.00s/);
  assert.match(app.nodes.get('detailStageTiming').innerHTML, /P3 RIFE 补帧.*22\.36s/);
  assert.match(app.nodes.get('detailStageTiming').innerHTML, /链路活跃合计.*566\.44s/);
  assert.match(app.nodes.get('taskParameters').innerHTML, /P1 生成活跃耗时.*61\.08s/);
});


