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

