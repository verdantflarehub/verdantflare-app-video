import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Execute the real event handler with a minimal DOM, without starting a service.
const script = readFileSync(new URL('../frontend/dashboard.js', import.meta.url), 'utf8');
function setup() {
  const listeners = new Map(), microtasks = [], messages = [];
  const node = () => ({ value: '', hidden: true, classList: { contains: () => false, toggle() {}, add() {}, remove() {} },
    elements: { route: { addEventListener() {} } }, addEventListener() {} });
  const nodes = new Map();
  const context = vm.createContext({
    window: { studioEmbedded: true, studioNavigate: path => messages.push(path), addEventListener: (type, fn) => listeners.set('window:' + type, fn) },
    document: { getElementById: id => { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id) }, addEventListener: (type, fn) => listeners.set(type, fn) },
    location: { pathname: '/dashboard', hash: '#tasks', search: '' }, URLSearchParams, URL,
    queueMicrotask: fn => microtasks.push(fn), setTimeout: () => 1, clearTimeout() {},
  });
  vm.runInContext(script, context);
  const flush = () => { while (microtasks.length) microtasks.shift()() };
  return { listeners, messages, flush, run: code => vm.runInContext(code, context) };
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
    assert.match(html, /id="backToTasksLink"/);
    assert.doesNotMatch(html, /id="detailTriSteps"|令牌直通/);
    assert.match(html, /id="detailOutputMeta"/);
    assert.match(html, /id="imagePreviewStage"/);
    assert.match(html, /id="zoomReadout"/);
    assert.match(html, /id="zoomReset"/);
    assert.match(html, /<section class="output-pane">[\s\S]*id="detailOutputMeta"[\s\S]*id="taskPrompt"[\s\S]*<\/section><div class="input-pane"><section class="input-factors-container"><div id="referenceGroups">/);
  }
  assert.doesNotMatch(css, /\.result-stage\{[^}]*width:640px/);
  assert.match(css, /\.result-stage\{[^}]*width:100%;height:380px/);
  assert.match(css, /\.result-stage video\{[^}]*max-height:348px[^}]*object-fit:contain[^}]*aspect-ratio:auto/);
  assert.match(css, /\.inspect-layout\{display:grid;grid-template-columns:minmax\(0,0\.82fr\) minmax\(0,1\.18fr\);align-items:start\}/);
  assert.match(css, /\.ref-video-grid\{display:grid!important;grid-template-columns:repeat\(3,minmax\(0,1fr\)\);gap:12px;width:100%\}/);
});
