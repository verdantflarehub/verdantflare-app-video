const root = document.getElementById('app');
const $ = id => root.querySelector(`#${id}`);
const {appSDK: sdk, dashboardHost: bridge} = window.$wujie?.props || {};
const pages = new Set(['overview', 'authorization', 'usage']);
const labels = {unconfigured:'未配置', pending_verification:'待验证', active:'已验证',
  invalid:'Key 无效或权限不足', unavailable:'验证暂不可用', revoked:'已撤销'};
let disposed = false, visible = true, snapshot = null, controller = null, timer = null;
const events = new AbortController(), unsubscribe = [];
function message(value) { $('message').textContent = value; $('message').hidden = !value; bridge?.resize?.(root.getBoundingClientRect().height); }
async function call(method, path, body, signal) {
  const response = await sdk.app.request({method, path, ...(body === undefined ? {} : {body})}, {signal});
  if (response.status < 200 || response.status >= 300) throw new Error(response.data?.error || 'request_unavailable');
  return response.data;
}
function render() {
  if (!snapshot || disposed) return;
  $('content').hidden = false;
  $('statusBadge').textContent = $('authStatus').textContent = $('authorizationState').textContent = labels[snapshot.status] || '待检查';
  $('enabledState').textContent = snapshot.enabled ? '允许' : '已暂停';
  $('dailyLimit').textContent = String(snapshot.daily_limit);
  $('checkedAt').textContent = snapshot.checked_at ? new Date(snapshot.checked_at).toLocaleString('zh-CN') : '尚未检查';
  $('keyVersion').textContent = snapshot.version == null ? '—' : String(snapshot.version);
  $('enabled').checked = snapshot.enabled;
  $('limit').value = snapshot.daily_limit;
  bridge?.resize?.(root.getBoundingClientRect().height);
}
async function refresh() {
  if (!visible || disposed) return;
  controller?.abort(); controller = new AbortController();
  try {
    snapshot = await call('GET', '/overview', undefined, controller.signal);
    if (disposed || !visible) return;
    render(); message('');
  } catch { if (!disposed && visible) message('无法读取渠道状态，请稍后重试。'); }
  clearTimeout(timer); if (!disposed && visible) timer = setTimeout(refresh, 30000);
}
function route(value) {
  const page = String(value?.path || '/overview').replace(/^\//, '');
  const selected = pages.has(page) ? page : 'overview';
  root.querySelectorAll('[data-page]').forEach(node => { node.hidden = node.dataset.page !== selected; });
  root.querySelectorAll('nav button').forEach(node => node.setAttribute('aria-current', node.dataset.route === selected ? 'page' : 'false'));
  if (selected === 'usage') void loadUsage();
  bridge?.resize?.(root.getBoundingClientRect().height);
}
async function loadUsage() {
  try {
    const usage = await call('GET', '/usage');
    if (!disposed && visible) $('submitted').textContent = String(Object.values(usage.local_submissions || {}).reduce((a,b) => a+b, 0));
  } catch { if (!disposed) $('submitted').textContent = '暂不可用'; }
}
function dispose() { disposed = true; clearTimeout(timer); controller?.abort(); events.abort(); unsubscribe.forEach(fn => fn?.()); }
if (!sdk?.app?.request || !sdk?.navigation?.get || !bridge) {
  message('请从 Studio 应用中心打开 fal.ai。');
} else {
  const origin = sdk.context.get();
  const identity = [origin.stationId, origin.installationId, origin.organizationId, origin.userId].join(':');
  document.body.dataset.theme = sdk.theme.get();
  unsubscribe.push(sdk.theme.subscribe(theme => { document.body.dataset.theme = theme; }));
  unsubscribe.push(sdk.navigation.subscribe(route));
  unsubscribe.push(sdk.context.subscribe(next => {
    if ([next.stationId, next.installationId, next.organizationId, next.userId].join(':') !== identity) dispose();
  }));
  root.addEventListener('click', async event => {
    const button = event.target.closest('button'); if (!button || disposed || !visible) return;
    if (button.dataset.route) await sdk.navigation.request({target:'app', path:'/'+button.dataset.route});
    if (button.hasAttribute('data-video')) await sdk.capabilities.call('navigation.openApp', {appId:'video-mcp-server', path:'/models'});
  }, {signal:events.signal});
  $('keyForm').addEventListener('submit', async event => {
    event.preventDefault(); if (!snapshot || disposed || !visible) return;
    const input = $('apiKey'), key = input.value; input.value = '';
    try { const value = await call('PUT', '/authorization', {api_key:key, expected_revision:snapshot.revision});
      snapshot = value; render(); message('Key 已保存并通过只读验证。新任务仍需手动开启。');
    } catch { message('授权未生效。原有效版本仍保留，请检查 Key 权限或网络后重试。'); }
  }, {signal:events.signal});
  $('recheck').addEventListener('click', async () => {
    try { snapshot = await call('POST', '/authorization/check', {}); render(); message('检查已完成。'); }
    catch { message('检查失败，当前状态请重新读取。'); }
  }, {signal:events.signal});
  $('revoke').addEventListener('click', async () => {
    if (!snapshot) return;
    try { snapshot = await call('DELETE', '/authorization', {expected_revision:snapshot.revision}); render(); message('已停止新提交。'); }
    catch { message('撤销结果未确认，请重新读取。'); }
  }, {signal:events.signal});
  $('limitsForm').addEventListener('submit', async event => {
    event.preventDefault(); if (!snapshot) return;
    try { snapshot = await call('PUT', '/limits', {enabled:$('enabled').checked,
      daily_limit:Number($('limit').value), expected_revision:snapshot.revision}); render(); message('限制已保存。'); }
    catch { message('保存结果未确认，请重新读取后核对。'); }
  }, {signal:events.signal});
  unsubscribe.push(sdk.lifecycle.subscribe(event => {
    if (event === 'hidden') { visible = false; clearTimeout(timer); controller?.abort(); }
    if (event === 'active') { visible = true; void refresh(); route(sdk.navigation.get()); }
    if (event === 'destroy') dispose();
  }));
  route(sdk.navigation.get()); bridge.ready(); void refresh();
}
