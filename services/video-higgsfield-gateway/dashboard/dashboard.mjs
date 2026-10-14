import {createLimitsEditor, errorCode, request} from './model.mjs';

const root = document.getElementById('dashboard'), $ = id => root.querySelector(`#${id}`);
const {appSDK: sdk, dashboardHost: bridge} = window.$wujie?.props || {};
const pages = new Set(['overview', 'authorization', 'usage', 'limits']);
const authNames = {authorized: '已授权', unconnected: '未连接', authorization_required: '需重新授权', unavailable: '暂时无法观测'};
let disposed = false, visible = true, overview = null, timer, reader, readVersion = 0;
let dirty = false, editor, usageCursor = null, usageNext = null, usageReader;
const events = new AbortController(), unsubscribe = [];
const text = (id, value) => { $(id).textContent = value ?? '—'; };
const time = value => value ? new Date(value).toLocaleString('zh-CN') : '尚无观测';
function size() { if (!disposed) bridge?.resize?.(Math.ceil(root.getBoundingClientRect().height)); }
function notice(value) { text('message', value); $('message').hidden = !value; size(); }
function denied(error) { return error.status === 401 || error.status === 403 || /DENIED|UNAUTHORIZED|SESSION_EXPIRED/.test(errorCode(error)); }
function failure(error, message) {
  if (disposed) return;
  if (denied(error)) {
    overview = null; $('business').hidden = true;
    notice('当前身份无权管理此渠道，或会话已失效。请从应用中心重新打开。');
  } else notice(message);
}
function updateEditor() {
  const state = editor?.snapshot(), pending = state?.pending;
  $('credits').disabled = $('submissions').disabled = Boolean(pending || state?.busy || !editor);
  $('save').disabled = !overview || !editor || Boolean(state?.busy);
  text('save', pending ? '重试原保存请求' : '保存设置');
  bridge?.setDirty?.(dirty || Boolean(pending));
}
function render() {
  if (!overview || disposed) return;
  const auth = overview.authorization, limits = overview.limits;
  $('business').hidden = false;
  text('authState', authNames[auth.status] || '待检查');
  text('authorizationBadge', authNames[auth.status] || '待检查');
  text('authorizationHeading', auth.status === 'authorized' ? '会员已连接到团队' : '检查或恢复会员连接');
  text('balance', auth.last_success ? `${auth.last_success.balance_credits} Credits` : '待检查');
  text('balanceTime', auth.last_success ? `${time(auth.last_success.observed_at)}${auth.stale ? ' · 旧观测' : ''}` : '检查授权后读取余额');
  text('reserved', `${limits.reserved_credits} Credits`);
  text('observed', time(auth.observed_at));
  text('workspaceState', auth.workspace_matches ? '与服务器绑定一致' : '等待校验原绑定');
  text('refreshState', auth.last_refresh_at ? time(auth.last_refresh_at) : '服务器维护 · 尚无可核验的刷新时间');
  text('generationState', ({paused: '新增生成已受限额暂停', channel_paused: 'Video 已暂停 Higgsfield 新增生成',
    not_verified: 'Video 渠道可用性待核验', authorization_unverified: '会员授权待检查，暂不能确认可生成',
    busy: '当前并发名额已占满', eligible: '可以提交，费用与项目权限仍按任务校验'})[overview.generation_status] || '新增生成状态待核验');
  text('submitted', `${limits.submitted_count} 条`);
  text('limitsReserved', `${limits.reserved_credits} Credits`);
  text('revision', limits.revision);
  const pending = editor?.snapshot().pending;
  if (pending) {
    $('credits').value = pending.credit_limit; $('submissions').value = pending.submission_limit;
    text('saveState', '存在结果尚未确认的保存请求。重试会使用原操作，不会另建修改。');
  } else if (!dirty) {
    $('credits').value = limits.credit_limit; $('submissions').value = limits.submission_limit;
  }
  $('check').disabled = Boolean(auth.pending_check_id);
  text('checkState', auth.pending_check_id ? '服务器正在检查，稍后自动更新。' : `最近检查：${time(auth.observed_at)}`);
  updateEditor(); size();
}
async function refresh() {
  if (disposed || !visible) return;
  clearTimeout(timer); reader?.abort();
  const version = ++readVersion;
  reader = new AbortController();
  try {
    const value = await request(sdk, 'GET', '/overview', undefined, reader.signal);
    if (disposed || !visible || version !== readVersion) return;
    overview = value; notice(''); render();
  } catch (error) {
    if (disposed || !visible || version !== readVersion) return;
    failure(error, '暂时无法更新渠道状态。已显示的数据可能过期，可重新读取。');
  } finally {
    if (!disposed && visible && version === readVersion) {
      timer = setTimeout(refresh, overview?.authorization.pending_check_id ? 2000 : 30000);
    }
  }
}
async function loadUsage() {
  if (!visible || disposed) return;
  usageReader?.abort(); const activeReader = usageReader = new AbortController();
  text('usage', '正在读取用量…'); $('usageNext').disabled = true;
  try {
    const value = await request(sdk, 'POST', '/usage/query', {cursor: usageCursor, page_size: 20}, activeReader.signal);
    if (disposed || !visible || usageReader !== activeReader) return;
    $('usage').replaceChildren();
    if (!value.items.length) text('usage', '暂无团队提交记录。');
    for (const item of value.items) {
      const row = document.createElement('article'); row.className = 'hf-task-row';
      const label = document.createElement('div'), title = document.createElement('strong'), detail = document.createElement('p');
      title.textContent = ({grok_video_v15_lite: 'Grok Imagine Video 1.5 Lite', seedance_2_5: 'Seedance 2.5', wan3_0: 'Wan 3.0'})[item.model] || item.model;
      detail.textContent = `Higgsfield · 报价预留 ${item.quoted_credits} Credits · ${time(item.created_at)}`;
      label.append(title, detail);
      const status = document.createElement('span'); status.className = 'hf-tag';
      status.textContent = item.status === 'completed' ? '已完成' : item.status === 'failed' ? '失败' : '处理中';
      row.append(label, status); $('usage').append(row);
    }
    const note = document.createElement('p'); note.className = 'hf-caption';
    note.textContent = '报价预留不代表实扣，任务与供应商流水尚待对账。任务与结果请在 Video 查看。';
    $('usage').append(note);
    const heading = document.createElement('h3'); heading.className = 'hf-spacing';
    heading.textContent = '会员近期流水'; $('usage').append(heading);
    const snapshot = value.provider_transactions, observed = snapshot?.last_success;
    const observation = document.createElement('p'); observation.className = 'hf-caption';
    observation.textContent = observed
      ? `最近 ${observed.window_size} 条以内 · ${time(observed.observed_at)}${snapshot.stale ? ' · 旧观测' : ''}。包含官网等入口使用，不等同团队任务总账。`
      : '尚无可用流水观测，请在会员授权页检查授权与用量。';
    $('usage').append(observation);
    for (const item of observed?.items || []) {
      const row = document.createElement('article'); row.className = 'hf-task-row';
      const label = document.createElement('div'), title = document.createElement('strong'), detail = document.createElement('p');
      title.textContent = ({grant: '额度入账', spend: '额度支出'})[item.action] || '其他额度变动';
      detail.textContent = time(item.created_at); label.append(title, detail);
      const amount = document.createElement('span'); amount.textContent = `${item.credits} Credits`;
      row.append(label, amount); $('usage').append(row);
    }
    if (observed && !observed.items.length) {
      const empty = document.createElement('p'); empty.className = 'hf-caption'; empty.textContent = '本次观测未返回近期流水。';
      $('usage').append(empty);
    }
    usageNext = value.next_cursor; $('usageNext').disabled = !usageNext;
  } catch (error) {
    if (disposed || !visible || usageReader !== activeReader) return;
    text('usage', '暂时无法读取用量，请重新读取。'); failure(error, '用量读取失败。');
  }
  size();
}
function route(value) {
  const selected = value.path.replace(/^\//, '') || 'overview';
  const page = pages.has(selected) ? selected : 'overview';
  root.querySelectorAll('[data-page]').forEach(el => { el.hidden = el.dataset.page !== page; });
  root.querySelectorAll('nav [data-route]').forEach(el => {
    if (el.dataset.route === page) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current');
  });
  if (page === 'usage') void loadUsage();
  size();
}

if (!sdk?.app || !sdk?.theme || !sdk?.navigation?.get || !bridge) {
  notice('请从 Studio 应用中心打开 Higgsfield。'); $('reload').disabled = true;
} else {
  const context = sdk.context.get();
  const identity = [context.stationId, context.installationId, context.organizationId, context.userId].join(':');
  try {
    editor = createLimitsEditor(sdk, window.sessionStorage, `higgsfield-limits-v1:${identity}`);
  } catch { text('saveState', '无法读取保存恢复记录。请检查浏览器会话存储；限额修改已禁用。'); }
  document.body.dataset.theme = sdk.theme.get();
  unsubscribe.push(sdk.theme.subscribe(theme => { document.body.dataset.theme = theme; }));
  unsubscribe.push(sdk.navigation.subscribe(route));
  unsubscribe.push(sdk.context.subscribe(next => {
    if ([next.stationId, next.installationId, next.organizationId, next.userId].join(':') !== identity) dispose();
  }));
  root.addEventListener('click', async event => {
    const target = event.target.closest('button');
    if (!target || target.disabled || !visible || disposed) return;
    try {
      if (target.dataset.route) await sdk.navigation.request({target: 'app', path: `/${target.dataset.route}`});
      if (target.hasAttribute('data-video')) await sdk.capabilities.call('navigation.openApp', {appId: 'video-mcp-server', path: '/'});
    } catch (error) { failure(error, '无法打开目标页面，请确认应用已安装且当前身份有访问权限。'); }
  }, {signal: events.signal});
  $('reload').onclick = () => { void refresh(); if (!root.querySelector('[data-page="usage"]').hidden) void loadUsage(); };
  $('connect').onclick = () => { $('authorizationFlow').hidden = false; $('connect').setAttribute('aria-expanded', 'true'); $('flowTitle').focus({preventScroll: true}); size(); };
  $('copy').onclick = async () => {
    try { await navigator.clipboard.writeText($('authorizationRequest').value); text('copyState', '已复制，请交给本地 Skills 执行。'); }
    catch { $('authorizationRequest').focus(); $('authorizationRequest').select(); text('copyState', '已选中，请按 Ctrl+C 复制。'); }
  };
  $('check').onclick = async () => {
    if (!visible || disposed) return;
    $('check').disabled = true;
    try { await request(sdk, 'POST', '/authorization/check', {}); await refresh(); }
    catch (error) { failure(error, errorCode(error) === 'AUTHORIZATION_CHECK_THROTTLED' ? '检查过于频繁，请稍后再试。' : '暂时无法开始检查，请重新读取确认状态。'); }
    finally { if (!disposed) $('check').disabled = Boolean(overview?.authorization.pending_check_id); }
  };
  for (const id of ['credits', 'submissions']) $(id).addEventListener('input', () => { dirty = true; updateEditor(); }, {signal: events.signal});
  $('limitsForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (!editor || !overview || !visible || disposed || editor.snapshot().busy) return;
    const operation = editor.save(overview.limits, $('credits').value, Number($('submissions').value));
    updateEditor();
    try {
      await operation;
      if (disposed) return;
      dirty = false; text('saveState', '设置已保存。'); await refresh();
    } catch (error) {
      if (disposed) return;
      if (errorCode(error) === 'LIMITS_REVISION_CONFLICT') {
        text('saveState', '设置已被其他操作更新。已保留填写值；请核对最新版本后再保存。'); await refresh();
      } else { text('saveState', editor.snapshot().pending ? '结果尚未确认，请重试原保存请求。' : '保存未完成，请检查输入并重试。'); if (denied(error)) failure(error, ''); }
    } finally { if (!disposed) updateEditor(); }
  }, {signal: events.signal});
  $('usageFirst').onclick = () => { usageCursor = null; void loadUsage(); };
  $('usageNext').onclick = () => { usageCursor = usageNext; void loadUsage(); };
  const resize = new ResizeObserver(size); resize.observe(root);
  function dispose() {
    if (disposed) return;
    disposed = true; clearTimeout(timer); reader?.abort(); usageReader?.abort(); events.abort(); resize.disconnect();
    for (const stop of unsubscribe) stop();
    for (const id of ['reload', 'connect', 'copy', 'check', 'usageFirst', 'usageNext']) $(id).onclick = null;
    $('business').hidden = true; overview = null;
  }
  unsubscribe.push(sdk.lifecycle.subscribe(event => {
    if (event === 'dispose') { dispose(); return; }
    if (event === 'hidden') { visible = false; ++readVersion; clearTimeout(timer); reader?.abort(); usageReader?.abort(); }
    if (event === 'active') { visible = true; void refresh(); route(sdk.navigation.get()); }
  }));
  route(sdk.navigation.get()); updateEditor(); bridge.ready(); void refresh();
}
