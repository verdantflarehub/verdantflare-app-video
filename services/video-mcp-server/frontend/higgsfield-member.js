"use strict";
(() => {
  const modes = { reference_to_video: ['R2V', '参考生视频'], text_to_video: ['T2V', '文生视频'], image_to_video: ['I2V', '图生视频'], first_last_frame_to_video: ['F2V', '首尾帧视频'] };
  const errors = { quote_required: '请先获取当前参数的报价。', quote_changed: '参数或价格已变化，请重新报价。', quote_expired: '报价已过期，请重新报价。', permission_denied: '当前项目不可访问，请选择其他项目。', account_budget_exceeded: '团队可用额度或条数不足，请联系维护人员。', account_concurrency_exceeded: '当前已有任务运行，请稍后报价。', account_reconciliation_required: '渠道有待确认的提交，请先查询原任务。', insufficient_credits: '渠道可用 Credits 不足。', authorization_required: '会员授权需要维护，请联系维护人员。', invalid_request: '请检查提示词、规格和参考素材是否满足当前模式要求。', image_to_video_not_verified: '该图生组合暂不可用。', channel_unavailable: '渠道暂不可用，请稍后重试。', not_found: '未找到任务或素材。' };
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' })[c]);
  const root = document.createElement('section'); root.className = 'hf-member'; root.hidden = true;
  document.querySelector('.market-page').append(root);
  let capabilities = [], projects = [], assets = [], selected = new Map(), quote = null, inputRevision = 0;
  let epoch = 0, poll, quoteTimer, pending = null, busy = false, page = '', task = null, restored = false;
  let draft = {}, filter = { q:'', status:'all', project:'' }, urls = new Map();
  const $ = id => root.querySelector('#hf-' + id);
  const active = () => /^#(?:create|higgsfield(?:-task-[a-zA-Z0-9_-]+)?)$/.test(location.hash);
  const uuid = () => crypto.randomUUID();
  let attempt = uuid();
  const previousState = window.studioViewState;
  window.studioViewState = () => ({ ...previousState?.(), higgsfield: { draft, selected: [...selected.values()], attempt, pending, filter } });
  function save() { if (window.studioEmbedded && active()) window.studioNavigate('/dashboard' + location.hash); }
  async function request(path, body, blob = false) {
    if (!window.studioRequest) throw Error('请从 Studio 打开 Video 应用。');
    const response = await window.studioRequest('/api/higgsfield/' + path, body === undefined ? {} : { method:'POST', body:JSON.stringify(body) });
    if (!response.ok) {
      const value = await response.json().catch(() => ({}));
      const failure = serviceFailure(value);
      const error = Error(failure ? failureMessage(failure) : errors[value.error] || (response.status === 401 ? 'Studio 会话已失效，请重新登录。' : '请求未完成，请查询原任务。'));
      if (failure) { const { message: _message, ...diagnostic } = failure; Object.assign(error, diagnostic); }
      else if (typeof value.error === 'string' && Object.hasOwn(errors, value.error)) error.code = value.error;
      error.status = response.status; throw error;
    }
    return blob ? response.blob() : response.json();
  }
  function message(text, error = false) { if ($('message')) { $('message').textContent = text; $('message').dataset.error = String(error); } }
  function releaseMedia() { for (const url of urls.values()) URL.revokeObjectURL(url); urls.clear(); }
  async function media(path, target, generation) {
    try { const blob = await request(path, undefined, true); if (epoch !== generation || !target.isConnected) return;
      const url = URL.createObjectURL(blob); urls.set(path, url); target.src = url;
    } catch { if (target.isConnected) target.alt = '预览暂不可用'; }
  }
  function go(hash) { location.hash = hash; }
  function shell(title, subtitle, content) {
    root.innerHTML = `<div class="hf-head"><div>${location.hash==='#higgsfield'?'':'<a href="#higgsfield">← Higgsfield 任务</a>'}<h1 tabindex="-1">${esc(title)}</h1><p class="hf-muted">${esc(subtitle)}</p></div><div class="hf-actions"><button class="button" id="hf-reload">刷新</button><a class="button" href="#tasks">全部任务</a>${location.hash==='#create'?'':'<a class="button primary" href="#create">＋ 新建视频</a>'}</div></div><p id="hf-message" class="hf-notice" role="status" aria-live="polite"></p>${content}`;
    $('reload').onclick = () => { if (!busy) route(); };
  }
  function options(values, labels = {}) { return values.map(v => `<option value="${esc(v)}">${esc(labels[v] || v)}</option>`).join(''); }
  function capture() {
    if (!$('project')) return;
    draft = Object.fromEntries(['project','model','mode','title','prompt','duration','resolution','ratio'].map(k => [k, $(k).value]));
  }
  function invalidate() { quote = null; inputRevision++; clearTimeout(quoteTimer); if ($('submit')) $('submit').disabled = true; if ($('price')) $('price').textContent = '待报价'; if ($('quote-state')) $('quote-state').textContent = '参数变化后需重新报价。'; }
  function modelChanged() {
    const cap = capabilities.find(c => c.model === $('model').value);
    if (!cap) return;
    $('mode').innerHTML = options(cap.modes, Object.fromEntries(Object.entries(modes).map(([k,v]) => [k,v[0]])));
    $('resolution').innerHTML = options(cap.resolutions);
    $('resolution').value = cap.resolutions.includes('480p') ? '480p' : cap.resolutions[0];
    $('ratio').innerHTML = options(cap.aspect_ratios);
    $('ratio').value = cap.aspect_ratios.includes('9:16') ? '9:16' : cap.aspect_ratios[0];
    $('duration').min = cap.duration_seconds.minimum; $('duration').max = cap.duration_seconds.maximum;
    $('duration').value = Math.max(cap.duration_seconds.minimum, Math.min(cap.duration_seconds.maximum, Number(draft.duration) || cap.duration_seconds.minimum));
    selected.clear(); renderAssets(); modeChanged();
  }
  function modeChanged() {
    const mode = $('mode').value, cap = capabilities.find(c => c.model === $('model').value);
    $('mode').title = modes[mode]?.[1] || '';
    $('references').hidden = mode === 'text_to_video';
    $('ref-help').textContent = mode === 'reference_to_video' ? `最多 ${cap?.reference_limits?.images || 0} 张图片 + ${cap?.reference_limits?.videos || 0} 段视频。按选择顺序使用 Image 1、Video 1 等参考编号；用途可分别指定人物、场景或动作。` : '选择一张图片作为输入。';
    if (mode === 'text_to_video') selected.clear();
    renderAssets();
  }
  function renderAssets() {
    if (!$('assets')) return;
    $('assets').innerHTML = assets.length ? assets.map(a => `<div class="hf-asset"><label><input type="checkbox" data-asset="${esc(a.artifact_id)}" ${selected.has(a.artifact_id)?'checked':''}>${esc(a.filename)}</label><p class="hf-muted">${a.media_type.startsWith('image/')?'图片':'视频'} · ${(a.size/1024/1024).toFixed(1)} MB</p>${selected.has(a.artifact_id)?`<label>参考用途<select data-purpose="${esc(a.artifact_id)}">${options(a.media_type.startsWith('image/')?['identity','scene','style']:['motion','camera'], {identity:'人物外观',scene:'场景',style:'风格',motion:'动作',camera:'镜头'})}</select></label>`:''}</div>`).join('') : '<p class="hf-muted">当前项目暂无已导入的参考素材，可从本机上传。</p>';
    for (const a of selected.values()) { const field = root.querySelector(`[data-purpose="${CSS.escape(a.artifact_id)}"]`); if (field) field.value = a.purpose; }
    const counts = { images:0, videos:0 };
    for (const a of selected.values()) {
      const input = root.querySelector(`[data-asset="${CSS.escape(a.artifact_id)}"]`);
      if (input) { const label = document.createElement('strong'); label.textContent = `${a.kind === 'images' ? 'Image' : 'Video'} ${++counts[a.kind]} · `; input.after(label); }
    }
    for (const card of $('assets').querySelectorAll('.hf-asset')) {
      const id = card.querySelector('[data-asset]').dataset.asset;
      const asset = assets.find(a => a.artifact_id === id);
      if (asset?.media_type.startsWith('image/')) {
        const img = document.createElement('img'); img.alt = asset.filename; card.prepend(img);
        const path = 'assets/' + id + '/content?project_id=' + encodeURIComponent($('project').value);
        if (urls.has(path)) img.src = urls.get(path); else media(path, img, epoch);
      }
    }
    $('assets').onchange = event => {
      const el = event.target;
      if (pending) { renderAssets(); return; }
      if (el.dataset.asset) {
        const a = assets.find(a => a.artifact_id === el.dataset.asset);
        if (el.checked) {
          const cap = capabilities.find(c => c.model === $('model').value), kind = a.media_type.startsWith('image/')?'images':'videos';
          const count = [...selected.values()].filter(a => a.kind === kind).length;
          const limit = $('mode').value === 'image_to_video' ? (kind==='images'?1:0) : cap.reference_limits?.[kind] || 0;
          if (count >= limit) { message('已达到当前模式的参考素材数量上限。', true); el.checked = false; return; }
          selected.set(a.artifact_id, { artifact_id:a.artifact_id, kind, purpose:kind==='images'?'identity':'motion' });
        } else selected.delete(a.artifact_id);
        renderAssets();
      } else if (el.dataset.purpose) selected.get(el.dataset.purpose).purpose = el.value;
      invalidate(); capture(); save();
    };
  }
  async function loadAssets(generation) {
    const project = $('project')?.value; if (!project) return;
    const value = await request('assets?project_id='+encodeURIComponent(project));
    if (epoch !== generation || $('project')?.value !== project) return;
    assets = value.items; renderAssets();
  }
  function body() {
    capture(); const references = {images:[],videos:[],audios:[]};
    for (const a of selected.values()) references[a.kind].push({artifact_id:a.artifact_id,purpose:draft.mode==='image_to_video'?'first_frame':a.purpose});
    if (!draft.project || !draft.model || !draft.prompt.trim()) throw Error('请选择项目、模型并填写提示词。');
    if (capabilities.find(c=>c.model===draft.model)?.can_submit !== true) throw Error('当前注册能力不可提交，请刷新模型状态。');
    if (draft.mode !== 'text_to_video' && !selected.size) throw Error('请至少选择一项参考素材。');
    return {project_id:draft.project,idempotency_key:attempt,title:draft.title||null,route:'higgsfield',model:draft.model,mode:draft.mode,prompt:draft.prompt,duration_seconds:Number(draft.duration),aspect_ratio:draft.ratio,resolution:draft.resolution,references};
  }
  function lockInputs(locked) { root.querySelectorAll('#hf-inputs input, #hf-inputs select, #hf-inputs textarea, #hf-upload').forEach(el => el.disabled = locked); }
  async function createPage(generation) {
    shell('新建视频', '模型 × 渠道 · 项目素材与会员额度统一使用', '<p id="hf-loading">正在读取可用项目与模型…</p>');
    const [ps, cs] = await Promise.all([request('projects'), request('capabilities')]);
    if (epoch !== generation) return;
    projects = ps.items; capabilities = cs.routes;
    const names = Object.fromEntries(capabilities.map(c => [c.model, c.display_name || '名称未登记']));
    const keepSelected = [...selected.values()];
    shell('新建视频', '模型 × 渠道 · 项目素材与会员额度统一使用', `<div class="hf-layout"><div id="hf-inputs"><section class="hf-panel"><h2>项目与模型</h2><div class="hf-fields"><label>项目<select id="hf-project"><option value="">请选择项目</option>${projects.map(p=>`<option value="${esc(p.project_id)}">${esc(p.name)}</option>`).join('')}</select></label><label>调用渠道<select disabled><option>Higgsfield</option></select></label><label>模型<select id="hf-model"><option value="">请选择模型</option>${capabilities.map(c=>`<option value="${esc(c.model)}" ${c.can_submit===true?'':'disabled'}>${esc(names[c.model])}${c.can_submit===true?'':' · 当前不可提交'}</option>`).join('')}</select></label><label>生成模式<select id="hf-mode"></select></label><label class="hf-wide">视频名称（可选）<input id="hf-title" maxlength="64" placeholder="为本次视频命名"></label></div><button class="button" id="hf-more-projects" ${ps.next_cursor?'':'hidden'}>加载更多项目</button></section><section class="hf-panel"><h2>输入内容</h2><label>提示词<textarea id="hf-prompt" maxlength="10000" placeholder="描述人物、动作、场景和镜头；说明每项参考素材的作用。"></textarea></label><div id="hf-references"><h3>参考素材</h3><p id="hf-ref-help" class="hf-muted"></p><div id="hf-assets" class="hf-assets"></div><label class="hf-actions">上传图片或视频<input id="hf-upload" type="file" accept="image/png,image/jpeg,image/webp,video/mp4" multiple></label><p class="hf-muted">图片最多 10 MB，视频最多 128 MB。上传中断后重新选择同一文件可续传。</p><p id="hf-upload-state" role="status"></p></div></section><section class="hf-panel"><h2>视频规格</h2><div class="hf-fields"><label>时长（秒）<input id="hf-duration" type="number" step="1"></label><label>分辨率<select id="hf-resolution"></select></label><label>画幅<select id="hf-ratio"></select></label></div></section></div><aside class="hf-panel"><h2>费用与可用性</h2><p id="hf-identity"></p><p class="hf-price"><span id="hf-price">待报价</span> <small>Credits</small></p><p id="hf-quote-state" class="hf-muted">报价有效期 60 秒，提交时再次核验。</p><div class="hf-actions"><button class="button" id="hf-quote">获取报价</button><button class="button primary" id="hf-submit" disabled>提交视频</button><button class="button" id="hf-recover" hidden>查询原提交</button></div><p class="hf-muted">预计消费不等于实际扣费。CLI 使用 Credits；网页 Unlimited 排队权益不适用于此入口。</p></aside></div>`);
    for (const key of ['project','model','title','prompt']) if (draft[key]) $(key).value = draft[key];
    if ($('model').value) { modelChanged(); for (const key of ['mode','duration','resolution','ratio']) if (draft[key]) $(key).value=draft[key]; modeChanged(); }
    selected = new Map(keepSelected.map(a=>[a.artifact_id,a]));
    const updateIdentity = () => { $('identity').textContent = `${names[$('model').value] || '请选择模型'} × Higgsfield`; };
    updateIdentity();
    $('inputs').addEventListener('input', event => { if (event.target.id !== 'hf-upload') { invalidate(); capture(); updateIdentity(); save(); } });
    $('model').onchange = () => { modelChanged(); capture(); updateIdentity(); save(); };
    $('mode').onchange = () => { modeChanged(); capture(); save(); };
    $('project').onchange = () => { selected.clear(); assets=[]; renderAssets(); capture(); save(); loadAssets(generation).catch(e=>message(e.message,true)); };
    let cursor=ps.next_cursor;
    $('more-projects').onclick = async () => { try { const p=await request('projects?cursor='+encodeURIComponent(cursor)); if(epoch!==generation)return; projects.push(...p.items); for(const item of p.items)$('project').add(new Option(item.name,item.project_id)); cursor=p.next_cursor; $('more-projects').hidden=!cursor; } catch(e){message(e.message,true);} };
    $('upload').onchange = () => upload([...$('upload').files], generation);
    $('quote').onclick = async () => {
      if (busy || pending) return; busy=true; $('quote').disabled=true;
      const revision=inputRevision;
      try { const value = await request('preflight', body()); if (epoch!==generation || revision!==inputRevision) return; quote=value;
        $('price').textContent=value.quoted_credits; $('quote-state').textContent=value.can_submit?'当前允许生成 · '+new Date(value.expires_at).toLocaleTimeString()+' 前有效':errors[value.blocked_reason]||'当前不可生成';
        $('submit').disabled=!value.can_submit; message(value.can_submit?'报价已更新。':'当前请求不可提交。',!value.can_submit);
        quoteTimer=setTimeout(()=>{invalidate(); message('报价已过期，请重新报价。');},Math.max(0,Date.parse(value.expires_at)-Date.now()));
      } catch(e){message(e.message,true);} finally {busy=false;if($('quote'))$('quote').disabled=!!pending;}
    };
    $('submit').onclick = async () => {
      if(busy||pending||!quote?.can_submit||Date.parse(quote.expires_at)<=Date.now())return;
      let submission; try{submission=body();}catch(e){message(e.message,true);return;}
      pending=submission; busy=true; lockInputs(true); $('submit').disabled=true; $('quote').disabled=true; save();
      try { const result=await request('tasks',submission); if(epoch!==generation)return; pending=null;attempt=uuid();save();go('#higgsfield-task-'+result.video_task_id); }
      catch(e){ if(epoch!==generation)return; if(e.status>=400&&e.status<500){pending=null;lockInputs(false);invalidate();$('quote').disabled=false;message(e.message,true);}else{message('提交结果待确认。请查询原提交，避免重复消费。',true);$('recover').hidden=false;} save(); }
      finally{busy=false;}
    };
    $('recover').onclick=()=>recover(generation);
    if(pending){lockInputs(true);$('recover').hidden=false;$('quote').disabled=true;message('上次提交待确认，请查询原提交。');}
    if(!projects.length)message('暂无可写项目，请先在项目管理中创建项目或获得项目访问权限。');
    await loadAssets(generation);
  }
  async function recover(generation) {
    if(!pending||busy)return;busy=true;
    try{const value=await request('attempt',{project_id:pending.project_id,idempotency_key:pending.idempotency_key});if(epoch!==generation)return;
      if(value.state==='found'){pending=null;attempt=uuid();save();go('#higgsfield-task-'+value.task.video_task_id);}
      else message('尚未查到持久化任务。保留本次提交，请稍后继续查询或联系维护人员；不会自动重发。');
    }catch(e){message(e.message,true);}finally{busy=false;}
  }
  async function sha(bytes) { return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(b=>b.toString(16).padStart(2,'0')).join(''); }
  async function upload(files,generation) {
    const project=$('project').value;if(!project){message('请先选择项目。',true);return;} if(busy||pending)return;
    busy=true;lockInputs(true);invalidate();$('quote').disabled=true;
    try{for(const file of files){
      if(file.size> (file.type.startsWith('image/')?10:128)*1024*1024)throw Error('素材超过大小限制。');
      const bytes=await file.arrayBuffer(),digest=await sha(bytes);
      let state=await request('imports/prepare',{project_id:project,idempotency_key:'web/'+digest,filename:file.name,size:file.size,sha256:digest,purpose:'video_reference'});
      state=await request('imports/status',{project_id:project,import_id:state.import_id});
      while(state.offset<file.size){if(epoch!==generation) return;
        const part=bytes.slice(state.offset,state.offset+state.chunk_max_bytes);let binary='';for(const byte of new Uint8Array(part))binary+=String.fromCharCode(byte);
        $('upload-state').textContent=`${file.name} · ${Math.floor(state.offset/file.size*100)}%`;
        state=await request('imports/chunk',{project_id:project,import_id:state.import_id,offset:state.offset,content_base64:btoa(binary),sha256:await sha(part)});
      }
      await request('imports/commit',{project_id:project,import_id:state.import_id});
      if(epoch!==generation)return;$('upload-state').textContent=file.name+' · 已导入，请在素材列表中选择';
    }await loadAssets(generation);
    }catch(e){if(epoch===generation)message(e.message+' 上传中断时重新选择原文件可续传。',true);}
    finally{busy=false;if(epoch===generation){lockInputs(!!pending);$('quote').disabled=!!pending;}}
  }
  async function listPage(generation) {
    shell('Higgsfield 任务', '查看有权访问的项目视频，恢复原任务或下载结果', `<section class="hf-panel"><div class="hf-fields"><label>搜索名称或提示词<input id="hf-search" type="search"></label><label>任务状态<select id="hf-status">${options(['all','queued','running','succeeded','failed'],{all:'全部状态',queued:'排队中',running:'处理中',succeeded:'已完成',failed:'失败'})}</select></label></div></section><div id="hf-cards" class="hf-cards"></div>`);
    const projectLabel=document.createElement('label');projectLabel.textContent='项目';
    const projectSelect=document.createElement('select');projectSelect.id='hf-filter-project';projectSelect.add(new Option('全部项目',''));projectLabel.append(projectSelect);$('search').closest('.hf-fields').append(projectLabel);
    const channelLabel=document.createElement('label');channelLabel.textContent='渠道';const channelSelect=document.createElement('select');channelSelect.disabled=true;channelSelect.add(new Option('Higgsfield'));channelLabel.append(channelSelect);projectLabel.after(channelLabel);
    let projectCursor='';do { const result=await request('projects'+(projectCursor?'?cursor='+encodeURIComponent(projectCursor):''));if(epoch!==generation)return;for(const p of result.items)projectSelect.add(new Option(p.name,p.project_id));projectCursor=result.next_cursor; } while(projectCursor);
    $('search').value=filter.q;$('status').value=filter.status;projectSelect.value=filter.project||'';let snapshot=[];
    const render=()=>{ releaseMedia();const tasks=snapshot.filter(t=>(!filter.project||t.project_id===filter.project)&&(filter.status==='all'||t.status===filter.status)&&`${t.title} ${t.prompt}`.toLowerCase().includes(filter.q.toLowerCase()));
      $('cards').innerHTML=tasks.length?tasks.map(t=>`<a class="hf-card" href="#higgsfield-task-${esc(t.video_task_id)}">${t.status==='succeeded'?`<img data-task-preview="${esc(t.video_task_id)}" alt="视频预览">`:''}<p>${esc(taskStatusLabel(t))}</p><h3>${esc(t.title||'视频任务')}</h3>${taskIdentity(t)}<p>${esc(t.resolution||'')} · ${esc(t.duration_seconds)}s · ${esc(t.aspect_ratio||'')}</p><p title="${esc(generationTimeHint)}">${esc(elapsed(t))}</p>${generationBadge(t)}</a>`).join(''):'<p class="hf-muted">暂无匹配任务。可以创建视频或调整筛选条件。</p>';
      root.querySelectorAll('[data-task-preview]').forEach(el=>media('tasks/'+el.dataset.taskPreview+'/thumbnail',el,generation));save();
    };
    $('search').oninput=()=>{filter.q=$('search').value;render();};$('status').onchange=()=>{filter.status=$('status').value;render();};projectSelect.onchange=()=>{filter.project=projectSelect.value;render();};
    const refresh=async()=>{if(epoch!==generation||document.hidden)return;try{const value=await request('tasks');if(epoch!==generation)return;snapshot=value.tasks;render();message('任务状态已更新。');}catch(e){if(epoch===generation)message(e.message+' 当前保留上次快照。',true);}finally{if(epoch===generation)poll=setTimeout(refresh,8000);}};
    await refresh();
  }
  async function detailPage(id,generation) {
    shell('视频任务','正在读取原任务…','');
    const refresh=async()=>{if(epoch!==generation||document.hidden)return;
      try{const value=await request('tasks/'+id);if(epoch!==generation)return;task=value;
        const existingVideo=$('video');const oldURL=existingVideo?.src;
        const facts=[['运行时间',elapsed(value)],['渠道确认完成',date(value.timing?.generation_completed_at)],['计时说明',generationTimeHint],['模型与渠道',(value.presentation?.model_label||'未记录模型')+' × '+(value.presentation?.channel_label||'Higgsfield')],['请求规格',`${value.duration_seconds}s / ${value.resolution} / ${value.aspect_ratio}`],['预计消费',value.billing?.quoted_credits!=null?value.billing.quoted_credits+' Credits':'未记录'],['实际扣费','待对账'],['实际尺寸',value.media?.width?value.media.width+' × '+value.media.height:'尚无结果'],['实际时长',value.media?.duration_ms!=null?value.media.duration_ms/1000+' 秒':'尚无结果'],['帧率',value.media?.frame_rate!=null?value.media.frame_rate+' FPS':'尚无结果']];
        shell(value.title||'视频任务',taskStatusLabel(value), `<section class="hf-panel"><h2>生成结果</h2>${value.status==='succeeded'?'<video id="hf-video" controls playsinline preload="metadata"></video><div class="hf-actions"><button class="button primary" id="hf-download">下载视频</button></div>':`<p>${value.runtime_stage==='submission_unknown'?'供应商提交状态待确认，请恢复原任务；不会自动重新生成。':value.status==='failed'?'本次生成未完成，任务和费用记录已保留。':value.runtime_stage==='collecting'?'视频已生成，可查询原任务获取结果。':'任务正在排队或生成，可以离开页面后再回来。'}</p><button class="button" id="hf-result">查询原任务</button>`}</section><section class="hf-panel"><h2>提示词</h2><p class="hf-prompt">${esc(value.prompt)}</p></section><section class="hf-panel"><h2>参考素材</h2>${value.references?.length?value.references.map(r=>`<p>${esc(r.filename||'素材暂不可用')} · ${esc(({identity:'人物外观',scene:'场景',motion:'动作',camera:'镜头',style:'风格'})[r.purpose]||r.purpose)}</p>`).join(''):'<p class="hf-muted">无参考素材</p>'}</section><section class="hf-panel"><h2>生成与费用</h2><dl class="hf-facts">${facts.map(([k,v])=>`<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl><details><summary>任务与归档信息</summary><p>${esc(id)}</p><p>${esc(value.media?.sha256||'尚无固定结果摘要')}</p></details></section>`);
        const diagnostic = serviceFailure({error:value.error});
        const failure = diagnostic ? failureMessage(diagnostic) : value.error?.code === 'media_contract_mismatch' ? '供应商输出未达到请求规格，任务和费用记录已保留。' : errors[value.error?.code] || '可保留原任务信息供排查，不会自动重提。';
        message(value.status==='failed'?'任务失败。'+failure:'任务状态已更新。',value.status==='failed');
        if($('video')){if(oldURL)$('video').src=oldURL;else await media('tasks/'+id+'/content',$('video'),generation);
          $('download').onclick=async()=>{if(busy)return;busy=true;try{const blob=await request('tasks/'+id+'/content',undefined,true);if(epoch!==generation)return;const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=(value.title||'video')+'.mp4';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){if(epoch===generation){if([401,403,404].includes(e.status)){releaseMedia();$('video')?.remove();$('download')?.remove();}message(e.message,true);}}finally{busy=false;}};
        }
        if($('result'))$('result').onclick=async()=>{if(busy)return;busy=true;try{await request('tasks/'+id+'/result',{});await refresh();}catch(e){message(e.message,true);}finally{busy=false;}};
      }catch(e){if(epoch===generation){if([401,403,404].includes(e.status)){releaseMedia();shell('视频任务','当前任务不可访问','');task=null;}message(e.message,true);}}
      finally{clearTimeout(poll);if(epoch===generation&&task&&!['succeeded','failed','cancelled'].includes(task.status))poll=setTimeout(refresh,8000);}
    };await refresh();
  }
  async function route() {
    const next=location.hash;const generation=++epoch;clearTimeout(poll);clearTimeout(quoteTimer);releaseMedia();quote=null;task=null;
    root.hidden=!active();document.querySelector('.business-intro').hidden=active();
    document.documentElement.dataset.hfMember=String(active());
    if(!active()){page='';if(authorized)refresh();return;}
    document.querySelectorAll('[data-business-view],#taskDetail').forEach(el=>el.hidden=true);
    page=next;save();
    try{if(next==='#create')await createPage(generation);else if(next==='#higgsfield')await listPage(generation);else await detailPage(next.slice('#higgsfield-task-'.length),generation);}catch(e){if(epoch===generation){$('loading')?.remove();message(e.message+' 可点击刷新重新读取。',true);}}
  }
  window.addEventListener('studio-restore',event=>{if(!restored&&event.detail?.higgsfield){const value=event.detail.higgsfield;draft=value.draft||{};attempt=value.attempt||uuid();pending=value.pending||null;filter=value.filter||filter;selected=new Map((value.selected||[]).map(a=>[a.artifact_id,a]));}restored=true;route();});
  window.addEventListener('hashchange',route);
  window.addEventListener('DOMContentLoaded',()=>{if(!window.studioEmbedded)route();});
  document.addEventListener('visibilitychange',()=>{clearTimeout(poll);if(!document.hidden&&active()&&page!=='#create')route();});
  window.addEventListener('pagehide',()=>{epoch++;clearTimeout(poll);clearTimeout(quoteTimer);releaseMedia();});
})();
