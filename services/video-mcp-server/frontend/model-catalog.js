/* Shared catalog view for the standalone dashboard and Studio embed. */
(() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const business = {video_generation:'视频生成',video_editing:'视频编辑',motion_control:'动作控制',video_upscale:'视频超分',frame_interpolation:'视频补帧',depth_estimation:'深度估计',video_processing:'视频处理'};
  const modes = {text_to_video:'文生视频',image_to_video:'图生视频',first_last_frame_to_video:'首尾帧视频',reference_to_video:'参考生视频',...business};
  const review = {pending:'待核实',approved:'资料已审核',rejected:'资料未通过'};
  const verification = {unverified:'未验证',verified:'技术已验证',failed:'技术验证失败'};
  const audio = {supported:'支持',unsupported:'不支持',unknown:'待核实'};
  const billing = {credits:'积分',unlimited:'会员权益',local_resources:'本地资源',provider_billed:'供应商计费'};
  const channelLogos = {
    local: '<svg class="channel-logo" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4"/></svg>',
    higgsfield: '<svg class="channel-logo" width="16" height="16" viewBox="0 0 1355 1355" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><rect width="1355" height="1355" rx="363" fill="#D1FE17"/><path d="M1088.73 670.726L1087.95 662.156C1080.52 579.906 1026.99 425.16 878.52 425.16C768.341 425.16 685.111 533.914 611.658 629.796C553.046 706.582 502.264 772.452 446.385 772.452C431.533 770.898 412.391 763.49 400.676 746.73C390.125 731.522 387.392 712.035 392.858 688.649C401.448 651.619 450.686 617.315 502.644 580.677C531.161 561.19 560.472 540.529 580.79 520.651C639.402 464.134 669.094 423.205 669.094 357.336C669.094 291.467 632.76 258.716 602.283 244.682C541.331 216.625 451.862 232.994 394.818 282.114C386.227 289.913 377.625 297.31 369.807 304.328C312.371 355.392 273.696 390.086 185 363.571V470.369C302.605 522.216 401.46 423.205 438.97 377.605C467.878 347.592 498.355 330.049 521.024 330.049H522.2C532.359 330.441 540.95 334.34 547.211 341.357C557.37 353.056 561.279 366.699 559.319 381.896C555.018 413.864 521.808 451.273 460.856 492.203C389.352 540.149 269.799 620.444 260.413 721.399C253.379 793.905 290.889 866.4 349.501 894.458C486.247 959.165 569.477 847.684 657.77 729.969C725.365 639.149 789.443 553.01 878.531 553.01C958.625 553.01 988.318 619.271 988.318 660.982V669.173L980.5 670.726C786.318 705.029 680.429 886.67 680.429 970.473C680.429 1054.28 751.541 1126 839.061 1126C941.433 1126 1068.02 1038.69 1088.34 793.123L1089.12 784.162H1170V670.737H1088.73V670.726ZM982.841 796.62C967.216 943.566 891.804 1012.17 846.095 1012.17C825.385 1012.17 796.477 995.021 796.477 963.064C796.477 927.208 850.003 818.454 970.341 786.106L984.409 782.597L982.841 796.632V796.62Z" fill="black"/></svg>',
    fal: '<svg class="channel-logo" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><rect width="24" height="24" rx="6" fill="#6C3AED"/><path d="M8.5 7H15.5M8.5 12H13.5M8.5 7V17" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="15.5" cy="15.5" r="1.8" fill="#4CE39A"/></svg>'
  };
  const channelBillingSummary = {
    local: '本地算力队列 · 免外部积分',
    higgsfield: '会员权益 / Credits 计费',
    fal: '供应商 API 按量计费 (FAL_KEY)'
  };
  const modelPresets = {
    'minimax-h3': { caps: [['R2V','参考生视频'],['I2V','图生视频']], quota: '最多 9张图 · 3个视频 · 3个音频', maxSize: '≤ 50MB', ratios: '9:16 / 16:9 / 1:1', audioDesc: '支持参考音轨驱动与有声视频输出', res: '768p', dur: '4S-15S', fps: '24FPS' },
    'seedvr2': { caps: [['V2V','视频超分']], quota: '源视频流 1个（支持音轨透传）', maxSize: '≤ 50MB', ratios: '保持源视频画幅', audioDesc: '保留原视频音轨或无声输出', res: '1080p', dur: '源视频时长', fps: '60FPS' },
    'rife': { caps: [['V2V','视频插帧']], quota: '源视频流 1个（2x / 3x / 4x 帧率倍增）', maxSize: '≤ 50MB', ratios: '保持源视频画幅', audioDesc: '保留原视频音轨或无声输出', res: '1080p', dur: '源视频时长', fps: '60FPS' },
    'video-depth-anything': { caps: [['V2V','深度估计']], quota: '源视频流 1个（提取时序深度图）', maxSize: '≤ 50MB', ratios: '保持源视频画幅', audioDesc: '无声深度视频输出', res: '1080p', dur: '源视频时长', fps: '30FPS' },
    'h3-latent-upscaler': { caps: [['V2V','潜空间超分']], quota: 'H3 潜空间任务 / 视频源 1个', maxSize: '≤ 50MB', ratios: '保持源视频画幅', audioDesc: '保留原视频音轨', res: '1080p', dur: '4S-15S', fps: '24FPS' },
    'seedance-2-5': { caps: [['T2V','文生视频'],['I2V','图生视频'],['R2V','参考生视频']], quota: '最多 9张图 · 3个视频 · 1个音频', maxSize: '≤ 50MB', ratios: '16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9', audioDesc: '支持音频参考与有声视频输出', res: '720p', dur: '4S-15S', fps: '24FPS' },
    'seedance-2-5-fast': { caps: [['T2V','文生视频'],['I2V','图生视频'],['R2V','参考生视频']], quota: '最多 9张图 · 3个视频 · 1个音频', maxSize: '≤ 50MB', ratios: '16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9', audioDesc: '支持音频参考与有声视频输出', res: '720p', dur: '4S-15S', fps: '24FPS' },
    'seedance-2-5-draft': { caps: [['R2V','参考生视频']], quota: '最多 9张图 · 3个视频 · 1个音频', maxSize: '≤ 50MB', ratios: '16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9', audioDesc: '支持音频参考与有声预览输出', res: '480p', dur: '4S-15S', fps: '24FPS' },
    'wan-3-0': { caps: [['T2V','文生视频'],['I2V','图生视频'],['F2V','首尾帧视频'],['R2V','参考生视频']], quota: '最多 4张图 · 1个视频 · 1个音频', maxSize: '≤ 50MB', ratios: '16:9 / 9:16 / 1:1 / 4:3 / 3:4', audioDesc: '支持原生音画同步与参考音轨', res: '720p', dur: '3S-15S', fps: '24FPS' },
    'grok-imagine-1-5-lite': { caps: [['T2V','文生视频'],['I2V','图生视频']], quota: '最多 1张参考图 · 0个视频 · 0个音频', maxSize: '≤ 20MB', ratios: '16:9 / 9:16 / 1:1 / 4:3 / 3:4', audioDesc: '支持原生音效输出', res: '720p', dur: '2S-10S', fps: '24FPS' },
    'genjutsu-1-0': { caps: [['T2V','文生视频'],['I2V','图生视频'],['R2V','参考生视频']], quota: '最多 4张图 · 1个视频 · 1个音频', maxSize: '≤ 50MB', ratios: '16:9 / 9:16 / 1:1', audioDesc: '支持原生有声视频输出', res: '1080p', dur: '5S-10S', fps: '24FPS' },
    'kling-3-0': { caps: [['T2V','文生视频'],['I2V','图生视频'],['F2V','首尾帧视频']], quota: '首帧 + 尾帧 (最多 2张图 · 0个视频)', maxSize: '≤ 30MB', ratios: '16:9 / 9:16 / 1:1', audioDesc: '支持原生音效生成', res: '4K', dur: '5S-10S', fps: '30FPS' },
    'kling-3-0-omni': { caps: [['T2V','文生视频'],['I2V','图生视频'],['F2V','首尾帧视频'],['R2V','参考生视频']], quota: '最多 7张图 · 1个视频 · 1个音频', maxSize: '≤ 50MB', ratios: '16:9 / 9:16 / 1:1', audioDesc: '支持多模态参考与有声输出', res: '4K', dur: '5S-10S', fps: '30FPS' },
    'sora-2': { caps: [['T2V','文生视频'],['I2V','图生视频']], quota: '最多 1张首帧图 · 0个视频 · 0个音频', maxSize: '≤ 30MB', ratios: '16:9 / 9:16 / 1:1', audioDesc: '支持原生环境音输出', res: '1080p', dur: '5S-20S', fps: '24FPS' },
    'veo-3-1': { caps: [['T2V','文生视频'],['I2V','图生视频'],['F2V','首尾帧视频'],['R2V','参考生视频']], quota: '最多 3张图 · 1个视频 · 1个音频', maxSize: '≤ 50MB', ratios: '16:9 / 9:16', audioDesc: '支持原生对白与环境音同步输出', res: '4K', dur: '4S-8S', fps: '24FPS' },
    'h3-reference-to-video': { caps: [['R2V','参考生视频'],['I2V','图生视频']], quota: '最多 9张图 · 3个视频 · 3个音频 (合计 ≤ 12)', maxSize: '≤ 50MB', ratios: 'adaptive / 16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9', audioDesc: '支持参考音频输入与有声输出', res: '480p', dur: '5S-15S', fps: '24FPS' },
    'h3-text-to-video': { caps: [['T2V','文生视频']], quota: '纯文本提示词驱动（0张图 · 0个视频 · 0个音频）', maxSize: '纯文本（无附件上限）', ratios: '16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9', audioDesc: '支持有声视频输出', res: '768p', dur: '5S-15S', fps: '24FPS' },
    'h3-image-to-video': { caps: [['I2V','图生视频']], quota: '最多 1张首帧图 · 0个视频 · 0个音频', maxSize: '≤ 20MB', ratios: '继承首帧图 / 16:9 / 9:16 / 1:1', audioDesc: '支持有声视频输出', res: '768p', dur: '5S-15S', fps: '24FPS' },
    'h3-fast-image-to-video': { caps: [['I2V','图生视频']], quota: '最多 1张首帧图 · 0个视频 · 0个音频', maxSize: '≤ 20MB', ratios: '继承首帧图 / 16:9 / 9:16 / 1:1', audioDesc: '支持快速图生预览', res: '512p', dur: '5S-10S', fps: '24FPS' }
  };
  const modeToCap = {
    text_to_video: ['T2V','文生视频'],
    image_to_video: ['I2V','图生视频'],
    first_last_frame_to_video: ['F2V','首尾帧视频'],
    reference_to_video: ['R2V','参考生视频'],
    video_upscale: ['V2V','视频超分'],
    frame_interpolation: ['V2V','视频插帧'],
    depth_estimation: ['V2V','深度估计'],
    video_editing: ['V2V','视频编辑'],
    motion_control: ['V2V','动作控制'],
    video_processing: ['V2V','潜空间超分']
  };
  const deriveSpecs = model => {
    const preset = modelPresets[model.id] || modelPresets[model.model_id] || {};
    const allProfiles = [...(model.team_profiles || []), ...(model.supplier_profiles || [])];
    const primaryProfiles = (model.supplier_profiles?.some(p => p.max_resolution) ? model.supplier_profiles : model.team_profiles) || [];
    const maxRes = [...new Set(primaryProfiles.map(p => p.max_resolution).filter(Boolean))].join(' / ') || preset.res || '768p';
    let dur = preset.dur || '';
    for (const p of allProfiles) {
      if (p.duration_seconds) { dur = `${p.duration_seconds.minimum}S-${p.duration_seconds.maximum}S`; break; }
      if (p.duration_choices_seconds?.length) {
        const min = Math.min(...p.duration_choices_seconds), max = Math.max(...p.duration_choices_seconds);
        dur = min === max ? `${min}S` : `${min}S-${max}S`;
        break;
      }
    }
    if (!dur) dur = '4S-15S';
    const fps = preset.fps || '24FPS';
    const caps = [];
    const seen = new Set();
    for (const p of allProfiles) {
      const pair = modeToCap[p.mode];
      if (pair && !seen.has(pair[0] + pair[1])) { seen.add(pair[0] + pair[1]); caps.push(pair); }
    }
    if (!caps.length && preset.caps) caps.push(...preset.caps);
    if (!caps.length) {
      const fallback = modeToCap[model.business_type] || ['T2V','文生视频'];
      caps.push(fallback);
    }
    let quota = preset.quota || '';
    if (!quota) {
      const lim = allProfiles.map(p => p.input_limits).find(Boolean);
      if (lim) {
        quota = (lim.images === 0 && lim.videos === 0 && lim.audio_files === 0)
          ? '纯文本提示词驱动（0张图 · 0个视频 · 0个音频）'
          : `最多 ${lim.images}张图 · ${lim.videos}个视频 · ${lim.audio_files}个音频`;
      } else {
        quota = '最多 9张图 · 3个视频 · 1个音频';
      }
    }
    const maxSize = preset.maxSize || '≤ 50MB';
    const ratios = [...new Set(allProfiles.flatMap(p => p.aspect_ratios || []))].join(' / ') || preset.ratios || '16:9 / 9:16 / 1:1';
    const audioDesc = preset.audioDesc || '支持原生音画同步输出';
    return { maxRes, dur, fps, caps, quota, maxSize, ratios, audioDesc };
  };
  function profiles(items, label) {
    if (!items.length) return `<p class="business-muted">${label}：规格待核实</p>`;
    return `<h5>${label}</h5>${items.map(p => {
      const duration = p.duration_seconds ? `${p.duration_seconds.minimum}–${p.duration_seconds.maximum} 秒` : p.duration_choices_seconds.length ? p.duration_choices_seconds.join(' / ')+' 秒' : '待核实';
      const limits = p.input_limits;
      return `<div class="catalog-profile"><strong>${esc(modes[p.mode] || p.mode)}</strong><dl class="catalog-meta"><dt>画幅</dt><dd>${esc(p.aspect_ratios.join(' · ') || '待核实')}</dd><dt>参考输入</dt><dd>${limits ? `图像 ${limits.images} · 视频 ${limits.videos} · 音频 ${limits.audio_files}` : '待核实'}</dd><dt>音频输入 / 输出</dt><dd>${audio[p.audio_input]} / ${audio[p.audio_output]}</dd><dt>资料 / 技术验证</dt><dd>${review[p.review_status]} · ${verification[p.verification_status]}</dd></dl></div>`;
    }).join('')}`;
  }
  class VideoCatalogView {
    constructor(root, fetchCatalog) {
      this.root = root; this.fetchCatalog = fetchCatalog; this.data = null;
      this.filters = {query:'',business:'all',channel:'all',view:'channels'};
      this.open = new Set(); this.generation = 0;
      this.onInput = event => {
        const field = event.target.dataset.catalogFilter;
        if (!field) return;
        this.filters[field] = event.target.value;
        this.renderResults();
      };
      this.onClick = event => { if (event.target.closest('[data-catalog-retry]')) this.load(); };
      this.onToggle = event => {
        const id = event.target.dataset.catalogEntry;
        if (id) event.target.open ? this.open.add(id) : this.open.delete(id);
      };
      root.addEventListener('input', this.onInput);
      root.addEventListener('click', this.onClick);
      root.addEventListener('toggle', this.onToggle, true);
      this.clear();
    }
    clear() {
      this.generation++; this.data = null; this.open.clear();
      this.root.innerHTML = '<p class="empty">连接后读取模型目录。</p>';
    }
    destroy() {
      this.clear();
      this.root.removeEventListener('input', this.onInput);
      this.root.removeEventListener('click', this.onClick);
      this.root.removeEventListener('toggle', this.onToggle, true);
    }
    failure() { this.root.innerHTML = '<div class="empty" role="alert">模型目录暂时不可用。<button class="button" data-catalog-retry>重试</button></div>'; }
    async load() {
      const generation = ++this.generation;
      if (!this.data) this.root.innerHTML = '<p class="empty" role="status">正在读取模型目录…</p>';
      try { const data = await this.fetchCatalog(); if (generation === this.generation) this.set(data); }
      catch { if (generation === this.generation) { this.data = null; this.failure(); } }
    }
    set(data) {
      if (data.state === 'error') { this.data = null; this.failure(); return; }
      if (this.data?.snapshot_id === data.snapshot_id &&
          JSON.stringify(this.data.execution || {}) === JSON.stringify(data.execution || {})) return;
      this.data = data;
      const option = (value,label) => `<option value="${esc(value)}">${esc(label)}</option>`;
      this.root.innerHTML = `<div class="catalog-toolbar"><label>查看方式<select data-catalog-filter="view">${option('channels','按渠道查看')}${option('models','按模型查看')}</select></label><label class="catalog-search">搜索模型<input type="search" data-catalog-filter="query" placeholder="模型名称、用途或模型 ID" /></label><label>业务<select data-catalog-filter="business">${option('all','全部业务')}${Object.entries(business).map(([id,label])=>option(id,label)).join('')}</select></label><label>渠道<select data-catalog-filter="channel">${option('all','全部渠道')}${data.channels.map(c=>option(c.id,c.display_name)).join('')}</select></label></div><p class="business-muted" data-catalog-count></p><div data-catalog-results></div>`;
      for (const input of this.root.querySelectorAll('[data-catalog-filter]')) input.value = this.filters[input.dataset.catalogFilter];
      this.renderResults();
    }
    executionStatus(entry) {
      if (entry.channel.id !== 'fal') return '';
      const capability = this.data.execution?.fal?.[entry.model.id];
      const state = capability?.state;
      const labels = {
        adapter_connected_unverified:'执行适配已接入 · 组织授权、限额及生成待验证',
        adapter_disconnected:'执行适配已接入 · 服务未连接，暂不可提交',
        catalog_only:'仅目录收录 · 暂不可提交'
      };
      const label = labels[state] || '调用状态待核实 · 暂不可提交';
      return `<p class="catalog-execution" data-execution-state="${esc(state || 'unknown')}">${esc(label)}</p>`;
    }
    card(group, entries) {
      const entry = entries[0];
      const model = entry.model;
      const info = deriveSpecs(model);
      const capPills = info.caps.map(([code, label]) => `<span class="cap-pill"><strong>${esc(code)}</strong> ${esc(label)}</span>`).join('');
      return `<article class="business-card catalog-card" data-catalog-card><div class="model-head"><div><span class="business-muted">${esc(business[model.business_type] || model.business_type)}</span><h3>${esc(model.display_name)}</h3></div></div><p class="model-id-line">模型 ID：<code>${esc(model.upstream_model_id ?? "待核实")}</code></p><p class="model-desc">${esc(model.description)}</p><div class="model-cap-row">${capPills}</div><div class="model-quota-strip"><div class="model-quota-row"><span>参考文件配额</span><strong>${esc(info.quota)}</strong></div><div class="model-quota-row"><span>单次任务最大容量</span><strong>${esc(info.maxSize)}</strong></div></div><dl class="model-params"><dt>支持画幅</dt><dd>${esc(info.ratios)}</dd><dt>音频能力</dt><dd>${esc(info.audioDesc)}</dd></dl>${entries.length===1?this.executionStatus(entry):''}<div class="model-foot"><div class="tags model-tags" aria-label="输出规格胶囊"><span>${esc(info.maxRes)}</span><span>${esc(info.dur)}</span><span>${esc(info.fps)}</span></div></div><details data-catalog-entry="${esc(group.id)}" ${this.open.has(group.id)?'open':''}><summary>能力规格与审核状态</summary><div class="catalog-detail">${entries.map(e=>`<section>${this.executionStatus(e)}<p class="business-muted">${esc(review[e.model.identity_review])} · ${esc(e.model.billing_modes.map(b=>billing[b]).join(' / '))}</p>${profiles(e.model.team_profiles,'团队适配范围')}${profiles(e.model.supplier_profiles,'供应商声明范围')}</section>`).join('')}</div></details></article>`;
    }
    renderResults() {
      if (!this.data) return;
      const lookup = new Map();
      for (const channel of this.data.channels) for (const model of channel.models) lookup.set(`${channel.id}/${model.id}`,{channel,model});
      const filtered = this.data.groups.map(group=>({group,entries:group.entries.map(e=>lookup.get(`${e.channel_id}/${e.entry_id}`)).filter(e=>{
        if (!e) return false;
        const {model,channel} = e, f = this.filters;
        return (f.channel==='all'||f.channel===channel.id) && (f.business==='all'||f.business===model.business_type) && `${model.display_name} ${model.description} ${model.upstream_model_id ?? ""}`.toLowerCase().includes(f.query.toLowerCase().trim());
      })})).filter(row=>row.entries.length);
      this.root.querySelector('[data-catalog-count]').textContent = `${filtered.length} 个模型 · ${this.data.channels.length} 个渠道${this.data.conflicts.length ? ' · 部分模型身份资料冲突，已分开展示' : ''}`;
      const results = this.root.querySelector('[data-catalog-results]');
      if (!filtered.length) { results.innerHTML = `<p class="empty">${this.data.groups.length?'暂无符合条件的模型。':'当前未收录模型。'}</p>`; return; }
      if (this.filters.view === 'channels') results.innerHTML = this.data.channels.map(channel=>{
        const order=new Map(channel.models.map((model,index)=>[model.id,index]));
        const rows = filtered.map(row=>({...row,entries:row.entries.filter(e=>e.channel.id===channel.id)})).filter(row=>row.entries.length).sort((a,b)=>order.get(a.entries[0].model.id)-order.get(b.entries[0].model.id));
        if (!rows.length) return '';
        const logo = channelLogos[channel.id] || channelLogos.local;
        const ver = `v${channel.version || channel.channel_version || '0.1.0'}`;
        const bill = channelBillingSummary[channel.id] || '按渠道计费';
        return `<section class="catalog-channel-section channel-group" data-channel-group="${esc(channel.id)}"><div class="channel-group-header"><div class="channel-group-title"><span class="channel-logo-badge">${logo}</span><h2>${esc(channel.display_name)}</h2></div><div class="channel-group-meta"><span class="channel-meta-pill">${esc(ver)}</span><span class="channel-meta-pill">${rows.length} 个模型</span><span class="channel-meta-pill">${esc(bill)}</span></div></div><div class="business-models">${rows.map(row=>this.card({...row.group,id:`${channel.id}/${row.group.id}`},row.entries)).join('')}</div></section>`;
      }).join('');
      else results.innerHTML = `<div class="business-models">${filtered.map(row=>this.card(row.group,row.entries)).join('')}</div>`;
    }
  }
  globalThis.VideoCatalogView = VideoCatalogView;
})();
