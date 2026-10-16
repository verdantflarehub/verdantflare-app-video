/* Shared catalog view for the standalone dashboard and Studio embed. */
(() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const business = {video_generation:'视频生成',video_editing:'视频编辑',motion_control:'动作控制',video_upscale:'视频超分',frame_interpolation:'视频补帧',depth_estimation:'深度估计',video_processing:'视频处理'};
  const modes = {text_to_video:'文生视频',image_to_video:'图生视频',first_last_frame_to_video:'首尾帧视频',reference_to_video:'参考生视频',...business};
  const review = {pending:'资料尚未审核',approved:'资料已审核',rejected:'资料未通过'};
  const verification = {unverified:'未验证',verified:'技术已验证',failed:'技术验证失败'};
  const audio = {supported:'支持',unsupported:'不支持',unknown:'未公布'};
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
  const profileFor = entry => {
    const model = entry.model;
    const team = (model.team_profiles || []).find(p => p.max_resolution || p.duration_seconds || p.duration_choices_seconds?.length || p.input_limits || p.aspect_ratios?.length);
    const supplier = (model.supplier_profiles || [])[0];
    return supplier || team || null;
  };
  const modeFor = entry => {
    const declared = profileFor(entry)?.mode || entry.model.team_profiles?.[0]?.mode || entry.model.supplier_profiles?.[0]?.mode;
    if (declared) return declared;
    if (entry.channel.id !== 'fal') return null;
    const suffix = String(entry.model.upstream_model_id || '').match(/\/(text-to-video|image-to-video|reference-to-video)$/)?.[1];
    return { 'text-to-video':'text_to_video', 'image-to-video':'image_to_video', 'reference-to-video':'reference_to_video' }[suffix] || null;
  };
  const familyOf = entry => {
    const {model, channel} = entry;
    const match = String(model.upstream_model_id || '').match(/^(.*)\/(text-to-video|image-to-video|reference-to-video)$/);
    const mode = { 'text-to-video':'text_to_video', 'image-to-video':'image_to_video', 'reference-to-video':'reference_to_video' }[match?.[2]];
    const base = String(model.display_name || '').split(' · ');
    if (!match || base.length !== 2 || modeFor(entry) !== mode || base[1] !== modes[mode]) return null;
    return `${channel.id}:${match[1]}:${base[0]}`;
  };
  const displayRows = rows => {
    const counts = new Map();
    for (const row of rows) if (row.entries.length === 1) {
      const key = familyOf(row.entries[0]);
      if (key) counts.set(key, (counts.get(key) || 0) + 1);
    }
    const merged = new Map(), result = [];
    for (const row of rows) {
      const key = row.entries.length === 1 && familyOf(row.entries[0]);
      if (!key || counts.get(key) < 2) { result.push(row); continue; }
      if (merged.has(key)) merged.get(key).entries.push(...row.entries);
      else {
        const item = {group:{id:`family:${key}`},entries:[...row.entries]};
        merged.set(key,item); result.push(item);
      }
    }
    return result;
  };
  const deriveSpecs = entries => {
    const profiles = entries.map(profileFor);
    const caps = [...new Map(entries.map(modeFor).map(mode => modeToCap[mode]).filter(Boolean).map(pair => [pair[0],pair])).values()];
    const summarize = values => {
      const grouped = new Map();
      values.forEach((value,index) => {
        if (!value) return;
        const code = modeToCap[modeFor(entries[index])]?.[0] || modes[modeFor(entries[index])];
        if (!grouped.has(value)) grouped.set(value, []);
        if (code && !grouped.get(value).includes(code)) grouped.get(value).push(code);
      });
      if (grouped.size === 1 && values.every(Boolean)) return [...grouped.keys()][0];
      return [...grouped].map(([value,codes]) => `${codes.join(' / ')}：${value}`).join('；');
    };
    const resolution = summarize(profiles.map(p => p?.max_resolution));
    const duration = summarize(profiles.map(p => p?.duration_seconds
      ? `${p.duration_seconds.minimum}S-${p.duration_seconds.maximum}S`
      : p?.duration_choices_seconds?.length ? `${Math.min(...p.duration_choices_seconds)}S-${Math.max(...p.duration_choices_seconds)}S` : null));
    const ratios = summarize(profiles.map(p => (p?.aspect_ratios || []).map(value => ['adaptive','auto'].includes(value) ? '自适应' : value).join(' / ')));
    const input = entries.map(entry => {
      if (modeFor(entry) !== 'reference_to_video') return null;
      const limit = (entry.model.supplier_profiles || []).map(p => p.input_limits).find(Boolean);
      const team = (entry.model.team_profiles || []).map(p => p.input_limits).find(Boolean);
      if (limit) return `最多 ${limit.videos} 段参考视频 · ${limit.images} 图 · ${limit.audio_files} 音频${team && team.videos < limit.videos ? `（适配器 ${team.videos} 段）` : ''}`;
      if (team) return `支持参考视频 · 适配器准入 ${team.videos} 段`;
      return '支持参考视频';
    }).filter(Boolean);
    const videoInput = [...new Set(input)].join('；');
    const audioDesc = summarize(profiles.map(p => p && (p.audio_input !== 'unknown' || p.audio_output !== 'unknown')
      ? `${audio[p.audio_input] || '未公布'} / ${audio[p.audio_output] || '未公布'}` : null));
    return { caps, videoInput, maxRes:resolution, dur:duration, ratios, audioDesc };
  };
  // Keep provenance in the catalog; the customer view contains known parameters only.
  const knownParameterText = value => String(value || '').split(/[；。]/).map(sentence =>
    sentence.split('，').filter(clause => !/(?:官方|所查官方资料)?未(?:公布|提供|给)|未公开/.test(clause)).join('，')
  ).filter(Boolean).join('；');
  const descriptionFor = entry => entry.channel.id === 'fal'
    ? knownParameterText(entry.model.description).replace(/^fal 官方 /, '') : entry.model.description;
  function inputGuidance(entries) {
    const rows = entries.map(entry => {
      const p = (entry.model.supplier_profiles || []).find(profile => profile.input_guidance);
      if (!p) return '';
      const g = p.input_guidance;
      let source;
      try { source = new URL(g.source_url); } catch { return ''; }
      if (source.protocol !== 'https:' || source.username || source.password) return '';
      const mode = modeToCap[p.mode]?.[0] || modes[p.mode] || p.mode;
      const quota = knownParameterText(g.reference_quota);
      const capacity = knownParameterText(g.capacity);
      if (!quota && !capacity) return '';
      return `<div class="catalog-profile"><strong>${esc(mode)}</strong><dl class="catalog-meta">${quota ? `<dt>参考文件配额</dt><dd>${esc(quota)}</dd>` : ''}${capacity ? `<dt>附件容量说明</dt><dd>${esc(capacity)}</dd>` : ''}</dl></div>`;
    }).filter(Boolean);
    return rows.length ? `<div class="model-quota-strip">${rows.join('')}</div>` : '';
  }
  function profiles(items, label, knownOnly = false) {
    if (!items.length) return knownOnly ? '' : `<p class="business-muted">${label}：暂无适配规格</p>`;
    return `<h5>${label}</h5>${items.map(p => {
      const duration = p.duration_seconds ? `${p.duration_seconds.minimum}–${p.duration_seconds.maximum} 秒` : p.duration_choices_seconds.length ? p.duration_choices_seconds.join(' / ')+' 秒' : '待核实';
      const limits = p.input_limits;
      const input = limits ? `图像 ${limits.images} · 视频 ${limits.videos} · 音频 ${limits.audio_files}` : p.mode === 'reference_to_video' ? '支持参考视频' : '无需参考视频';
      if (knownOnly) {
        const rows = [
          ['画幅', p.aspect_ratios.join(' · ')],
          ['视频输入', input],
          ['视频输出', [p.max_resolution, duration !== '待核实' ? duration : null].filter(Boolean).join(' · ')],
          ['音频输入', p.audio_input !== 'unknown' ? audio[p.audio_input] : ''],
          ['音频输出', p.audio_output !== 'unknown' ? audio[p.audio_output] : ''],
          ['资料 / 技术验证', `${review[p.review_status]} · ${verification[p.verification_status]}`]
        ].filter(([,value]) => value);
        return `<div class="catalog-profile"><strong>${esc(modes[p.mode] || p.mode)}</strong><dl class="catalog-meta">${rows.map(([name,value]) => `<dt>${esc(name)}</dt><dd>${esc(value)}</dd>`).join('')}</dl></div>`;
      }
      return `<div class="catalog-profile"><strong>${esc(modes[p.mode] || p.mode)}</strong><dl class="catalog-meta"><dt>画幅</dt><dd>${esc(p.aspect_ratios.join(' · ') || '未公布')}</dd><dt>视频输入</dt><dd>${esc(input)}</dd><dt>视频输出</dt><dd>${esc([p.max_resolution, duration !== '待核实' ? duration : null].filter(Boolean).join(' · ') || '未公布')}</dd><dt>音频输入 / 输出</dt><dd>${audio[p.audio_input]} / ${audio[p.audio_output]}</dd><dt>资料 / 技术验证</dt><dd>${review[p.review_status]} · ${verification[p.verification_status]}</dd></dl></div>`;
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
        adapter_connected_unverified:'执行适配已接入 · 调用受组织授权与额度限制',
        adapter_disconnected:'执行适配已接入 · 服务未连接，暂不可提交',
        catalog_only:'仅目录收录 · 暂不可提交'
      };
      const label = labels[state] || '暂不可提交';
      return `<p class="catalog-execution" data-execution-state="${esc(state || 'unknown')}">${esc(label)}</p>`;
    }
    card(group, entries) {
      const entry = entries[0];
      const model = entry.model;
      const info = deriveSpecs(entries);
      const familyKey = familyOf(entry);
      const family = entries.length > 1 && familyKey && entries.every(e => familyOf(e) === familyKey);
      const title = family ? model.display_name.split(' · ')[0] : model.display_name;
      const upstreamId = family ? model.upstream_model_id.slice(0, model.upstream_model_id.lastIndexOf('/')) : model.upstream_model_id;
      const description = family ? `${info.caps.map(([,label]) => label).join("、")}；各模式参数如下。` : descriptionFor(entry);
      const connected = entries.filter(e => this.data.execution?.fal?.[e.model.id]?.state === 'adapter_connected_unverified').length;
      const catalogOnly = entries.filter(e => this.data.execution?.fal?.[e.model.id]?.state === 'catalog_only').length;
      const summary = family && entry.channel.id === 'fal' ? `<p class="catalog-execution">${connected} 个端点已接入执行适配 · ${catalogOnly} 个仅目录收录</p>` : '';
      const capPills = info.caps.map(([code, label]) => `<span class="cap-pill"><strong>${esc(code)}</strong> ${esc(label)}</span>`).join('');
      const videoInput = info.videoInput ? `<div class="model-quota-strip"><div class="model-quota-row"><span>视频参考输入</span><strong>${esc(info.videoInput)}</strong></div></div>` : '';
      const ratios = info.ratios ? `<dt>支持画幅</dt><dd>${esc(info.ratios)}</dd>` : '';
      const audioRow = info.audioDesc ? `<dt>音频输入 / 输出</dt><dd>${esc(info.audioDesc)}</dd>` : '';
      const output = [info.maxRes,info.dur].filter(Boolean);
      const outputLabel = entry.channel.id === 'fal' ? '供应商视频输出' : '视频输出';
      const outputFooter = output.length ? `<div class="model-foot"><span class="business-muted">${outputLabel}</span><div class="tags model-tags" aria-label="输出规格胶囊">${output.map(value=>`<span>${esc(value)}</span>`).join('')}</div></div>` : '';
      return `<article class="business-card catalog-card" data-catalog-card><div class="model-head"><div><span class="business-muted">${esc(business[model.business_type] || model.business_type)}</span><h3>${esc(title)}</h3></div></div><p class="model-id-line">模型 ID：<code>${esc(upstreamId ?? "待核实")}</code></p><p class="model-desc">${esc(description)}</p><div class="model-cap-row">${capPills}</div>${inputGuidance(entries) || videoInput}<dl class="model-params">${ratios}${audioRow}</dl>${summary || (entries.length===1?this.executionStatus(entry):'')}${outputFooter}<details data-catalog-entry="${esc(group.id)}" ${this.open.has(group.id)?'open':''}><summary>能力规格与审核状态</summary><div class="catalog-detail">${entries.map(e=>`<section><h4>${esc(e.model.display_name)}</h4><p class="model-id-line">端点 ID：<code>${esc(e.model.upstream_model_id ?? '待核实')}</code></p><p class="model-desc">${esc(descriptionFor(e))}</p>${this.executionStatus(e)}<p class="business-muted">${esc(review[e.model.identity_review])} · ${esc(e.model.billing_modes.map(b=>billing[b]).join(' / '))}</p>${profiles(e.model.team_profiles,'团队适配范围',e.channel.id === 'fal')}${profiles(e.model.supplier_profiles,'供应商声明范围',e.channel.id === 'fal')}</section>`).join('')}</div></details></article>`;
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
      const visible = displayRows(filtered);
      this.root.querySelector('[data-catalog-count]').textContent = `${visible.length} 个模型卡片 · ${filtered.reduce((sum,row)=>sum+row.entries.length,0)} 条目录能力 · ${this.data.channels.length} 个渠道${this.data.conflicts.length ? ' · 部分模型身份资料冲突，已分开展示' : ''}`;
      const results = this.root.querySelector('[data-catalog-results]');
      if (!filtered.length) { results.innerHTML = `<p class="empty">${this.data.groups.length?'暂无符合条件的模型。':'当前未收录模型。'}</p>`; return; }
      if (this.filters.view === 'channels') results.innerHTML = this.data.channels.map(channel=>{
        const order=new Map(channel.models.map((model,index)=>[model.id,index]));
        const rows = displayRows(filtered.map(row=>({...row,entries:row.entries.filter(e=>e.channel.id===channel.id)})).filter(row=>row.entries.length).sort((a,b)=>order.get(a.entries[0].model.id)-order.get(b.entries[0].model.id)));
        if (!rows.length) return '';
        const logo = channelLogos[channel.id] || channelLogos.local;
        const ver = `v${channel.version || channel.channel_version || '0.1.0'}`;
        const bill = channelBillingSummary[channel.id] || '按渠道计费';
        return `<section class="catalog-channel-section channel-group" data-channel-group="${esc(channel.id)}"><div class="channel-group-header"><div class="channel-group-title"><span class="channel-logo-badge">${logo}</span><h2>${esc(channel.display_name)}</h2></div><div class="channel-group-meta"><span class="channel-meta-pill">${esc(ver)}</span><span class="channel-meta-pill">${rows.length} 个模型</span><span class="channel-meta-pill">${esc(bill)}</span></div></div><div class="business-models">${rows.map(row=>this.card({...row.group,id:`${channel.id}/${row.group.id}`},row.entries)).join('')}</div></section>`;
      }).join('');
      else results.innerHTML = `<div class="business-models">${visible.map(row=>this.card(row.group,row.entries)).join('')}</div>`;
    }
  }
  globalThis.VideoCatalogView = VideoCatalogView;
})();
