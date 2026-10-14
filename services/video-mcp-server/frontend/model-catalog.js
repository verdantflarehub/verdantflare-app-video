/* Shared catalog view for the standalone dashboard and Studio embed. */
(() => {
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const business = {video_generation:'视频生成',video_editing:'视频编辑',motion_control:'动作控制',video_upscale:'视频超分',frame_interpolation:'视频补帧',depth_estimation:'深度估计',video_processing:'视频处理'};
  const modes = {text_to_video:'文生视频',image_to_video:'图生视频',first_last_frame_to_video:'首尾帧视频',reference_to_video:'参考生视频',...business};
  const review = {pending:'待核实',approved:'资料已审核',rejected:'资料未通过'};
  const verification = {unverified:'未验证',verified:'技术已验证',failed:'技术验证失败'};
  const audio = {supported:'支持',unsupported:'不支持',unknown:'待核实'};
  const billing = {credits:'积分',unlimited:'会员权益',local_resources:'本地资源',provider_billed:'供应商计费'};
  const specs = model => [...new Set((model.supplier_profiles.some(p=>p.max_resolution) ? model.supplier_profiles : model.team_profiles).map(p => p.max_resolution).filter(Boolean))].join(' / ') || '规格待核实';
  function profiles(items, label) {
    if (!items.length) return `<p class="business-muted">${label}：规格待核实</p>`;
    return `<h5>${label}</h5>${items.map(p => {
      const duration = p.duration_seconds ? `${p.duration_seconds.minimum}–${p.duration_seconds.maximum} 秒` : p.duration_choices_seconds.length ? p.duration_choices_seconds.join(' / ')+' 秒' : '待核实';
      const limits = p.input_limits;
      return `<div class="catalog-profile"><strong>${esc(modes[p.mode] || p.mode)}</strong><dl class="catalog-meta"><dt>分辨率</dt><dd>${esc(p.resolutions.join(' / ') || '待核实')}</dd><dt>时长</dt><dd>${esc(duration)}</dd><dt>画幅</dt><dd>${esc(p.aspect_ratios.join(' · ') || '待核实')}</dd><dt>参考输入</dt><dd>${limits ? `图像 ${limits.images} · 视频 ${limits.videos} · 音频 ${limits.audio_files}` : '待核实'}</dd><dt>音频输入 / 输出</dt><dd>${audio[p.audio_input]} / ${audio[p.audio_output]}</dd><dt>资料 / 技术验证</dt><dd>${review[p.review_status]} · ${verification[p.verification_status]}</dd></dl></div>`;
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
      if (this.data?.snapshot_id === data.snapshot_id) return;
      this.data = data;
      const option = (value,label) => `<option value="${esc(value)}">${esc(label)}</option>`;
      this.root.innerHTML = `<div class="catalog-toolbar"><label>查看方式<select data-catalog-filter="view">${option('channels','按渠道查看')}${option('models','按模型查看')}</select></label><label class="catalog-search">搜索模型<input type="search" data-catalog-filter="query" placeholder="模型名称、用途或模型 ID" /></label><label>业务<select data-catalog-filter="business">${option('all','全部业务')}${Object.entries(business).map(([id,label])=>option(id,label)).join('')}</select></label><label>渠道<select data-catalog-filter="channel">${option('all','全部渠道')}${data.channels.map(c=>option(c.id,c.display_name)).join('')}</select></label></div><p class="business-muted" data-catalog-count></p><div data-catalog-results></div>`;
      for (const input of this.root.querySelectorAll('[data-catalog-filter]')) input.value = this.filters[input.dataset.catalogFilter];
      this.renderResults();
    }
    card(group, entries) {
      const model = entries[0].model;
      const channels = [...new Map(entries.map(e=>[e.channel.id,e.channel])).values()];
      return `<article class="business-card catalog-card" data-catalog-card><span class="business-muted">${esc(business[model.business_type] || model.business_type)}</span><h3>${esc(model.display_name)}</h3><p>${esc(model.description)}</p><div class="catalog-channels">${channels.slice(0,3).map(c=>`<span class="catalog-chip">${esc(c.display_name)}</span>`).join('')}${channels.length>3?`<span class="catalog-chip">另有 ${channels.length-3} 个渠道</span>`:''}</div><p class="catalog-spec">${entries.slice(0,3).map(e=>`${esc(e.channel.display_name)} · ${esc(specs(e.model))}`).join('　')}</p><details data-catalog-entry="${esc(group.id)}" ${this.open.has(group.id)?'open':''}><summary>查看模型与渠道详情</summary><div class="catalog-detail">${entries.map(e=>`<section><h4>${esc(e.channel.display_name)} · ${esc(e.model.display_name)}</h4><p class="business-muted">${esc(review[e.model.identity_review])} · ${esc(e.model.billing_modes.map(b=>billing[b]).join(' / '))}</p><p>模型 ID：<code>${esc(e.model.upstream_model_id ?? "待核实")}</code></p>${profiles(e.model.team_profiles,'团队适配范围')}${profiles(e.model.supplier_profiles,'供应商声明范围')}</section>`).join('')}<p class="business-muted">目录收录不表示当前可调用，执行范围以现有生成入口为准。</p></div></details></article>`;
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
        return rows.length ? `<section class="catalog-channel-section"><h2>${esc(channel.display_name)} <small>${rows.length} 个模型</small></h2><div class="business-models">${rows.map(row=>this.card({...row.group,id:`${channel.id}/${row.group.id}`},row.entries)).join('')}</div></section>` : '';
      }).join('');
      else results.innerHTML = `<div class="business-models">${filtered.map(row=>this.card(row.group,row.entries)).join('')}</div>`;
    }
  }
  globalThis.VideoCatalogView = VideoCatalogView;
})();
