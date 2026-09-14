"use strict";
let businessBusy = false, resourceSelection = null, resourceWindow = "15m", resourceOrigin = null, resourceRevision = 0;
const deploymentLabels = {online:"已上线",partial:"部分就绪",not_ready:"未就绪",not_deployed:"未部署",scaled_zero:"已部署 · 0 实例",unknown:"状态未知"};
const resourceButton = (kind, model, id, text, gpu="") => `<button class="button" data-resource="${escapeHTML(kind)}" data-model="${escapeHTML(model)}" data-instance="${escapeHTML(id)}" data-gpu="${escapeHTML(gpu)}">${escapeHTML(text)}</button>`;
function clearBusiness(){
  resourceRevision++;
  resourceSelection=null;
  $("resourceDrawer").close();
  $("resourceBody").replaceChildren();
  $("resourceCrumbs").replaceChildren();
  $("inventoryTime").textContent="";
  $("mcpStatus").innerHTML='<button class="business-card" data-resource="mcp"><h3>MCP</h3><p>认证后查看服务状态 →</p></button>';
  $("modelServices").innerHTML=`<div class="business-card"><h3>minimax-h3-ref2va</h3><p>业务模型类型 / 接口契约</p></div>`;
  $("channelServices").innerHTML=`<div class="business-card"><p>h3 · h3-sol · h3-vdn</p></div>`;
}
function renderMCP(data){
  const protocol=data.protocol.state==='fresh'?(data.protocol.status==='ready'?'通过':'不可达'):'检查状态未知';
  $("mcpStatus").innerHTML=`<button class="business-card business-mcp" data-resource="mcp"><div><h3>MCP</h3><span class="business-state">服务可达</span></div><div>协议检查：${protocol}<p>${date(data.protocol.sampled_at)}</p></div><div>已记录请求：${data.requests.count}<p>HTTP 错误：${data.requests.errors} · 查看详情 →</p></div></button>`;
}
function renderModels(data){
  const routes=new Map(data.models.map(m=>[m.id,m]));
  for(const id of ['h3','h3-sol','h3-vdn']){ const option=$("dispatchForm").elements.route.querySelector(`option[value="${id}"]`), row=routes.get(id); if(!option)continue; const available=data.state==='fresh'&&row?.route_status==='connected'&&row.ready>0; option.disabled=!available; option.textContent=available?id:`${id} · 暂不可用`; }
  $("inventoryTime").textContent=`部署采集：${date(data.sampled_at)}${data.state!=='fresh'?' · 当前部署状态未知':''}`;
  const modelCards=data.models.map(m=>`<button class="business-card" data-resource="model" data-model="${escapeHTML(m.id)}"><h3>${escapeHTML(m.name)}</h3><span class="business-state ${data.state==='fresh'&&m.deployment_status==='online'?'':'stale'}">${deploymentLabels[m.deployment_status]||'未知'}</span><div class="instance-counts"><div><strong>${m.ready??'—'}</strong><span>就绪实例</span></div><div><strong>${m.current??'—'}</strong><span>当前实例</span></div><div><strong>${m.desired??'—'}</strong><span>期望实例</span></div></div><p>模型：${escapeHTML(m.model_type||'minimax-h3-ref2va')} · 渠道：${escapeHTML(m.route)}</p><p>MCP 路由：${m.route_status==='connected'?'已接入':m.route_status==='not_connected'?'未接入':'未知'}</p><div class="business-foot">查看 ${escapeHTML(m.name)} 实例列表 →</div></button>`).join('');
  $("modelServices").innerHTML=`<div class="business-card model-contract"><h3>minimax-h3-ref2va</h3><p>业务模型类型 / 接口契约</p></div>`;
  $("channelServices").innerHTML=modelCards;
}
async function refreshBusiness(){
  if(businessBusy||!authorized)return;
  businessBusy=true;
  const version=revision;
  try{
    const responses=await Promise.allSettled([api('/api/mcp/status'),api('/api/models')]);
    if(version!==revision||!authorized)return;
    if(responses[0].status==='fulfilled')renderMCP(responses[0].value);
    else $("mcpStatus").innerHTML='<button class="business-card" data-resource="mcp"><h3>MCP</h3><p>连接失败，当前状态未知 →</p></button>';
    if(responses[1].status==='fulfilled')renderModels(responses[1].value);
    else renderModels({state:'unavailable',sampled_at:null,models:['h3','h3-sol','h3-vdn'].map(id=>({id,name:names[id],route:id,model_type:'minimax-h3-ref2va',deployment_status:'unknown'}))});
    if($("resourceDrawer").open&&resourceSelection)await inspectResource(...resourceSelection,false);
  }finally{businessBusy=false;}
}
const emptyResource=(text)=>`<div class="empty">${escapeHTML(text)}</div>`;
const resourceDetails=(rows)=>`<dl>${rows.map(([k,v])=>`<dt>${escapeHTML(k)}</dt><dd>${escapeHTML(v??'未知')}</dd>`).join('')}</dl>`;
function gpuChart(data,field,max,label,memory=false){
  const points=data.history||[];
  if(!points.some(p=>p[field]!=null))return emptyResource(`${label}：暂无有效历史`);
  const end=Date.now(), start=end-data.window_minutes*60000;
  let segments=[],segment=[],last=null,circles=[];
  for(const p of points){
    const t=Date.parse(p.sampled_at),v=p[field];
    if(v==null||!Number.isFinite(v)||t<start||t>end){if(segment.length)segments.push(segment);segment=[];last=null;continue;}
    if(last!==null&&t-last>30000){if(segment.length)segments.push(segment);segment=[];}
    const x=(t-start)/(end-start)*560,y=130-Math.min(max,Math.max(0,v))/max*120;
    segment.push(`${x.toFixed(2)},${y.toFixed(2)}`);circles.push(`<circle class="${memory?'gpu-memory-point':'gpu-point'}" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="2"><title>${escapeHTML(date(p.sampled_at))}: ${v}</title></circle>`);last=t;
  }
  if(segment.length)segments.push(segment);
  return `<p>${label} · 0–${max}${memory?' GiB':'%'}</p><svg viewBox="0 0 560 140" role="img" aria-label="${label}历史曲线">${segments.map(s=>`<polyline class="gpu-line ${memory?'gpu-memory':''}" points="${s.join(' ')}"/>`).join('')}${circles.join('')}</svg><div class="chart-range"><span>${data.window_minutes} 分钟前</span><span>现在</span></div>`;
}
async function inspectResource(kind,model='',id='',gpu='',focus=true){
  if(!authorized){openModal('tokenModal');return;}
  const generation=++resourceRevision,authVersion=revision;
  resourceSelection=[kind,model,id,gpu];
  const drawer=$("resourceDrawer");
  if(!drawer.open){resourceOrigin=document.activeElement;drawer.showModal();}
  if(focus){$("resourceBody").textContent='正在读取…';$("resourceTitle").textContent='业务详情';$("resourceTitle").focus();}
  const path=`/api/models/${encodeURIComponent(model)}/instances`;
  let title,html,crumbs='';
  try{
    if(kind==='mcp'){
      const d=await api('/api/mcp/status');title='MCP 服务详情';
      html=resourceDetails([['版本',d.version],['服务启动',date(d.started_at)],['协议状态',d.protocol.state==='fresh'?(d.protocol.status==='ready'?'通过':'不可达'):'未知'],['协议检查',date(d.protocol.sampled_at)],['已记录 HTTP 请求',d.requests.count],['HTTP 错误',d.requests.errors],['最近请求',date(d.requests.last_at)],['最近 HTTP 错误',date(d.requests.last_error_at)]]);
      html+='<p class="business-muted">请求统计自当前实例启动累计，包含 MCP 协议检查及任务写入请求；HTTP 成功不代表生成质量合格，工具业务错误请查看任务详情。</p>';
    }else if(kind==='model'){
      const d=await api(path);title=`${names[model]||model} · 实例列表`;
      crumbs='<span>模型服务</span>';
      html=d.state!=='fresh'?emptyResource('部署信息不可用，实例数量未知。'):!d.instances.length?emptyResource('当前没有实例。上线状态与期望数量请见模型概览。'):d.instances.map(i=>`<div class="business-card"><h3>${escapeHTML(i.name)}</h3><span class="business-state">${i.ready?'就绪':'未就绪'} · ${escapeHTML(i.phase)}</span><p>${escapeHTML(i.node||'等待调度')} · ${escapeHTML(i.versions.join(', '))}</p>${resourceButton('instance',model,i.id,'查看实例与 GPU →')}</div>`).join('');
      html+=`<p class="business-muted">部署采集：${date(d.sampled_at)}</p>`;
    }else if(kind==='instance'){
      const d=await api(`${path}/${encodeURIComponent(id)}`);title='模型实例详情';
      crumbs=resourceButton('model',model,'',names[model]||model);
      if(d.state!=='fresh')html=emptyResource('实例分配信息已过期或不可用。');
      else{
        const i=d.instance;title=i.name;
        html=resourceDetails([['所属模型',names[model]],['实例身份',i.id],['运行状态',i.phase],['就绪状态',i.ready?'就绪':'未就绪'],['模型阶段',i.model_phase],['节点',i.node],['版本',i.versions.join(', ')],['启动时间',date(i.started_at)],['重启次数',i.restart_count],['采集时间',date(d.sampled_at)]]);
        html+='<h3>实例绑定的 GPU</h3>'+(d.gpus.length?`<div class="business-gpus">${d.gpus.map(g=>`<div class="business-card"><h3>${escapeHTML(g.metrics?.name||g.last_sample?.name||'GPU')}</h3><p>${escapeHTML(g.id)}</p><p>利用率：${g.metrics?.utilization_percent??'—'} %</p><p>显存：${g.metrics?.memory_used_gib??'—'} / ${g.metrics?.memory_total_gib??'—'} GiB</p><p>${g.state==='fresh'?'最近采样':g.state==='stale'?'数据过期':'指标未接入'}：${date(g.sampled_at)}</p>${resourceButton('gpu',model,id,'查看指标与趋势 →',g.id)}</div>`).join('')}</div>`:emptyResource('暂无已确认的物理 GPU 分配。'));
        html+='<h3>关联任务</h3>'+((d.tasks||[]).length?d.tasks.map(t=>`<p><button class="button" data-task="${escapeHTML(t.video_task_id)}">${escapeHTML(t.video_task_id)} · ${labels[t.status]||escapeHTML(t.status)}</button></p>`).join(''):'<p class="business-muted">暂无可确认的关联任务；未上报执行实例身份的任务不推断归属。</p>');
      }
    }else if(kind==='gpu'){
      const d=await api(`${path}/${encodeURIComponent(id)}/gpus/${encodeURIComponent(gpu)}?window=${resourceWindow}`);title=d.metrics?.name||d.last_sample?.name||'GPU 详情';
      crumbs=resourceButton('model',model,'',names[model]||model)+resourceButton('instance',model,id,'返回实例');
      const m=d.metrics||{};
      html=resourceDetails([['物理标识',gpu],['数据状态',d.state==='fresh'?'最近采样':d.state==='stale'?'已过期':'不可用'],['采样时间',date(d.sampled_at)],['利用率',m.utilization_percent==null?'未知':`${m.utilization_percent} %`],['显存已用 / 总量',`${m.memory_used_gib??'—'} / ${m.memory_total_gib??'—'} GiB`],['温度',m.temperature_celsius==null?'未知':`${m.temperature_celsius} °C`],['功耗',m.power_watts==null?'未知':`${m.power_watts} W`]]);
      if(d.state!=='fresh'&&d.last_sample)html+=`<p class="business-muted">最后快照：${date(d.last_sample.sampled_at)}，利用率 ${d.last_sample.utilization_percent??'—'} %，显存 ${d.last_sample.memory_used_gib??'—'} GiB。</p>`;
      html+=`<div class="business-window">${['15m','60m'].map(w=>`<button class="button" data-resource-window="${w}" aria-pressed="${w===resourceWindow}">${w==='15m'?'15':'60'} 分钟</button>`).join('')}</div>`;
      const memoryMax=d.last_sample?.memory_total_gib||Math.max(1,...(d.history||[]).map(p=>p.memory_total_gib||p.memory_used_gib||0));
      html+=gpuChart(d,'utilization_percent',100,'GPU 利用率')+gpuChart(d,'memory_used_gib',memoryMax,'显存已用',true)+'<p class="business-muted">历史从本次采集开始，缺口不补零；重启后重新积累。显存占用与计算活动分别展示。</p>';
    }else throw new Error('未知资源类型');
    if(generation!==resourceRevision||authVersion!==revision||!authorized||!drawer.open)return;
    const scroll=drawer.scrollTop,activeWindow=document.activeElement?.dataset.resourceWindow;
    $("resourceTitle").textContent=title;$("resourceCrumbs").innerHTML=crumbs;$("resourceBody").innerHTML=html;
    if(focus){drawer.scrollTop=0;$("resourceTitle").focus();}else{drawer.scrollTop=scroll;if(activeWindow)drawer.querySelector(`[data-resource-window="${activeWindow}"]`)?.focus();}
  }catch(error){
    if(generation!==resourceRevision||!authorized||!drawer.open)return;
    $("resourceTitle").textContent='资源详情暂不可用';$("resourceCrumbs").innerHTML=model?resourceButton('model',model,'','返回模型实例列表'):'';
    $("resourceBody").textContent=error.message+' 执行实例可能已退出，当前设备关联不能推断。';
  }
}
document.addEventListener('click',e=>{
  const b=e.target.closest('[data-resource]');
  if(b){if(!$("inspectorModal").hidden)closeModal('inspectorModal');inspectResource(b.dataset.resource,b.dataset.model||'',b.dataset.instance||'',b.dataset.gpu||'');}
  const w=e.target.closest('[data-resource-window]');
  if(w&&resourceSelection){resourceWindow=w.dataset.resourceWindow;inspectResource(...resourceSelection,false);}
});
$("closeResource").onclick=()=>$("resourceDrawer").close();
$("resourceDrawer").addEventListener('close',()=>{resourceSelection=null;resourceRevision++;resourceOrigin?.focus();});
$("resourceDrawer").addEventListener('click',e=>{if(e.target===$("resourceDrawer")){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
$("resourceDrawer").addEventListener('keydown',e=>{
  if(e.key!=='Tab')return;
  const controls=[...e.currentTarget.querySelectorAll('button:not(:disabled),a[href]')].filter(el=>el.getClientRects().length),active=document.activeElement;
  if(e.shiftKey&&(active===controls[0]||!controls.includes(active))){e.preventDefault();controls.at(-1)?.focus();}
  else if(!e.shiftKey&&(active===controls.at(-1)||!controls.includes(active))){e.preventDefault();controls[0]?.focus();}
});
