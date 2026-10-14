"use strict";

// Tool descriptions remain separate from the reviewed channel model catalog.
const videoCapabilityCatalog=[
  {
    "id": "artifact",
    "title": "素材导入",
    "description": "登记项目输入，校验来源与内容哈希。",
    "input": "项目 · HTTPS 素材地址 · 文件名 · SHA-256",
    "output": "Artifact 元数据",
    "tools": [
      "artifact.import"
    ]
  },
  {
    "id": "video",
    "title": "视频生成",
    "description": "参考素材驱动的视频生成，按项目保存任务与结果。",
    "input": "提示词 · 图像 / 视频 / 可选音频参考",
    "output": "视频任务 · 视频产物",
    "tools": [
      "video.generate",
      "video.status",
      "video.result"
    ]
  },
  {
    "id": "video.depth",
    "title": "深度估计",
    "description": "把项目中的 RGB 视频转换为时序深度视频。",
    "input": "项目内已登记的视频 Artifact",
    "output": "深度视频 · RGB / 深度对比预览",
    "tools": [
      "video.depth.generate",
      "video.depth.status",
      "video.depth.result",
      "video.depth.preview"
    ]
  },
  {
    "id": "video.sr",
    "title": "视频超分",
    "description": "对项目视频进行清晰度恢复，保留时序与画面比例。",
    "input": "视频 Artifact · 目标宽高",
    "output": "超分视频 · 对比预览",
    "tools": [
      "video.sr.generate",
      "video.sr.status",
      "video.sr.result",
      "video.sr.preview"
    ]
  },
  {
    "id": "video.interpolate",
    "title": "视频插帧",
    "description": "提高视频帧率，保持画面尺寸与原始时长。",
    "input": "视频 Artifact · 目标帧率",
    "output": "插帧视频 · 对比预览",
    "tools": [
      "video.interpolate.generate",
      "video.interpolate.status",
      "video.interpolate.result",
      "video.interpolate.preview"
    ]
  },
  {
    "id": "video.h3.latent.upscale",
    "title": "H3 Latent 后处理",
    "description": "基于已完成 H3 任务保留的同节点 latent 资源做后处理；不接受 MP4 输入。",
    "input": "已完成 H3 任务 ID · 可选 profile / 目标尺寸",
    "output": "后处理视频 · 对比预览",
    "tools": [
      "video.h3.latent.upscale.generate",
      "video.h3.latent.upscale.status",
      "video.h3.latent.upscale.result",
      "video.h3.latent.upscale.preview"
    ],
    "pending": false
  }
];

function catalogCapabilities(){return `<h2 class="service-subheading">能力目录</h2><p class="business-muted">当前发布定义 · 工具就绪情况取决于对应渠道</p><div class="business-models">${videoCapabilityCatalog.map(c=>`<article class="business-card catalog-card"><h3>${escapeHTML(c.title)}</h3><p>${escapeHTML(c.description)}</p><dl class="catalog-meta"><dt>输入</dt><dd>${escapeHTML(c.input)}</dd><dt>输出</dt><dd>${escapeHTML(c.output)}</dd></dl><details><summary>${c.tools.length} 个工具</summary>${c.tools.map(t=>`<p><code>${escapeHTML(t)}</code></p>`).join('')}</details></article>`).join('')}</div>`}

const modelCatalogView = new VideoCatalogView($('modelServices'), () => api('/api/model-catalog'));
let businessBusy = false, resourceSelection = null, resourceOrigin = null, resourceRevision = 0;
function clearBusiness(){
  resourceRevision++;
  resourceSelection=null;
  $("resourceDrawer").close();
  $("resourceBody").replaceChildren();
  $("resourceCrumbs").replaceChildren();
  $("mcpStatus").innerHTML='<button class="business-card" data-resource="mcp"><h3>MCP</h3><p>认证后查看服务状态 →</p></button>';
  modelCatalogView.clear();
}
function renderMCP(data){
  const protocol=data.protocol.state==='fresh'?(data.protocol.status==='ready'?'通过':'不可达'):'检查状态未知';
  $("mcpStatus").innerHTML=`<button class="business-card business-mcp" data-resource="mcp"><div><h3>MCP</h3><span class="business-state">服务可达</span></div><div>协议检查：${protocol}<p>${date(data.protocol.sampled_at)}</p></div><div>已记录请求：${data.requests.count}<p>HTTP 错误：${data.requests.errors} · 查看详情 →</p></div></button>`+catalogCapabilities();
}
function renderRouteCapabilities(data){
  const routes=new Map((data.routes||[]).map(r=>[r.route,r]));
  for(const id of ['h3','h3-sol','h3-vdn']){
    const option=$("dispatchForm").elements.route.querySelector(`option[value="${id}"]`);
    if(!option)continue;
    const route=routes.get(id)||(id==='h3'?routes.get(data.default_route):null);
    const available=route?.configured===true&&!['not_ready','unreachable'].includes(route.readiness);
    option.disabled=!available;
    option.textContent=available?id:`${id} · 暂不可用`;
  }
}
async function refreshBusiness(){
  if(businessBusy||!authorized)return;
  businessBusy=true;
  const version=revision;
  if(!modelCatalogView.data)$("modelServices").innerHTML='<p class="empty" role="status">正在读取模型目录…</p>';
  try{
    const responses=await Promise.allSettled([api('/api/mcp/status'),api('/api/capabilities'),api('/api/model-catalog')]);
    if(version!==revision||!authorized)return;
    if(responses[0].status==='fulfilled')renderMCP(responses[0].value);
    else $("mcpStatus").innerHTML='<button class="business-card" data-resource="mcp"><h3>MCP</h3><p>连接失败，当前状态未知 →</p></button>';
    renderRouteCapabilities(responses[1].status==='fulfilled'?responses[1].value:{routes:[]});
    if(responses[2].status==='fulfilled')modelCatalogView.set(responses[2].value);
    else { modelCatalogView.data=null; modelCatalogView.failure(); }
    if($("resourceDrawer").open&&resourceSelection)await inspectResource(...resourceSelection,false);
  }finally{businessBusy=false;}
}
const resourceDetails=(rows)=>`<dl>${rows.map(([k,v])=>`<dt>${escapeHTML(k)}</dt><dd>${escapeHTML(v??'未知')}</dd>`).join('')}</dl>`;
async function inspectResource(kind,model='',id='',gpu='',focus=true){
  if(kind!=='mcp')return;
  if(!authorized){openModal('tokenModal');return;}
  const generation=++resourceRevision,authVersion=revision;
  resourceSelection=[kind,model,id,gpu];
  const drawer=$("resourceDrawer");
  if(!drawer.open){resourceOrigin=document.activeElement;drawer.showModal();}
  if(focus){$("resourceBody").textContent='正在读取…';$("resourceTitle").textContent='业务详情';$("resourceTitle").focus();}
  let title,html,crumbs='';
  try{
    if(kind==='mcp'){
      const d=await api('/api/mcp/status');title='MCP 服务详情';
      html=resourceDetails([['版本',d.version],['服务启动',date(d.started_at)],['协议状态',d.protocol.state==='fresh'?(d.protocol.status==='ready'?'通过':'不可达'):'未知'],['协议检查',date(d.protocol.sampled_at)],['已记录 HTTP 请求',d.requests.count],['HTTP 错误',d.requests.errors],['最近请求',date(d.requests.last_at)],['最近 HTTP 错误',date(d.requests.last_error_at)]]);
      html+='<p class="business-muted">请求统计自当前实例启动累计，包含 MCP 协议检查及任务写入请求；HTTP 成功不代表生成质量合格，工具业务错误请查看任务详情。</p>';
    }
    if(generation!==resourceRevision||authVersion!==revision||!authorized||!drawer.open)return;
    const scroll=drawer.scrollTop;
    $("resourceTitle").textContent=title;$("resourceCrumbs").innerHTML=crumbs;$("resourceBody").innerHTML=html;
    if(focus){drawer.scrollTop=0;$("resourceTitle").focus();}else{drawer.scrollTop=scroll;}
  }catch(error){
    if(generation!==resourceRevision||!authorized||!drawer.open)return;
    $("resourceTitle").textContent='MCP 详情暂不可用';$("resourceCrumbs").replaceChildren();
    $("resourceBody").textContent=error.message;
  }
}
document.addEventListener('click',e=>{
  const b=e.target.closest('[data-resource]');
  if(b){inspectResource(b.dataset.resource,b.dataset.model||'',b.dataset.instance||'',b.dataset.gpu||'');}
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
