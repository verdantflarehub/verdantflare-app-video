"use strict";
// Explicit hosted mode. Never access host DOM, cookies, storage or credentials.
window.studioEmbedded = new URLSearchParams(location.search).get('embed') === '1' && window.parent !== window;
if (window.studioEmbedded) {
 document.documentElement.dataset.embedded='true';
 document.documentElement.dataset.theme=new URLSearchParams(location.search).get('theme')==='light'?'light':'dark';
 const view=new URLSearchParams(location.search).get('view');
 const hostOrigin=new URL(location.href).origin;
 let next=0;const pending=new Map();
 window.studioRequest=(path,options={})=>new Promise((resolve,reject)=>{
  const id=++next;const timer=setTimeout(()=>{pending.delete(id);reject(Error('Video 连接超时，请重新连接。'))},95000);
  pending.set(id,{resolve,reject,timer});
  parent.postMessage({channel:'vf-video',view,type:'request',id,path,method:options.method||'GET',body:options.body},hostOrigin);
 });
 window.addEventListener('message',e=>{
  if(e.source!==parent||e.origin!==hostOrigin||e.data?.channel!=='vf-studio')return;
  const d=e.data;
  if(d.type==='theme'){document.documentElement.dataset.theme=d.theme==='light'?'light':'dark';return}
  if(d.type==='restore'){window.dispatchEvent(new CustomEvent('studio-restore',{detail:d.state}));return}
  if(d.type==='response'&&pending.has(d.id)){
   const item=pending.get(d.id);pending.delete(d.id);clearTimeout(item.timer);
   item.resolve(new Response(d.body,{status:d.status,headers:{'Content-Type':d.contentType||'application/json'}}));
  }
 });
 window.studioNavigate=path=>parent.postMessage({channel:'vf-video',view,type:'navigate',path,state:window.studioViewState?.()},hostOrigin);
 document.addEventListener('click',e=>{
  const a=e.target.closest('a');if(!a)return;
  const u=new URL(a.href,location.href);const path=u.pathname.replace(/^.*(?=\/dashboard)/,'');
  if(/^\/dashboard(?:\/tasks\/[a-zA-Z0-9_-]+)?$/.test(path)){e.preventDefault();window.studioNavigate(path+u.hash)}
 });
 window.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('[data-nav]').forEach(a=>{a.textContent={tasks:'任务',models:'模型',mcp:'MCP'}[a.dataset.nav]});
  parent.postMessage({channel:'vf-video',view,type:'ready'},hostOrigin);
 });
}
