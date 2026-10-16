"use strict";
// Explicit hosted mode. Never access host DOM, cookies, storage or credentials.
if (/\/dashboard\/tasks\/[^/]+$/.test(location.pathname)) document.documentElement.dataset.page = 'task';
window.studioEmbedded = new URLSearchParams(location.search).get('embed') === '1' && window.parent !== window;
if (window.studioEmbedded) {
 document.documentElement.dataset.embedded='true';
 document.documentElement.dataset.theme=new URLSearchParams(location.search).get('theme')==='light'?'light':'dark';
 const view=new URLSearchParams(location.search).get('view');
 const hostOrigin=new URL(location.href).origin;
 let next=0;const pending=new Map();
 const previews=new Map(),previewFocus=new Map(); let previewRevision=0;
 const mediaBlobs=new Map();
 window.studioRememberMedia=(url,blob)=>mediaBlobs.set(url,blob);
 window.studioReleaseMedia=()=>{mediaBlobs.clear();window.studioClosePreview?.()};
 window.studioPreviewImage=async(url,title)=>{
  if(!url.startsWith('blob:'))throw Error('图片尚未准备好。');
  const epoch=++previewRevision,focus=document.activeElement;
  const blob=mediaBlobs.get(url);
  if(!blob)throw Error('图片尚未准备好，请重新打开详情。');
  if(blob.size>128*1024*1024)throw Error('图片过大，请下载查看。');
  const bytes=await blob.arrayBuffer(),id=++next;
  if(epoch!==previewRevision)return;
  previewFocus.set(id,focus);
  return new Promise((resolve,reject)=>{
   const timer=setTimeout(()=>{previews.delete(id);previewFocus.delete(id);reject(Error('全局预览未响应，请刷新 Studio 后重试。'))},15000);
   previews.set(id,{resolve,reject,timer});
   parent.postMessage({channel:'vf-video',view,type:'media-preview',id,bytes,mime:blob.type,title:String(title).slice(0,256)},hostOrigin,[bytes]);
  });
 };
 window.studioClosePreview=()=>{previewRevision++;parent.postMessage({channel:'vf-video',view,type:'media-close'},hostOrigin)};
 window.studioRequest=(path,options={})=>new Promise((resolve,reject)=>{
  const id=++next;const timer=setTimeout(()=>{pending.delete(id);reject(Error('Video 连接超时，请重新连接。'))},95000);
  pending.set(id,{resolve,reject,timer});
  parent.postMessage({channel:'vf-video',view,type:'request',id,path,method:options.method||'GET',body:options.body},hostOrigin);
 });
 window.addEventListener('message',e=>{
  if(e.source!==parent||e.origin!==hostOrigin||e.data?.channel!=='vf-studio')return;
  const d=e.data;
  if(d.type==='media-result'&&previews.has(d.id)){
   const item=previews.get(d.id);previews.delete(d.id);clearTimeout(item.timer);
   if(d.ok)item.resolve();else {previewFocus.delete(d.id);item.reject(Error('无法打开图片预览，请刷新后重试。'))}return;
  }
  if(d.type==='media-closed'){const focus=previewFocus.get(d.id);if(focus?.isConnected)focus.focus({preventScroll:true});previewFocus.delete(d.id);return}
  if(d.type==='theme'){document.documentElement.dataset.theme=d.theme==='light'?'light':'dark';return}
  if(d.type==='route'&&typeof d.hash==='string'){if(location.hash!=='#'+d.hash)location.hash='#'+d.hash;return}
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
  if(/^\/dashboard(?:\/tasks\/[a-zA-Z0-9_-]+)?$/.test(path)){
   e.preventDefault();
   if(u.pathname===location.pathname&&u.hash){
    if(location.hash!==u.hash)location.hash=u.hash;
   }
   window.studioNavigate(path+u.hash);
  }
 });
 window.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('[data-nav]').forEach(a=>{a.textContent={tasks:'任务',models:'模型',mcp:'MCP'}[a.dataset.nav]});
  parent.postMessage({channel:'vf-video',view,type:'ready'},hostOrigin);
  let lastHeight=0;
  const postHeight=()=>{
   // Root scrollHeight is at least the current iframe viewport and cannot
   // shrink after switching from a long task list to a shorter view.
   const h=Math.ceil(document.body.getBoundingClientRect().height);
   if(h>0&&h!==lastHeight){lastHeight=h;parent.postMessage({channel:'vf-video',view,type:'resize',height:h},hostOrigin);}
  };
  postHeight();
  window.addEventListener('load',postHeight);
  window.addEventListener('resize',postHeight);
  if(window.ResizeObserver){const ro=new ResizeObserver(postHeight);ro.observe(document.body);}
  window.addEventListener('wheel',e=>{parent.postMessage({channel:'vf-video',view,type:'wheel',deltaY:e.deltaY},hostOrigin)},{passive:true});
 });
}
