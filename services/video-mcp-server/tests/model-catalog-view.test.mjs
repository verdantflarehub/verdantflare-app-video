import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

test('default channel view follows channel order and shows supplier tiers separately from team limits', () => {
  const context = vm.createContext({ URL });
  vm.runInContext(readFileSync(new URL('../frontend/model-catalog.js', import.meta.url), 'utf8'), context);
  const nodes = {};
  const root = { innerHTML: '', addEventListener() {}, removeEventListener() {},
    querySelectorAll: () => [], querySelector: selector => nodes[selector] ??= {} };
  const view = new context.VideoCatalogView(root, async () => ({}));
  const profile = max_resolution => ({ mode: 'video_generation', max_resolution, resolutions: [max_resolution],
    duration_seconds: null, duration_choices_seconds: [], aspect_ratios: [], input_limits: null,
    audio_input: 'unknown', audio_output: 'unknown', review_status: 'pending', verification_status: 'unverified' });
  const model = (id, name, supplier, team, upstream_model_id = id) => ({ id, display_name: name,
    business_type: 'video_generation', description: '', identity_review: 'pending', billing_modes: [],
    upstream_model_id, supplier_profiles: supplier ? [profile(supplier)] : [], team_profiles: team ? [profile(team)] : [] });
  const hf = { id: 'higgsfield', display_name: 'Higgsfield', models: [
    model('seedance', 'Seedance 2.5', '1080p', '720p'),
    model('genjutsu', 'Higgsfield Genjutsu', '1080p', null, null),
    model('h3', 'MiniMax H3', '2K', null),
  ] };
  // Global groups intentionally place H3 first, as shared identities used to do.
  view.set({ state: 'ready', snapshot_id: 'one', channels: [hf], conflicts: [],
    groups: ['h3', 'genjutsu', 'seedance'].map(id => ({ id, entries: [{ channel_id: 'higgsfield', entry_id: id }] })) });
  assert.equal(view.filters.view, 'channels');
  assert.ok(root.innerHTML.indexOf('按渠道查看') < root.innerHTML.indexOf('按模型查看'));
  const html = nodes['[data-catalog-results]'].innerHTML;
  assert.ok(html.indexOf('<h3>Seedance 2.5</h3>') < html.indexOf('<h3>Higgsfield Genjutsu</h3>'));
  assert.ok(html.indexOf('<h3>Higgsfield Genjutsu</h3>') < html.indexOf('<h3>MiniMax H3</h3>'));
  assert.match(html, /class="channel-group-header"[\s\S]*<h2>Higgsfield<\/h2>/);
  assert.doesNotMatch(html, /class="catalog-channels"/);
  assert.doesNotMatch(html, /Higgsfield · 1080p/);
  assert.match(html, /aria-label="输出规格胶囊"><span>1080p<\/span>/);
  assert.match(html, /aria-label="输出规格胶囊"><span>2K<\/span>/);
  assert.doesNotMatch(html, /24FPS|≤ 50MB/);
  assert.match(html, /模型 ID：<code>待核实<\/code>/);
  view.filters.view = 'models';
  view.set({ ...view.data, snapshot_id: 'two' });
  assert.equal(view.filters.view, 'models');
});

test('fal H3 modes share one card without borrowing quotas or claiming a task-size maximum', () => {
  const context = vm.createContext({ URL });
  vm.runInContext(readFileSync(new URL('../frontend/model-catalog.js', import.meta.url), 'utf8'), context);
  const nodes = {};
  const root = { innerHTML:'', addEventListener() {}, removeEventListener() {},
    querySelectorAll: () => [], querySelector: selector => nodes[selector] ??= {} };
  const view = new context.VideoCatalogView(root, async () => ({}));
  const make = (mode,suffix,name,limit) => ({ id:`fal-h3-${suffix}`, display_name:`MiniMax H3 · ${name}`,
    business_type:'video_generation', description:'', identity_review:'pending', billing_modes:[],
    upstream_model_id:`minimax/h3/${suffix}`, supplier_profiles:[{ mode, max_resolution:'2K', duration_seconds:null,
      duration_choices_seconds:[], aspect_ratios:[], input_limits:null, audio_input:'unknown', audio_output:'unknown',
      review_status:'pending', verification_status:'unverified' }],
    team_profiles:limit?[{ mode, max_resolution:'480p', duration_seconds:null, duration_choices_seconds:[],
      aspect_ratios:['16:9'], input_limits:limit, audio_input:'supported', audio_output:'unknown',
      review_status:'pending', verification_status:'unverified' }]:[] });
  const fal = { id:'fal', display_name:'fal.ai', models:[
    make('text_to_video','text-to-video','文生视频',null),
    make('image_to_video','image-to-video','图生视频',null),
    make('reference_to_video','reference-to-video','参考生视频',{ images:9,videos:3,audio_files:3 })
  ] };
  view.set({ state:'ready',snapshot_id:'h3',channels:[fal],conflicts:[],
    execution:{ fal:Object.fromEntries(fal.models.map((m,i)=>[m.id,{state:i===2?'adapter_connected_unverified':'catalog_only'}])) },
    groups:fal.models.map(m=>({id:m.id,entries:[{channel_id:'fal',entry_id:m.id}]})) });
  const html = nodes['[data-catalog-results]'].innerHTML;
  assert.equal((html.match(/data-catalog-card/g) || []).length,1);
  assert.match(html, /<h3>MiniMax H3<\/h3>/);
  assert.match(html, /<strong>T2V<\/strong>/);
  assert.match(html, /<strong>I2V<\/strong>/);
  assert.match(html, /<strong>R2V<\/strong>/);
  assert.match(html, /视频参考输入<\/span><strong>支持参考视频 · 适配器准入 3 段<\/strong>/);
  assert.doesNotMatch(html, /单次任务最大容量/);
  assert.match(html, /1 个端点已接入执行适配 · 2 个仅目录收录/);
  assert.match(html, /端点 ID：<code>minimax\/h3\/reference-to-video<\/code>/);
  assert.doesNotMatch(html, /≤ 50MB|24FPS/);
  assert.match(nodes['[data-catalog-count]'].textContent, /1 个模型卡片 · 3 条目录能力/);
  view.filters.query = '图生视频';
  view.renderResults();
  assert.equal((nodes['[data-catalog-results]'].innerHTML.match(/data-catalog-card/g) || []).length,1);
  assert.doesNotMatch(nodes['[data-catalog-results]'].innerHTML, /<strong>T2V<\/strong>/);
});

test('fal official catalog candidate shows mode badges and sourced input/output specs', () => {
  const context = vm.createContext({ URL });
  vm.runInContext(readFileSync(new URL('../frontend/model-catalog.js', import.meta.url), 'utf8'), context);
  const nodes = {};
  const root = { innerHTML:'', addEventListener() {}, removeEventListener() {},
    querySelectorAll: () => [], querySelector: selector => nodes[selector] ??= {} };
  const candidate = readFileSync(new URL('../../../../deploys/k8s.dev.verdantflarehub.com/verdantflare-video/video-models-fal-v0-1-1.cm.yaml', import.meta.url), 'utf8');
  const catalog = JSON.parse(candidate.split('  channel-catalog.yaml: |\n')[1].split('  review.yaml: |\n')[0].replace(/^    /gm, ''));
  const channel = { ...catalog.channel, version:catalog.channel_version, models:catalog.models };
  const view = new context.VideoCatalogView(root, async () => ({}));
  view.set({ state:'ready',snapshot_id:'official-parameters',channels:[channel],conflicts:[],execution:{fal:{}},
    groups:channel.models.map(model=>({ id:model.id, entries:[{channel_id:'fal',entry_id:model.id}] })) });
  const html = nodes['[data-catalog-results]'].innerHTML;
  assert.match(html, /<strong>T2V<\/strong>/);
  assert.match(html, /<strong>I2V<\/strong>/);
  assert.equal((html.match(/class="cap-pill"><strong>R2V<\/strong>/g) || []).length,3);
  assert.match(html, /最多 9 图、3 视频、3 音频，合计 ≤12 件/);
  assert.doesNotMatch(html, /参考文件配额|<div class="catalog-profile"><strong>(T2V|I2V|R2V)<\/strong>|无需参考附件|此模式不接收参考附件/);
  assert.match(html, /起始帧 1 张，可选结束帧 1 张/);
  assert.doesNotMatch(html, /官方资料|官方来源|官方未|所查官方|未公布|未公开/);
  assert.match(html, /图片 ≤30 MB\/张；视频合计 &lt;50 MB；音频 ≤15 MB\/个/);
  assert.match(html, /参考视频、音频各自合计 ≤15 秒，每段 2–15 秒/);
  assert.doesNotMatch(html, /href="https:\/\/fal.ai/);
  assert.doesNotMatch(html, /待验证|待核实|各模式见详情|满额实测/);
  assert.match(html, /支持参考视频/);
  assert.match(html, /MP4\/MOV/);
  assert.match(html, /1080p/);
  assert.match(html, /4K/);
  assert.match(html, /T2V \/ R2V：自适应/);
  assert.match(html, /I2V：跟随输入图/);
  assert.match(html, /T2V \/ I2V：不支持 \/ 支持；R2V：支持 \/ 支持/);
  assert.doesNotMatch(html, /单次任务最大容量/);
});

test('fal cards distinguish the one adapted endpoint from catalog-only entries', () => {
  const context = vm.createContext({ URL });
  vm.runInContext(readFileSync(new URL('../frontend/model-catalog.js', import.meta.url), 'utf8'), context);
  const nodes = {};
  const root = { innerHTML: '', addEventListener() {}, removeEventListener() {},
    querySelectorAll: () => [], querySelector: selector => nodes[selector] ??= {} };
  const view = new context.VideoCatalogView(root, async () => ({}));
  const model = (id, name) => ({ id, display_name:name, business_type:'video_generation', description:'',
    identity_review:'pending', billing_modes:[], upstream_model_id:id, supplier_profiles:[], team_profiles:[] });
  const fal = { id:'fal', display_name:'fal.ai', models:[model('ref','H3 参考生'),model('text','H3 文生')] };
  view.set({ state:'ready', snapshot_id:'fal-one', channels:[fal], conflicts:[],
    execution:{ fal:{ ref:{ state:'adapter_connected_unverified' }, text:{ state:'catalog_only' } } },
    groups:fal.models.map(item=>({id:item.id,entries:[{channel_id:'fal',entry_id:item.id}]})) });
  const html = nodes['[data-catalog-results]'].innerHTML;
  assert.match(html, /执行适配已接入 · 调用受组织授权与额度限制/);
  assert.match(html, /仅目录收录 · 暂不可提交/);
  assert.doesNotMatch(html, /可生成/);
  view.set({ ...view.data, execution:{ fal:{
    ref:{ state:'adapter_disconnected' }, text:{ state:'catalog_only' }
  } } });
  assert.match(nodes['[data-catalog-results]'].innerHTML, /服务未连接，暂不可提交/);
});
