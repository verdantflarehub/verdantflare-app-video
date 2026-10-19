import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../frontend/resources.js', import.meta.url), 'utf8');
const code = source.slice(source.indexOf('const FalCapabilityCards ='), source.indexOf('// Tool descriptions'));
const context = vm.createContext({});
vm.runInContext(code, context);
const cards = snapshot => vm.runInContext('FalCapabilityCards.cardsFrom(snapshot)', Object.assign(context, { snapshot }));
const render = snapshot => vm.runInContext('FalCapabilityCards.render(snapshot)', Object.assign(context, { snapshot }));

function capability(model, endpoint, group, title, state = 'catalog_only') {
  return {
    model_id: model, endpoint_id: `${model}/${endpoint}`, display_name: endpoint,
    card_group: group, card_title: title, input_summary: '提示词', output_summary: '视频',
    execution_state: state, source_url: `https://fal.ai/models/${model}/${endpoint}/api`,
  };
}

test('fal cards follow the channel directory and group modes by model', () => {
  const h3Text = capability('minimax/h3', 'text-to-video', 'standard-generation', '标准视频生成');
  const h3Ref = capability('minimax/h3', 'reference-to-video', 'standard-generation', '标准视频生成', 'adapter_connected_unverified');
  const maxText = capability('minimax/h3-max', 'text-to-video', 'standard-generation', '标准视频生成');
  const camera = capability('minimax/h3-max', 'camera-controls', 'camera-controls', '镜头控制');
  const snapshot = { channel_capabilities: { fal: { schema_version: 1, channel_id: 'fal',
    capabilities: [h3Text, h3Ref, maxText, camera] } } };
  assert.equal(cards(snapshot).length, 3);
  assert.equal(cards(snapshot)[0].inputs.size, 1);
  const html = render(snapshot);
  assert.match(html, /fal\.ai 补充能力/);
  assert.match(html, /标准视频生成/);
  assert.match(html, /<dt>输入<\/dt>/);
  assert.match(html, /<dt>输出<\/dt>/);
  assert.doesNotMatch(html, /已接入适配|仅目录资料|API 页面|href=|endpoint-state/);
  snapshot.channel_capabilities.fal.capabilities.pop();
  assert.equal(cards(snapshot).length, 2);
  assert.doesNotMatch(render(snapshot), /镜头控制/);
});

test('unavailable directory does not invent cards and capability text remains escaped', () => {
  assert.match(render(null), /目录暂不可读/);
  const entry = capability('minimax/h3-max', 'recast', 'recast', '<script>');
  const snapshot = { channel_capabilities: { fal: { schema_version: 1, channel_id: 'fal',
    capabilities: [entry] } } };
  const html = render(snapshot);
  assert.match(html, /&lt;script&gt;/);
  assert.doesNotMatch(html, /minimax\/h3-max\/recast|fal\.ai\/models/);
  assert.doesNotMatch(html, /href=/);
});
