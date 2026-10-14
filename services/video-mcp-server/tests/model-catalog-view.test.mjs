import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

test('default channel view follows channel order and shows supplier tiers separately from team limits', () => {
  const context = vm.createContext({});
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
  assert.match(html, /class="catalog-spec">Higgsfield · 1080p/);
  assert.match(html, /class="catalog-spec">Higgsfield · 2K/);
  assert.match(html, /模型 ID：<code>待核实<\/code>/);
  view.filters.view = 'models';
  view.set({ ...view.data, snapshot_id: 'two' });
  assert.equal(view.filters.view, 'models');
});
