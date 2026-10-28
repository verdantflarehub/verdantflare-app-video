import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

test('restored Draft stays unavailable through quote and upload callbacks, and Wan requires an explicit choice', async () => {
  const nodes = new Map(), calls = [];
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, {value:'', innerHTML:'', textContent:'', disabled:false,
      hidden:false, dataset:{}, files:[], addEventListener() {}, querySelectorAll:() => []});
    return nodes.get(id);
  };
  const root = {innerHTML:'', querySelector:selector => node(selector.slice(4)), querySelectorAll:() => []};
  const make = (model, display_name, can_submit, modes, reason_code) => ({
    model, display_name, can_submit, modes, reason_code, resolutions:['480p'], aspect_ratios:['16:9'],
    duration_seconds:{minimum:4,maximum:10}, reference_limits:{images:4,videos:1}
  });
  const routes = [make('seedance_2_5_draft','Seedance 2.5 Draft',false,['reference_to_video'],'provider_not_found'),
    make('wan3_0','Wan 3.0',true,['text_to_video','reference_to_video'])];
  const context = vm.createContext({
    document:{createElement:() => root, querySelector:() => ({append() {}}), addEventListener() {}},
    window:{addEventListener() {}, studioRequest:async path => {
      calls.push(path);
      const value = path.endsWith('/projects') ? {items:[{project_id:'project',name:'Project'}]} :
        path.endsWith('/capabilities') ? {routes} : path.endsWith('/preflight') ?
          {quoted_credits:2,can_submit:true,expires_at:new Date(Date.now()+60000).toISOString()} : {items:[]};
      return {ok:true,json:async () => value};
    }},
    location:{hash:'#create'}, crypto:{randomUUID:() => 'attempt'}, setTimeout:() => 1, clearTimeout() {}
  });
  const script = readFileSync(new URL('../frontend/higgsfield-member.js', import.meta.url), 'utf8');
  vm.runInContext(script.replace(/\}\)\(\);\s*$/, `window.openSavedCreate = async saved => {
    draft=saved; await createPage(epoch);
  }; })();`), context);
  await context.window.openSavedCreate({project:'project',model:'seedance_2_5_draft',prompt:'A paper boat',
    mode:'reference_to_video',resolution:'480p',ratio:'16:9',duration:4});
  assert.equal(node('model').value, 'seedance_2_5_draft');
  assert.equal(node('quote').disabled, true);
  assert.match(root.innerHTML, /Seedance 2.5 Draft：供应商暂不可用/);
  assert.match(root.innerHTML, /参考生成可自行选择 Wan 3.0/);
  assert.match(root.innerHTML, /value="seedance_2_5_draft" disabled/);
  await node('quote').onclick();
  assert.match(node('message').textContent, /供应商暂不可用/);
  assert.equal(node('quote').disabled, true);
  await node('upload').onchange();
  assert.equal(node('quote').disabled, true);
  assert.equal(calls.filter(path => path.endsWith('/preflight')).length, 0);
  node('model').value = 'wan3_0';
  node('model').onchange();
  node('mode').value = 'text_to_video';
  assert.equal(node('quote').disabled, false);
  assert.equal(node('price').textContent, '待报价');
  await node('quote').onclick();
  assert.equal(calls.filter(path => path.endsWith('/preflight')).length, 1);
  assert.equal(node('submit').disabled, false);
});
