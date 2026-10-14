import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

test('generation choices use capabilities, preserve the default alias and fail closed', () => {
  const source = readFileSync(new URL('../frontend/resources.js', import.meta.url), 'utf8');
  const render = source.slice(source.indexOf('function renderRouteCapabilities('), source.indexOf('async function refreshBusiness('));
  const options = Object.fromEntries(['h3', 'h3-sol', 'h3-vdn'].map(id => [id, {}]));
  const context = vm.createContext({ $: () => ({ elements: { route: {
    querySelector: selector => options[selector.match(/value="([^"]+)"/)[1]],
  } } }) });
  vm.runInContext(render, context);
  context.data = { default_route: 'h3-singularity', routes: [
    { route: 'h3-singularity', configured: true, readiness: 'ready' },
    { route: 'h3-sol', configured: true, readiness: 'not_ready' },
    { route: 'h3-vdn', configured: false, readiness: 'not_checked' },
  ] };
  vm.runInContext('renderRouteCapabilities(data)', context);
  assert.equal(options.h3.disabled, false);
  assert.equal(options['h3-sol'].disabled, true);
  assert.equal(options['h3-vdn'].disabled, true);
  context.data.routes[0].readiness = 'unreachable';
  vm.runInContext('renderRouteCapabilities(data)', context);
  assert.equal(options.h3.disabled, true);
  vm.runInContext('renderRouteCapabilities({routes:[]})', context);
  assert.ok(Object.values(options).every(option => option.disabled));
});
