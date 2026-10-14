import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Execute the real event handler with a minimal DOM, without starting a service.
const script = readFileSync(new URL('../frontend/dashboard.js', import.meta.url), 'utf8');
function setup() {
  const listeners = new Map(), microtasks = [], messages = [];
  const node = () => ({ value: '', hidden: true, classList: { contains: () => false, toggle() {}, add() {}, remove() {} },
    elements: { route: { addEventListener() {} } }, addEventListener() {} });
  const nodes = new Map();
  const context = vm.createContext({
    window: { studioEmbedded: true, studioNavigate: path => messages.push(path), addEventListener: (type, fn) => listeners.set('window:' + type, fn) },
    document: { getElementById: id => { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id) }, addEventListener: (type, fn) => listeners.set(type, fn) },
    location: { pathname: '/dashboard', hash: '#tasks', search: '' }, URLSearchParams, URL,
    queueMicrotask: fn => microtasks.push(fn), setTimeout: () => 1, clearTimeout() {},
  });
  vm.runInContext(script, context);
  const flush = () => { while (microtasks.length) microtasks.shift()() };
  return { listeners, messages, flush, run: code => vm.runInContext(code, context) };
}

test('restoring table view cannot signal list readiness before the first render', () => {
  const app = setup();
  app.run('authorized = true;');
  app.listeners.get('click')(); app.flush();
  assert.equal(app.messages.length, 0);
  app.run('listReady = true;');
  for (const type of ['click', 'change', 'input']) { app.listeners.get(type)(); app.flush(); }
  assert.deepEqual(app.messages, ['/dashboard#tasks', '/dashboard#tasks', '/dashboard#tasks']);
  app.run('authorized = false;');
  app.listeners.get('input')(); app.flush();
  assert.equal(app.messages.length, 3);
});
