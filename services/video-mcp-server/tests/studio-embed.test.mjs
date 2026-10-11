import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

test('hosted resize follows natural content, deduplicates and shrinks across views', () => {
  const messages = [], listeners = new Map();
  let height = 1200, observer;
  const parent = { postMessage(message) { messages.push(message); } };
  const document = {
    documentElement: { dataset: {}, scrollHeight: 100000 },
    body: { scrollHeight: 100000, getBoundingClientRect: () => ({ height }) },
    querySelectorAll: () => [],
    addEventListener: (name, handler) => listeners.set(name, handler),
  };
  class ResizeObserver { constructor(callback) { observer = callback; } observe() {} }
  const window = { parent, ResizeObserver, addEventListener: (name, handler) => listeners.set(name, handler) };
  vm.runInNewContext(readFileSync(new URL('../frontend/studio-embed.js', import.meta.url), 'utf8'), {
    window, parent, document, ResizeObserver, URL, URLSearchParams,
    location: { search: '?embed=1&view=7', href: 'https://studio.invalid/apps/video/dashboard?embed=1&view=7', hash: '' },
  });
  listeners.get('DOMContentLoaded')();
  const sizes = () => messages.filter(m => m.type === 'resize').map(m => m.height);
  assert.deepEqual(sizes(), [1200]);
  document.documentElement.scrollHeight = 200000;
  observer(); listeners.get('resize')();
  assert.deepEqual(sizes(), [1200]);
  height = 399.2; observer();
  assert.deepEqual(sizes(), [1200, 400]);
  height = 2600; listeners.get('load')();
  assert.deepEqual(sizes(), [1200, 400, 2600]);
  assert.ok(messages.every(m => m.channel === 'vf-video' && m.view === '7'));
});
