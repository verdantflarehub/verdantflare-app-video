import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';

test('isolated image preview transfers retained bytes without a CSP-blocked blob fetch', async () => {
  const handlers = {}, posts = [], parent = { postMessage: message => posts.push(message) };
  const window = { parent, addEventListener: (name, fn) => { handlers[name] = fn; } };
  const context = { window, parent, document: { documentElement: { dataset: {} }, activeElement: null, addEventListener() {} },
    location: { pathname: '/dashboard/tasks/task-1', search: '?embed=1&view=1', href: 'https://studio.invalid/dashboard/tasks/task-1?embed=1&view=1' },
    URL, URLSearchParams, Response, setTimeout: () => 1, clearTimeout() {}, fetch: () => { throw Error('connect-src none'); } };
  vm.runInNewContext(readFileSync(new URL('../frontend/studio-embed.js', import.meta.url), 'utf8'), context);
  window.studioRememberMedia('blob:null/test', new Blob([new Uint8Array([1, 2, 3])], { type: 'image/webp' }));
  const pending = window.studioPreviewImage('blob:null/test', '参考图');
  await new Promise(resolve => setImmediate(resolve));
  const message = posts.find(p => p.type === 'media-preview');
  assert.equal(message.mime, 'image/webp'); assert.equal(message.bytes.byteLength, 3);
  handlers.message({ source: parent, origin: 'https://studio.invalid', data: { channel: 'vf-studio', type: 'media-result', id: message.id, ok: true } });
  await pending;
  window.studioReleaseMedia();
  await assert.rejects(window.studioPreviewImage('blob:null/test', '参考图'), /尚未准备好/);
  assert.equal(posts.at(-1).type, 'media-close');
});
