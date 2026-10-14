export function errorCode(error) {
  return error?.code || error?.message || 'REQUEST_UNAVAILABLE';
}

export async function request(sdk, method, path, body, signal) {
  const input = {method, path};
  if (body !== undefined) input.body = body;
  const response = await sdk.app.request(input, {signal});
  if (response.status < 200 || response.status >= 300) {
    const error = new Error(response.data?.code || 'REQUEST_UNAVAILABLE');
    error.code = error.message;
    error.status = response.status;
    throw error;
  }
  return response.data;
}

export function createLimitsEditor(sdk, storage, key, uuid = () => crypto.randomUUID()) {
  let pending = null, busy = false;
  // A failed storage read must not discard an unknown previously sent write.
  const saved = storage.getItem(key);
  if (saved) {
    pending = JSON.parse(saved);
    if (!valid(pending)) throw new Error('LIMITS_RECOVERY_INVALID');
  }
  function valid(value) {
    return value && Object.keys(value).sort().join(',') ===
      'binding_id,credit_limit,operation_id,revision,submission_limit' &&
      typeof value.binding_id === 'string' && value.binding_id.length > 0 && value.binding_id.length <= 128 &&
      typeof value.operation_id === 'string' && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value.operation_id) &&
      typeof value.credit_limit === 'string' && /^(0|[1-9][0-9]{0,11})(\.[0-9]{1,6})?$/.test(value.credit_limit) &&
      Number.isSafeInteger(value.revision) && value.revision > 0 &&
      Number.isSafeInteger(value.submission_limit) && value.submission_limit >= 0 && value.submission_limit <= 2147483647;
  }
  function snapshot() { return {pending: pending && structuredClone(pending), busy}; }
  async function save(limits, credits, submissions) {
    if (busy) throw new Error('LIMITS_SAVE_BUSY');
    if (!pending) {
      const next = {binding_id: limits.binding_id, revision: limits.revision, credit_limit: credits,
        submission_limit: submissions, operation_id: uuid()};
      if (!valid(next)) throw new Error('INVALID_REQUEST');
      storage.setItem(key, JSON.stringify(next));
      pending = next;
    }
    if (pending.binding_id !== limits.binding_id) throw new Error('LIMITS_BINDING_CHANGED');
    busy = true;
    try {
      const result = await request(sdk, 'PUT', '/limits', structuredClone(pending));
      storage.removeItem(key);
      pending = null;
      return result;
    } catch (error) {
      // Only a definite rejection retires the operation. A cancelled request or
      // missing response is retried with exactly the same payload and identity.
      if ([409, 422].includes(error.status) || ['LIMITS_REVISION_CONFLICT', 'IDEMPOTENCY_CONFLICT', 'INVALID_REQUEST'].includes(errorCode(error))) {
        storage.removeItem(key);
        pending = null;
      }
      throw error;
    } finally { busy = false; }
  }
  return {snapshot, save};
}
