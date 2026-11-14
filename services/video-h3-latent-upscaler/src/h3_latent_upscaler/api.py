"""Authenticated internal HTTP API; only trusted MCP task identities are accepted."""
import hmac
import json
import os
from pathlib import Path
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from .queue import Queue, QueueError
from .resources import SourceIndex, ResourceError


def create_app(queue, sources, engine_factory, token):
    if not token:
        raise ValueError('runtime_token_required')
    state = {'ready': False, 'error': None}
    stop = threading.Event()

    def worker():
        try:
            state['engine'] = engine_factory()
            state['ready'] = True
            while not stop.wait(.25):
                task = queue.take()
                if task is None:
                    continue
                try:
                    request = task['request']
                    resources = sources.resolve(request['project_id'], request['source_video_task_id'])
                    directory = queue.root / task['video_task_id']
                    directory.mkdir(exist_ok=False)
                    result = state['engine'].generate(resources, request, directory)
                    queue.finish(task['video_task_id'], result=result)
                except Exception as exc:
                    code = exc.code if isinstance(exc, ResourceError) else type(exc).__name__
                    queue.finish(task['video_task_id'], error=code)
                    # Do not reuse a possibly corrupt CUDA context after failure.
                    if not isinstance(exc, ResourceError):
                        state.update(ready=False, error=code)
                        return
        except Exception as exc:
            state.update(ready=False, error=type(exc).__name__)

    @asynccontextmanager
    async def lifespan(app):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        yield
        stop.set()

    app = FastAPI(lifespan=lifespan)

    @app.middleware('http')
    async def auth(request: Request, call_next):
        if request.url.path not in {'/health', '/live'}:
            supplied = request.headers.get('Authorization', '')
            if not hmac.compare_digest(supplied.encode(), ('Bearer ' + token).encode()):
                from fastapi.responses import JSONResponse
                return JSONResponse({'error': 'unauthorized'}, status_code=401)
        return await call_next(request)

    @app.get('/health')
    def health():
        if not state['ready']:
            raise HTTPException(503, 'not_ready')
        return {'ready': True, 'gpu_count': 1, 'cpu_offload': False, 'profile_id': state['engine'].profile['id']}

    @app.get('/live')
    def live():
        return {'live': True, 'ready': state['ready']}

    @app.post('/v1/upscales')
    async def submit(request: Request):
        try:
            length = int(request.headers.get('content-length', '0'))
        except ValueError:
            raise HTTPException(400, 'invalid_content_length') from None
        if length < 0:
            raise HTTPException(400, 'invalid_content_length')
        if length > 16384:
            raise HTTPException(413, 'body_too_large')
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > 16384:
                raise HTTPException(413, 'body_too_large')
            data.extend(chunk)
        try:
            payload = json.loads(data)
            required = {'idempotency_key', 'project_id', 'source_video_task_id'}
            optional = {'profile_id', 'target_width', 'target_height', 'seed'}
            if not isinstance(payload, dict) or not required <= payload.keys() or payload.keys() - required - optional:
                raise ValueError('invalid_fields')
            # Recover an already accepted task even if the model is currently unhealthy.
            try:
                queue.get(payload['idempotency_key'])
            except KeyError:
                pass
            else:
                return queue.submit(payload)
            if not state['ready']:
                raise HTTPException(503, 'not_ready')
            resources = sources.resolve(payload['project_id'], payload['source_video_task_id'])
            state['engine'].validate_request(resources, payload)
            return queue.submit(payload)
        except (ValueError, ResourceError, QueueError) as exc:
            code = str(exc)
            status = 429 if code == 'queue_full' else 409 if code == 'idempotency_conflict' else 422
            raise HTTPException(status, code) from None

    @app.get('/v1/upscales/{task_id}')
    def status(task_id: str):
        try:
            return queue.get(task_id)
        except (KeyError, QueueError):
            raise HTTPException(404, 'not_found') from None

    @app.get('/v1/upscales/{task_id}/{kind}')
    def output(task_id: str, kind: str):
        if kind == 'result':
            result = status(task_id)
            if result['status'] != 'succeeded':
                raise HTTPException(409, 'not_succeeded')
            return result['result']
        if kind not in {'content', 'preview'} or status(task_id)['status'] != 'succeeded':
            raise HTTPException(404, 'not_found')
        path = queue.root / task_id / ('output.mp4' if kind == 'content' else 'preview.mp4')
        return FileResponse(path, media_type='video/mp4')
    return app


def main():
    import uvicorn
    from .engine import Engine
    queue = Queue(os.environ.get('H3_LATENT_TASK_ROOT', '/projects/h3-latent/tasks'))
    sources = SourceIndex(Path(os.environ['H3_LATENT_SOURCE_INDEX']), Path(os.environ['H3_LATENT_SOURCE_ROOT']), os.environ['H3_LATENT_NODE_NAME'])
    app = create_app(queue, sources, lambda: Engine(os.environ['H3_LATENT_MODELS'], os.environ['H3_LATENT_PROFILE'],
                                                  os.environ['H3_LATENT_BACKEND_LOCK']), os.environ['H3_LATENT_RUNTIME_TOKEN'])
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)


if __name__ == '__main__':
    main()
