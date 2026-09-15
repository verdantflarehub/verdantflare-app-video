"""Browser integration using isolated fixtures, never GPU acceptance inputs."""
from __future__ import annotations

import os
import hashlib
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from playwright.sync_api import sync_playwright, expect
from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.routing import Route

from verdantflare_video_mcp.artifacts import ArtifactStore
from verdantflare_video_mcp.dashboard import Dashboard
from verdantflare_video_mcp.executor import VideoExecutor
from verdantflare_video_mcp.tasks import TaskStore


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        os.environ['VIDEO_ARTIFACT_ROOT'] = str(root)
        os.environ['VIDEO_MCP_BEARER_TOKEN'] = 'browser-test-token'
        from verdantflare_video_mcp.server import BearerAuthMiddleware
        artifacts, tasks = ArtifactStore(root), TaskStore(root)
        video = root/'fixture.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=darkgreen:s=240x420:r=24',
                        '-t','5','-c:v','libx264','-pix_fmt','yuv420p',str(video)], check=True)
        image = root/'fixture.png'
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-frames:v','1',str(image)], check=True)
        reference = artifacts.create_from_chunks(project_id='demo', operation='test', filename='reference.png', media_type='image/png', chunks=[image.read_bytes()])
        result = artifacts.create_from_chunks(project_id='demo', operation='test', filename='fixture.mp4', media_type='video/mp4', chunks=[video.read_bytes()])
        audio = root/'fixture.wav'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=1',str(audio)],check=True)
        audio_ref=artifacts.create_from_chunks(project_id='demo',operation='test',filename='fixture.wav',media_type='audio/wav',chunks=[audio.read_bytes()])
        for i in range(27):
            record = tasks.create(project_id='demo', idempotency_key=f'shot-{i:02d}/attempt-1',
                                 input_digest='sha256:test', runtime_task_id=f'private-{i}', status='succeeded' if i == 26 else 'queued',
                                 request={'prompt':'Continuous orbit <img src=x onerror=alert(1)> <Picture 1> <Video 1> <Audio 1>', 'model':'minimax-h3-ref2va',
                                          'duration_seconds':5,'aspect_ratio':'9:16','references':{'images':[{'artifact_id':reference.artifact_id,'purpose':'identity'}], 'videos':[{'artifact_id':result.artifact_id,'purpose':'motion'}], 'audios':[{'artifact_id':audio_ref.artifact_id,'purpose':'rhythm'}]}})
            if i == 26:
                tasks.update(record, artifact_id=result.artifact_id, media={'frame_rate':24,'width':240,'height':420})
        executor = VideoExecutor(artifacts,tasks,httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200,content=image.read_bytes()) if r.method == 'GET' else httpx.Response(200,json={'id':'test-submission'}))))
        executor.allowed_origins = frozenset({'https://assets.example.com'})
        dashboard = Dashboard(executor)
        dashboard.services['h3'] = 'ready'
        from test_resources import Source
        from verdantflare_video_mcp.resources import Resources
        dashboard.resources = Resources(Source())
        dashboard.resources.sync_inventory(); dashboard.resources.sync_metrics()
        async def content(request):
            artifact = artifacts.get(request.path_params['artifact_id'])
            return FileResponse(artifacts.content_path(artifact),media_type=artifact.media_type)
        app = Starlette(routes=[*dashboard.routes(),Route('/artifacts/{artifact_id}/content',content)])
        app.add_middleware(BearerAuthMiddleware)
        async def ingress(scope, receive, send):
            if scope['type'] == 'http' and scope['path'].startswith('/video/'):
                scope = dict(scope, path=scope['path'][6:], raw_path=scope['raw_path'][6:])
            await app(scope, receive, send)
        sock = socket.socket(); sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(ingress, log_level='error'))
        thread = threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True); thread.start()
        try:
            for _ in range(100):
                if server.started: break
                time.sleep(.05)
            with sync_playwright() as p:
                browser = p.chromium.launch(args=['--no-sandbox'])
                context = browser.new_context(viewport={'width':1440,'height':1000})
                page = context.new_page()
                errors=[]; page.on('pageerror',lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{port}/video/dashboard/')
                expect(page.locator('#view-tasks h1')).to_contain_text('任务')
                page.locator('#tokenButton').click(); page.locator('#tokenInput').fill('browser-test-token'); page.locator('#tokenForm button[type=submit]').click()
                expect(page.locator('#pageLabel')).to_contain_text('27 个任务')
                expect(page.locator('#channelServices')).to_contain_text('h3-sol')
                expect(page.locator('#channelServices')).to_contain_text('未部署')
                expect(page.locator('main [data-resource=gpu]')).to_have_count(0)
                page.locator('[data-nav=models]').click()
                page.locator('#channelServices [data-model=h3]').click()
                expect(page.locator('#resourceBody')).to_contain_text('h3-instance')
                page.locator('#resourceBody [data-resource=instance]').click()
                expect(page.locator('#resourceBody [data-resource=gpu]')).to_have_count(2)
                page.locator('#resourceBody [data-resource=gpu]').first.click()
                expect(page.locator('#resourceBody')).to_contain_text('18 / 24 GiB')
                page.locator('[data-resource-window="60m"]').click()
                expect(page.locator('[data-resource-window="60m"]')).to_have_attribute('aria-pressed','true')
                page.locator('#resourceCrumbs [data-resource=instance]').click()
                expect(page.locator('#resourceBody')).to_contain_text('未上报执行实例身份')
                page.keyboard.press('Escape')
                page.locator('[data-nav=tasks]').click()
                expect(page.locator('.model-card')).to_have_count(24)
                expect(page.locator('.model-card img').first).to_be_visible(timeout=15000)
                page.locator('#nextPage').click(); expect(page.locator('.model-card')).to_have_count(3)
                page.locator('#prevPage').click(); expect(page.locator('.model-card')).to_have_count(24)
                page.locator('#searchInput').fill('shot-26'); expect(page.locator('.model-card')).to_have_count(1)
                page.locator('#btnViewTable').click(); expect(page.locator('#tableContainer')).to_be_visible()
                page.locator('#taskRows button').click(); expect(page.locator('#taskPrompt')).to_contain_text('Continuous orbit <img')
                assert '/dashboard/tasks/' in page.url
                expect(page.locator('#videoArea video')).to_be_visible(); expect(page.locator('#videoArea video')).to_have_js_property('readyState', 4)
                page.locator('#videoArea video').evaluate('(v) => v.play()'); expect(page.locator('#videoArea video')).to_have_js_property('paused', False)
                expect(page.locator('#resultActions a')).to_have_attribute('download','fixture.mp4')
                expect(page.locator('#reference-images-1 img')).to_be_visible()
                expect(page.locator('#reference-videos-1 video')).to_be_visible()
                expect(page.locator('#reference-audios-1 audio')).to_be_visible()
                page.locator('#taskPrompt button').last.click()
                expect(page.locator('#reference-audios-1')).to_have_class('reference-card audios reference-focus')
                page.locator('#reference-images-1 button').click()
                expect(page.locator('#imagePreview')).to_be_visible()
                page.keyboard.press('Escape')
                page.reload()
                expect(page.locator('#taskPrompt')).to_contain_text('Continuous orbit <img')
                for width in [1920,1440,800,390]:
                    page.set_viewport_size({'width':width,'height':1000})
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    result_box=page.locator('#videoArea').bounding_box()
                    assert abs(result_box['width']-result_box['height'])<1
                    img_box=page.locator('#reference-images-1 .reference-preview').bounding_box()
                    vid_box=page.locator('#reference-videos-1 .reference-preview').bounding_box()
                    assert abs(img_box['width']-vid_box['width'])<1
                    assert abs(img_box['height']-vid_box['height'])<1
                page.set_viewport_size({'width':1440,'height':1000})

                page.locator('#backToTasks').click(); expect(page.locator('.model-card')).to_have_count(24)
                page.locator('#btnViewGallery').click()
                output = Path(os.environ.get('BROWSER_OUTPUT_DIR','/tmp/video-dashboard-browser')); output.mkdir(parents=True, exist_ok=True)
                page.evaluate('window.scrollTo({top:0,behavior:"instant"})')
                page.screenshot(path=str(output/'desktop.png'))
                page.locator('#filterEngine').select_option('h3-sol'); expect(page.locator('#emptyState')).to_contain_text('暂无符合条件')
                page.locator('#filterEngine').select_option('all'); expect(page.locator('.model-card')).to_have_count(24)
                page.locator('[data-action=dispatch]').first.click()
                page.locator('#dispatchForm [name=project_id]').fill('demo'); page.locator('#dispatchForm [name=prompt]').fill('A browser test task')
                page.locator('#openImport').click()
                page.locator('#importForm [name=filename]').fill('reference.png')
                page.locator('#importForm [name=source_url]').fill('https://assets.example.com/reference.png')
                page.locator('#importForm [name=expected_sha256]').fill(hashlib.sha256(image.read_bytes()).hexdigest())
                page.locator('#importForm button[type=submit]').click()
                expect(page.locator('#importMessage')).to_contain_text('导入成功')
                page.locator('[data-close=importModal]').click()
                expect(page.locator('#referenceInput')).to_contain_text('')
                assert page.locator('#referenceInput').input_value().startswith('image | art_')
                expect(page.locator('#dispatchForm [name=duration_seconds]')).to_have_attribute('step','5')
                def models_connected(route):
                    route.fulfill(json={'state':'fresh','sampled_at':'2026-09-10T00:00:00Z','models':[
                        {'id':'h3','name':'H3','deployment_status':'online','ready':1,'current':1,'desired':1,'route_status':'connected'},
                        {'id':'h3-vdn','name':'H3-VDN','deployment_status':'online','ready':1,'current':1,'desired':1,'route_status':'connected'},
                        {'id':'h3-sol','name':'H3-Sol','deployment_status':'online','ready':1,'current':1,'desired':1,'route_status':'connected'}]})
                page.route('**/api/models',models_connected)
                page.evaluate('refreshBusiness()')
                expect(page.locator('#dispatchForm option[value="h3-sol"]')).to_be_enabled()
                page.locator('#dispatchForm [name=route]').select_option('h3-sol')
                expect(page.locator('#dispatchForm [name=duration_seconds]')).to_have_attribute('step','5')
                page.locator('#dispatchForm [name=route]').select_option('h3')
                expect(page.locator('#dispatchForm [name=duration_seconds]')).to_have_attribute('step','1')
                page.evaluate('refreshBusiness()')
                expect(page.locator('#dispatchForm [name=route]')).to_have_value('h3')
                page.locator('#dispatchForm [name=route]').select_option('h3-vdn')
                expect(page.locator('#dispatchForm [name=duration_seconds]')).to_have_attribute('step','5')
                page.locator('#dispatchForm [name=route]').select_option('h3')
                page.unroute('**/api/models',models_connected)
                page.locator('#submitTask').click(); expect(page.locator('#taskPrompt')).to_contain_text('A browser test task')
                page.locator('#backToTasks').click()
                page.set_viewport_size({'width':390,'height':844})
                page.evaluate('window.scrollTo({top:0,behavior:"instant"})')
                page.screenshot(path=str(output/'mobile.png'))
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
                page.locator('[data-nav=models]').click()
                page.locator('#channelServices [data-model=h3]').click()
                page.locator('#resourceBody [data-resource=instance]').click()
                expect(page.locator('#resourceBody [data-resource=gpu]')).to_have_count(2)
                assert page.locator('#resourceDrawer').evaluate('(e)=>e.scrollWidth<=e.clientWidth'), 'Mobile instance overflow'
                page.locator('#resourceBody [data-resource=gpu]').first.click()
                expect(page.locator('#resourceBody svg')).to_have_count(2)
                assert page.locator('#resourceDrawer').evaluate('(e)=>e.scrollWidth<=e.clientWidth'), 'Mobile GPU overflow'
                for _ in range(8):
                    page.keyboard.press('Tab')
                    assert page.evaluate('document.getElementById("resourceDrawer").contains(document.activeElement)')
                page.keyboard.press('Escape')
                assert page.evaluate('localStorage.length === 1 && sessionStorage.length === 0')
                page.reload()
                expect(page.locator('#mcpStatus')).to_contain_text('服务可达', timeout=30000)
                expect(page.locator('.model-card')).not_to_have_count(0)
                expect(page.locator('#tokenButton')).to_contain_text('已设置')
                context = page.context
                dashboard_url = page.url
                page.close()
                page = context.new_page()
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(dashboard_url)
                expect(page.locator('.model-card')).not_to_have_count(0)
                expect(page.locator('#tokenButton')).to_contain_text('已设置')
                page.locator('#tokenButton').click(); page.locator('#clearToken').click(); expect(page.locator('.model-card')).to_have_count(0)
                expect(page.locator('#resourceBody')).to_be_empty()
                expect(page.locator('#channelServices')).to_contain_text('h3-sol')
                assert page.evaluate('localStorage.length === 0 && sessionStorage.length === 0')
                page.reload()
                expect(page.locator('#tokenButton')).to_contain_text('未配置')
                expect(page.locator('#channelServices')).to_contain_text('h3-sol')
                page.locator('#tokenButton').click(); page.locator('#tokenInput').fill('invalid-token')
                page.locator('#tokenForm button[type=submit]').click()
                expect(page.locator('#tokenButton')).to_contain_text('请重新设置')
                assert page.evaluate('localStorage.length === 0 && sessionStorage.length === 0')
                page.reload()
                expect(page.locator('#tokenButton')).to_contain_text('未配置')
                assert not errors, errors
                browser.close()
            print('Browser integration passed: prefixed routes, auth, pagination, filters, views, previews, playback, submission, mobile, token cleanup')
        finally:
            server.should_exit = True; thread.join(timeout=10); sock.close()

if __name__ == '__main__':
    main()
