from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, uuid

app = FastAPI(title="Video Depth Anything API")

class DepthRequest(BaseModel):
    project_id: str
    idempotency_key: str
    source_artifact_id: str
    model: str = "video-depth-anything"
    output_format: str = "mp4"

@app.get('/health')
def health():
    return {'status': 'ok', 'model': os.getenv('DEPTH_MODEL_VERSION', 'video-depth-anything')}

@app.post('/v1/depth')
def generate(req: DepthRequest):
    if req.model != 'video-depth-anything':
        raise HTTPException(400, 'unsupported model')
    # Runtime integration point: worker consumes the registered Artifact and writes
    # an immutable grayscale MP4. Keep the MCP contract stable while weights are deployed.
    return {'depth_task_id': f'depth_task_{uuid.uuid4().hex}', 'status': 'queued',
            'project_id': req.project_id, 'source_artifact_id': req.source_artifact_id,
            'model': req.model, 'output_format': req.output_format}
