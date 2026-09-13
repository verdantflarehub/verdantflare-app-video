from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, uuid, tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import pipeline

app = FastAPI(title="Video Depth Anything API")
MODEL_ID = os.getenv("DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf")
DEPTH_PIPE = None

def _pipe():
    global DEPTH_PIPE
    if DEPTH_PIPE is None:
        DEPTH_PIPE = pipeline("depth-estimation", model=MODEL_ID,
                              device=0 if torch.cuda.is_available() else -1)
    return DEPTH_PIPE

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
    # The worker contract expects the MCP to provide a local source path. This
    # endpoint accepts that path for the first cluster smoke test.
    source = Path(req.source_artifact_id)
    if not source.is_file():
        raise HTTPException(404, 'source artifact path not found')
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(tempfile.gettempdir()) / f'depth-{uuid.uuid4().hex}.mp4'
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), False)
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            result = _pipe()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            depth = np.asarray(result['depth'].resize((width, height)))
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            writer.write(depth); count += 1
    finally:
        cap.release(); writer.release()
    return {'depth_task_id': f'depth_task_{uuid.uuid4().hex}', 'status': 'succeeded',
            'project_id': req.project_id, 'source_artifact_id': req.source_artifact_id,
            'model': req.model, 'output_format': req.output_format, 'frames': count,
            'fps': fps, 'width': width, 'height': height, 'output_path': str(out_path)}
