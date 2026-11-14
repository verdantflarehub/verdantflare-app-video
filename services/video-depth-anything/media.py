"""CFR media validation and full-resolution grayscale/side-by-side export."""
import json
import subprocess
from fractions import Fraction
from pathlib import Path
import cv2
import numpy as np


class MediaError(ValueError):
    pass


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
                             '-show_frames', '-show_entries',
                             'stream=width,height,avg_frame_rate,r_frame_rate,codec_name:frame=best_effort_timestamp_time',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True, timeout=120)
    data = json.loads(result.stdout)
    if len(data.get('streams', [])) != 1:
        raise MediaError('video_stream_missing')
    stream = data['streams'][0]
    fps = Fraction(stream['avg_frame_rate'])
    timestamps = [float(f['best_effort_timestamp_time']) for f in data.get('frames', [])]
    w, h, count = stream['width'], stream['height'], len(timestamps)
    if not 0 < fps <= 120 or count == 0 or w % 2 or h % 2:
        raise MediaError('unsupported_video_geometry_or_rate')
    # Timestamp quantization is allowed; variable frame intervals are not.
    if any(abs((b - a) - float(1 / fps)) > .0001 for a, b in zip(timestamps, timestamps[1:])):
        raise MediaError('variable_frame_rate_unsupported')
    return {'width': w, 'height': h, 'frames': count, 'frame_rate': str(fps),
            'duration_seconds': float(count / fps), 'video_codec': stream['codec_name']}


def read_frames(path, expected):
    if expected['frames'] > 1800 or max(expected['width'], expected['height']) > 1920 or expected['frames'] * expected['width'] * expected['height'] > 2_000_000_000:
        raise MediaError('video_exceeds_resource_limit')
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if len(frames) >= expected['frames']:
                raise MediaError('decoded_frame_count_mismatch')
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if len(frames) != expected['frames']:
        raise MediaError('decoded_frame_count_mismatch')
    return np.stack(frames)


def encode(frames, path, fps, width, height, pixel_format):
    with tempfile_log(path) as log:
        process = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo',
                                    '-pixel_format', pixel_format, '-video_size', f'{width}x{height}',
                                    '-framerate', fps, '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                                    '-threads', '4', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                                    '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            process.stdin.close()
            if process.wait(timeout=300):
                raise MediaError('video_encode_failed')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def tempfile_log(path):
    return Path(str(path) + '.ffmpeg.log').open('wb')


def convert(engine, source, directory):
    media = probe(source)
    frames = read_frames(source, media)
    depth = engine.infer(frames, float(Fraction(media['frame_rate'])))
    if depth.shape != frames.shape[:3] or not np.isfinite(depth).all():
        raise MediaError('invalid_depth_tensor')
    lo, hi = float(depth.min()), float(depth.max())
    gray = np.clip((depth - lo) * (255 / (hi - lo) if hi > lo else 0), 0, 255).astype(np.uint8)
    del depth
    output, preview = directory / 'depth.mp4', directory / 'preview.mp4'
    encode(gray, output, media['frame_rate'], media['width'], media['height'], 'gray')
    pairs = (np.concatenate((rgb, np.repeat(g[:, :, None], 3, axis=2)), axis=1) for rgb, g in zip(frames, gray))
    encode(pairs, preview, media['frame_rate'], media['width'] * 2, media['height'], 'rgb24')
    actual = probe(output)
    preview_media = probe(preview)
    for key in ('frames', 'frame_rate', 'width', 'height'):
        if actual[key] != media[key] or preview_media[key] != (media[key] * 2 if key == 'width' else media[key]):
            raise MediaError('output_media_mismatch')
    for path in (output, preview):
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True, capture_output=True, timeout=300)
    return {'input_media': media, 'media': actual, 'preview_media': preview_media,
            'parameters': {'input_size': 518, 'fp32': False, 'normalization': 'whole_video_minmax',
                           'depth_min': lo, 'depth_max': hi, 'audio': False, 'crf': 18}}
