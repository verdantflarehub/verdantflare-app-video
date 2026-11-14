"""Bounded CFR processing, frame-accurate output and source-audio remux."""
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import tempfile
from .windowing import restore_windows
import cv2
import numpy as np

class MediaError(ValueError):
    pass

def command(args, timeout=300):
    return subprocess.run(args, check=True, capture_output=True, timeout=timeout)

def probe(path):
    try:
        return _probe(path)
    except (subprocess.CalledProcessError, KeyError, ValueError, ZeroDivisionError) as exc:
        raise MediaError('invalid_or_unsupported_media') from exc

def _probe(path):
    data = json.loads(command(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-f', 'mov', '-show_streams', '-of', 'json', str(path)]).stdout)
    videos = [s for s in data['streams'] if s['codec_type'] == 'video']
    audio = [s for s in data['streams'] if s['codec_type'] == 'audio']
    if len(videos) != 1 or len(audio) > 1:
        raise MediaError('unsupported_stream_count')
    stream = videos[0]
    fps = Fraction(stream['avg_frame_rate'])
    (w, h) = (stream['width'], stream['height'])
    if not 0 < fps <= 120 or not 0 < w <= 4096 or (not 0 < h <= 2048) or w % 2 or h % 2:
        raise MediaError('unsupported_video_geometry_or_rate')
    duration = float(stream.get('duration', 'inf'))
    if not math.isfinite(duration) or duration <= 0:
        raise MediaError('video_exceeds_resource_limit')
    if stream.get('color_transfer') not in (None, 'unknown', 'bt709') or stream.get('pix_fmt') not in ('yuv420p', 'yuvj420p'):
        raise MediaError('only_8bit_sdr_yuv420_input_supported')
    if stream.get('sample_aspect_ratio') not in (None, 'N/A', '1:1') or any((s.get('rotation', 0) for s in stream.get('side_data_list', []))):
        raise MediaError('non_square_pixels_or_rotation_unsupported')
    if any((abs(float(s.get('start_time', 0))) > 0.001 for s in [stream, *audio])):
        raise MediaError('nonzero_stream_start_unsupported')
    (count, previous) = (0, None)
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-f', 'mov', '-select_streams', 'v:0', '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time:side_data=', '-of', 'compact=p=0:nk=0', str(path)], stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            for line in proc.stdout:
                fields = dict((part.split('=', 1) for part in line.strip().split('|') if '=' in part))
                if 'best_effort_timestamp_time' not in fields:
                    continue
                timestamp = float(fields['best_effort_timestamp_time'])
                if not math.isfinite(timestamp) or (previous is None and abs(timestamp) > 0.001) or (previous is not None and abs(timestamp - previous - float(1 / fps)) > 0.0001):
                    raise MediaError('variable_frame_rate_unsupported')
                count += 1
                previous = timestamp
            if proc.wait(timeout=300) or not count:
                raise MediaError('invalid_video_timestamps')
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
    return dict(width=w, height=h, frames=count, frame_rate=str(fps), duration_seconds=float(count / fps), video_codec=stream['codec_name'], audio=bool(audio))

def read_frames(path, media):
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            (ok, frame) = cap.read()
            if not ok:
                break
            count += 1
            if count > media['frames'] or frame.shape != (media['height'], media['width'], 3):
                raise MediaError('decoded_media_mismatch')
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    if count != media['frames']:
        raise MediaError('decoded_frame_count_mismatch')

def encode(frames, path, fps, width, height):
    with Path(str(path) + '.ffmpeg.log').open('wb') as log:
        process = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', str(path)], stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in frames:
                if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                    raise MediaError('invalid_output_frame')
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            process.stdin.close()
            if process.wait(timeout=300):
                raise MediaError('video_encode_failed')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if not process.stdin.closed:
                process.stdin.close()

def validate_request(req, media):
    if max(media['width'], media['height']) > 2048:
        raise MediaError('input_exceeds_resource_limit')
    if req.target_width * media['height'] != req.target_height * media['width']:
        raise MediaError('aspect_ratio_change_unsupported')
    if req.target_width < media['width'] or req.target_height < media['height']:
        raise MediaError('downscale_unsupported')
    return {**media, 'width': req.target_width, 'height': req.target_height}

def is_scene_cut(left, right):
    a = cv2.resize(left, (64, 64)).astype(np.float32) / 255
    b = cv2.resize(right, (64, 64)).astype(np.float32) / 255
    return float(np.abs(a - b).mean()) > 0.3

def process(engine, source, directory, req):
    media = probe(source)
    expected = validate_request(req, media)
    window_report = None
    policy = engine.select_window(media, req)
    window_report = {**policy, 'chunks': 0, 'max_chunk_frames': 0, 'output_frames': 0}
    frames = restore_windows(engine, read_frames(source, media), req, policy, is_scene_cut, window_report)
    silent = directory / 'silent.mp4'
    encode(frames, silent, expected['frame_rate'], expected['width'], expected['height'])
    (output, preview) = (directory / 'output.mp4', directory / 'preview.mp4')
    media_timeout = max(300, math.ceil(expected['duration_seconds'] * 10))
    command(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(silent), '-i', str(source), '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-t', str(expected['duration_seconds']), '-movflags', '+faststart', str(output)], timeout=media_timeout)
    (w, h, fps) = (expected['width'], expected['height'], expected['frame_rate'])
    command(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(source), '-i', str(output), '-filter_complex_threads', '1', '-filter_complex', f'[0:v]scale={w}:{h}:flags=bicubic,fps={fps},setsar=1[left];[left][1:v]hstack=inputs=2:shortest=1[v]', '-map', '[v]', '-map', '1:a:0?', '-c:v', 'libx264', '-threads', '4', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', str(preview)], timeout=media_timeout)
    (actual, preview_media) = (probe(output), probe(preview))
    for key in ('width', 'height', 'frames', 'frame_rate', 'audio'):
        if actual[key] != expected[key] or preview_media[key] != (expected[key] * 2 if key == 'width' else expected[key]):
            raise MediaError('output_media_mismatch')
    for path in (output, preview):
        command(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], timeout=media_timeout)
    silent.unlink()
    parameters = req.model_dump(exclude={'project_id', 'idempotency_key', 'source_artifact_id', 'source_sha256'})
    parameters.update(audio='aac_if_present', video_codec='h264', pixel_format='yuv420p', crf=18)
    parameters.update(sample_steps=1, cfg_scale=1.0, normalization='torch_rms_layer', color_fix=False, windowing=window_report)
    return {'input_media': media, 'media': actual, 'preview_media': preview_media, 'parameters': parameters}
