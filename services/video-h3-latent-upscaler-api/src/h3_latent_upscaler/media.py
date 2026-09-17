"""Encode decoded latent frames and retain the source audio, with full validation."""
import json
from pathlib import Path
import subprocess
import tempfile
from .resources import ResourceError


def probe(path):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-show_format', '-of', 'json', str(path)]))
    video = [s for s in data['streams'] if s['codec_type'] == 'video']
    audio = [s for s in data['streams'] if s['codec_type'] == 'audio']
    if len(video) != 1 or len(audio) > 1:
        raise ResourceError('invalid_media_streams')
    stream = video[0]
    if stream.get('r_frame_rate') != '24/1' or stream.get('avg_frame_rate') != '24/1':
        raise ResourceError('invalid_frame_rate')
    if stream.get('codec_name') != 'h264' or stream.get('pix_fmt') != 'yuv420p':
        raise ResourceError('invalid_video_format')
    if abs(float(stream.get('start_time', 0))) > 0.001 or any(abs(float(s.get('start_time', 0))) > .001 for s in audio):
        raise ResourceError('nonzero_media_start')
    if audio and audio[0]['codec_name'] != 'aac':
        raise ResourceError('unsupported_audio_codec')
    return {'width': stream['width'], 'height': stream['height'], 'frames': int(stream['nb_read_frames']),
            'frame_rate': '24/1', 'video_codec': 'h264', 'audio': bool(audio),
            'duration_seconds': int(stream['nb_read_frames']) / 24}


def decode_check(path):
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def finish_video(frames, source, directory):
    import torch
    source_media = probe(source)
    if frames.ndim != 4 or frames.shape[-1] != 3 or not torch.isfinite(frames).all():
        raise ResourceError('invalid_decoded_frames')
    count, height, width, _ = frames.shape
    if count != source_media['frames']:
        raise ResourceError('timeline_mismatch')
    directory = Path(directory)
    output, preview = directory / 'output.mp4', directory / 'preview.mp4'
    command = ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', '24', '-i', 'pipe:0', '-i', str(source),
               '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
               '-crf', '18', '-c:a', 'copy', '-movflags', '+faststart', str(output)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for frame in frames:
                process.stdin.write((frame.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise ResourceError('video_encode_failed')
        except BaseException:
            process.kill()
            process.wait()
            raise
    measured = probe(output)
    if measured != {**source_media, 'width': width, 'height': height}:
        raise ResourceError('output_media_mismatch')
    decode_check(output)
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(source), '-i', str(output),
        '-filter_complex', f'[0:v]scale={width}:{height}:flags=lanczos[left];[left][1:v]hstack=inputs=2[v]',
        '-map', '[v]', '-map', '1:a:0?', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        '-c:a', 'copy', '-movflags', '+faststart', str(preview)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    preview_media = probe(preview)
    if preview_media != {**measured, 'width': width * 2}:
        raise ResourceError('preview_media_mismatch')
    decode_check(preview)
    return measured
