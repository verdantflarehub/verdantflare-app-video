"""Download Higgsfield media, then register it through Video output handling."""
import hashlib
import os

from ..artifact_output import publish, validate_media
from ..artifact_compat import higgsfield_output_path


def collect(adapter, record, media):
    tasks = adapter.executor.tasks
    path = higgsfield_output_path(tasks, record)
    validate_media(media)
    if path.is_symlink():
        raise ValueError("invalid_media_file")
    if not path.exists():
        temporary = path.with_suffix(".part")
        try:
            digest, size = hashlib.sha256(), 0
            with adapter.client.stream("GET", adapter.url + "/v1/jobs/" + record.video_task_id + "/content",
                                       headers=adapter.headers()) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > media["size"]:
                            raise ValueError("media_size_mismatch")
                        stream.write(chunk)
                        digest.update(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            if size != media["size"] or digest.hexdigest() != media["sha256"]:
                raise ValueError("media_integrity_failed")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return publish(tasks, record, path, media)
