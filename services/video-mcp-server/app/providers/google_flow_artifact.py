"""Verify the original job's MP4 before resuming its durable Station upload."""
import hashlib
import os
import re
import subprocess
from tempfile import NamedTemporaryFile
from pathlib import Path

from ..artifact_output import publish


def collect(adapter, record, headers):
    tasks = adapter.executor.tasks
    path = tasks.root / (record.video_task_id + ".google-flow.mp4")
    media = (record.provider_state or {}).get("produced_media")
    if path.is_symlink():
        raise ValueError("invalid_media_file")
    if media is None or not path.exists():
        url = adapter.url + "/v1/jobs/" + record.video_task_id + "/content"
        with adapter.client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            digest = response.headers.get("X-Content-SHA256", "")
            if response.headers.get("content-type","").split(";")[0] != "video/mp4" or not re.fullmatch("[a-f0-9]{64}",digest):
                raise ValueError("invalid_media_metadata")
            if media and digest != media["sha256"]:
                raise ValueError("media_integrity_failed")
            with NamedTemporaryFile(dir=tasks.root, delete=False) as out:
                temp = Path(out.name)
            try:
                size, observed = 0, hashlib.sha256()
                with temp.open("wb") as out:
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > 128 * 1024 * 1024:
                            raise ValueError("media_size_mismatch")
                        observed.update(chunk)
                        out.write(chunk)
                with temp.open("rb") as stream:
                    if stream.read(8)[4:] != b"ftyp" or observed.hexdigest() != digest:
                        raise ValueError("media_integrity_failed")
                from .grok import GrokVideoAdapter
                try:
                    details = GrokVideoAdapter.probe(temp)
                except (subprocess.SubprocessError, StopIteration, ArithmeticError, KeyError):
                    raise ValueError("media_integrity_failed") from None
                width, height = details["width"], details["height"]
                a, b = map(int, record.request["aspect_ratio"].split(":"))
                if (min(width, height) <= 0 or abs(width*b - height*a) > max(a,b)
                        or abs(details["duration_ms"] - 1000*record.request["duration_seconds"]) > 100):
                    raise ValueError("media_spec_mismatch")
                os.replace(temp, path)
                media = {"mime":"video/mp4", "sha256":digest, "size":size, **details}
            finally:
                temp.unlink(missing_ok=True)
    return publish(tasks,record,path,media)
