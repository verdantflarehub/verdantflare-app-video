from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from .artifacts import MAX_ARTIFACT_BYTES, ArtifactError
from .tasks import TaskConflict, TaskRecord


FAL_PUBLIC_MODEL = "minimax-h3-ref2va"
FAL_MODEL_ID = "minimax/h3/reference-to-video"
FAL_RUNTIME_VERSION = "video-fal-adapter-v0.2.1"
FAL_RESOLUTION = "480P"
FAL_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
MAX_FAL_IMAGE_BYTES = 20 * 1024 * 1024
MAX_FAL_VIDEO_BYTES = 256 * 1024 * 1024
MAX_FAL_AUDIO_BYTES = 15 * 1024 * 1024
FAL_ASPECT_RATIOS = {"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}


class FalError(RuntimeError):
    pass


class FalAdapter:
    """Server-side fal Queue adapter; provider credentials never enter public records."""

    def __init__(self, executor) -> None:
        self.executor = executor
        self.key = os.environ.get("FAL_KEY", "").strip()
        self.base_url = self._origin(
            os.environ.get("FAL_QUEUE_BASE_URL", "https://queue.fal.run")
        )
        self.model_id = FAL_MODEL_ID
        self.runtime_version = FAL_RUNTIME_VERSION
        self.resolution = FAL_RESOLUTION

    @staticmethod
    def _origin(value: str) -> str:
        parsed = urlsplit(value.strip())
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("fal origins must be absolute HTTPS origins")
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("fal origin has an invalid port") from error
        host = parsed.hostname.lower()
        return f"https://{host}" if port in {None, 443} else f"https://{host}:{port}"

    def connected(self) -> bool:
        return bool(self.key)

    def _headers(self) -> dict[str, str]:
        if not self.key:
            raise FalError("provider_auth_not_configured")
        return {"Authorization": f"Key {self.key}", "Content-Type": "application/json"}

    def _request_url(self, suffix: str = "") -> str:
        return f"{self.base_url}/{self.model_id}{suffix}"

    @staticmethod
    def _safe_error(code: str, message: str) -> dict[str, str]:
        return {"code": code, "message": message}

    @classmethod
    def _response_error(cls, response: httpx.Response, *, submission: bool = False) -> dict[str, str]:
        if response.status_code in {401, 403}:
            return cls._safe_error("provider_auth_failed", "fal rejected the configured API credential")
        if response.status_code == 429:
            return cls._safe_error("provider_rate_limited", "fal temporarily rate limited the request")
        if response.status_code in {400, 404, 409, 422}:
            return cls._safe_error("provider_rejected", "fal rejected the submitted request")
        return cls._safe_error(
            "submission_unconfirmed" if submission else "provider_failed",
            "fal did not return a confirmed provider response",
        )

    @staticmethod
    def _data_uri(media_type: str, payload: bytes, maximum: int) -> str:
        if not payload or len(payload) > maximum:
            raise ValueError("fal reference exceeds the approved size limit")
        return f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"

    @staticmethod
    def _reference_duration(payload: bytes, suffix: str) -> float:
        with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
            temp.write(payload)
            temp.flush()
            try:
                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "json",
                        temp.name,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                duration = float(json.loads(probe.stdout)["format"]["duration"])
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
                raise ArtifactError("timed reference metadata is invalid") from error
        if not 2 <= duration <= 15:
            raise ArtifactError("timed fal references must be between 2 and 15 seconds")
        return duration

    def _artifact_data_uri(self, artifact, maximum: int, *, timed: bool = False) -> tuple[str, float | None]:
        content = self.executor.artifacts.content_path(artifact).read_bytes()
        if len(content) != artifact.size or hashlib.sha256(content).hexdigest() != artifact.sha256:
            raise ArtifactError("artifact content does not match its immutable metadata")
        duration = self._reference_duration(content, os.path.splitext(artifact.filename)[1]) if timed else None
        return self._data_uri(artifact.media_type, content, maximum), duration

    def normalize(
        self,
        *,
        project_id: str,
        model: str,
        prompt: str,
        duration_seconds: int,
        aspect_ratio: str,
        references: dict[str, list[dict[str, str]]],
    ) -> tuple[dict[str, object], str]:
        if model != FAL_PUBLIC_MODEL:
            raise ValueError(f"model must be {FAL_PUBLIC_MODEL} for route fal")
        if not prompt.strip() or not 5 <= duration_seconds <= 15:
            raise ValueError("prompt or duration_seconds is invalid for route fal")
        images = references.get("images", [])
        videos = references.get("videos", [])
        audios = references.get("audios", [])
        if len(images) > 9 or len(videos) > 3 or len(audios) > 3:
            raise ValueError("fal supports at most nine images, three videos, and three audio references")
        reference_count = len(images) + len(videos) + len(audios)
        if reference_count > 12:
            raise ValueError("fal supports at most twelve references in total")
        if not reference_count:
            raise ValueError("fal reference-to-video requires at least one reference")
        if aspect_ratio not in FAL_ASPECT_RATIOS:
            raise ValueError("aspect_ratio is invalid for fal reference-to-video")

        normalized: dict[str, list[dict[str, str]]] = {"images": [], "videos": [], "audios": []}
        for kind, items, prefix in (
            ("images", images, "image/"),
            ("videos", videos, "video/"),
            ("audios", audios, "audio/"),
        ):
            for item in items:
                artifact = self.executor.artifacts.get(item["artifact_id"], project_id)
                purpose = item.get("purpose", "").strip()
                if not artifact.media_type.startswith(prefix) or not purpose:
                    raise ValueError(f"invalid {kind} reference")
                normalized[kind].append(
                    {"artifact_id": artifact.artifact_id, "purpose": purpose}
                )

        request: dict[str, object] = {
            "project_id": project_id,
            "model": model,
            "prompt": prompt.strip(),
            "duration_seconds": duration_seconds,
            "aspect_ratio": aspect_ratio,
            "references": normalized,
            "route": "fal",
        }
        canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return request, "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    def _payload(self, record: TaskRecord) -> dict[str, object]:
        request = record.request
        references = request["references"]
        payload: dict[str, object] = {
            "duration": request["duration_seconds"],
            "resolution": self.resolution,
            "enable_safety_checker": True,
            "sync_mode": False,
            "prompt_expansion_mode": "disabled",
            "aspect_ratio": request["aspect_ratio"],
        }
        tags: list[str] = []
        mapping = (
            ("images", "Image", "reference_image_urls", MAX_FAL_IMAGE_BYTES, False),
            ("videos", "Video", "reference_video_urls", MAX_FAL_VIDEO_BYTES, True),
            ("audios", "Audio", "reference_audio_urls", MAX_FAL_AUDIO_BYTES, True),
        )
        for kind, label, field, maximum, timed in mapping:
            values: list[str] = []
            combined_duration = 0.0
            for index, item in enumerate(references[kind], start=1):
                artifact = self.executor.artifacts.get(item["artifact_id"], record.project_id)
                value, reference_duration = self._artifact_data_uri(artifact, maximum, timed=timed)
                values.append(value)
                combined_duration += reference_duration or 0.0
                tags.append(f"{label} {index} is the approved {item['purpose']} reference")
            if timed and combined_duration > 15:
                raise ArtifactError(f"combined fal {kind} duration exceeds 15 seconds")
            if values:
                payload[field] = values
        payload["prompt"] = "; ".join(tags) + ". " + str(request["prompt"])
        return payload

    def generate(
        self,
        *,
        project_id: str,
        idempotency_key: str,
        model: str,
        prompt: str,
        duration_seconds: int,
        aspect_ratio: str,
        references: dict[str, list[dict[str, str]]],
    ) -> TaskRecord:
        request, digest = self.normalize(
            project_id=project_id,
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
            references=references,
        )
        existing = self.executor.tasks.find_idempotency(project_id, idempotency_key)
        if existing:
            if existing.input_digest != digest:
                raise TaskConflict("idempotency key already exists with different input")
            return existing
        record = self.executor.tasks.create(
            project_id=project_id,
            idempotency_key=idempotency_key,
            input_digest=digest,
            request=request,
            runtime_task_id="",
            status="queued",
        )
        record = self.executor.tasks.update(
            record,
            service="fal",
            runtime_version=self.runtime_version,
            runtime_route="fal",
        )
        if not self.key:
            return self.executor.tasks.update(
                record,
                status="failed",
                error=self._safe_error(
                    "provider_auth_not_configured", "fal API credentials are not configured"
                ),
            )
        try:
            payload = self._payload(record)
        except (ArtifactError, OSError, ValueError):
            return self.executor.tasks.update(
                record,
                status="failed",
                error=self._safe_error(
                    "source_integrity_failed", "a fal reference artifact could not be read safely"
                ),
            )
        try:
            response = self.executor.client.post(
                self._request_url(),
                headers=self._headers(),
                json=payload,
                timeout=60,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self.executor.tasks.update(
                record,
                status="failed",
                error=self._safe_error("provider_unavailable", "fal is unreachable; generation was not submitted"),
            )
        except httpx.HTTPError:
            return self.executor.tasks.update(
                record,
                status="failed",
                error=self._safe_error(
                    "submission_unconfirmed", "fal submission could not be confirmed; do not resubmit automatically"
                ),
            )
        if not response.is_success:
            return self.executor.tasks.update(
                record, status="failed", error=self._response_error(response, submission=True)
            )
        try:
            request_id = response.json()["request_id"]
            if not isinstance(request_id, str) or not FAL_REQUEST_ID_PATTERN.fullmatch(request_id):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return self.executor.tasks.update(
                record,
                status="failed",
                error=self._safe_error(
                    "submission_unconfirmed", "fal submission could not be confirmed; do not resubmit automatically"
                ),
            )
        return self.executor.tasks.update(
            record,
            runtime_task_id=request_id,
            dispatched_at=datetime.now(UTC).isoformat(),
        )

    def status(self, record: TaskRecord) -> TaskRecord:
        if record.status in {"succeeded", "failed", "cancelled"}:
            return record
        if not record.runtime_task_id or not FAL_REQUEST_ID_PATTERN.fullmatch(record.runtime_task_id):
            return self.executor.tasks.update(
                record,
                status="failed",
                error=self._safe_error("remote_state_unknown", "fal request state is unavailable"),
            )
        if not self.key:
            return self.executor.tasks.update(
                record,
                status="failed",
                error=self._safe_error(
                    "provider_auth_not_configured", "fal API credentials are not configured"
                ),
            )
        try:
            response = self.executor.client.get(
                self._request_url(f"/requests/{record.runtime_task_id}/status"),
                headers=self._headers(),
                timeout=15,
            )
        except httpx.HTTPError as error:
            raise FalError("provider_unavailable") from error
        if response.status_code == 404:
            return self.executor.tasks.update(
                record,
                status="failed",
                error=self._safe_error("runtime_task_lost", "fal no longer recognizes the remote task"),
            )
        if not response.is_success:
            error = self._response_error(response)
            if error["code"] in {"provider_auth_failed", "provider_rejected"}:
                return self.executor.tasks.update(record, status="failed", error=error)
            raise FalError(error["code"])
        try:
            data = response.json()
            remote_status = data["status"]
        except (ValueError, KeyError, TypeError) as error:
            raise FalError("provider_failed") from error
        if remote_status == "IN_QUEUE":
            return self.executor.tasks.update(record, status="queued", runtime_stage="remote_queued")
        if remote_status == "IN_PROGRESS":
            return self.executor.tasks.update(record, status="running", runtime_stage="remote_running")
        if remote_status != "COMPLETED":
            raise FalError("provider_failed")
        if data.get("error") or data.get("error_type"):
            return self.executor.tasks.update(
                record,
                status="failed",
                runtime_stage="remote_failed",
                error=self._safe_error("provider_failed", "fal reported that video generation failed"),
            )
        metrics = data.get("metrics")
        runtime_metrics = None
        if isinstance(metrics, dict) and isinstance(metrics.get("inference_time"), (int, float)):
            runtime_metrics = {"runtime_total_seconds": float(metrics["inference_time"])}
        return self.executor.tasks.update(
            record,
            status="succeeded",
            runtime_stage="completed",
            runtime_metrics=runtime_metrics,
        )

    def _allowed_result_url(self, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise FalError("result_download_failed")
        try:
            port = parsed.port
        except ValueError as error:
            raise FalError("result_download_failed") from error
        host = parsed.hostname.lower()
        origin = f"https://{host}" if port in {None, 443} else f"https://{host}:{port}"
        official = host == "fal.media" or host.endswith(".fal.media")
        if not official:
            raise FalError("result_download_failed")
        return value

    def _download(self, url: str) -> bytes:
        data = bytearray()
        try:
            with self.executor.client.stream(
                "GET", self._allowed_result_url(url), headers={"Accept": "video/*"}, timeout=600
            ) as response:
                if not response.is_success:
                    raise FalError("result_download_failed")
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
                if content_type not in {"video/mp4", "application/octet-stream"}:
                    raise FalError("result_integrity_failed")
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError as error:
                        raise FalError("result_integrity_failed") from error
                    if not 0 < declared_size <= MAX_ARTIFACT_BYTES:
                        raise FalError("result_integrity_failed")
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_ARTIFACT_BYTES:
                        raise FalError("result_integrity_failed")
        except httpx.HTTPError as error:
            raise FalError("result_download_failed") from error
        if not data:
            raise FalError("result_integrity_failed")
        return bytes(data)

    @staticmethod
    def _probe(payload: bytes) -> dict[str, object]:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp:
            temp.write(payload)
            temp.flush()
            try:
                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
                        "-of",
                        "json",
                        temp.name,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                data = json.loads(probe.stdout)
            except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
                raise FalError("result_integrity_failed") from error
        video = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), None)
        audio = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"), None)
        if not video or not video.get("codec_name") or not video.get("width") or not video.get("height"):
            raise FalError("result_integrity_failed")
        try:
            duration_ms = round(float(data["format"]["duration"]) * 1000)
            numerator, denominator = str(video.get("r_frame_rate", "0/1")).split("/", 1)
            frame_rate = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError, KeyError, TypeError) as error:
            raise FalError("result_integrity_failed") from error
        if duration_ms <= 0 or frame_rate <= 0:
            raise FalError("result_integrity_failed")
        return {
            "duration_ms": duration_ms,
            "width": int(video["width"]),
            "height": int(video["height"]),
            "frame_rate": int(frame_rate) if frame_rate.is_integer() else round(frame_rate, 3),
            "video_codec": video["codec_name"],
            "audio_codec": audio.get("codec_name") if audio else None,
            "audio_sample_rate": int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
            "audio_channels": audio.get("channels") if audio else None,
        }

    def result(self, record: TaskRecord) -> TaskRecord:
        record = self.status(record)
        if record.status != "succeeded":
            raise FalError("video_task_not_succeeded")
        if record.artifact_id:
            return record
        try:
            response = self.executor.client.get(
                self._request_url(f"/requests/{record.runtime_task_id}"),
                headers=self._headers(),
                timeout=30,
            )
        except httpx.HTTPError as error:
            raise FalError("provider_unavailable") from error
        if not response.is_success:
            raise FalError(self._response_error(response)["code"])
        try:
            video_url = response.json()["video"]["url"]
            if not isinstance(video_url, str):
                raise ValueError
        except (ValueError, KeyError, TypeError) as error:
            raise FalError("result_integrity_failed") from error
        payload = self._download(video_url)
        media = self._probe(payload)
        artifact = self.executor.artifacts.create_from_chunks(
            project_id=record.project_id,
            operation="video.result",
            filename=f"{record.video_task_id}.mp4",
            media_type="video/mp4",
            chunks=(payload,),
        )
        return self.executor.tasks.update(record, artifact_id=artifact.artifact_id, media=media)
