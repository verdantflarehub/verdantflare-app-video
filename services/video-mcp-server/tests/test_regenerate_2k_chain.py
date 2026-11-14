from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from jsonschema import Draft202012Validator
from PIL import Image
import yaml

from app.artifacts import ArtifactStore
from app.catalog import ModelCatalog
from app.dashboard import Dashboard, task_channel, task_presentation
from app.delivery import CALLER, OPTION, normalize
from app.executor import ExecutionError, VideoExecutor
from app.gpu_lease import LeaseBusy, LocalGpuStageSlot
from app.regenerate import (
    LOCKED_P2_CODE_REVISION,
    LOCKED_P2_WEIGHTS_SHA256,
    is_2k_canvas,
    preview_geometry,
    resolve_2k_target_geometry,
    temporal_windows,
)
from app.tasks import TaskConflict, TaskStore


REPO_ROOT = Path(__file__).resolve().parents[4]
APP_VIDEO_ROOT = Path(__file__).resolve().parents[3]
FASTH3_SRC = APP_VIDEO_ROOT / "services" / "video-minimax-h3-fasth3" / "src"
REGEN2K_SRC = APP_VIDEO_ROOT / "services" / "video-minimax-h3-regenerate-2k" / "src"
SINGULARITY_SRC = APP_VIDEO_ROOT / "services" / "video-minimax-h3-singularity" / "src"
RIFE_SRC = APP_VIDEO_ROOT / "services" / "video-frame-interpolation-api"


def _png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(20, 40, 80)).save(buf, format="PNG")
    return buf.getvalue()


def _load_sibling_module(package_root: Path, dotted_name: str):
    import types
    added = False
    stubbed: list[str] = []
    root_str = str(package_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added = True
    for mod_name in ("cv2", "numpy"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
            stubbed.append(mod_name)
    try:
        return importlib.import_module(dotted_name)
    finally:
        if added and root_str in sys.path:
            sys.path.remove(root_str)
        for mod_name in stubbed:
            sys.modules.pop(mod_name, None)


def _make_p2_result_payload(
    *,
    input_sha256: str,
    output_sha256: str,
    preview_sha256: str,
    src_w: int = 1344,
    src_h: int = 768,
    frames: int = 124,
    grid_mode: str = "budget_1440p",
    prompt: str = "cinematic portrait in rain",
    seed: int = 7,
    has_audio: bool = True,
) -> dict:
    geo = resolve_2k_target_geometry(src_w, src_h, grid_mode)
    target_w, target_h = geo["target_w"], geo["target_h"]
    pw, ph = preview_geometry(target_w, target_h)
    duration = frames / 24.0
    manifest_sha256 = hashlib.sha256(
        json.dumps(LOCKED_P2_WEIGHTS_SHA256, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    audio_mode = "copy" if has_audio else "none"
    input_media = {
        "width": src_w,
        "height": src_h,
        "frames": frames,
        "frame_rate": "24",
        "duration_seconds": duration,
        "video_codec": "h264",
        "audio": has_audio,
    }
    media = {
        "width": target_w,
        "height": target_h,
        "frames": frames,
        "frame_rate": "24",
        "duration_seconds": duration,
        "video_codec": "h264",
        "audio": has_audio,
        "audio_mode": audio_mode,
    }
    if has_audio:
        media["audio_sha256"] = "a" * 64
    preview_media = {
        "width": pw,
        "height": ph,
        "frames": frames,
        "frame_rate": "24",
        "duration_seconds": duration,
        "video_codec": "h264",
        "audio": has_audio,
    }
    return {
        "input_media": input_media,
        "media": media,
        "preview_media": preview_media,
        "geometry": {
            "source_w": src_w,
            "source_h": src_h,
            "target_w": target_w,
            "target_h": target_h,
            "levels": geo["levels"],
        },
        "windows": [[ws, we] for ws, we in temporal_windows(frames)],
        "parameters": {
            "profile": "open_h3_2k_ic",
            "grid_mode": grid_mode,
            "seed": seed,
            "prompt_sha256": prompt_sha256,
            "audio": audio_mode,
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "crf": 16,
            "tiles": 2,
        },
        "model": {
            "backend": "open_h3_2k_ic",
            "profile": "open_h3_2k_ic",
            "code_revision": LOCKED_P2_CODE_REVISION,
            "weights_revision": f"sha256:{manifest_sha256}",
            "manifest_sha256": manifest_sha256,
            "weights_sha256": dict(LOCKED_P2_WEIGHTS_SHA256),
        },
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "preview_sha256": preview_sha256,
        "runtime_metrics": {
            "p2a_cond_seconds": 1.2,
            "p2b_upscale_seconds": 0.4,
            "p2c_refine_seconds": 95.0,
            "p2d_decode_encode_seconds": 18.0,
            "vm_hwm_bytes": 16 * 1024**3,
            "gpu_peak_reserved_bytes": 30 * 1024**3,
            "gpu_uuid": "GPU-test-5090",
            "free_vram_mib_before": 31800,
        },
    }


def _make_p3_result_payload(
    *,
    input_sha256: str,
    output_sha256: str,
    preview_sha256: str,
    width: int,
    height: int,
    src_frames: int = 124,
    has_audio: bool = True,
) -> dict:
    pw, ph = preview_geometry(width, height)
    duration = src_frames / 24.0
    source = {
        "width": width,
        "height": height,
        "frames": src_frames,
        "frame_rate": "24",
        "duration_seconds": duration,
        "video_codec": "h264",
        "audio": has_audio,
    }
    media = {
        **source,
        "frames": src_frames * 2,
        "frame_rate": "48",
        "audio_mode": "copy" if has_audio else "none",
    }
    preview = {
        **media,
        "width": pw,
        "height": ph,
    }
    return {
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "preview_sha256": preview_sha256,
        "media": media,
        "input_media": source,
        "preview_media": preview,
        "model": {
            "backend": "rife",
            "code_revision": "a" * 40,
            "weights_revision": "sha256:" + "b" * 64,
            "manifest_sha256": "c" * 64,
            "weights_sha256": {"flownet.pkl": "d" * 64},
        },
        "parameters": {
            "target_fps_num": 48,
            "target_fps_den": 1,
            "backend": "rife",
            "audio": "aac_copy_if_aligned",
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "crf": 18,
            "scene_cut_threshold": 0.30,
            "last_frame": "hold",
        },
    }


class CrossRepoPureContractTest(unittest.TestCase):
    def test_cross_repo_preview_geometry_2k_grid_and_windows_are_byte_identical(self):
        fasth3_contract = _load_sibling_module(FASTH3_SRC, "h3_fasth3.contract")
        regen_contract = _load_sibling_module(REGEN2K_SRC, "h3_regenerate_2k.contract")
        sing_contract = _load_sibling_module(SINGULARITY_SRC, "h3_singularity.contract")
        rife_media = _load_sibling_module(RIFE_SRC, "video_interpolation.media")

        golden_previews = {
            (768, 1344): (1536, 1344),
            (1344, 768): (2688, 768),
            (2464, 1408): (3840, 1096),
            (1408, 2464): (2468, 2160),
            (1536, 2688): (2468, 2160),
            (3072, 1344): (3840, 840),
        }
        for (w, h), expected in golden_previews.items():
            self.assertEqual(preview_geometry(w, h), expected)
            self.assertEqual(regen_contract.preview_geometry(w, h), expected)
            self.assertEqual(rife_media.preview_geometry(w, h), expected)

        self.assertEqual(fasth3_contract.resolve_geometry("16:9", None), sing_contract.resolve_geometry("16:9", None))
        self.assertEqual(fasth3_contract.resolve_timing(15), sing_contract.resolve_timing(15))

        from types import SimpleNamespace
        rife_req = SimpleNamespace(target_fps_num=48, target_fps_den=1)
        validated_3072 = rife_media.validate_request(rife_req, {"width": 3072, "height": 1344, "frame_rate": "24", "frames": 124})
        self.assertEqual((validated_3072["width"], validated_3072["frames"]), (3072, 248))
        with self.assertRaises(rife_media.MediaError):
            rife_media.validate_request(rife_req, {"width": 3074, "height": 1344, "frame_rate": "24", "frames": 124})

        native_canvases = [
            (1536, 672),
            (1344, 768),
            (1024, 768),
            (768, 768),
            (768, 1024),
            (768, 1344),
        ]
        for sw, sh in native_canvases:
            for gmode in ("budget_1440p", "exact_2x"):
                mcp_geo = resolve_2k_target_geometry(sw, sh, gmode)
                regen_geo = regen_contract.resolve_2k_target_geometry(sw, sh, gmode)
                self.assertEqual(mcp_geo, regen_geo)
                self.assertLessEqual(max(mcp_geo["target_w"], mcp_geo["target_h"]), 3072)
                self.assertTrue(is_2k_canvas(mcp_geo["target_w"], mcp_geo["target_h"]))

        for frames in range(107, 363, 17):
            self.assertEqual(temporal_windows(frames), regen_contract.temporal_windows(frames))


class Regenerate2kChainAndTimeoutTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env_patch = patch.dict(
            os.environ,
            {
                "H3_RUNTIME_URL": "http://fasth3.local:8000",
                "H3_RUNTIME_ROUTE": "h3-fasth3",
                "H3_RUNTIME_VERSION": "video-minimax-h3-fasth3-v0.1.0",
                "H3_FASTH3_RUNTIME_TOKEN": "fasth3-token",
                "H3_SINGULARITY_RUNTIME_TOKEN": "sing-token",
                "H3_RUNTIME_ROUTES": json.dumps(
                    {
                        "h3-singularity": {
                            "url": "http://singularity.local:8000",
                            "version": "video-minimax-h3-singularity-v0.1.40",
                            "requires_token": True,
                        }
                    }
                ),
                "VIDEO_REGENERATE_2K_RUNTIME_URL": "http://regenerate2k.local:8000",
                "VIDEO_REGENERATE_2K_RUNTIME_TOKEN": "regen2k-token",
                "VIDEO_REGENERATE_2K_RUNTIME_VERSION": "video-minimax-h3-regenerate-2k-v0.1.0",
                "VIDEO_INTERPOLATION_RUNTIME_URL": "http://rife.local:8080",
                "VIDEO_PROJECT_SERVICE_URL": "http://project.local",
                "VIDEO_PROJECT_SERVICE_TOKEN": "proj-token",
                "VIDEO_RUNTIME_ARTIFACT_BASE_URL": "http://mcp.local:8080",
            },
            clear=False,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

        self.caller_token = CALLER.set({"user_id": "u-1", "organization_id": "org-1"})
        self.addCleanup(lambda: CALLER.reset(self.caller_token))

        self.requests: list[httpx.Request] = []
        self.p1_status = "completed"
        self.p1_started_at = datetime.now(UTC).isoformat()
        self.p1_worker_settled = True
        self.p2_status = "succeeded"
        self.p2_started_at = datetime.now(UTC).isoformat()
        self.p2_worker_settled = True
        self.p3_status = "succeeded"
        self.p3_started_at = datetime.now(UTC).isoformat()
        self.p3_worker_settled = True
        self.health_default_profile = "fasth3-nvfp4"
        self.fasth3_gpu_active = False
        self.regen2k_gpu_active = False
        self.regen2k_ready = True
        self.singularity_reachable = True
        self.singularity_ready = True
        self.singularity_gpu_active = False
        self.rife_gpu_active = False

        self.m1_bytes = b"m1-native-768p-mp4"
        self.m1_sha = hashlib.sha256(self.m1_bytes).hexdigest()
        self.m2_bytes = b"m2-regenerate-2k-24fps-mp4"
        self.m2_sha = hashlib.sha256(self.m2_bytes).hexdigest()
        self.m3_bytes = b"m3-rife-2k-48fps-mp4"
        self.m3_sha = hashlib.sha256(self.m3_bytes).hexdigest()

        self.artifacts = ArtifactStore(self.root)
        self.tasks = TaskStore(self.root)
        self.ref_image = self.artifacts.create_from_chunks(
            project_id="proj_1",
            operation="import",
            filename="ref.png",
            media_type="image/png",
            chunks=[_png_bytes(1344, 768)],
        )
        self.client = httpx.Client(transport=httpx.MockTransport(self._transport))
        self.addCleanup(self.client.close)
        self.executor = VideoExecutor(self.artifacts, self.tasks, self.client)

        self.p2_submitted: dict[str, dict] = {}
        self.p3_submitted: dict[str, dict] = {}
        self.deleted_paths: list[str] = []

    def _transport(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        path = request.url.path

        if host == "project.local" and path == "/internal/v1/project/open":
            return httpx.Response(200, json={"ok": True})

        if host == "fasth3.local":
            if path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "ready": True,
                        "default_quality_profile": self.health_default_profile,
                        "gpu_worker_active": self.fasth3_gpu_active,
                        "active_task_ids": ["fasth3-active-1"] if self.fasth3_gpu_active else [],
                        "capabilities": {
                            "quality_profiles": ["fasth3-nvfp4"],
                            "image_guidance_modes": ["reference", "first_frame"],
                            "audio_guidance_modes": ["reference", "timeline"],
                        },
                    },
                )
            if path == "/v1/videos" and request.method == "POST":
                return httpx.Response(202, json={"id": "fasth3_" + "1" * 32, "status": "queued"})
            if path.startswith("/v1/videos/") and request.method == "DELETE":
                self.deleted_paths.append(path)
                return httpx.Response(
                    200,
                    json={
                        "id": path.rsplit("/", 1)[-1],
                        "status": "cancelled",
                        "started_at": self.p1_started_at,
                        "worker_settled": self.p1_worker_settled,
                    },
                )
            if path.endswith("/content") and "/v1/videos/" in path:
                return httpx.Response(200, content=self.m1_bytes, headers={"content-type": "video/mp4"})
            if path.startswith("/v1/videos/") and request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "id": "fasth3_" + "1" * 32,
                        "status": self.p1_status,
                        "stage": "completed" if self.p1_status == "completed" else "generating",
                        "started_at": self.p1_started_at,
                        "worker_settled": self.p1_worker_settled,
                        "result": {
                            "content_sha256": self.m1_sha,
                            "generate_seconds": 19.3,
                            "width": 1344,
                            "height": 768,
                            "frames": 124,
                            "fps": 24,
                        },
                    },
                )

        if host == "singularity.local":
            if not self.singularity_reachable:
                raise httpx.ConnectError("cold standby replicas=0", request=request)
            if path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "ready": self.singularity_ready,
                        "default_quality_profile": "du-0",
                        "gpu_worker_active": self.singularity_gpu_active,
                        "active_task_ids": ["sing-active-1"] if self.singularity_gpu_active else [],
                        "capabilities": {
                            "quality_profiles": ["du-0"],
                            "image_guidance_modes": ["reference", "first_frame"],
                            "audio_guidance_modes": ["reference", "timeline"],
                        },
                    },
                )
            if path == "/v1/videos" and request.method == "POST":
                return httpx.Response(202, json={"id": "singularity_" + "1" * 32, "status": "queued"})
            if path.startswith("/v1/videos/") and request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "id": path.rsplit("/", 1)[-1],
                        "status": self.p1_status,
                        "stage": "completed" if self.p1_status == "completed" else "generating",
                        "started_at": self.p1_started_at,
                        "worker_settled": self.p1_worker_settled,
                    },
                )

        if host == "regenerate2k.local":
            if path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "ready": self.regen2k_ready,
                        "status": "ok" if self.regen2k_ready else "degraded",
                        "service": "regenerate_2k",
                        "gpu_worker_active": self.regen2k_gpu_active,
                        "active_task_ids": ["regen-active-1"] if self.regen2k_gpu_active else [],
                        "capabilities": {
                            "regenerate_2k_profiles": ["open_h3_2k_ic"],
                            "regenerate_2k_grid_modes": ["budget_1440p", "exact_2x"],
                        },
                    },
                )
            if path == "/v1/regenerate-2k" and request.method == "POST":
                body = json.loads(request.content)
                tid = body["idempotency_key"]
                self.p2_submitted[tid] = body
                return httpx.Response(202, json={"video_task_id": tid, "status": "queued", "worker_settled": False})
            if path.startswith("/v1/regenerate-2k/") and request.method == "DELETE":
                self.deleted_paths.append(path)
                tid = path.rsplit("/", 1)[-1]
                return httpx.Response(
                    200,
                    json={
                        "video_task_id": tid,
                        "status": "cancelled",
                        "started_at": self.p2_started_at,
                        "worker_settled": self.p2_worker_settled,
                    },
                )
            if path.startswith("/v1/regenerate-2k/") and path.endswith(("/content", "/preview")):
                return httpx.Response(200, content=self.m2_bytes, headers={"content-type": "video/mp4"})
            if path.startswith("/v1/regenerate-2k/") and request.method == "GET":
                tid = path.rsplit("/", 1)[-1]
                rec = self.tasks.get(tid)
                frames = 124
                if isinstance(rec.provider_state, dict) and isinstance(rec.provider_state.get("source_frames"), int):
                    frames = rec.provider_state["source_frames"]
                payload = {
                    "video_task_id": tid,
                    "status": self.p2_status,
                    "started_at": self.p2_started_at,
                    "worker_settled": self.p2_worker_settled,
                }
                if self.p2_status == "succeeded":
                    payload["result"] = _make_p2_result_payload(
                        input_sha256=rec.request["source_sha256"],
                        output_sha256=self.m2_sha,
                        preview_sha256=self.m2_sha,
                        frames=frames,
                        grid_mode=rec.request["grid_mode"],
                        prompt=rec.request["prompt"],
                        seed=rec.request["seed"],
                    )
                elif self.p2_status == "failed":
                    payload["error"] = {"code": "engine_failed", "message": "P2 failed"}
                return httpx.Response(200, json=payload)

        if host == "rife.local":
            if path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "service": "interpolate",
                        "gpu_worker_active": self.rife_gpu_active,
                        "active_task_ids": ["rife-1"] if self.rife_gpu_active else [],
                    },
                )
            if path == "/v1/interpolate" and request.method == "POST":
                body = json.loads(request.content)
                tid = body["idempotency_key"]
                self.p3_submitted[tid] = body
                return httpx.Response(202, json={"video_task_id": tid, "status": "queued", "worker_settled": False})
            if path.startswith("/v1/interpolate/") and request.method == "DELETE":
                self.deleted_paths.append(path)
                tid = path.rsplit("/", 1)[-1]
                return httpx.Response(
                    200,
                    json={
                        "video_task_id": tid,
                        "status": "cancelled",
                        "started_at": self.p3_started_at,
                        "worker_settled": self.p3_worker_settled,
                    },
                )
            if path.startswith("/v1/interpolate/") and path.endswith(("/content", "/preview")):
                return httpx.Response(200, content=self.m3_bytes, headers={"content-type": "video/mp4"})
            if path.startswith("/v1/interpolate/") and request.method == "GET":
                tid = path.rsplit("/", 1)[-1]
                rec = self.tasks.get(tid)
                is_2k = ":2k" in rec.idempotency_key
                width, height = (2464, 1408) if is_2k else (1344, 768)
                payload = {
                    "video_task_id": tid,
                    "status": self.p3_status,
                    "started_at": self.p3_started_at,
                    "worker_settled": self.p3_worker_settled,
                }
                if self.p3_status == "succeeded":
                    payload["result"] = _make_p3_result_payload(
                        input_sha256=rec.request["source_sha256"],
                        output_sha256=self.m3_sha,
                        preview_sha256=self.m3_sha,
                        width=width,
                        height=height,
                    )
                elif self.p3_status == "failed":
                    payload["error"] = {"code": "processing_failed", "message": "P3 failed"}
                return httpx.Response(200, json=payload)

        raise AssertionError(f"Unhandled mock request: {request.method} {request.url}")

    def _base_gen_kwargs(self, key: str = "k-1", duration_seconds: int = 5, route: str | None = None) -> dict:
        kw = {
            "project_id": "proj_1",
            "idempotency_key": key,
            "model": "minimax-h3-ref2va",
            "prompt": "cinematic portrait in rain",
            "duration_seconds": duration_seconds,
            "aspect_ratio": "16:9",
            "references": {
                "images": [{"artifact_id": self.ref_image.artifact_id, "purpose": "character"}],
                "videos": [],
                "audios": [],
            },
        }
        if route is not None:
            kw["route"] = route
        return kw

    def _seed_native_h3_task(
        self,
        key: str = "seed-m1",
        frames: int = 124,
        *,
        route: str = "h3-singularity",
    ) -> tuple[str, str]:
        m1_art = self.artifacts.create_from_chunks(
            project_id="proj_1",
            operation="video.generate",
            filename="m1.mp4",
            media_type="video/mp4",
            chunks=[self.m1_bytes],
        )
        prefix = "fasth3_" if route == "h3-fasth3" else "singularity_"
        with self.tasks.exclusive():
            rec = self.tasks.create(
                project_id="proj_1",
                idempotency_key=key,
                input_digest=f"sha256:{key}",
                request={
                    "project_id": "proj_1",
                    "route": route,
                    "model": "minimax-h3-ref2va",
                    "prompt": "cinematic portrait in rain",
                    "resolved_output": {"width": 1344, "height": 768, "frames": frames, "fps": 24},
                },
                runtime_task_id=prefix + "2" * 32,
                status="succeeded",
            )
            rec = self.tasks.update(
                rec,
                artifact_id=m1_art.artifact_id,
                media={
                    "width": 1344,
                    "height": 768,
                    "frames": frames,
                    "frame_rate": "24",
                    "duration_seconds": frames / 24.0,
                    "video_codec": "h264",
                    "audio": True,
                },
            )
        return rec.video_task_id, m1_art.artifact_id

    def test_v1_v2_v7_default_fasth3_route_profile_rejection_and_capabilities(self):
        self.assertEqual(normalize({"interpolate": {"profile": "rife_2x"}}, "h3-fasth3"), OPTION)
        self.assertEqual(normalize({"interpolate": {"profile": "rife_2x"}}, "h3-singularity"), OPTION)

        # V2: mismatched quality_profile rejected before persistence or HTTP dispatch.
        with self.assertRaisesRegex(ValueError, "quality_profile_not_available"):
            self.executor.generate(**self._base_gen_kwargs("bad-fh3-du0", route="h3-fasth3"), quality_profile="du-0")
        with self.assertRaisesRegex(ValueError, "quality_profile_not_available"):
            self.executor.generate(
                **self._base_gen_kwargs("bad-sing-fh3", route="h3-singularity"),
                quality_profile="fasth3-nvfp4",
            )
        self.assertEqual(len(list(self.tasks.root.glob("video_task_*.json"))), 0)
        self.assertEqual(len([r for r in self.requests if r.method == "POST"]), 0)

        # V1 & V7: video.create without route defaults to h3-fasth3 + fasth3-nvfp4 + 8 NFE.
        with patch.object(self.executor, "catalog_view", return_value={}):
            caps = self.executor.capabilities()
        self.assertEqual(caps["default_route"], "h3-fasth3")
        self.assertEqual(caps["postprocess"]["regenerate_2k"]["generation_routes"], ["h3-fasth3", "h3-singularity"])
        self.assertEqual(caps["postprocess"]["interpolate"]["generation_routes"], ["h3-fasth3", "h3-singularity"])

        row = self.executor.generate(**self._base_gen_kwargs("v1-default"))
        self.assertEqual(row.service, "h3-fasth3")
        self.assertEqual(row.runtime_route, "h3-fasth3")
        self.assertEqual(row.request["route"], "h3-fasth3")
        self.assertEqual(row.request["quality_profile"], "fasth3-nvfp4")
        post_reqs = [r for r in self.requests if r.url.path == "/v1/videos" and r.method == "POST"]
        self.assertEqual(len(post_reqs), 1)
        self.assertEqual(post_reqs[0].url.host, "fasth3.local")
        self.assertEqual(post_reqs[0].headers["Authorization"], "Bearer fasth3-token")
        body = json.loads(post_reqs[0].content)
        self.assertEqual(body["quality_profile"], "fasth3-nvfp4")
        self.assertEqual(body["num_inference_steps"], 8)

    def test_v5_v6_i1_i2_i3_i5_i6_cold_standby_isolation_and_regenerate_2k_decoupling(self):
        # V5: route_unavailable (unregistered h3-singularity) writes zero record and acquires zero lease/slot.
        with patch.dict(os.environ, {"H3_RUNTIME_ROUTES": "{}"}):
            unreg_exec = VideoExecutor(self.artifacts, self.tasks, self.client)
            with patch.object(unreg_exec, "acquire_local_gpu") as acq:
                with self.assertRaisesRegex(ExecutionError, "route_unavailable"):
                    unreg_exec.generate(**self._base_gen_kwargs("cold-unreg", route="h3-singularity"))
                acq.assert_not_called()
        self.assertEqual(len(list(self.tasks.root.glob("video_task_*.json"))), 0)

        # V5: runtime_not_ready (registered h3-singularity with replicas=0 / unreachable or ready=False) writes zero record & acquires zero lease.
        self.singularity_reachable = False
        with patch.object(self.executor, "acquire_local_gpu") as acq:
            with self.assertRaisesRegex(ExecutionError, "runtime_not_ready"):
                self.executor.generate(**self._base_gen_kwargs("cold-down", route="h3-singularity"))
            acq.assert_not_called()
        self.singularity_reachable = True
        self.singularity_ready = False
        with patch.object(self.executor, "acquire_local_gpu") as acq:
            with self.assertRaisesRegex(ExecutionError, "runtime_not_ready"):
                self.executor.generate(**self._base_gen_kwargs("cold-not-ready", route="h3-singularity"))
            acq.assert_not_called()
        self.assertEqual(len(list(self.tasks.root.glob("video_task_*.json"))), 0)
        self.singularity_ready = True

        # I6: cold standby unreachable /health with zero active h3-singularity tasks must NOT hold gpu-task-h3.
        self.singularity_reachable = False
        self.assertFalse(self.executor._local_h3_active())

        # I1, I2, V6 & I3: historical 0.17.67 h3-singularity record remains byte-identical on disk, readable offline,
        # and standalone regenerate_2k inherits parent prompt while posting strictly to VIDEO_REGENERATE_2K_RUNTIME_URL.
        legacy_tid, m1_id = self._seed_native_h3_task("legacy-sing-01767", route="h3-singularity")
        disk_before = (self.tasks.root / f"{legacy_tid}.json").read_bytes()
        self.assertEqual(self.executor.status(legacy_tid).status, "succeeded")
        self.assertEqual(self.executor.result(legacy_tid).artifact_id, m1_id)
        self.assertEqual((self.tasks.root / f"{legacy_tid}.json").read_bytes(), disk_before)

        regen_rec = self.executor.processing["regenerate_2k"].generate(
            project_id="proj_1",
            idempotency_key="v6-from-legacy-sing",
            source_artifact_id=m1_id,
            profile="open_h3_2k_ic",
            grid_mode="budget_1440p",
            prompt=None,
        )
        self.assertEqual(regen_rec.runtime_route, "h3-regenerate-2k")
        self.assertEqual(regen_rec.request["prompt"], "cinematic portrait in rain")
        regen_posts = [r for r in self.requests if r.url.path == "/v1/regenerate-2k" and r.method == "POST"]
        self.assertEqual(len(regen_posts), 1)
        self.assertEqual(regen_posts[0].url.host, "regenerate2k.local")
        self.assertEqual(regen_posts[0].headers["Authorization"], "Bearer regen2k-token")

        # V6 negative: missing VIDEO_REGENERATE_2K_RUNTIME_URL never falls back to H3_RUNTIME_URL or SINGULARITY_RUNTIME_URL.
        with patch.dict(
            os.environ,
            {
                "VIDEO_REGENERATE_2K_RUNTIME_URL": "",
                "SINGULARITY_RUNTIME_URL": "http://singularity.local:8000",
            },
        ):
            no_regen_exec = VideoExecutor(self.artifacts, self.tasks, self.client)
            regen_cap = no_regen_exec.delivery.regenerate_capability()
            self.assertFalse(regen_cap["configured"])
            self.assertNotEqual(regen_cap["readiness"], "ready")
            with self.assertRaises(ExecutionError):
                no_regen_exec.processing["regenerate_2k"].generate(
                    project_id="proj_1",
                    idempotency_key="v6-no-fallback",
                    source_artifact_id=m1_id,
                )

        # I5: non-terminal record on unregistered route settles to failed(route_unavailable) without raising;
        # registered unreachable route preserves retryable transient handling without faking worker_settled.
        with self.tasks.exclusive():
            orphan = self.tasks.create(
                project_id="proj_1",
                idempotency_key="i5-orphan",
                input_digest="sha256:i5",
                request={"project_id": "proj_1", "route": "h3-singularity", "quality_profile": "du-0"},
                runtime_task_id="singularity_" + "7" * 32,
                status="running",
            )
            orphan = self.tasks.update(orphan, provider_state={"worker_settled": False})
        with self.assertRaises(ExecutionError):
            self.executor.status(orphan.video_task_id)
        self.assertEqual(self.tasks.get(orphan.video_task_id).status, "running")
        self.assertFalse(self.tasks.get(orphan.video_task_id).provider_state.get("worker_settled", True))

        with patch.dict(os.environ, {"H3_RUNTIME_ROUTES": "{}"}):
            unreg_exec2 = VideoExecutor(self.artifacts, self.tasks, self.client)
            settled = unreg_exec2.status(orphan.video_task_id)
        self.assertEqual(settled.status, "failed")
        self.assertEqual(settled.error["code"], "route_unavailable")
        self.assertFalse(settled.error["retryable"])

    def test_legacy_interpolate_only_postprocess_is_byte_identical_and_uses_legacy_child_key(self):
        self.assertEqual(normalize({"interpolate": {"profile": "rife_2x"}}, "h3-fasth3"), OPTION)
        parent = self.executor.generate(**self._base_gen_kwargs("p1-p3"), postprocess=OPTION)
        self.assertEqual(parent.service, "h3-fasth3")
        self.assertEqual(parent.request["quality_profile"], "fasth3-nvfp4")
        p1_posts = [r for r in self.requests if r.url.path == "/v1/videos" and r.method == "POST"]
        self.assertEqual(json.loads(p1_posts[0].content)["num_inference_steps"], 8)
        self.assertEqual(parent.delivery["profiles"], ["rife_2x"])
        self.assertEqual(parent.delivery["timing"]["active_limit_seconds"], {"stage": 900.0, "chain": 1200.0})

        with patch.object(self.executor, "result", side_effect=lambda tid: self._complete_p1(tid)):
            self.executor.delivery.tick(parent.video_task_id)
        updated = self.tasks.get(parent.video_task_id)
        self.assertEqual(updated.delivery["status"], "succeeded")
        interp_child = self.tasks.get(updated.delivery["interpolation_task_id"])
        self.assertEqual(interp_child.idempotency_key, f"delivery:{parent.video_task_id}:rife_2x")

    def _complete_p1(self, tid: str):
        rec = self.tasks.get(tid)
        if not rec.artifact_id:
            art = self.artifacts.create_from_chunks(
                project_id=rec.project_id,
                operation="video.generate",
                filename="m1.mp4",
                media_type="video/mp4",
                chunks=[self.m1_bytes],
            )
            rec = self.tasks.update(
                rec,
                status="succeeded",
                artifact_id=art.artifact_id,
                media={
                    "width": 1344,
                    "height": 768,
                    "frames": 124,
                    "frame_rate": "24",
                    "duration_seconds": 124 / 24.0,
                    "video_codec": "h264",
                    "audio": True,
                },
            )
        return rec

    def test_p1_plus_p2_success_and_failure_preserve_m1(self):
        post = {"regenerate_2k": {"profile": "open_h3_2k_ic"}}
        parent = self.executor.generate(**self._base_gen_kwargs("p1-p2-ok"), postprocess=post)
        self.assertEqual(parent.request["postprocess"], {"regenerate_2k": {"profile": "open_h3_2k_ic", "grid_mode": "budget_1440p"}})
        self.assertEqual(parent.delivery["profiles"], ["open_h3_2k_ic"])
        self.assertEqual(parent.delivery["timing"]["active_limit_seconds"], {"stage": 900.0, "chain": 1800.0})

        with patch.object(self.executor, "result", side_effect=lambda tid: self._complete_p1(tid)):
            done = self.executor.delivery.tick(parent.video_task_id)
        self.assertEqual(done.delivery["status"], "succeeded")
        self.assertEqual(done.delivery["stage"], "completed")
        self.assertIsNotNone(done.delivery["regenerate_2k_artifact_id"])
        self.assertEqual(done.delivery["final_artifact_id"], done.delivery["regenerate_2k_artifact_id"])
        p2_child = self.tasks.get(done.delivery["regenerate_2k_task_id"])
        self.assertEqual(p2_child.idempotency_key, f"delivery:{parent.video_task_id}:open_h3_2k_ic:budget_1440p")

        self.p2_status = "failed"
        parent_fail = self.executor.generate(**self._base_gen_kwargs("p1-p2-fail"), postprocess=post)
        with patch.object(self.executor, "result", side_effect=lambda tid: self._complete_p1(tid)):
            failed = self.executor.delivery.tick(parent_fail.video_task_id)
        self.assertEqual(failed.status, "succeeded")
        self.assertIsNotNone(failed.artifact_id)
        self.assertEqual(failed.delivery["status"], "failed")
        self.assertEqual(failed.delivery["stage"], "regenerate_2k")
        self.assertEqual(failed.delivery["source_artifact_id"], failed.artifact_id)
        self.assertIsNone(failed.delivery["final_artifact_id"])

    def test_p1_p2_p3_full_chain_success_and_p3_failure_preserve_m1_and_m2_with_null_final(self):
        post = {
            "regenerate_2k": {"profile": "open_h3_2k_ic", "grid_mode": "budget_1440p"},
            "interpolate": {"profile": "rife_2x"},
        }
        parent = self.executor.generate(**self._base_gen_kwargs("p1-p2-p3-ok"), postprocess=post)
        self.assertEqual(parent.delivery["profiles"], ["open_h3_2k_ic", "rife_2x"])
        self.assertEqual(parent.delivery["profile"], "rife_2x")

        with patch.object(self.executor, "result", side_effect=lambda tid: self._complete_p1(tid)):
            done = self.executor.delivery.tick(parent.video_task_id)
        self.assertEqual(done.delivery["status"], "succeeded")
        self.assertEqual(done.delivery["stage"], "completed")
        self.assertEqual(done.delivery["source_artifact_id"], done.artifact_id)
        self.assertIsNotNone(done.delivery["regenerate_2k_artifact_id"])
        self.assertIsNotNone(done.delivery["final_artifact_id"])
        self.assertNotEqual(done.delivery["final_artifact_id"], done.delivery["regenerate_2k_artifact_id"])
        p3_child = self.tasks.get(done.delivery["interpolation_task_id"])
        self.assertEqual(p3_child.idempotency_key, f"delivery:{parent.video_task_id}:rife_2x:2k")
        self.assertEqual(p3_child.request["source_artifact_id"], done.delivery["regenerate_2k_artifact_id"])

        self.p3_status = "failed"
        parent_p3_fail = self.executor.generate(**self._base_gen_kwargs("p1-p2-p3-fail"), postprocess=post)
        with patch.object(self.executor, "result", side_effect=lambda tid: self._complete_p1(tid)):
            degraded = self.executor.delivery.tick(parent_p3_fail.video_task_id)
        self.assertEqual(degraded.status, "succeeded")
        self.assertEqual(degraded.delivery["status"], "failed")
        self.assertEqual(degraded.delivery["stage"], "interpolation")
        self.assertEqual(degraded.delivery["source_artifact_id"], degraded.artifact_id)
        self.assertIsNotNone(degraded.delivery["regenerate_2k_artifact_id"])
        self.assertIsNone(degraded.delivery["final_artifact_id"])

    def test_standalone_regenerate_2k_auto_prompt_idempotency_and_chained_rife(self):
        from app import server

        _, m1_id = self._seed_native_h3_task("source-15s", frames=362)
        with patch.object(server, "executor", self.executor), patch.object(
            server, "authorize", return_value={"user_id": "u-1", "organization_id": "org-1"}
        ):
            pre = server.video_preflight(
                project_id="proj_1",
                source_artifact_id=m1_id,
                grid_mode="exact_2x",
                postprocess=OPTION,
            ).structured_content
            self.assertEqual(pre["status"], "validated")
            self.assertEqual(pre["prompt"], "cinematic portrait in rain")
            self.assertEqual(pre["resolved_output"]["width"], 2688)
            self.assertEqual(pre["resolved_output"]["height"], 1536)
            self.assertEqual(pre["delivery_output"]["fps"], 48)
            self.assertEqual(pre["delivery_output"]["frames"], 724)

            created = server.video_regenerate_2k(
                project_id="proj_1",
                idempotency_key="standalone-p2-p3",
                source_artifact_id=m1_id,
                grid_mode="budget_1440p",
                postprocess=OPTION,
            ).structured_content
            again = server.video_regenerate_2k(
                project_id="proj_1",
                idempotency_key="standalone-p2-p3",
                source_artifact_id=m1_id,
                prompt="cinematic portrait in rain",
                grid_mode="budget_1440p",
                postprocess=OPTION,
            ).structured_content
            self.assertEqual(created["video_task_id"], again["video_task_id"])

        rec = self.tasks.get(created["video_task_id"])
        self.assertEqual(rec.request["model"], "open-h3-regenerate-2k-ic")
        self.assertEqual(rec.delivery["timing"]["window_count"], 3)
        self.assertEqual(rec.delivery["timing"]["active_limit_seconds"], {"stage": 2700.0, "chain": 3600.0})

        done = self.executor.delivery.tick(rec.video_task_id)
        self.assertEqual(done.status, "succeeded")
        self.assertEqual(done.delivery["status"], "succeeded")
        self.assertEqual(done.delivery["source_artifact_id"], m1_id)
        self.assertEqual(done.delivery["regenerate_2k_artifact_id"], done.artifact_id)
        self.assertIsNotNone(done.delivery["final_artifact_id"])
        p3_child = self.tasks.get(done.delivery["interpolation_task_id"])
        self.assertEqual(p3_child.idempotency_key, f"delivery:{rec.video_task_id}:rife_2x:2k")

    def test_stage_and_chain_active_timeouts_gate_on_worker_settled_before_terminal_failure(self):
        post = {
            "regenerate_2k": {"profile": "open_h3_2k_ic"},
            "interpolate": {"profile": "rife_2x"},
        }
        parent = self.executor.generate(**self._base_gen_kwargs("timeout-p2"), postprocess=post)
        self.p2_status = "running"
        self.p2_started_at = (datetime.now(UTC) - timedelta(seconds=950)).isoformat()
        self.p2_worker_settled = False

        with patch.object(self.executor, "result", side_effect=lambda tid: self._complete_p1(tid)):
            first_tick = self.executor.delivery.tick(parent.video_task_id)
            self.assertTrue(any(p.startswith("/v1/regenerate-2k/") for p in self.deleted_paths))
            self.assertEqual(first_tick.delivery["status"], "running")

            self.p2_status = "cancelled"
            self.p2_worker_settled = True
            second_tick = self.executor.delivery.tick(parent.video_task_id)
            self.assertEqual(second_tick.delivery["status"], "failed")
            self.assertEqual(second_tick.delivery["error"]["code"], "stage_timeout_regenerate_2k")
            self.assertIsNone(second_tick.delivery["final_artifact_id"])

        # Chain cumulative cap first-hit: P1 (850s) + P2 (850s) + P3 (150s < 300s stage cap) = 1850s > 1800s chain cap.
        now = datetime.now(UTC)
        self.p1_status = "completed"
        self.p1_started_at = (now - timedelta(seconds=1900)).isoformat()
        self.p2_status = "succeeded"
        self.p2_started_at = (now - timedelta(seconds=1000)).isoformat()
        self.p2_worker_settled = True
        self.p3_status = "running"
        self.p3_started_at = (now - timedelta(seconds=150)).isoformat()
        self.p3_worker_settled = True

        parent_chain = self.executor.generate(**self._base_gen_kwargs("timeout-chain"), postprocess=post)
        parent_chain = self.tasks.update(
            parent_chain,
            provider_state={
                "started_at": (now - timedelta(seconds=1900)).isoformat(),
                "finished_at": (now - timedelta(seconds=1050)).isoformat(),
                "worker_settled": True,
            },
        )
        with patch.object(self.executor, "result", side_effect=lambda tid: self._complete_p1(tid)):
            # Run once to complete P2 with 850s active duration.
            p2_finish = (now - timedelta(seconds=150)).isoformat()
            orig_transport = self._transport

            def _chain_transport(req: httpx.Request) -> httpx.Response:
                resp = orig_transport(req)
                if (
                    req.url.host == "regenerate2k.local"
                    and req.url.path.startswith("/v1/regenerate-2k/")
                    and not req.url.path.endswith(("/content", "/preview"))
                    and req.method == "GET"
                ):
                    data = resp.json()
                    data["finished_at"] = p2_finish
                    return httpx.Response(200, json=data)
                return resp

            with httpx.Client(transport=httpx.MockTransport(_chain_transport)) as chain_client:
                self.executor.client = chain_client
                self.executor.processing["regenerate_2k"].client = chain_client
                self.executor.processing["interpolate"].client = chain_client
                outcome = self.executor.delivery.tick(parent_chain.video_task_id)
        self.assertEqual(outcome.delivery["status"], "failed")
        self.assertEqual(outcome.delivery["error"]["code"], "chain_active_timeout")
        self.assertIsNotNone(outcome.delivery["regenerate_2k_artifact_id"])
        self.assertIsNone(outcome.delivery["final_artifact_id"])

    def test_lease_busy_marks_task_failed_retryable_and_stage_slot_waits_for_worker_settled(self):
        self.executor.stage_slot.enabled = True
        with patch.object(self.executor, "acquire_local_gpu", side_effect=LeaseBusy("gpu_busy")):
            row = self.executor.generate(**self._base_gen_kwargs("busy-1"))
        self.assertEqual(row.status, "failed")
        self.assertEqual(row.error["code"], "gpu_busy")
        self.assertTrue(row.error["retryable"])
        self.assertFalse(self.executor._local_h3_active())

        self.p1_worker_settled = False
        slot = LocalGpuStageSlot(self.tasks, enabled=True, queue_seconds=0.05)
        with self.tasks.exclusive():
            holder = self.tasks.create(
                project_id="proj_1",
                idempotency_key="holder-1",
                input_digest="sha256:h1",
                request={"route": "h3-singularity"},
                runtime_task_id="singularity_" + "9" * 32,
                status="succeeded",
            )
            holder = self.tasks.update(holder, provider_state={"worker_settled": False})
        slot.acquire_for_queue(holder.video_task_id, "h3-singularity")
        self.assertTrue(self.executor._local_h3_active())
        with self.assertRaises(LeaseBusy):
            slot.acquire_for_queue("video_task_" + "f" * 32, "interpolate", reconcile_holder=self.executor._reconcile_slot_holder)

        self.p1_worker_settled = True
        slot.acquire_for_queue("video_task_" + "f" * 32, "interpolate", reconcile_holder=self.executor._reconcile_slot_holder)
        slot.release("video_task_" + "f" * 32, force=True)

        self.rife_gpu_active = True
        self.assertTrue(self.executor._local_h3_active())
        self.rife_gpu_active = False
        self.assertFalse(self.executor._local_h3_active())

    def test_pre_upgrade_du0_idempotency_is_preserved_and_preflight_checks_default_profile(self):
        from app import server

        # V4 & I4: existing h3-singularity (du-0) task retried WITHOUT route or quality_profile after default route
        # switched to h3-fasth3 MUST inherit stored route & quality_profile and return original record with zero new POST.
        legacy = self.executor.generate(**self._base_gen_kwargs("pre-upgrade", route="h3-singularity"))
        self.assertEqual(legacy.service, "h3-singularity")
        self.assertEqual(legacy.request["quality_profile"], "du-0")
        post_count_before = len([r for r in self.requests if r.method == "POST"])

        self.singularity_reachable = False
        retried_no_route = self.executor.generate(**self._base_gen_kwargs("pre-upgrade"))
        self.assertEqual(retried_no_route.video_task_id, legacy.video_task_id)
        self.assertEqual(len([r for r in self.requests if r.method == "POST"]), post_count_before)
        self.singularity_reachable = True

        # Explicit conflicting route or quality_profile still raises TaskConflict.
        with self.assertRaises(TaskConflict):
            self.executor.generate(**self._base_gen_kwargs("pre-upgrade", route="h3-fasth3"))

        self.health_default_profile = "du-0"
        args = self._base_gen_kwargs("preflight-mismatch")
        args.pop("idempotency_key")
        with patch.object(server, "executor", self.executor):
            res = server.video_preflight(**args)
        self.assertTrue(res.is_error)
        self.assertIn("runtime_not_ready", json.dumps(res.structured_content))

    def test_local_gpu_queue_never_blocks_cloud_create_or_peer_delivery_tick(self):
        import time as _time

        self.executor.stage_slot.enabled = True
        self.executor.stage_slot.queue_seconds = 1200.0
        self.p1_status = "in_progress"
        self.p1_worker_settled = False

        # Task A claims the single local GPU stage slot on h3-fasth3.
        task_a = self.executor.generate(
            **self._base_gen_kwargs("slot-holder-a"),
            postprocess={"regenerate_2k": {"profile": "open_h3_2k_ic"}},
        )
        self.assertTrue(task_a.runtime_task_id)
        self.assertEqual(self.executor.stage_slot.current_holder(), (task_a.video_task_id, "h3-fasth3"))

        # Task B (local H3) and Task C (standalone regenerate_2k) must return immediately as queued (gpu_dispatched=False).
        t0 = _time.monotonic()
        task_b = self.executor.generate(**self._base_gen_kwargs("slot-waiter-b"))
        _, m1_id = self._seed_native_h3_task("seed-for-c")
        task_c = self.executor.processing["regenerate_2k"].generate(
            project_id="proj_1",
            idempotency_key="slot-waiter-c",
            source_artifact_id=m1_id,
            profile="open_h3_2k_ic",
            grid_mode="budget_1440p",
            prompt="rainy portrait",
        )
        # Concurrent cloud fal route must also return without waiting on local GPU slot.
        with patch.object(
            self.executor.fal,
            "generate",
            side_effect=lambda **kw: self.tasks.create(
                project_id="proj_1",
                idempotency_key=kw["idempotency_key"],
                input_digest="sha256:fal",
                request={"route": "fal"},
                runtime_task_id="fal_1",
                status="queued",
            ),
        ):
            cloud_row = self.executor.generate(
                project_id="proj_1",
                idempotency_key="cloud-fal-1",
                model="minimax-h3-ref2va",
                prompt="cloud test",
                duration_seconds=5,
                aspect_ratio="16:9",
                references={"images": [], "videos": [], "audios": []},
                route="fal",
                quality_profile="du-0",
            )
        elapsed_ms = (_time.monotonic() - t0) * 1000.0
        self.assertLess(elapsed_ms, 500.0)
        self.assertEqual(cloud_row.status, "queued")
        self.assertEqual(task_b.status, "queued")
        self.assertEqual(task_b.runtime_task_id, "")
        self.assertIs(task_b.provider_state.get("gpu_dispatched"), False)
        self.assertEqual(task_b.provider_state.get("waiting_reason"), "gpu_stage_slot")
        self.assertEqual(task_c.status, "queued")
        self.assertIs(task_c.provider_state.get("gpu_dispatched"), False)
        self.assertEqual(len(self.p2_submitted), 0)

        # Peer delivery_poll (sync) is NOT starved: Task A hits active P1 timeout while B & C wait!
        now = datetime.now(UTC)
        self.p1_started_at = (now - timedelta(seconds=950)).isoformat()
        self.executor.delivery.sync()
        refreshed_a = self.tasks.get(task_a.video_task_id)
        self.assertTrue(self.deleted_paths)
        self.assertEqual(refreshed_a.delivery["status"], "running")

        # Once Task A worker settles, sync() settles Task A and dispatches queued Task B automatically.
        self.p1_worker_settled = True
        self.p1_started_at = now.isoformat()
        self.executor.delivery.sync()
        refreshed_a = self.tasks.get(task_a.video_task_id)
        self.assertEqual(refreshed_a.delivery["status"], "failed")
        self.assertEqual(refreshed_a.delivery["error"]["code"], "stage_timeout_generation")

        refreshed_b = self.tasks.get(task_b.video_task_id)
        self.assertTrue(refreshed_b.runtime_task_id)
        self.assertIs(refreshed_b.provider_state.get("gpu_dispatched"), True)
        # Task C remains queued because Task B now owns the stage_slot!
        self.assertIs(self.tasks.get(task_c.video_task_id).provider_state.get("gpu_dispatched"), False)

        # While Task B holds the slot (unsettled), expire Task C's queue budget -> gpu_busy (retryable=True).
        self.p1_worker_settled = False
        stale_c = self.tasks.get(task_c.video_task_id)
        self.tasks.update(
            stale_c,
            provider_state={
                **(stale_c.provider_state or {}),
                "queue_started_at": (now - timedelta(seconds=1205)).isoformat(),
                "_queue_started_mono": _time.monotonic() - 1205.0,
            },
        )
        expired_c = self.executor.status(task_c.video_task_id)
        self.assertEqual(expired_c.status, "failed")
        self.assertEqual(expired_c.error["code"], "gpu_busy")
        self.assertTrue(expired_c.error["retryable"])

    def test_p2_timeout_target_isolation_and_delete_retry_on_network_failure(self):
        now = datetime.now(UTC)
        parent = self.executor.generate(
            **self._base_gen_kwargs("p2-timeout-retry"),
            postprocess={"regenerate_2k": {"profile": "open_h3_2k_ic"}, "interpolate": {"profile": "rife_2x"}},
        )
        self.p2_status = "running"
        self.p2_started_at = (now - timedelta(seconds=950)).isoformat()
        self.p2_worker_settled = False

        fail_delete_once = {"failed": False}
        orig_transport = self._transport

        def _flaky_delete_transport(req: httpx.Request) -> httpx.Response:
            if (
                req.method == "DELETE"
                and req.url.path.startswith("/v1/regenerate-2k/")
                and not fail_delete_once["failed"]
            ):
                fail_delete_once["failed"] = True
                raise httpx.ConnectError("transient network drop on DELETE", request=req)
            return orig_transport(req)

        with (
            patch.object(self.executor, "result", side_effect=lambda tid: self._complete_p1(tid)),
            httpx.Client(transport=httpx.MockTransport(_flaky_delete_transport)) as flaky_client,
        ):
            self.executor.client = flaky_client
            self.executor.processing["regenerate_2k"].client = flaky_client
            self.executor.processing["interpolate"].client = flaky_client

            # Tick 1: P1 completes (worker_settled=True), P2 exceeds 900s, first DELETE fails on ConnectError.
            t1 = self.executor.delivery.tick(parent.video_task_id)
            p2_id = t1.delivery["regenerate_2k_task_id"]
            self.assertEqual(t1.delivery["status"], "running")
            self.assertEqual(t1.delivery["stage"], "regenerate_2k")
            self.assertEqual(t1.delivery["_pending_timeout_task_id"], p2_id)
            self.assertEqual(len(self.deleted_paths), 0)
            p2_rec_1 = self.tasks.get(p2_id)
            self.assertFalse(p2_rec_1.provider_state.get("cancel_delivered", False))

            # Tick 2: P1 (settled) MUST NOT hijack _pending_timeout_error; P2 retries DELETE and delivers it,
            # while remaining running because p2_worker_settled is still False.
            t2 = self.executor.delivery.tick(parent.video_task_id)
            self.assertEqual(t2.delivery["status"], "running")
            self.assertEqual(t2.delivery["stage"], "regenerate_2k")
            self.assertEqual(self.deleted_paths, [f"/v1/regenerate-2k/{p2_id}"])
            p2_rec_2 = self.tasks.get(p2_id)
            self.assertTrue(p2_rec_2.provider_state.get("cancel_delivered"))
            self.assertFalse(p2_rec_2.provider_state.get("worker_settled", True))

            # Tick 3: Worker settles -> delivery fails on stage="regenerate_2k" (NOT "generation"), preserving M1!
            self.p2_worker_settled = True
            t3 = self.executor.delivery.tick(parent.video_task_id)
            self.assertEqual(t3.delivery["status"], "failed")
            self.assertEqual(t3.delivery["stage"], "regenerate_2k")
            self.assertEqual(t3.delivery["error"]["code"], "stage_timeout_regenerate_2k")
            self.assertIsNotNone(t3.delivery["source_artifact_id"])
            self.assertIsNone(t3.delivery["final_artifact_id"])
            self.assertEqual(t3.status, "succeeded")

    def test_seed_zero_queue_timing_and_e2e_video_client_download_mcp_result(self):
        import importlib.util
        import io
        import urllib.request

        now = datetime.now(UTC)
        parent = self.executor.generate(
            **self._base_gen_kwargs("seed-zero-e2e"),
            postprocess={"regenerate_2k": {"profile": "open_h3_2k_ic"}, "interpolate": {"profile": "rife_2x"}},
        )
        # Explicitly set seed=0 and simulated orchestration + runtime queue intervals on parent.
        parent = self.tasks.update(
            parent,
            created_at=(now - timedelta(seconds=60)).isoformat(),
            dispatched_at=(now - timedelta(seconds=48)).isoformat(),
            request={**parent.request, "seed": 0},
            provider_state={
                "started_at": (now - timedelta(seconds=40)).isoformat(),
                "finished_at": (now - timedelta(seconds=20)).isoformat(),
                "worker_settled": True,
            },
        )
        self.p2_started_at = (now - timedelta(seconds=18)).isoformat()
        self.p3_started_at = (now - timedelta(seconds=5)).isoformat()

        with patch.object(self.executor, "result", side_effect=lambda tid: self._complete_p1(tid)):
            done = self.executor.delivery.tick(parent.video_task_id)

        self.assertEqual(done.delivery["status"], "succeeded")
        p2_id = done.delivery["regenerate_2k_task_id"]
        self.assertEqual(self.p2_submitted[p2_id]["seed"], 0)
        # Parent queue seconds = (60 - 48) orch wait + (48 - 40) runtime wait = 20s (>= 20s total across stages).
        self.assertGreaterEqual(done.delivery["timing"]["queue_seconds"], 20.0)

        # Load real verdantflare-video client and verify download_mcp_result accepts both standalone P2 and chained P1+P2+P3 results.
        skill_client_path = (
            REPO_ROOT / "verdantflare-skills" / "skills" / "verdantflare-video" / "scripts" / "video_client.py"
        )
        skill_scripts_dir = str(skill_client_path.parent)
        sys.path.insert(0, skill_scripts_dir)
        try:
            spec = importlib.util.spec_from_file_location("vf_skill_video_client", skill_client_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["vf_skill_video_client"] = mod
            spec.loader.exec_module(mod)
        finally:
            if sys.path and sys.path[0] == skill_scripts_dir:
                sys.path.pop(0)

        p2_rec = self.tasks.get(p2_id)
        standalone_pub = self.executor.processing["regenerate_2k"].public_result(p2_rec)
        chained_pub = {
            "video_task_id": done.video_task_id,
            "project_id": done.project_id,
            "status": "succeeded",
            "artifact_id": done.artifact_id,
            "download_path": self.artifacts.download_path(done.artifact_id),
            "media": {"sha256": self.m1_sha, "size": len(self.m1_bytes)},
            "delivery": done.delivery,
        }

        class _FakeResp:
            def __init__(self, payload: bytes):
                self._buf = io.BytesIO(payload)
                self.status = 200
                self.headers = {
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(payload)),
                }

            def read(self, n: int = -1) -> bytes:
                return self._buf.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _FakeOpener:
            def __init__(self, store: ArtifactStore):
                self._store = store

            def open(self, req: urllib.request.Request, timeout: float = 120):
                art_id = req.full_url.split("/artifacts/", 1)[1].split("/content", 1)[0]
                art = self._store.get(art_id, "proj_1")
                return _FakeResp(self._store.content_path(art).read_bytes())

        cfg = mod.MCPConfig(url="https://studio.example.com/mcp/video", token="tok_test", state_dir=self.root)
        with (
            patch.object(mod, "_validate_mcp_url", side_effect=lambda u: u),
            patch.object(mod, "_mcp_opener", return_value=_FakeOpener(self.artifacts)),
            patch.object(mod, "_validate_downloaded_video", return_value=None),
        ):
            out_p2 = self.root / "downloaded_p2.mp4"
            dl_p2 = mod.download_mcp_result(cfg, standalone_pub, str(out_p2))
            self.assertTrue(dl_p2["download"]["verified"])
            self.assertEqual(dl_p2["download"]["sha256"], self.m2_sha)
            self.assertEqual(dl_p2["download"]["size"], len(self.m2_bytes))
            self.assertEqual(out_p2.read_bytes(), self.m2_bytes)

            out_chain = self.root / "downloaded_chain.mp4"
            dl_chain = mod.download_mcp_result(cfg, chained_pub, str(out_chain))
            self.assertTrue(dl_chain["download"]["verified"])
            self.assertEqual(dl_chain["download"]["sha256"], self.m3_sha)
            self.assertEqual(dl_chain["download"]["size"], len(self.m3_bytes))
            self.assertEqual(out_chain.read_bytes(), self.m3_bytes)


class CatalogAndPresentation2kTest(unittest.TestCase):
    def test_presentation_vsr2va_and_2k_resolution_labels(self):
        temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        tasks = TaskStore(root)
        artifacts = ArtifactStore(root)
        executor = VideoExecutor(artifacts, tasks, httpx.Client())
        self.addCleanup(executor.client.close)
        dashboard = Dashboard(executor)

        with tasks.exclusive():
            rec = tasks.create(
                project_id="proj_1",
                idempotency_key="regen-ui",
                input_digest="sha256:ui",
                request={
                    "project_id": "proj_1",
                    "service": "regenerate_2k",
                    "model": "open-h3-regenerate-2k-ic",
                    "grid_mode": "budget_1440p",
                },
                runtime_task_id="video_task_" + "1" * 32,
                status="succeeded",
            )
            rec = tasks.update(rec, media={"width": 2464, "height": 1408, "frames": 124, "frame_rate": "24"})

        pres = task_presentation(rec)
        self.assertEqual(task_channel(rec), "local")
        self.assertEqual(pres["channel_id"], "local")
        self.assertEqual(pres["channel_label"], "本地")
        self.assertEqual(pres["model_label"], "H3-Regenerate-2K")
        self.assertEqual(pres["generation_code"], "VSR2VA")
        pub = dashboard._public_task(rec, tasks)
        self.assertEqual(pub["grid_mode"], "budget_1440p")

        with tasks.exclusive():
            fh3_rec = tasks.create(
                project_id="proj_1",
                idempotency_key="fasth3-ui",
                input_digest="sha256:fh3ui",
                request={
                    "project_id": "proj_1",
                    "route": "h3-fasth3",
                    "model": "minimax-h3-ref2va",
                    "quality_profile": "fasth3-nvfp4",
                },
                runtime_task_id="fasth3_" + "3" * 32,
                status="succeeded",
            )
            fh3_rec = tasks.update(fh3_rec, media={"width": 1344, "height": 768, "frames": 124, "frame_rate": "24"})
        fh3_pres = task_presentation(fh3_rec)
        self.assertEqual(task_channel(fh3_rec), "local")
        self.assertEqual(fh3_pres["channel_id"], "local")
        self.assertEqual(fh3_pres["channel_label"], "本地")

        for w, h in [(2464, 1408), (1408, 2464), (2560, 1120), (3072, 1344), (1536, 2688)]:
            self.assertTrue(is_2k_canvas(w, h), f"Expected 2K canvas for {w}x{h}")
        for w, h in [(1920, 1080), (1344, 768), (3840, 2160)]:
            self.assertFalse(is_2k_canvas(w, h), f"Did not expect 2K canvas for {w}x{h}")

    def test_v0_1_3_catalog_three_way_sha256_schema_immutability_and_pending_vs_approved_load(self):
        v012_cm_path = (
            REPO_ROOT
            / "deploys"
            / "k8s.dev.verdantflarehub.com"
            / "verdantflare-app"
            / "video-models-local-v0-1-2.yaml"
        )
        v012_cm = yaml.safe_load(v012_cm_path.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(v012_cm["data"]["channel-catalog.yaml"].encode("utf-8")).hexdigest(),
            "125552bd18b2f4d738240f3a74c44c3afb74eebc58787a34b7a9d1ecf750d48c",
        )

        local_cm_path = (
            REPO_ROOT
            / "deploys"
            / "k8s.dev.verdantflarehub.com"
            / "verdantflare-app"
            / "video-models-local-v0-1-3.yaml"
        )
        channels_cm_path = (
            REPO_ROOT
            / "deploys"
            / "k8s.dev.verdantflarehub.com"
            / "verdantflare-app"
            / "video-model-channels-v0-1-6.yaml"
        )
        schema_path = (
            APP_VIDEO_ROOT
            / "services"
            / "video-mcp-server"
            / "app"
            / "channel-catalog.schema.json"
        )

        local_cm = yaml.safe_load(local_cm_path.read_text(encoding="utf-8"))
        channels_cm = yaml.safe_load(channels_cm_path.read_text(encoding="utf-8"))
        catalog_bytes = local_cm["data"]["channel-catalog.yaml"].encode("utf-8")
        computed_sha = hashlib.sha256(catalog_bytes).hexdigest()
        review_doc = yaml.safe_load(local_cm["data"]["review.yaml"])
        channels_doc = yaml.safe_load(channels_cm["data"]["channels.yaml"])
        local_sel = next(ch for ch in channels_doc["channels"] if ch["channel_id"] == "local")

        self.assertEqual(local_sel["channel_version"], "0.1.3")
        self.assertEqual(review_doc["candidate_sha256"], computed_sha)
        self.assertEqual(local_sel["sha256"], computed_sha)
        self.assertIn(review_doc["decision"], ("pending", "approved"))
        if review_doc["decision"] == "approved":
            self.assertEqual(review_doc["approved_by"], "Mengsk")

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        catalog_doc = yaml.safe_load(catalog_bytes.decode("utf-8"))
        Draft202012Validator(schema).validate(catalog_doc)
        r5_src = next(s for s in catalog_doc["sources"] if s["id"] == "h3-5090-r5-service-split")
        design_sha = hashlib.sha256((REPO_ROOT / r5_src["evidence_ref"]).read_bytes()).hexdigest()
        self.assertEqual(r5_src["sha256"], design_sha)

        # Mount simulation: pending review fails closed; live approved manifest loads ready with local-h3-regenerate-2k-ic.
        temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temp.cleanup)
        mount = Path(temp.name)
        (mount / "channels.yaml").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "channels": [local_sel],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        local_dir = mount / "local"
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "channel-catalog.yaml").write_bytes(catalog_bytes)

        pending_review = {
            **review_doc,
            "decision": "pending",
            "approved_by": "pending",
        }
        (local_dir / "review.yaml").write_text(
            yaml.safe_dump(pending_review, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        pending_view = ModelCatalog(mount).snapshot()
        self.assertNotEqual(pending_view["state"], "ready")

        if review_doc["decision"] == "approved":
            (local_dir / "review.yaml").write_text(local_cm["data"]["review.yaml"], encoding="utf-8")
        else:
            approved_review = {
                **review_doc,
                "decision": "approved",
                "approved_by": "Mengsk",
                "approved_at": "2026-11-14T15:00:00Z",
            }
            (local_dir / "review.yaml").write_text(
                yaml.safe_dump(approved_review, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        ready_view = ModelCatalog(mount).snapshot()
        self.assertEqual(ready_view["state"], "ready")
        ids = [m["id"] for m in ready_view["channels"][0]["models"]]
        self.assertIn("local-h3-regenerate-2k-ic", ids)


if __name__ == "__main__":
    unittest.main()
