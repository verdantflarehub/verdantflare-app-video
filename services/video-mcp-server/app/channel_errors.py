"""Safe business errors shared by MCP tools and channel adapters."""
from functools import wraps
import json
import re

import httpx
from mcp import types
from .tasks import TaskConflict, TaskNotFound


MESSAGES = {
    "authorization_required": "Channel membership authorization has expired or is missing.",
    "workspace_binding_mismatch": "The authorized workspace does not match the channel binding.",
    "limits_unavailable": "The channel limit configuration is unavailable.",
    "account_unavailable": "The channel account information could not be read.",
    "account_disabled": "The channel account is disabled.",
    "account_required": "No channel account was selected and no default account is available.",
    "account_access_denied": "The current account does not permit this operation.",
    "historical_cost_unknown": "Historical channel costs require reconciliation before a new submission.",
    "billing_access_denied": "The channel credential does not have permission to read billing records.",
    "provider_call_failed": "The provider rejected the channel operation. No new generation was retried.",
    "provider_timeout": "The provider request timed out; recover the original operation if already submitted.",
    "provider_protocol_error": "The provider returned an unsupported response format.",
    "cli_unavailable": "The channel command-line runtime is unavailable.",
    "upload_protocol_error": "The provider returned an invalid reference upload receipt.",
    "reference_upload_expired": "A reference upload has expired; revalidate the original media before submission.",
    "reference_upload_failed": "The provider could not accept the reference media.",
    "insufficient_credits": "The membership has insufficient credits for this request.",
    "account_budget_exceeded": "The channel account budget would be exceeded.",
    "account_concurrency_exceeded": "The channel has reached its concurrent submission limit.",
    "account_reconciliation_required": "An earlier submission has an unknown outcome and requires reconciliation.",
    "quote_changed": "The current provider quote exceeds the authorized price ceiling.",
    "quote_expired": "The quote has expired; obtain a new quote before submitting.",
    "model_retired": "The provider has retired this model. Select an explicitly supported replacement.",
    "capability_not_supported": "The selected model or operation is not enabled on this channel.",
    "channel_unavailable": "The channel is unavailable for new submissions.",
    "proxy_unavailable": "The configured channel proxy is unavailable.",
    "rate_limited": "The provider rate limit has been reached.",
    "quota_exhausted": "The channel quota has been exhausted.",
    "video_daily_limit_exceeded": "The configured daily video limit has been reached.",
    "video_monthly_limit_exceeded": "The configured monthly video limit has been reached.",
    "video_concurrency_limit_exceeded": "The configured concurrent video limit has been reached.",
    "budget_below_estimate": "The authorized budget is lower than the current request estimate.",
    "cost_unknown": "The request cost cannot be estimated under the current channel rules.",
    "unauthorized": "The channel rejected the credential.",
    "forbidden": "The current identity does not have permission for this operation.",
    "project_access_denied": "The current identity cannot access this project.",
    "higgsfield_account_access_denied": "The current identity cannot use this membership account.",
    "unsupported_duration": "The requested duration is not supported by this model.",
    "unsupported_resolution": "The requested resolution is not supported by this model.",
    "unsupported_aspect_ratio": "The requested aspect ratio is not supported by this model.",
    "unsupported_references": "The reference media types or counts are not supported by this model.",
    "unsupported_reference_media": "The reference format or file size exceeds this channel's limits.",
    "mode_reference_mismatch": "The selected mode does not match the supplied reference media.",
    "reference_to_video_required": "This model requires reference media.",
    "first_frame_reference_required": "This model requires a single first-frame reference.",
    "invalid_reference": "A reference identifier or purpose is invalid for this model.",
    "invalid_prompt": "The prompt is empty or exceeds the allowed length.",
    "invalid_request": "The request contains invalid or unsupported parameters.",
    "idempotency_conflict": "This idempotency key already belongs to a different request.",
    "source_integrity_failed": "The reference bytes do not match the saved media identity.",
    "result_not_ready": "The original task result is not ready for collection.",
    "submission_unknown": "The provider may have accepted this request; query the original task without resubmitting.",
    "acceptance_unknown": "The provider may have accepted this request; query the original task without resubmitting.",
}
RETRYABLE = {"provider_timeout", "account_unavailable", "limits_unavailable", "channel_unavailable", "proxy_unavailable", "rate_limited", "result_not_ready"}
MESSAGES.update({
    "provider_not_found": "The provider returned not found for this operation.",
    "budget_exceeded": "The channel account budget is insufficient.",
    "daily_limit_reached": "The channel daily task limit has been reached.",
    "authorization_not_verified": "The channel account authorization has not been verified.",
    "price_table_expired": "The channel price table has expired.",
    "price_unknown": "The channel has no supported price for this request.",
    "invalid_video_payload": "The video parameters do not match the channel contract.",
    "provider_rejected": "The provider rejected the request.",
    "provider_generation_failed": "The provider reported generation failure.",
    "source_size_exceeded": "The reference file exceeds the permitted size.",
    "unsupported_parameter": "The channel does not support this parameter.",
    "invalid_channel_account": "The channel account identifier is invalid.",
    "invalid_budget_limit": "The budget ceiling is invalid.",
    "provider_not_configured": "The channel connection is not configured.",
    "unsupported_mode": "The channel does not support this generation mode.",
    "invalid_reference_count": "The number of references is not supported.",
    "media_contract_mismatch": "The generated media does not match the requested specification.",
    "provider_auth_failed": "The provider rejected the configured credential.",
    "provider_rate_limited": "The provider rate limit was reached.",
    "provider_insufficient_credits": "The provider reports insufficient credits.",
    "provider_quota_exceeded": "The provider quota is exhausted.",
    "provider_content_rejected": "The provider rejected the content under its policy.",
    "provider_validation_failed": "The provider rejected the request parameters.",
    "pricing_review_required": "The channel price evidence requires review.",
    "provider_usage_required": "Actual provider usage is needed to determine the total charge.",
    "remote_not_found": "The provider no longer recognizes the original task.",
    "result_expired": "The provider result has expired.",
    "result_integrity_failed": "The returned media failed integrity verification.",
    "result_collection_failed": "The original result could not be collected; query the same task again.",
    "higgsfield_recovery_pending": "The original Higgsfield operation is awaiting recovery.",
    "higgsfield_binding_changed": "The Higgsfield binding changed; recover with the original account.",
    "provider_unavailable": "The provider is unavailable.",
    "submission_unconfirmed": "Acceptance is unknown; recover the original task without resubmitting.",
    "provider_invalid_request": "The provider rejected these request parameters.",
    "provider_model_unavailable": "The selected provider model is unavailable.",
    "provider_quota_exhausted": "The provider reports that the account quota is exhausted.",
    "provider_failed": "The provider reported that the operation failed.",
    "operation_failed": "The operation could not be completed.",
    "project_authorization_unavailable": "Project authorization is temporarily unavailable.",
    "channel_budget_not_supported": "This channel does not accept that budget or account parameter.",
    "invalid_price_ceiling": "The Credits price ceiling is invalid.",
    "invalid_cost": "The provider returned an invalid cost quote.",
    "higgsfield_not_configured": "The Higgsfield membership connection is not configured.",
    "higgsfield_result_not_ready": "The original Higgsfield result is not ready for collection.",
    "task_not_found": "The requested video task is unavailable to the current identity.",
})


# Exact validator diagnostics only; arbitrary ValueError text stays private.
MESSAGES.update({
    "reference_image_count_exceeded": "This route accepts at most 9 reference images.",
    "reference_video_count_exceeded": "This route accepts at most 3 reference videos.",
    "reference_audio_count_exceeded": "This route accepts at most 3 reference audio files.",
    "reference_total_count_exceeded": "This route accepts at most 12 reference files in total.",
    "visual_reference_required": "This route requires at least one image or video reference.",
    "invalid_h3_duration_or_prompt": "H3 duration must be an integer from 4 to 15 seconds and the prompt must not be empty.",
    "reference_video_duration_total_exceeded": "Reference videos must total at most 15 seconds.",
    "reference_audio_duration_total_exceeded": "Reference audio files must total at most 15 seconds.",
    "import_size_must_be_1_byte_to_1_GiB": "Import file size must be between 1 byte and 1 GiB (1,073,741,824 bytes), inclusive.",
    "unsupported_import_format": "Import supports PNG, JPEG, WebP, MP4, WAV and MP3 files.",
    "invalid_import_idempotency_key": "Import requires a nonempty idempotency key of at most 128 characters.",
    "invalid_import_sha256": "Import requires a SHA-256 digest with 64 lowercase hexadecimal characters.",
    "import_purpose_required": "Import purpose must contain 1 to 1000 characters.",
    "fixed_content_ref_required": "Import source must use a fixed store, artifact and version reference.",
    "invalid_import_id": "The import identifier is invalid.",
    "import_not_found": "The original import is unavailable in this project.",
    "import_idempotency_conflict": "This import idempotency key already belongs to different metadata.",
    "invalid_import_offset": "The import offset must be a nonnegative integer.",
    "import_chunk_too_large_or_empty": "Each import chunk must contain 1 to 524,288 decoded bytes.",
    "invalid_import_base64": "The import chunk is not valid Base64.",
    "import_chunk_size_or_hash_mismatch": "The import chunk exceeds 524,288 bytes, is empty, or does not match its SHA-256 digest.",
    "import_chunk_exceeds_size": "The import chunk exceeds the declared total file size.",
    "import_chunk_overlap": "The chunk overlaps the confirmed import offset; resume from the original import status.",
    "import_repeated_chunk_mismatch": "Repeated chunk bytes differ from the original upload.",
    "import_offset_or_status_conflict": "The chunk offset or import state has changed; query the original import status.",
    "import_committed_bytes_missing": "Previously confirmed upload bytes are unavailable; recover the original import before retrying.",
    "import_incomplete": "The upload is incomplete. Query the original import status and resume from its confirmed offset.",
    "import_final_size_or_hash_mismatch": "The complete upload does not match the declared file size or SHA-256 digest.",
    "import_published_artifact_conflict": "The existing artifact does not match the original import identity.",
    "reference_duration_must_be_2_to_15_seconds": "Reference media must be 2 to 15 seconds long. Use a valid source interval before importing again.",
    "reference_video_duration_must_be_2_to_15_seconds": "The video stream itself must be 2 to 15 seconds long; a longer audio track does not satisfy this limit.",
    "reference_pixel_budget_exceeded": "Reference image or video dimensions exceed the supported bounds. Check the media dimensions.",
    "reference_video_decode_budget_exceeded": "The reference video exceeds the supported frame-rate or decoded-pixel budget.",
    "reference_media_decode_failed": "The reference media could not be decoded. Check its format and completeness.",
    "reference_requires_one_visual_stream": "Reference images and videos must contain exactly one visual stream.",
    "reference_image_format_mismatch": "The decoded image format must match PNG, JPEG or WebP and contain no audio.",
    "reference_image_must_be_still": "Reference images must contain a single still frame.",
    "reference_video_requires_mp4_h264_or_hevc": "Reference video must use an MP4 container and H.264 or H.265 encoding.",
    "reference_video_rotation_must_be_right_angle": "Reference video rotation must be a multiple of 90 degrees.",
    "reference_requires_one_audio_stream": "Reference audio must contain one audio stream and no visual stream.",
    "reference_audio_requires_wav_pcm_or_mp3": "Reference audio must use WAV/PCM or MP3 encoding.",
    "reference_audio_requires_mono_or_stereo": "Reference audio must contain one mono or stereo stream.",
    "reference_audio_sample_rate_exceeded": "Reference audio sample rate must be positive and at most 192,000 Hz.",
    "unsupported_reference_type": "This reference media type is not supported.",
})

VALIDATION_ALIASES = {
    "verified Studio identity required": "unauthorized",
    "project access denied": "project_access_denied",
    "too many images references": "reference_image_count_exceeded",
    "too many videos references": "reference_video_count_exceeded",
    "too many audios references": "reference_audio_count_exceeded",
    "Selected route supports at most 12 references": "reference_total_count_exceeded",
    "at least one reference is required": "reference_to_video_required",
    "at least one image or video reference is required": "visual_reference_required",
    "duration_seconds must be an integer from 4 to 15 and prompt must not be empty": "invalid_h3_duration_or_prompt",
    "Selected channel duration must be an integer from 5 to 15 seconds": "unsupported_duration",
    "Unsupported Singularity aspect_ratio": "unsupported_aspect_ratio",
    "references must contain only images, videos and audios arrays": "unsupported_references",
    "reference requires only artifact_id and purpose": "invalid_reference",
    "invalid images reference": "invalid_reference",
    "invalid videos reference": "invalid_reference",
    "invalid audios reference": "invalid_reference",
    "videos references exceed 15 seconds in total": "reference_video_duration_total_exceeded",
    "audios references exceed 15 seconds in total": "reference_audio_duration_total_exceeded",
}


def safe_code(code, source):
    return isinstance(code, str) and (code in MESSAGES or
        (source == "google" and re.fullmatch(r"google_(?:[a-z_]{1,80}|http_[1-5][0-9]{2})", code)) or
        code in {"google_result_unavailable", "google_file_failed", "no_video_returned"})


class ChannelError(ValueError):
    def __init__(self, code, *, source, http_status=None, accepted=None, message=None, retryable=None):
        self.code = code if safe_code(code, source) else "channel_http_error"
        self.source = source if source in {"higgsfield", "fal", "grok", "google", "video"} else "video"
        self.http_status = http_status
        self.accepted = accepted if type(accepted) is bool else None
        self.message = message
        self.retryable = retryable if type(retryable) is bool else self.code in RETRYABLE
        super().__init__(self.code)

    def public(self):
        value = {"code": self.code, "message": self.message or MESSAGES.get(self.code, "The channel returned an unrecognized error response."),
                 "source": self.source, "retryable": self.retryable}
        if type(self.http_status) is int:
            value["http_status"] = self.http_status
        if self.accepted is not None:
            value["accepted"] = self.accepted
        return value


def channel_error(payload, *, source, accepted=None, http_status=None):
    """Consume only a channel's safe envelope, never its arbitrary upstream body."""
    payload = payload if isinstance(payload, dict) else {}
    envelope = payload.get("error")
    if not isinstance(envelope, dict) and isinstance(payload.get('diagnostic'), dict):
        envelope = payload['diagnostic']
    envelope = envelope or {}
    code = envelope.get("code") if isinstance(envelope, dict) else envelope
    code = code or payload.get("reason_code")
    message, retryable = None, None
    if isinstance(envelope, dict) and envelope.get("source") == source and safe_code(code, source):
        message = envelope.get("message")
        if isinstance(message, str):
            # Defense in depth around channel-side sanitization.
            message = re.sub(r"https?://\S+", "[redacted URL]", message, flags=re.I)
            message = re.sub(r'''(?i)(["']?(?:api[_-]?key|access_token|refresh_token|token|password|secret|authorization)["']?\s*[:=]\s*)["']?[^\s,}\]]+''', r"\1[redacted]", message)
            message = re.sub(r"(?i)(bearer\s+|(?:api[_-]?key|token|password|secret)\s*[:=]\s*)\S+", r"\1[redacted]", message)
            message = re.sub(r'(?<!\w)/(?:[\w.-]+/)+[\w./-]*', '[redacted path]', message)
            message = re.sub(r'(?i)\b[a-z]:[\\/][^\s,;]+', '[redacted path]', message)
            message = re.sub(r"\b[A-Za-z0-9_-]{40,}\b", "[redacted]", message)[:512]
        else:
            message = None
        retryable = envelope.get("retryable")
        if type(accepted) is not bool:
            accepted = payload.get("accepted", envelope.get("accepted"))
    else:
        if type(accepted) is not bool:
            accepted = payload.get("accepted")
    return ChannelError(code, source=source, http_status=http_status, accepted=accepted,
                        message=message, retryable=retryable)


def check_response(response, source, *, accepted=None):
    if not response.is_error:
        return response
    payload = {}
    try:
        if not response.is_closed and not response.is_stream_consumed:
            # Error responses should be small; do not read successful video streams.
            raw = bytearray()
            for chunk in response.iter_bytes():
                raw.extend(chunk)
                if len(raw) > 65536:
                    break
            if len(raw) <= 65536:
                payload = json.loads(raw)
        elif len(response.content) <= 65536:
            payload = response.json()
    except (ValueError, AttributeError, httpx.HTTPError):
        pass
    raise channel_error(payload, source=source, http_status=response.status_code, accepted=accepted)


def tool_errors(function):
    """Keep tool schemas/signatures unchanged while returning useful business failures."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ChannelError as exc:
            error = exc.public()
        except httpx.HTTPStatusError as exc:
            error = ChannelError(None, source=kwargs.get("route", "video"), http_status=exc.response.status_code).public()
            try:
                check_response(exc.response, kwargs.get("route", "video"))
            except ChannelError as channel:
                error = channel.public()
        except httpx.RequestError:
            error = ChannelError("channel_unavailable", source=kwargs.get("route", "video")).public()
        except TaskNotFound:
            error = ChannelError('task_not_found', source='video').public()
        except TaskConflict:
            error = ChannelError('idempotency_conflict', source='video').public()
        except PermissionError as exc:
            code = str(exc)
            error = ChannelError(code if code in MESSAGES else 'project_access_denied', source='video').public()
        except ValueError as exc:
            code = str(exc)
            code = VALIDATION_ALIASES.get(code, code)
            error = ChannelError(code if code in MESSAGES else "invalid_request", source="video").public()
        except RuntimeError as exc:
            code = str(exc)
            code = {'project authorization temporarily unavailable':'project_authorization_unavailable',
                    'project authorization is not configured':'project_authorization_unavailable'}.get(code, code)
            error = ChannelError(code if code in MESSAGES else 'operation_failed', source='video').public()
        value = {"error": error}
        return types.CallToolResult(isError=True, content=[types.TextContent(type="text", text=json.dumps(value, ensure_ascii=False))], structuredContent=value)
    return wrapped
