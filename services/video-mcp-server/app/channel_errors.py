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
            code = {'verified Studio identity required':'unauthorized', 'project access denied':'project_access_denied'}.get(code, code)
            error = ChannelError(code if code in MESSAGES else "invalid_request", source="video").public()
        except RuntimeError as exc:
            code = str(exc)
            code = {'project authorization temporarily unavailable':'project_authorization_unavailable',
                    'project authorization is not configured':'project_authorization_unavailable'}.get(code, code)
            error = ChannelError(code if code in MESSAGES else 'operation_failed', source='video').public()
        value = {"error": error}
        return types.CallToolResult(isError=True, content=[types.TextContent(type="text", text=json.dumps(value, ensure_ascii=False))], structuredContent=value)
    return wrapped
