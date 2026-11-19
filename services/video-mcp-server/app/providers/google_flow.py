"""Video member adapter for the Google Flow jobs mapping."""
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import re
import ssl
from urllib.parse import urlsplit

import httpx

from ..channel_errors import ChannelError, channel_error
from ..tasks import TaskConflict
from .membership import MembershipAccess

MODELS = ("veo-lite", "veo-fast", "veo-quality", "omni-flash")
ROUTE = "google-flow"


class GoogleFlowAdapter(MembershipAccess):
    route = ROUTE

    def __init__(self, executor):
        self.executor = executor
        self.url = os.getenv("GOOGLE_FLOW_GATEWAY_URL", "").rstrip("/")
        self.token = os.getenv("GOOGLE_FLOW_INTERNAL_TOKEN", "")
        self.organization = os.getenv("GOOGLE_FLOW_ORGANIZATION_ID", "")
        self.members = frozenset()
        self.enabled = os.getenv("GOOGLE_FLOW_ENABLED", "true").lower() == "true"
        self.profile = hashlib.sha256(json.dumps([self.url, self.organization]).encode()).hexdigest()
        self.client = httpx.Client(trust_env=False, follow_redirects=False, timeout=120,
                                   verify=ssl.create_default_context(cafile=os.getenv("VIDEO_PROJECT_CA_FILE") or None))

    def configured(self):
        url = urlsplit(self.url)
        return bool(self.organization and len(self.token) >= 32 and url.scheme in {"http", "https"}
                    and url.hostname and not (url.username or url.password or url.query or url.fragment))

    def channel_health(self):
        return {"configured": self.configured(), "enabled": self.enabled}

    def capabilities(self):
        return [dict(route=ROUTE, provider=ROUTE, model=model, modes=["text_to_video", "reference_to_video"],
                     aspect_ratios=["16:9", "9:16"], duration_seconds={"minimum": 4, "maximum": 8, "enum": [4,6,8]},
                     poll=True, cancel=False, readiness="not_checked", **self.channel_health()) for model in MODELS]

    def normalize(self, project_id, prompt, model, duration_seconds, aspect_ratio, references,
                  mode=None, resolution=None, quality_profile=None, budget_limit=None, channel_account_id=None):
        if model not in MODELS or quality_profile is not None or resolution is not None or channel_account_id is not None:
            raise ValueError("unsupported_parameter")
        if type(duration_seconds) is not int or duration_seconds not in (4, 6, 8):
            raise ValueError("unsupported_duration")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 5000:
            raise ValueError("invalid_prompt")
        if aspect_ratio is not None and aspect_ratio not in ("16:9", "9:16"):
            raise ValueError("unsupported_aspect_ratio")
        if not isinstance(references, dict) or set(references) - {"images","videos","audios"}:
            raise ValueError("invalid_references")
        images = references.get("images", [])
        if references.get("videos") or references.get("audios") or not isinstance(images,list) or len(images)>3:
            raise ValueError("unsupported_references")
        sources = []
        for ref in images:
            if (not isinstance(ref, dict) or set(ref) != {"artifact_id", "purpose"}
                    or not isinstance(ref["purpose"], str) or not ref["purpose"].strip()
                    or ref["purpose"] in {"first_frame", "last_frame"}):
                raise ValueError("invalid_reference")
            _, info = self.reference_source(project_id, ref["artifact_id"], "image")
            sources.append(info)
        resolved_mode = "reference_to_video" if images else "text_to_video"
        if mode not in (None, resolved_mode):
            raise ValueError("mode_reference_mismatch")
        budget = None
        if budget_limit is not None:
            try:
                value = Decimal(str(budget_limit))
            except InvalidOperation:
                raise ValueError("invalid_budget_limit") from None
            if not value.is_finite() or not 0 < value <= 2**31-1 or value != value.to_integral_value():
                raise ValueError("invalid_budget_limit")
            budget = int(value)
        request = dict(route=ROUTE, model=model, gateway_profile=self.profile, project_id=project_id,
                       duration_seconds=duration_seconds, aspect_ratio=aspect_ratio or "16:9", mode=resolved_mode,
                       prompt=prompt, references={"images":images, "videos":[], "audios":[]}, sources=sources,
                       budget_limit=budget)
        return request, "sha256:" + hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    def headers(self, project, actor):
        return {"Authorization":"Bearer "+self.token, "X-User-Id":actor["user_id"],
                "X-Organization-Id":actor["organization_id"], "X-Project-Id":project, "X-Project-Write":"true"}

    def response(self, response):
        if response.is_error:
            try:
                code = response.json().get("code")
            except (ValueError, AttributeError):
                code = None
            raise ChannelError(code or "channel_http_error", source=ROUTE, http_status=response.status_code)
        return response.json()

    def payload(self, request):
        args = {"prompt":request["prompt"], "duration":request["duration_seconds"], "aspect":request["aspect_ratio"]}
        if request["sources"]:
            args["reference_images"] = [i["sha256"] for i in request["sources"]]
        result = dict(model=request["model"], capability="video", parameters=args)
        if request["budget_limit"] is not None:
            result["budget_limit"] = request["budget_limit"]
        return result

    def check(self, request, actor):
        if not self.enabled or not self.configured():
            raise ValueError("channel_not_configured")
        self.executor.require_channel(ROUTE, request["model"])
        headers = self.headers(request["project_id"],actor)
        catalog = self.response(self.client.get(self.url+"/v1/capabilities",headers=headers))
        if request["model"] not in catalog["models"]["video"]:
            raise ValueError("capability_not_supported")
        account = self.response(self.client.get(self.url+"/v1/account",headers=headers))
        if account.get("authorized") is not True:
            raise ChannelError("authorization_required", source=ROUTE, accepted=False)
        quote = self.response(self.client.post(self.url+"/v1/cost",headers=headers,json=self.payload(request))).get("credits")
        if type(quote) is not int or quote < 0 or type(account.get("credits")) is not int:
            raise ValueError("invalid_cost")
        if quote > account["credits"] or (request["budget_limit"] is not None and quote > request["budget_limit"]):
            raise ChannelError("quota_exhausted", source=ROUTE, accepted=False)
        return quote

    def preflight(self, **kwargs):
        actor = self.authorize(kwargs["project_id"])
        request,digest = self.normalize(**kwargs)
        quote = self.check(request,actor)
        return dict(status="validated", can_submit=True, generation_verified=False, input_digest=digest,
                    resolved_output={k:request[k] for k in ("route","model","mode","duration_seconds","aspect_ratio")},
                    reference_mapping=request["references"], quoted_credits=quote, unit="Credits")

    def generate(self, idempotency_key, **kwargs):
        actor = self.authorize(kwargs["project_id"])
        request,digest = self.normalize(**kwargs)
        tasks = self.executor.tasks
        old = tasks.find_idempotency(request["project_id"],idempotency_key)
        if old:
            if old.input_digest != digest or old.authority != actor:
                raise TaskConflict("idempotency_conflict")
            return old
        quote = self.check(request,actor)
        return tasks.create(project_id=request["project_id"], idempotency_key=idempotency_key,
                            input_digest=digest, request=request, runtime_task_id="", status="queued",
                            authority=actor, runtime_route=ROUTE, provider_state={"quoted_credits":quote})

    def status(self, record):
        tasks = self.executor.tasks
        with tasks.task_lock(record.video_task_id):
            record = tasks.get(record.video_task_id)
            if record.status in {"succeeded","failed","cancelled"}:
                return record
            try:
                if record.request.get("gateway_profile") != self.profile:
                    raise ValueError("binding_changed")
                actor = self.authorize(record.project_id, record.authority)
                headers = self.headers(record.project_id,actor)
                if not (record.provider_state or {}).get("submission_intent"):
                    self.check(record.request,actor)
                    for ref, expected in zip(record.request["references"]["images"],record.request["sources"],strict=True):
                        path, info = self.reference_source(record.project_id, ref["artifact_id"],"image")
                        if info != expected:
                            raise ValueError("source_integrity_failed")
                        with path.open("rb") as stream:
                            receipt = self.response(self.client.put(self.url+"/v1/inputs", content=stream, headers={
                                **headers,"Content-Type":"image/"+info["format"],"X-Content-SHA256":info["sha256"]}))
                        if receipt.get("sha256") != info["sha256"] or receipt.get("size") != info["size"]:
                            raise ValueError("input_receipt_mismatch")
                    record = tasks.update(record, status="running", runtime_stage="submission_unknown",
                                          provider_state={**record.provider_state, "submission_intent":True})
                    data = self.response(self.client.post(self.url+"/v1/jobs",headers=headers,
                                         json=self.payload(record.request) | {"request_id":record.video_task_id}))
                else:
                    data = self.response(self.client.get(self.url+"/v1/jobs/"+record.video_task_id,headers=headers))
                identity = data.get("call_id")
                if (data.get("request_id") != record.video_task_id or not isinstance(identity,str)
                        or not re.fullmatch("[a-f0-9]{32}",identity) or record.runtime_task_id not in ("",identity)):
                    raise ValueError("gateway_identity_mismatch")
                accepted = data["accepted"]
                stage = data["state"]
                state = dict(record.provider_state, accepted=accepted, actual_credits=data.get("credits"))
                if accepted is None:
                    stage = "submission_unknown"
                elif type(accepted) is not bool or stage not in {"pending","running","failed","succeeded"}:
                    raise ValueError("invalid_gateway_state")
                record = tasks.update(record,runtime_task_id=identity,runtime_stage=stage,provider_state=state,
                                      status="failed" if stage=="failed" else "running",
                                      error=channel_error(data,source=ROUTE,accepted=accepted).public() if stage=="failed" else None)
                if stage == "succeeded" and accepted is True:
                    from .google_flow_artifact import collect
                    return collect(self,record,headers)
                return record
            except (httpx.HTTPError,ValueError,KeyError,TypeError,OSError,PermissionError) as exc:
                record = tasks.get(record.video_task_id)
                err = exc if isinstance(exc,ChannelError) else ChannelError("channel_unavailable",source=ROUTE)
                return tasks.update(record,error=err.public())

    def result(self, record):
        from ..executor import ExecutionError
        record = self.status(record)
        if record.status != "succeeded" or not (record.provider_state or {}).get("content_ref"):
            raise ExecutionError("result_not_ready")
        return record

    def public_result(self, record):
        state = record.provider_state
        return dict(task_id=record.video_task_id, video_task_id=record.video_task_id, status=record.status,
                    model=record.request["model"], runtime_route=ROUTE, media=record.media, content_ref=state["content_ref"],
                    download_path=state["download_path"], quoted_credits=state.get("quoted_credits"),
                    actual_credits=state.get("actual_credits"),input_digest=record.input_digest)
