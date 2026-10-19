"""Read existing native-output evidence without rewriting historical task records.

Only Higgsfield has this native Artifact format today. Unknown formats stay denied.
This module has no provider client and never submits or refreshes a remote task.
"""
import os


def higgsfield_output_path(tasks, record):
    return tasks.root / (record.video_task_id + ".higgsfield.mp4")


def read_output_evidence(tasks, record):
    if record.service != "higgsfield" or not record.authority:
        return None
    if record.authority.get("organization_id") != os.getenv("HIGGSFIELD_ORGANIZATION_ID", ""):
        return None
    members = set(filter(None, os.getenv("HIGGSFIELD_ALLOWED_USERS", "").split(",")))
    if members and record.authority.get("user_id") not in members:
        return None
    media = (record.provider_state or {}).get("produced_media")
    if not isinstance(media, dict):
        return None
    return higgsfield_output_path(tasks, record), media
