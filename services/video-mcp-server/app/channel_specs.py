"""Effective Video adapter limits, keyed by the exact execution identity."""
from copy import deepcopy

from .channel_errors import safe_code

from .providers.fal import (FAL_PUBLIC_MODEL, FAL_SD2_PUBLIC_MODEL, FAL_LIP_PUBLIC_MODEL, MAX_FAL_IMAGE_BYTES,
                            MAX_FAL_VIDEO_BYTES, MAX_FAL_AUDIO_BYTES)
from .providers.google import MAX_SOURCE_IMAGE_BYTES, MAX_SOURCE_IMAGE_PIXELS


def reference_limits(route, model):
    mib = 1024 * 1024
    value = {'scope': 'video_adapter', 'max_total_bytes': None}
    if route == 'fal' and model == FAL_LIP_PUBLIC_MODEL:
        value.update(images=1, videos=0, audios=1, total=2, minimum_total=2,
            max_bytes={'images': MAX_FAL_IMAGE_BYTES, 'videos': 0, 'audios': MAX_FAL_AUDIO_BYTES},
            formats={'images': ['PNG', 'JPEG', 'WebP'], 'videos': [], 'audios': ['WAV/PCM']},
            duration_seconds={'audios': {'minimum': 5, 'maximum': 15}},
            image_aspect_ratio={'minimum': 0.4, 'maximum': 2.5},
            audio_duration_tolerance_seconds=0.05, request_body_max_bytes=96 * mib,
            request_body_encoding='json_with_base64_media')
    elif route == 'fal':
        sd2 = model == FAL_SD2_PUBLIC_MODEL
        value.update(images=9, videos=3, audios=3, total=12, minimum_total=1,
            max_bytes={'images': 30 * mib if sd2 else MAX_FAL_IMAGE_BYTES,
                       'videos': 50 * mib - 1 if sd2 else MAX_FAL_VIDEO_BYTES,
                       'audios': MAX_FAL_AUDIO_BYTES},
            duration_seconds={'videos': {'minimum': 2, 'maximum': 15},
                              'audios': {'minimum': 2, 'maximum': 15}},
            combined_duration_seconds={'videos': 15, 'audios': 15},
            request_body_max_bytes=96 * mib,
            request_body_encoding='json_with_base64_media')
        if sd2:
            value['required_media_any_of'] = ['images', 'videos']
    elif route in {'grok', 'google'}:
        value.update(images=1, videos=0, audios=0, total=1,
            max_bytes={'images': MAX_SOURCE_IMAGE_BYTES if route == 'google' else 10 * mib,
                       'videos': 0, 'audios': 0},
            formats={'images': ['PNG', 'JPEG', 'WebP'], 'videos': [], 'audios': []},
            max_image_pixels=MAX_SOURCE_IMAGE_PIXELS if route == 'google' else 16 * mib,
            by_mode={'text_to_video': {'images': 0, 'videos': 0, 'audios': 0, 'total': 0},
                     'image_to_video': {'images': 1, 'videos': 0, 'audios': 0, 'total': 1, 'minimum_total': 1}})
    elif route == 'higgsfield':
        mixed = model in {'wan3_0', 'seedance_2_5_draft'}
        value.update(images=4 if mixed else 0, videos=1 if mixed else 0, audios=0, total=5 if mixed else 0,
            minimum_total=1 if model == 'seedance_2_5_draft' else 0,
            max_bytes={'images': 10 * mib if mixed else 0, 'videos': 128 * mib if mixed else 0, 'audios': 0},
            formats={'images': ['PNG', 'JPEG', 'WebP'] if mixed else [],
                     'videos': ['MP4'] if mixed else [], 'audios': []})
        if mixed:
            value['by_mode'] = {'reference_to_video': {'images': 4, 'videos': 1, 'audios': 0,
                                                      'total': 5, 'minimum_total': 1}}
            if model == 'wan3_0':
                value['by_mode']['text_to_video'] = {'images': 0, 'videos': 0, 'audios': 0, 'total': 0}
    return value


def effective_route(template, declaration, entry):
    route, model = template['route'], entry['model']
    value = {**deepcopy(template), 'model': model, 'execution_state': entry['state'],
             'can_submit': entry.get('can_submit', False), 'readiness': 'not_checked',
             'capability_id': entry['capability_id'], 'display_name': declaration['display_name']}
    value.pop('models', None)
    reason = entry.get('reason_code')
    if not value['can_submit'] and safe_code(reason, route):
        value['reason_code'] = reason
    channel_schema = deepcopy(declaration.get('parameter_schema', {}))
    props = channel_schema.get('properties', {})
    props = props if isinstance(props, dict) else {}
    effective = {}
    for name, values in [('mode', value.get('modes')), ('resolution', value.get('resolutions')),
                         ('aspect_ratio', value.get('aspect_ratios'))]:
        source = props.get(name, {})
        source = source if isinstance(source, dict) else {}
        if route == 'fal' and model == FAL_LIP_PUBLIC_MODEL and name == 'resolution' and 'enum' in source:
            source = {**source, 'enum': [item.lower() for item in source['enum'] if isinstance(item, str)]}
        if values is not None:
            enum = [item for item in values if 'enum' not in source or item in source['enum']]
            effective[name] = {**source, 'enum': enum}
            value[{'mode': 'modes', 'resolution': 'resolutions', 'aspect_ratio': 'aspect_ratios'}[name]] = enum
    if 'duration_seconds' in value:
        bounds = deepcopy(value['duration_seconds'])
        source = props.get('duration_seconds', {})
        if isinstance(source, dict):
            if 'minimum' in source:
                bounds['minimum'] = max(bounds['minimum'], source['minimum'])
            if 'maximum' in source:
                bounds['maximum'] = min(bounds['maximum'], source['maximum'])
        value['duration_seconds'] = bounds
        effective['duration_seconds'] = {'type': 'integer', **bounds}
    if isinstance(props.get('prompt'), dict):
        effective['prompt'] = deepcopy(props['prompt'])
    limits = reference_limits(route, model)
    value['reference_limits'] = limits
    effective['references'] = {'type': 'object', 'additionalProperties': False, 'properties': {
        kind: {'type': 'array', 'maxItems': limits[kind]} for kind in ('images', 'videos', 'audios')}}
    value['channel_parameter_schema'] = channel_schema
    value['parameter_schema'] = {'type': 'object', 'properties': effective}
    value['defaults'] = {'resolution': '480p' if model == 'seedance_2_5_draft' else '720p'}
    if route in {'google', 'grok'}:
        value['defaults']['by_mode'] = {'text_to_video': {'aspect_ratio': '16:9'}}
    elif route == 'higgsfield':
        value['defaults']['aspect_ratio'] = '16:9'
    elif route == 'fal':
        value['defaults']['mode'] = 'reference_to_video'
        if model == FAL_LIP_PUBLIC_MODEL:
            value['defaults'].update(mode='lip_sync', resolution='768p', aspect_ratio='adaptive')
        if model == FAL_PUBLIC_MODEL:
            value['defaults'].pop('resolution')
            value['output_resolutions'] = ['480P']
            value['parameter_schema']['properties']['resolution'] = {'type': 'null'}
    return value


def external_routes(templates, catalog):
    exact = {}
    for item in templates:
        for model in item.get('models', [item['model']]):
            exact[item['route'], model] = item
    result = []
    for channel, entries in catalog.get('execution', {}).items():
        declarations = {cap['capability_id']: cap for cap in
                        catalog.get('channel_capabilities', {}).get(channel, {}).get('capabilities', [])}
        for entry in entries.values():
            template = exact.get((channel, entry.get('model')))
            declaration = declarations.get(entry.get('capability_id'))
            if template and declaration:
                result.append(effective_route(template, declaration, entry))
    return result
