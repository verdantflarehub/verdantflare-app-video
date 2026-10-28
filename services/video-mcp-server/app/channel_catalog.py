"""Project installed channel declarations without rewriting reviewed snapshots."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .channel_registry import ChannelRegistry, RegistryUnavailable


def connections(executor):
    return {
        'fal': (executor.fal.gateway_url, executor.fal.gateway_token),
        'grok': (executor.grok.channel_url, executor.grok.channel_token),
        'google': (executor.google.channel_url, executor.google.channel_token),
        'higgsfield': (executor.higgsfield.url, executor.higgsfield.token),
    }


def validate_directory(value, channel):
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int or value.get('schema_version') != 1
            or value.get('channel_id') != channel or not isinstance(value.get('version'), str)
            or not isinstance(value.get('capabilities'), list) or len(value['capabilities']) > 64):
        raise ValueError('invalid_channel_capabilities')
    seen = set()
    for cap in value['capabilities']:
        if (not isinstance(cap, dict) or any(not isinstance(cap.get(k), str) or not cap[k].strip()
                for k in ('capability_id', 'type', 'model_id', 'display_name', 'operation', 'execution_state'))
                or cap['capability_id'] in seen or cap['type'] != 'video'
                or not isinstance(cap.get('parameter_schema'), dict)
                or cap['execution_state'] not in {'enabled', 'disabled', 'catalog_only', 'adapter_connected_unverified'}
                or '\ufffd' in cap['display_name'] or cap['display_name'] == cap['model_id']
                or any(ord(char) < 32 for char in cap['display_name'])
                or ('endpoint_id' in cap and (not isinstance(cap['endpoint_id'], str) or not cap['endpoint_id']))):
            raise ValueError('invalid_channel_capabilities')
        seen.add(cap['capability_id'])
        try:
            Draft202012Validator.check_schema(cap['parameter_schema'])
        except SchemaError:
            raise ValueError('invalid_channel_parameter_schema') from None
    # Only contract fields cross into the public model catalog.
    result = {key: deepcopy(value[key]) for key in ('schema_version', 'channel_id', 'version',
              'channel_name', 'state', 'capabilities') if key in value}
    public_fields = {'capability_id', 'type', 'model_id', 'endpoint_id', 'display_name', 'operation',
        'parameter_schema', 'execution_state', 'card_group', 'card_title', 'input_summary', 'output_summary',
        'protocol', 'output_resolutions', 'source_url', 'reason_code', 'retired_on'}
    result['capabilities'] = [{key: val for key, val in cap.items() if key in public_fields}
                              for cap in result['capabilities']]
    return result


def read_directory(channel, url, token, *, transport=None):
    parsed = urlsplit(url)
    if (not url or len(token) < 32 or parsed.scheme not in {'http', 'https'}
            or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment):
        return {'schema_version': 1, 'channel_id': channel, 'state': 'unavailable',
                'reason_code': 'channel_not_configured', 'capabilities': []}
    try:
        with httpx.Client(timeout=3, trust_env=False, follow_redirects=False, transport=transport) as client:
            with client.stream('GET', url.rstrip('/') + '/v1/capabilities',
                    headers={'Authorization': 'Bearer ' + token, 'X-Caller-Id': 'video-mcp'}) as response:
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 256 * 1024:
                        raise ValueError('channel_capabilities_too_large')
                if response.status_code != 200:
                    raise ValueError('channel_capabilities_unavailable')
            value = validate_directory(json.loads(raw), channel)
            value.setdefault('state', 'available')
            return value
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        return {'schema_version': 1, 'channel_id': channel, 'state': 'unavailable',
                'reason_code': 'channel_capabilities_unavailable', 'capabilities': []}


def supported_models(executor):
    from .providers.fal import FAL_MODEL_ID, FAL_PUBLIC_MODEL, FAL_SD2_MODEL_ID, FAL_SD2_PUBLIC_MODEL
    from .providers.grok import MODELS as GROK_MODELS
    from .providers.google import MODELS as GOOGLE_MODELS
    from .providers.higgsfield import MODELS as HF_MODELS
    return {'fal': {FAL_MODEL_ID: FAL_PUBLIC_MODEL, FAL_SD2_MODEL_ID: FAL_SD2_PUBLIC_MODEL},
            'grok': {model: model for model in GROK_MODELS},
            'google': {model: model for model in GOOGLE_MODELS},
            'higgsfield': {model: model for model in HF_MODELS}}


def supported_operation(channel, operation):
    return operation == ('video.generate.reference' if channel == 'fal' else 'video.generate')


def adapter_facts(executor):
    facts = {}
    for channel in ('grok', 'google'):
        adapter = getattr(executor, channel)
        if hasattr(adapter, 'capability'):
            facts[channel] = adapter.capability()
    if hasattr(executor.higgsfield, 'capabilities'):
        facts['higgsfield'] = {item['model']: item for item in executor.higgsfield.capabilities()}
    return facts


def declaration_profiles(capability, adapter=None):
    props = capability['parameter_schema'].get('properties', {})
    if not isinstance(props, dict):
        return []
    field = lambda name: props.get(name) if isinstance(props.get(name), dict) else {}
    modes = field('mode').get('enum', [])
    duration = field('duration_seconds')
    bounds = ({'minimum': duration['minimum'], 'maximum': duration['maximum']}
              if 'minimum' in duration and 'maximum' in duration else None)
    resolutions = field('resolution').get('enum', [])
    ratios = field('aspect_ratio').get('enum', [])
    if not all(isinstance(values, list) for values in (modes, resolutions, ratios)):
        return []
    if adapter:
        modes = [mode for mode in modes if mode in adapter.get('modes', [])]
        resolutions = [value for value in resolutions if value in adapter.get('resolutions', [])]
        ratios = [value for value in ratios if value in adapter.get('aspect_ratios', [])]
        if bounds and adapter.get('duration_seconds'):
            bounds = {'minimum': max(bounds['minimum'], adapter['duration_seconds']['minimum']),
                      'maximum': min(bounds['maximum'], adapter['duration_seconds']['maximum'])}
    return [{'id': 'declared-' + mode, 'mode': mode, 'resolutions': resolutions,
             'max_resolution': resolutions[-1] if resolutions else None,
             'duration_seconds': bounds, 'duration_choices_seconds': [],
             'aspect_ratios': ratios, 'input_limits': None,
             'audio_input': 'unknown', 'audio_output': 'unknown', 'review_status': 'pending',
             'verification_status': 'unverified'} for mode in modes if isinstance(mode, str)]


def project(executor):
    from .catalog import group_entries
    catalog = executor.catalog.snapshot()
    installed = connections(executor)
    registry = ChannelRegistry({channel: url.rstrip('/') for channel, (url, _) in installed.items() if url})
    try:
        registrations = registry.snapshot()
        registry_error = None
    except (RegistryUnavailable, httpx.HTTPError, ValueError, TypeError, KeyError, OverflowError):
        registrations, registry_error = {}, 'channel_registry_unavailable'
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {channel: pool.submit(read_directory, channel, *connection) for channel, connection in installed.items()}
        directories = {channel: future.result() for channel, future in futures.items()}
    supported = supported_models(executor)
    facts = adapter_facts(executor)
    previous = {channel['id']: channel for channel in catalog.get('channels', [])}
    channels = [deepcopy(channel) for channel in catalog.get('channels', []) if channel['id'] not in installed]
    execution = {}
    for channel, directory in directories.items():
        registered = registrations.get(channel)
        directory['registration_state'] = 'enabled' if registered else 'unavailable'
        directory['registry_error'] = registry_error
        source = deepcopy(previous.get(channel, {'id': channel, 'display_name': directory.get('channel_name') or channel,
                                                 'sort_order': len(channels), 'models': []}))
        source['display_name'] = directory.get('channel_name') or source['display_name']
        source['directory_state'] = directory['state']
        source['registration_state'] = directory['registration_state']
        execution[channel] = {}
        if directory['state'] != 'available':
            for model in source['models']:
                execution[channel][model['id']] = {'state': 'unavailable'}
                model['declaration_source'] = 'historical_review'
        else:
            old = {model.get('upstream_model_id'): model for model in source['models']}
            source['models'] = []
            source['version'] = directory['version']
            for cap in directory['capabilities']:
                identity = cap.get('endpoint_id') or cap['model_id']
                model = deepcopy(old.get(identity, {}))
                model.update(id=cap['capability_id'], model_id=cap['model_id'], upstream_model_id=identity,
                    display_name=cap['display_name'], entry_kind='model', business_type='video_generation',
                    identity_review='pending', description=' · '.join(filter(None, [cap.get('input_summary'), cap.get('output_summary')])),
                    declaration_source='installed_channel', parameter_schema=cap['parameter_schema'],
                    execution_status=cap['execution_state'], capability_id=cap['capability_id'])
                for field in ('team_profiles', 'supplier_profiles', 'billing_modes'):
                    model.setdefault(field, [])
                adapter = facts.get(channel, {})
                if channel == 'higgsfield':
                    adapter = adapter.get(identity, {})
                if declaration_profiles(cap):
                    model['supplier_profiles'] = declaration_profiles(cap)
                    model['team_profiles'] = declaration_profiles(cap, adapter) if adapter else []
                public_model = supported[channel].get(identity)
                state = cap['execution_state']
                adapter_ready = (executor.fal.connected() if channel == 'fal' else
                                 adapter.get('configured') is True and adapter.get('enabled') is True)
                if state in {'disabled', 'catalog_only'}:
                    pass
                elif public_model is None or not supported_operation(channel, cap['operation']):
                    state = 'catalog_only'
                elif not registered:
                    state = 'unavailable'
                elif not adapter_ready:
                    state = 'adapter_disconnected'
                can_submit = (public_model is not None and supported_operation(channel, cap['operation'])
                              and state in {'enabled', 'adapter_connected_unverified'}
                              and adapter_ready
                              and registry.permits(channel, cap['capability_id'], registrations)
                              and (state == 'enabled' or channel in registry.allow_unverified))
                execution[channel][model['id']] = {'state': state, 'route': channel, 'model': public_model,
                                                   'can_submit': can_submit, 'capability_id': cap['capability_id'],
                                                   'reason_code': cap.get('reason_code')}
                source['models'].append(model)
        if source['models'] or directory['state'] == 'available' or installed[channel][0]:
            channels.append(source)
    groups, conflicts = group_entries(channels)
    digest = hashlib.sha256(json.dumps([channels, directories, execution], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {**catalog, 'state': 'ready' if groups else 'empty', 'snapshot_id': digest,
            'channels': channels, 'groups': groups, 'conflicts': conflicts, 'execution': execution,
            'channel_capabilities': directories, 'registry_state': 'unavailable' if registry_error else 'available'}
