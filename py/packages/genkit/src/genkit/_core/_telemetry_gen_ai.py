# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Map Genkit span facts onto OpenTelemetry GenAI semantic conventions.

Owned by the ``otel_ai_semantic_conventions`` handler (not by ``annotate``):

* **Start** — :func:`apply_gen_ai_start_attrs` before the renderer opens the span.
* **Mid / end** — handler binds :func:`project_gen_ai_from_frame` as the annotate
  projector for the span lifetime so later Genkit writes project too.

Swap or remove that handler to change / disable GenAI projection. Pure Dev UI
keys (``genkit:state``, snapshot ids, …) stay ``genkit:*``.

Message bodies follow the GenAI opt-in:
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``.
"""

from __future__ import annotations

import json
import os
from typing import Any, cast

from pydantic import BaseModel

from ._telemetry_contract import current_frame
from ._trace._attrs import TYPE_FACT, Attr, metadata_key

CAPTURE_CONTENT_ENV = 'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'

# Genkit plugin / action name prefixes → gen_ai.provider.name
_PROVIDER_BY_PREFIX: dict[str, str] = {
    'googleai': 'gcp.gen_ai',
    'google-genai': 'gcp.gen_ai',
    'google_genai': 'gcp.gen_ai',
    'vertexai': 'gcp.vertex_ai',
    'vertex': 'gcp.vertex_ai',
    'openai': 'openai',
    'anthropic': 'anthropic',
    'ollama': 'ollama',
    'azureopenai': 'azure.ai.openai',
    'azure': 'azure.ai.openai',
    'bedrock': 'aws.bedrock',
    'aws': 'aws.bedrock',
}

_GOOGLE_PREFIXES = frozenset({'googleai', 'google-genai', 'google_genai', 'vertexai', 'vertex'})


def capture_message_content() -> bool:
    """True when GenAI message-body capture is explicitly opted in."""
    return os.environ.get(CAPTURE_CONTENT_ENV, '').strip().lower() in {'1', 'true', 'yes', 'on'}


def genkit_span_role(attrs: dict[str, Any]) -> str:
    """Logical Genkit role for GenAI mapping (model/tool/flow/…).

    Action spans use ``genkit:type=action`` + ``genkit:metadata:subtype=<kind>``.
    Legacy / test attrs may set ``genkit.type`` / ``genkit:type`` directly to the
    kind (e.g. ``model``).
    """
    type_ = str(attrs.get(TYPE_FACT) or attrs.get(Attr.TYPE) or '')
    subtype = str(attrs.get(Attr.SUBTYPE) or '')
    if type_ == 'model' or subtype == 'model':
        return 'model'
    if type_ == 'agent-turn':
        return 'agent-turn'
    if subtype:
        return subtype
    return type_


def is_model_span(attrs: dict[str, Any]) -> bool:
    return genkit_span_role(attrs) == 'model'


def split_model_ref(ref: str) -> tuple[str | None, str]:
    """Split ``plugin/model`` into ``(prefix, model_id)``; bare names keep no prefix."""
    if not ref:
        return None, ''
    if '/' in ref:
        prefix, rest = ref.split('/', 1)
        return prefix or None, rest or ref
    return None, ref


def provider_name(prefix: str | None) -> str | None:
    if not prefix:
        return None
    return _PROVIDER_BY_PREFIX.get(prefix.lower(), prefix.lower())


def model_operation_name(prefix: str | None) -> str:
    """Google-family model APIs → ``generate_content``; otherwise chat-shaped."""
    if prefix and prefix.lower() in _GOOGLE_PREFIXES:
        return 'generate_content'
    return 'chat'


def _parse_json_obj(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True, exclude_none=True)
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None
    return None


def _as_dict(value: object) -> dict[str, Any] | None:
    return _parse_json_obj(value)


def _set_num(attrs: dict[str, Any], key: str, value: object) -> None:
    if value is None:
        return
    try:
        if isinstance(value, bool):
            return
        if isinstance(value, float):
            attrs[key] = value
        else:
            attrs[key] = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return


def _set_float(attrs: dict[str, Any], key: str, value: object) -> None:
    if value is None:
        return
    try:
        attrs[key] = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return


def _config_get(config: dict[str, Any], *keys: str) -> object:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def apply_model_request_attrs(attrs: dict[str, Any], *, span_name: str) -> None:
    """Stamp start-time ``gen_ai.*`` attrs for a model span from name + input."""
    model_ref = str(attrs.get('model') or attrs.get(Attr.MODEL) or span_name or '')
    prefix, model_id = split_model_ref(model_ref)
    op = model_operation_name(prefix)
    attrs['gen_ai.operation.name'] = op
    if model_id or model_ref:
        attrs['gen_ai.request.model'] = model_id or model_ref
        attrs.setdefault(Attr.MODEL, model_ref)
    provider = provider_name(prefix)
    if provider:
        attrs['gen_ai.provider.name'] = provider

    req = _parse_json_obj(attrs.get(Attr.INPUT))
    if not req:
        return

    config = req.get('config')
    if isinstance(config, dict):
        _set_float(attrs, 'gen_ai.request.temperature', _config_get(config, 'temperature'))
        _set_float(attrs, 'gen_ai.request.top_p', _config_get(config, 'topP', 'top_p'))
        _set_num(attrs, 'gen_ai.request.top_k', _config_get(config, 'topK', 'top_k'))
        _set_num(
            attrs,
            'gen_ai.request.max_tokens',
            _config_get(config, 'maxOutputTokens', 'max_output_tokens', 'maxTokens', 'max_tokens'),
        )
        stops = _config_get(config, 'stopSequences', 'stop_sequences')
        if isinstance(stops, list) and stops:
            attrs['gen_ai.request.stop_sequences'] = [str(s) for s in stops]
        seed = _config_get(config, 'seed')
        _set_num(attrs, 'gen_ai.request.seed', seed)
        freq = _config_get(config, 'frequencyPenalty', 'frequency_penalty')
        _set_float(attrs, 'gen_ai.request.frequency_penalty', freq)
        presence = _config_get(config, 'presencePenalty', 'presence_penalty')
        _set_float(attrs, 'gen_ai.request.presence_penalty', presence)
        # Config version sometimes carries the resolved model revision.
        version = _config_get(config, 'version')
        if isinstance(version, str) and version and 'gen_ai.request.model' not in attrs:
            attrs['gen_ai.request.model'] = version

    output = req.get('output')
    if isinstance(output, dict):
        fmt = output.get('format') or output.get('contentType') or output.get('content_type')
        if isinstance(fmt, str) and fmt:
            # Low-cardinality modality hint when the caller asked for a format.
            if fmt in {'json', 'text', 'image', 'jsonl', 'array', 'enum'}:
                attrs['gen_ai.output.type'] = 'json' if fmt in {'json', 'jsonl', 'array', 'enum'} else fmt

    tools = req.get('tools')
    if isinstance(tools, list) and tools:
        defs: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get('name')
            if not name:
                continue
            entry: dict[str, Any] = {'type': 'function', 'name': str(name)}
            if tool.get('description') is not None:
                entry['description'] = tool['description']
            schema = tool.get('inputSchema') or tool.get('input_schema')
            if schema is not None:
                entry['parameters'] = schema
            defs.append(entry)
        if defs:
            attrs['gen_ai.tool.definitions'] = json.dumps(defs)

    if capture_message_content():
        messages = req.get('messages')
        if isinstance(messages, list) and messages:
            attrs['gen_ai.input.messages'] = json.dumps(messages)
        system_parts: list[Any] = []
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get('role') == 'system':
                    content = msg.get('content')
                    if content is not None:
                        system_parts.append(content)
        if system_parts:
            attrs['gen_ai.system_instructions'] = json.dumps(system_parts)


def apply_gen_ai_start_attrs(*, name: str, attrs: dict[str, Any]) -> None:
    """Enrich ``attrs`` with GenAI semconv fields known at span start."""
    role = genkit_span_role(attrs)

    if role == 'model':
        apply_model_request_attrs(attrs, span_name=name)
        return

    if role == 'tool':
        attrs['gen_ai.operation.name'] = 'execute_tool'
        attrs['gen_ai.tool.name'] = name
        attrs['gen_ai.tool.type'] = 'function'
        if capture_message_content():
            tool_input = attrs.get(Attr.INPUT)
            if tool_input is not None:
                attrs['gen_ai.tool.call.arguments'] = (
                    tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
                )
        return

    if role == 'embedder':
        attrs['gen_ai.operation.name'] = 'embeddings'
        prefix, model_id = split_model_ref(name)
        if model_id:
            attrs['gen_ai.request.model'] = model_id
        provider = provider_name(prefix)
        if provider:
            attrs['gen_ai.provider.name'] = provider
        return

    if role in {'agent-turn', 'agent'}:
        attrs['gen_ai.operation.name'] = 'invoke_agent'
        attrs['gen_ai.agent.name'] = name
        return

    if role in {'flow', 'flowStep'}:
        attrs['gen_ai.operation.name'] = 'invoke_workflow'
        attrs['gen_ai.workflow.name'] = name
        return

    if role == 'retriever':
        attrs['gen_ai.operation.name'] = 'retrieval'
        return

    if role == 'reranker':
        attrs['gen_ai.operation.name'] = 'retrieval'
        return

    if role == 'evaluator':
        attrs['gen_ai.operation.name'] = 'genkit.evaluator'
        attrs['gen_ai.evaluation.name'] = name
        return

    if role:
        attrs['gen_ai.operation.name'] = f'genkit.{role}'
    else:
        attrs['gen_ai.operation.name'] = 'genkit'


def gen_ai_span_name(name: str, attrs: dict[str, Any]) -> str:
    """OTel span display name; model spans follow ``{operation} {model}``."""
    if not is_model_span(attrs):
        return name
    op = str(attrs.get('gen_ai.operation.name') or 'chat')
    model = str(attrs.get('gen_ai.request.model') or attrs.get(Attr.MODEL) or name)
    return f'{op} {model}'.rstrip()


def _model_output_gen_ai(value: object, attrs: dict[str, Any]) -> dict[str, Any]:
    """Derive response/usage ``gen_ai.*`` attrs from a model-shaped output value."""
    data = _as_dict(value)
    if data is None:
        return {}

    role = genkit_span_role(attrs)
    looks_like_model = any(key in data for key in ('finishReason', 'finish_reason', 'usage', 'message', 'candidates'))
    if role != 'model' and not looks_like_model:
        return {}

    out: dict[str, Any] = {}
    usage = data.get('usage')
    if isinstance(usage, dict):
        for gen_key, src_keys in (
            ('gen_ai.usage.input_tokens', ('inputTokens', 'input_tokens')),
            ('gen_ai.usage.output_tokens', ('outputTokens', 'output_tokens')),
            ('gen_ai.usage.reasoning.output_tokens', ('thoughtsTokens', 'thoughts_tokens')),
            ('gen_ai.usage.cache_read.input_tokens', ('cachedContentTokens', 'cached_content_tokens')),
        ):
            raw = next((usage[k] for k in src_keys if k in usage and usage[k] is not None), None)
            if raw is None:
                continue
            try:
                out[gen_key] = int(raw)
            except (TypeError, ValueError):
                continue

    finish = data.get('finishReason', data.get('finish_reason'))
    if finish is not None:
        reason = finish.value if hasattr(finish, 'value') else finish
        out['gen_ai.response.finish_reasons'] = [str(reason)]

    custom = data.get('custom')
    if isinstance(custom, dict):
        response_id = custom.get('id') or custom.get('responseId') or custom.get('response_id')
        if isinstance(response_id, str) and response_id:
            out['gen_ai.response.id'] = response_id

    if capture_message_content():
        message = data.get('message')
        if message is not None:
            payload = message if isinstance(message, list) else [message]
            out['gen_ai.output.messages'] = json.dumps(payload)

    response_model = attrs.get('gen_ai.request.model') or attrs.get(Attr.MODEL)
    if response_model:
        out['gen_ai.response.model'] = str(response_model)
    return out


def _mid_gen_ai_from_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Project mid-span Genkit metadata onto GenAI attrs."""
    out: dict[str, Any] = {}
    session_id = attrs.get(metadata_key('agent:sessionId'))
    if session_id is not None and str(session_id):
        out['gen_ai.conversation.id'] = str(session_id)

    role = genkit_span_role(attrs)
    if role == 'tool' and capture_message_content():
        tool_input = attrs.get(Attr.INPUT)
        if tool_input is not None:
            out['gen_ai.tool.call.arguments'] = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
        tool_output = attrs.get(Attr.OUTPUT)
        if tool_output is not None:
            out['gen_ai.tool.call.result'] = tool_output if isinstance(tool_output, str) else json.dumps(tool_output)
    return out


def project_gen_ai_from_frame(frame: Any) -> None:  # noqa: ANN401
    """Sync ``gen_ai.*`` from all mappable Genkit facts currently on ``frame``.

    Called after every Genkit ``annotate`` so mid/end facts project the same way
    start facts do. Writes go through ``write_span_attr`` (no re-entrant project).
    """
    from ._tracing import write_span_attr

    attrs = frame.attrs
    name = str(attrs.get(Attr.NAME) or frame.name or '')

    # Re-derive start-shaped projections from current identity + input.
    scratch = dict(attrs)
    apply_gen_ai_start_attrs(name=name, attrs=scratch)
    desired: dict[str, Any] = {k: v for k, v in scratch.items() if str(k).startswith('gen_ai.')}

    desired.update(_mid_gen_ai_from_attrs(attrs))

    raw_output = attrs.get(Attr.OUTPUT)
    if raw_output is not None:
        desired.update(_model_output_gen_ai(raw_output, attrs))

    for key, value in desired.items():
        if attrs.get(key) == value:
            continue
        write_span_attr(key, value)


def annotate_gen_ai_from_output(value: object) -> None:
    """Project a model output onto ``gen_ai.*`` (used by ``annotate_output``)."""
    frame = current_frame()
    if frame is None:
        return
    # annotate_output already wrote genkit:output; sync projections from the frame.
    project_gen_ai_from_frame(frame)
