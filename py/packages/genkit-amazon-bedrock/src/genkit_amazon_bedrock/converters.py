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

"""Pure conversion functions between Genkit types and the Bedrock Converse API.

Everything here is side-effect free and testable without AWS: request builders
return the keyword arguments for ``client.converse(...)``, response parsers take
the raw response dict.
"""

import base64
import json
from typing import Any

from pydantic import ValidationError

from genkit import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
)
from genkit.plugin_api import GenkitError
from genkit_amazon_bedrock.config import BedrockConfig

# Metadata keys used to round-trip Bedrock reasoning ("thinking") content back
# into a follow-up request. Bedrock returns signed and sometimes redacted
# reasoning that must be replayed verbatim on the next turn or the model
# rejects it, so both are stashed on the part metadata: the signature verbatim
# (it is a string on the wire), the redacted blob as a base64 string so the
# part stays JSON-serializable. These keys are Bedrock-specific: a generic
# reasoning part (without these) is intentionally NOT round-tripped, so
# foreign reasoning can't corrupt a Bedrock conversation.
REASONING_SIGNATURE_METADATA_KEY = 'bedrockReasoningSignature'
REDACTED_CONTENT_METADATA_KEY = 'bedrockRedactedContent'

# Custom-part key marking a prompt cache point.
CACHE_POINT_CUSTOM_KEY = 'bedrockCachePointType'
DEFAULT_CACHE_POINT_TYPE = 'default'

IMAGE_FORMATS = {
    'image/png': 'png',
    'image/jpeg': 'jpeg',
    # Common alias; Bedrock's format enum has no "jpg".
    'image/jpg': 'jpeg',
    'image/gif': 'gif',
    'image/webp': 'webp',
}

DOCUMENT_FORMATS = {
    'application/pdf': 'pdf',
    'text/html': 'html',
    'text/plain': 'txt',
    'text/markdown': 'md',
    'text/csv': 'csv',
    'application/msword': 'doc',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/vnd.ms-excel': 'xls',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
}

# Bedrock stopReason values → Genkit finish reasons. Unknown values map to
# OTHER instead of raising: AWS adds stop reasons without notice.
STOP_REASON_MAP = {
    'end_turn': FinishReason.STOP,
    'stop_sequence': FinishReason.STOP,
    # Genkit's core loop drives tool re-entry by inspecting parts, not the
    # finish reason, so a tool-use turn is a normal stop.
    'tool_use': FinishReason.STOP,
    'max_tokens': FinishReason.LENGTH,
    'model_context_window_exceeded': FinishReason.LENGTH,
    'content_filtered': FinishReason.BLOCKED,
    'guardrail_intervened': FinishReason.BLOCKED,
    'malformed_model_output': FinishReason.OTHER,
    'malformed_tool_use': FinishReason.OTHER,
}


def cache_point_part(cache_type: str = DEFAULT_CACHE_POINT_TYPE) -> Part:
    """Builds a prompt cache-point part.

    A cache point should be inserted after a big static prompt that is reused
    across multiple requests.

    Args:
        cache_type: Bedrock cache-point type; only ``default`` exists today.

    Returns:
        A custom Part that converts to a Converse ``cachePoint`` block.
    """
    return Part.model_validate({'custom': {CACHE_POINT_CUSTOM_KEY: cache_type}})


def _cache_point_type(root: Any) -> str | None:  # noqa: ANN401
    custom = getattr(root, 'custom', None)
    if not isinstance(custom, dict):
        return None
    value = custom.get(CACHE_POINT_CUSTOM_KEY)
    # Compare by value: the marker arrives as a plain string after any JSON
    # round-trip (resumed/serialized flows).
    return value if isinstance(value, str) and value else None


def _metadata_bytes(metadata: dict[str, Any] | None, key: str) -> bytes | None:  # noqa: ANN401
    """Reads a bytes metadata value stored in its base64-string form.

    Raw bytes are also tolerated for programmatically-built parts; invalid
    base64 yields None.
    """
    if not metadata:
        return None
    value = metadata.get(key)
    if isinstance(value, bytes):
        return value
    if isinstance(value, str) and value:
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, TypeError):
            return None
    return None


def _metadata_str(metadata: dict[str, Any] | None, key: str) -> str | None:  # noqa: ANN401
    """Reads a string metadata value, tolerating a UTF-8 bytes form.

    The reasoning signature is a string on the Converse wire and must be
    replayed verbatim, never re-encoded.
    """
    if not metadata:
        return None
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return value
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8') or None
        except UnicodeDecodeError:
            return None
    return None


def normalize_config(config: Any) -> BedrockConfig | None:  # noqa: ANN401
    """Coerces the request config into a BedrockConfig.

    Accepts a BedrockConfig, any pydantic model (e.g. the core ModelConfig the
    framework validates configs into), or a plain dict — the historical shape
    on resumed/serialized flows.

    Args:
        config: The raw ``request.config`` value.

    Returns:
        A BedrockConfig, or None when no config was given.

    Raises:
        GenkitError: INVALID_ARGUMENT for unsupported config types, unknown
            config keys, and values that fail validation.
    """
    if config is None:
        return None
    if isinstance(config, BedrockConfig):
        return config
    if isinstance(config, dict):
        payload = config
    else:
        dump = getattr(config, 'model_dump', None)
        if not callable(dump):
            raise GenkitError(
                message=(
                    f'bedrock: unexpected config type {type(config).__name__}, want BedrockConfig, ModelConfig, or dict'
                ),
                status='INVALID_ARGUMENT',
            )
        payload = dump(exclude_none=True)
    try:
        return BedrockConfig.model_validate(payload)
    except ValidationError as e:
        unknown = sorted({str(error['loc'][0]) for error in e.errors() if error['type'] == 'extra_forbidden'})
        if unknown:
            raise GenkitError(
                message=(
                    f'bedrock: unknown config key(s): {", ".join(unknown)}; only declared BedrockConfig '
                    'fields (e.g. maxOutputTokens) are sent, model-specific options go in '
                    'additionalModelRequestFields'
                ),
                status='INVALID_ARGUMENT',
            ) from e
        raise GenkitError(message=f'bedrock: invalid config: {e}', status='INVALID_ARGUMENT') from e


def _effective_max_tokens(config: BedrockConfig | None) -> int | None:
    if config is None or config.max_output_tokens is None or config.max_output_tokens <= 0:
        return None
    return int(config.max_output_tokens)


def build_inference_config(config: BedrockConfig | None) -> dict[str, Any] | None:  # noqa: ANN401
    """Builds the Converse ``inferenceConfig`` member; None when empty."""
    if config is None:
        return None
    inference_config: dict[str, Any] = {}
    max_tokens = _effective_max_tokens(config)
    if max_tokens is not None:
        inference_config['maxTokens'] = max_tokens
    if config.temperature is not None:
        inference_config['temperature'] = config.temperature
    if config.top_p is not None:
        inference_config['topP'] = config.top_p
    if config.stop_sequences:
        inference_config['stopSequences'] = config.stop_sequences
    return inference_config or None


def to_bedrock_role(role: Role | str) -> str:
    """Maps a Genkit role to a Converse role (only user/assistant exist).

    Genkit's TOOL role becomes ``user``: tool results travel back to Bedrock
    inside a user message.
    """
    if role in (Role.USER, Role.TOOL):
        return 'user'
    if role == Role.MODEL:
        return 'assistant'
    raise GenkitError(message=f'bedrock: unsupported role {role!r}', status='INVALID_ARGUMENT')


def _media_mime(media: Any) -> str:  # noqa: ANN401
    content_type = (getattr(media, 'content_type', None) or '').strip()
    url = getattr(media, 'url', '') or ''
    if not content_type and url.startswith('data:'):
        header = url.split(',', 1)[0].removeprefix('data:')
        content_type = header.split(';', 1)[0].strip()
    if not content_type:
        raise GenkitError(message='bedrock: media part has no content type', status='INVALID_ARGUMENT')
    return content_type.split(';', 1)[0].strip().lower()


def _decode_media_payload(url: str) -> bytes:
    """Decodes media to raw bytes; boto3 base64-encodes them for the wire.

    Accepts a ``data:<mime>;base64,...`` URL or a bare base64 string.
    Double-encoding (sending the base64 string as bytes) is the classic bug.
    """
    data = (url or '').strip()
    if not data:
        raise GenkitError(message='bedrock: media part has empty data', status='INVALID_ARGUMENT')
    # Substring search tolerates multi-parameter data-URL headers.
    marker = data.find(';base64,')
    if marker != -1:
        payload = data[marker + len(';base64,') :].strip()
    elif data.startswith('data:'):
        raise GenkitError(
            message="bedrock: data URL must be base64-encoded (use ';base64,' prefix)",
            status='INVALID_ARGUMENT',
        )
    elif data.startswith(('http://', 'https://')):
        raise GenkitError(
            message='bedrock: remote URLs are not supported; use a data URL or base64-encoded data',
            status='INVALID_ARGUMENT',
        )
    else:
        payload = data
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as e:
        raise GenkitError(message=f'bedrock: decode base64 media: {e}', status='INVALID_ARGUMENT') from e


def media_to_block(root: Any) -> dict[str, Any]:  # noqa: ANN401
    """Converts a media part to an image or document content block."""
    media = root.media
    mime = _media_mime(media)
    payload = _decode_media_payload(media.url)
    # Document formats are checked first: text/plain and text/html are
    # documents to Converse, and no MIME type belongs to both tables.
    document_format = DOCUMENT_FORMATS.get(mime)
    if document_format is not None:
        return {
            'document': {
                'format': document_format,
                'name': 'document',
                'source': {'bytes': payload},
            }
        }
    image_format = IMAGE_FORMATS.get(mime)
    if image_format is not None:
        return {'image': {'format': image_format, 'source': {'bytes': payload}}}
    raise GenkitError(
        message=(
            f'bedrock: unsupported media MIME type {mime!r} '
            '(must be png/jpeg/gif/webp or one of pdf/csv/doc/docx/xls/xlsx/html/txt/md)'
        ),
        status='INVALID_ARGUMENT',
    )


def _tool_response_text(output: Any) -> str:  # noqa: ANN401
    if output is None:
        return ''
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output)
    except (TypeError, ValueError) as e:
        raise GenkitError(message=f'bedrock: marshal tool response: {e}', status='INVALID_ARGUMENT') from e


def _reasoning_part_to_blocks(root: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Converts a reasoning part back to Converse reasoningContent blocks.

    Only Bedrock-originated reasoning (carrying the signature and/or redacted
    metadata) is emitted; a generic reasoning part produces no blocks so it
    cannot corrupt the follow-up request.
    """
    metadata = getattr(root, 'metadata', None)
    blocks: list[dict[str, Any]] = []
    redacted = _metadata_bytes(metadata, REDACTED_CONTENT_METADATA_KEY)
    if redacted:
        blocks.append({'reasoningContent': {'redactedContent': redacted}})
    signature = _metadata_str(metadata, REASONING_SIGNATURE_METADATA_KEY)
    text = getattr(root, 'reasoning', None) or ''
    if text and signature:
        blocks.append({'reasoningContent': {'reasoningText': {'text': text, 'signature': signature}}})
    return blocks


def _tool_use_id(ref: str | None, label: str) -> str:
    """Bedrock's toolUseId rejects empty strings, so a missing ref cannot be sent."""
    if not ref:
        raise GenkitError(
            message=f'bedrock: {label} requires a ref to send as toolUseId',
            status='INVALID_ARGUMENT',
        )
    return ref


def _part_to_blocks(part: Part | Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Converts one Genkit part to Converse content blocks.

    Unknown part kinds are silently dropped so a foreign part cannot fail a
    request; the response side fails loud instead.
    """
    root = part.root if isinstance(part, Part) else part
    if getattr(root, 'media', None) is not None:
        return [media_to_block(root)]
    tool_request = getattr(root, 'tool_request', None)
    if tool_request is not None:
        return [
            {
                'toolUse': {
                    'toolUseId': _tool_use_id(tool_request.ref, 'tool request'),
                    'name': tool_request.name,
                    'input': tool_request.input,
                }
            }
        ]
    tool_response = getattr(root, 'tool_response', None)
    if tool_response is not None:
        return [
            {
                'toolResult': {
                    'toolUseId': _tool_use_id(tool_response.ref, 'tool response'),
                    'content': [{'text': _tool_response_text(tool_response.output)}],
                    'status': 'success',
                }
            }
        ]
    cache_type = _cache_point_type(root)
    if cache_type is not None:
        return [{'cachePoint': {'type': cache_type}}]
    # `is not None`: a redacted-only reasoning part has reasoning == ''.
    if getattr(root, 'reasoning', None) is not None:
        return _reasoning_part_to_blocks(root)
    if getattr(root, 'text', None) is not None:
        return [{'text': root.text}]
    return []


def _system_blocks(message: Message | Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """System messages keep only text and cache points; the rest is dropped."""
    blocks: list[dict[str, Any]] = []
    for part in message.content or []:
        root = part.root if isinstance(part, Part) else part
        cache_type = _cache_point_type(root)
        if cache_type is not None:
            blocks.append({'cachePoint': {'type': cache_type}})
        # Truthiness, not `is not None`: Bedrock rejects empty system text.
        elif getattr(root, 'text', None):
            blocks.append({'text': root.text})
    return blocks


def convert_messages(
    messages: list[Message] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Splits Genkit messages into Converse messages and top-level system blocks.

    Genkit's SYSTEM role becomes the separate top-level ``system`` array, not
    a conversation message. Messages that convert to zero blocks are dropped
    entirely — Bedrock rejects empty content arrays.

    Returns:
        A ``(messages, system)`` tuple of Converse-shaped dicts.
    """
    converse_messages: list[dict[str, Any]] = []
    system: list[dict[str, Any]] = []
    for message in messages or []:
        if message is None:
            continue
        if message.role == Role.SYSTEM:
            system.extend(_system_blocks(message))
            continue
        # Validate the role before converting parts, so an unsupported role
        # errors even when its message converts to nothing.
        role = to_bedrock_role(message.role)
        blocks: list[dict[str, Any]] = []
        for part in message.content or []:
            if part is None:
                continue
            blocks.extend(_part_to_blocks(part))
        if not blocks:
            continue
        converse_messages.append({'role': role, 'content': blocks})
    return converse_messages, system


def _normalize_tool_schema(schema: Any) -> dict[str, Any]:  # noqa: ANN401
    """Normalizes a tool input schema for ``toolSpec.inputSchema.json``.

    Bedrock requires an input schema; a missing one becomes the empty object
    schema, and ``type``/``properties``/``$schema`` are filled in when absent.
    """
    if schema is None:
        normalized: dict[str, Any] = {'type': 'object', 'properties': {}}
    elif isinstance(schema, dict):
        normalized = dict(schema)
    elif isinstance(schema, (str, bytes)):
        try:
            parsed = json.loads(schema)
        except (ValueError, TypeError):
            return {'type': 'object', 'properties': {}}
        normalized = parsed if isinstance(parsed, dict) else {'type': 'object', 'properties': {}}
    else:
        return {'type': 'object', 'properties': {}}
    normalized.setdefault('type', 'object')
    if normalized.get('type') == 'object':
        normalized.setdefault('properties', {})
    normalized.setdefault('$schema', 'http://json-schema.org/draft-07/schema#')
    return normalized


def to_bedrock_tool(tool: ToolDefinition | None) -> dict[str, Any]:  # noqa: ANN401
    """Converts a Genkit tool definition to a Converse toolSpec."""
    if tool is None:
        raise GenkitError(message='bedrock: tool definition required', status='INVALID_ARGUMENT')
    if not tool.name:
        raise GenkitError(message='bedrock: tool name required', status='INVALID_ARGUMENT')
    tool_spec: dict[str, Any] = {
        'name': tool.name,
        'inputSchema': {'json': _normalize_tool_schema(tool.input_schema)},
    }
    if tool.description:
        # Omitted when empty: description is optional, but Bedrock rejects ''.
        tool_spec['description'] = tool.description
    return {'toolSpec': tool_spec}


def _to_bedrock_tool_choice(tool_choice: str, tools: list[ToolDefinition]) -> dict[str, Any]:  # noqa: ANN401
    if tool_choice in ('', 'auto'):
        return {'auto': {}}
    if tool_choice in ('required', 'any'):
        return {'any': {}}
    for tool in tools:
        if tool.name == tool_choice:
            return {'tool': {'name': tool_choice}}
    raise GenkitError(
        message=f'bedrock: tool_choice {tool_choice!r} does not match any declared tool',
        status='INVALID_ARGUMENT',
    )


def build_converse_request(model_id: str, request: ModelRequest[Any]) -> dict[str, Any]:  # noqa: ANN401
    """Builds the keyword arguments for ``client.converse(...)``.

    The model ID is sent verbatim — inference-profile prefixes and ARNs are
    preserved; only capability lookup ever strips them.

    Args:
        model_id: Bedrock model ID, inference-profile ID, or ARN.
        request: The Genkit model request.

    Returns:
        Keyword arguments for the Converse call.
    """
    if request is None:
        raise GenkitError(message='bedrock: model request is missing', status='INVALID_ARGUMENT')
    config = normalize_config(request.config)
    messages, system = convert_messages(request.messages)

    tools = request.tools or []
    tool_choice = _requested_tool_choice(request, config) if tools else ''
    # "none" means omit toolConfig entirely — Bedrock has no none mode.
    send_tool_config = bool(tools) and tool_choice != 'none'
    # Bedrock rejects a trailing assistant message only when a toolConfig is
    # sent. Refusing here, with the constraint named, beats both the opaque
    # ValidationException and silently dropping the caller's prefill.
    if send_tool_config and messages and messages[-1]['role'] == 'assistant':
        raise GenkitError(
            message=(
                'bedrock: the Converse API rejects a conversation ending with an assistant message when tools '
                "are configured; end with a user or tool message, or set tool_choice='none'"
            ),
            status='INVALID_ARGUMENT',
        )

    kwargs: dict[str, Any] = {'modelId': model_id, 'messages': messages}
    if system:
        kwargs['system'] = system

    inference_config = build_inference_config(config)
    if inference_config:
        kwargs['inferenceConfig'] = inference_config

    if config is not None and config.additional_model_request_fields:
        # Forwarded verbatim (e.g. Claude extended thinking budgets).
        kwargs['additionalModelRequestFields'] = config.additional_model_request_fields

    if send_tool_config:
        tool_config: dict[str, Any] = {'tools': [to_bedrock_tool(tool) for tool in tools]}
        if tool_choice:
            tool_config['toolChoice'] = _to_bedrock_tool_choice(tool_choice, tools)
        kwargs['toolConfig'] = tool_config
    return kwargs


def _requested_tool_choice(request: ModelRequest[Any], config: BedrockConfig | None) -> str:
    # The Bedrock-specific config wins over the core request field so callers
    # can name a specific tool, which the core enum cannot express.
    if config is not None and config.tool_choice:
        return config.tool_choice
    if request.tool_choice:
        return str(request.tool_choice)
    return ''


def _coerce_value(value: Any, schema: Any) -> Any:  # noqa: ANN401
    """Coerces a tool-input value toward its declared schema type.

    Models occasionally return numbers or booleans as strings and floats for
    integers; coercing them keeps tool dispatch alive.
    """
    if not isinstance(schema, dict):
        return value
    schema_type = schema.get('type')
    if schema_type in ('number', 'integer') and isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if schema_type == 'integer' else number
    if schema_type == 'integer' and isinstance(value, float):
        return int(value)
    if schema_type == 'number' and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if schema_type == 'string' and isinstance(value, (int, float)) and not isinstance(value, bool):
        # A raw int against a string field fails pydantic validation.
        return str(value)
    if schema_type == 'boolean' and isinstance(value, str):
        # Anything outside this vocabulary is left alone for the tool to reject.
        lowered = value.strip().lower()
        if lowered in ('true', 't', '1'):
            return True
        if lowered in ('false', 'f', '0'):
            return False
        return value
    if schema_type == 'array' and isinstance(value, list):
        return [_coerce_value(item, schema.get('items')) for item in value]
    if schema_type == 'object' and isinstance(value, dict):
        return _coerce_map(value, schema)
    return value


def _coerce_map(value: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    # Only object schemas coerce; non-object schemas pass through.
    if schema.get('type') != 'object':
        return value
    properties = schema.get('properties')
    if not isinstance(properties, dict):
        return value
    # Keys without a schema are kept as-is.
    return {key: _coerce_value(item, properties.get(key)) for key, item in value.items()}


def coerce_tool_input(name: str, value: Any, tools: list[ToolDefinition] | None) -> Any:  # noqa: ANN401
    """Coerces a decoded tool input toward the named tool's declared schema."""
    if not isinstance(value, dict):
        return value
    for tool in tools or []:
        if tool is not None and tool.name == name and isinstance(tool.input_schema, dict):
            return _coerce_map(value, tool.input_schema)
    return value


def _reasoning_block_to_part(block: dict[str, Any]) -> Part | None:  # noqa: ANN401
    reasoning_text = block.get('reasoningText')
    if reasoning_text is not None:
        # A ReasoningTextBlock structure on the wire, so botocore parses it to a dict.
        text = reasoning_text.get('text') or ''
        signature = reasoning_text.get('signature')
        if not text and not signature:
            return None
        return bedrock_reasoning_part(text, signature, None)
    redacted = block.get('redactedContent')
    if redacted is not None:
        if not redacted:
            return None
        return bedrock_reasoning_part('', None, redacted)
    raise GenkitError(
        message=f'bedrock: unhandled reasoning content variant {sorted(block.keys())!r}',
        status='INTERNAL',
    )


def bedrock_reasoning_part(text: str, signature: str | None, redacted: bytes | None) -> Part:
    """Builds a reasoning part carrying the Bedrock replay metadata.

    Args:
        text: Reasoning text; empty for a redacted-only part.
        signature: Signature string, replayed verbatim (it is a string on the
            Converse wire, never bytes).
        redacted: Redacted reasoning blob, stored base64-encoded.

    Returns:
        A Part wrapping a ReasoningPart.
    """
    metadata: dict[str, Any] = {}
    if signature:
        # Also stored under the generic key so framework-level consumers see it.
        metadata['signature'] = signature
        metadata[REASONING_SIGNATURE_METADATA_KEY] = signature
    if redacted:
        # Base64 string, not raw bytes: part metadata must stay JSON-serializable.
        metadata[REDACTED_CONTENT_METADATA_KEY] = base64.b64encode(redacted).decode('ascii')
    return Part(root=ReasoningPart(reasoning=text, metadata=metadata or None))


def content_blocks_to_parts(
    blocks: list[dict[str, Any]],
    tools: list[ToolDefinition] | None = None,
) -> list[Part]:
    """Converts Converse response content blocks to Genkit parts.

    Unlike the request side, unknown response blocks fail loud: silently
    dropping model output would corrupt conversations.
    """
    parts: list[Part] = []
    for block in blocks:
        if 'text' in block:
            parts.append(Part(root=TextPart(text=block['text'])))
        elif 'toolUse' in block:
            tool_use = block['toolUse']
            tool_input = tool_use.get('input')
            if tool_input is None:
                tool_input = {}
            parts.append(
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(
                            ref=tool_use.get('toolUseId'),
                            name=tool_use.get('name') or '',
                            input=coerce_tool_input(tool_use.get('name') or '', tool_input, tools),
                        )
                    )
                )
            )
        elif 'reasoningContent' in block:
            part = _reasoning_block_to_part(block['reasoningContent'])
            if part is not None:
                parts.append(part)
        else:
            raise GenkitError(
                message=f'bedrock: unhandled response content variant {sorted(block.keys())!r}',
                status='INTERNAL',
            )
    return parts


def map_finish_reason(stop_reason: str | None) -> FinishReason:
    """Maps a Bedrock stopReason to a Genkit finish reason (unknown → OTHER)."""
    return STOP_REASON_MAP.get(stop_reason or '', FinishReason.OTHER)


def usage_from_response(usage: dict[str, Any] | None) -> ModelUsage | None:  # noqa: ANN401
    """Maps Converse token usage; totals are trusted verbatim from AWS.

    Only ``cacheReadInputTokens`` surfaces (as ``cached_content_tokens``);
    Genkit has no field for cache-write tokens.
    """
    if usage is None:
        return None
    return ModelUsage(
        input_tokens=usage.get('inputTokens'),
        output_tokens=usage.get('outputTokens'),
        total_tokens=usage.get('totalTokens'),
        cached_content_tokens=usage.get('cacheReadInputTokens'),
    )


def usage_log_fields(usage: dict[str, Any] | None) -> dict[str, Any]:  # noqa: ANN401
    """Token counts as structlog keys; empty when the response reported none.

    Cache writes are here but not on ``ModelUsage``, so the log is the only
    place they surface.
    """
    if not usage:
        return {}
    return {
        'input_tokens': usage.get('inputTokens'),
        'output_tokens': usage.get('outputTokens'),
        'cache_read_tokens': usage.get('cacheReadInputTokens'),
        'cache_write_tokens': usage.get('cacheWriteInputTokens'),
    }


def to_model_response(response: dict[str, Any] | None, request: ModelRequest[Any]) -> ModelResponse:  # noqa: ANN401
    """Converts a raw Converse response to a Genkit ModelResponse."""
    if response is None:
        raise GenkitError(message='bedrock: converse response is missing', status='INTERNAL')
    output = response.get('output') or {}
    message = output.get('message')
    if output and message is None:
        raise GenkitError(
            message=f'bedrock: unexpected output variant {sorted(output.keys())!r}',
            status='INTERNAL',
        )
    parts = content_blocks_to_parts((message or {}).get('content') or [], request.tools)
    if not parts:
        # Guardrail-blocked responses have no content; return a well-formed
        # empty message rather than erroring.
        parts = [Part(root=TextPart(text=''))]
    return ModelResponse(
        message=Message(role=Role.MODEL, content=parts),
        finish_reason=map_finish_reason(response.get('stopReason')),
        usage=usage_from_response(response.get('usage')),
        request=request,
    )
