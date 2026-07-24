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

"""Converters between Genkit message parts and Interactions API wire shapes."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from pydantic import BaseModel

from genkit import CustomPart, Media, MediaPart, Part, ReasoningPart, TextPart, ToolRequestPart, ToolResponsePart
from genkit.model import Error, FinishReason, Message, ModelResponse, ModelUsage, Operation, ToolDefinition
from genkit_google_genai._interactions.options import ClientOptions

logger = logging.getLogger(__name__)

# Wire payloads are plain dicts (SDK accepts them on create; responses are normalized
# to dicts so converter tests can stay fixture-driven).
Content = dict[str, Any]
Step = dict[str, Any]
InteractionTool = dict[str, Any]
InteractionDict = dict[str, Any]


def as_interaction_dict(interaction: BaseModel | InteractionDict) -> InteractionDict:
    """Normalize an SDK Interaction or dict to a plain interaction dict."""
    if isinstance(interaction, BaseModel):
        return interaction.model_dump(mode='python')
    return dict(interaction)


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip schema keys the Interactions API rejects."""
    out = copy.deepcopy(schema)
    for key in list(out):
        if key in ('$schema', 'additionalProperties'):
            del out[key]
            continue
        value = out[key]
        if isinstance(value, dict):
            out[key] = clean_schema(value)
        elif key == 'type' and isinstance(value, list):
            out[key] = next((item for item in value if item != 'null'), value[0] if value else value)
    return out


def ensure_tool_ids(messages: list[Message]) -> list[Message]:
    """Assign stable tool call IDs so wire payloads stay pairable."""
    generated_ids: list[str] = []
    next_id_counter = 0

    new_messages = [message.model_copy(deep=True) for message in messages]

    for message in new_messages:
        for part in message.content:
            root = part.root
            if isinstance(root, ToolRequestPart) and root.tool_request and not root.tool_request.ref:
                new_id = f'genkit-auto-id-{next_id_counter}'
                next_id_counter += 1
                root.tool_request.ref = new_id
                generated_ids.append(new_id)

    # Responses without refs reuse request IDs in order; unmatched ones get orphan IDs.
    for message in new_messages:
        for part in message.content:
            root = part.root
            if isinstance(root, ToolResponsePart) and root.tool_response and not root.tool_response.ref:
                matched_id = generated_ids.pop(0) if generated_ids else None
                if matched_id:
                    root.tool_response.ref = matched_id
                else:
                    root.tool_response.ref = f'genkit-orphan-id-{next_id_counter}'
                    next_id_counter += 1

    return new_messages


def to_interaction_tool(tool: ToolDefinition) -> InteractionTool:
    """Convert a Genkit tool definition to an Interactions function tool."""
    func: InteractionTool = {
        'type': 'function',
        'name': tool.name,
        'description': tool.description,
    }
    if tool.input_schema is not None:
        if isinstance(tool.input_schema, dict):
            func['parameters'] = clean_schema(tool.input_schema)
        else:
            func['parameters'] = clean_schema(dict(tool.input_schema))
    return func


def to_interaction_content(part: Part) -> Content | None:
    """Convert a Genkit part to an Interactions content block."""
    root = part.root
    if isinstance(root, TextPart):
        return {'type': 'text', 'text': root.text}
    if isinstance(root, MediaPart) and root.media is not None:
        return to_interaction_media(root)
    logger.warning('Unsupported part type for Interaction input: %s', part.model_dump(by_alias=True))
    return None


def to_interaction_media(part: MediaPart) -> Content:
    """Convert a media part to an Interactions image/audio/video/document block."""
    if part.media is None:
        raise ValueError('Media part missing media')
    url = part.media.url
    content_type = part.media.content_type
    if not content_type:
        raise ValueError('Media part missing contentType')

    data: str | None = None
    uri: str | None = None
    if url.startswith('data:'):
        data = url[url.index(',') + 1 :]
    else:
        uri = url

    out: dict[str, Any] = {'mime_type': content_type}
    if data is not None:
        out['data'] = data
    if uri is not None:
        out['uri'] = uri

    if content_type.startswith('image/'):
        out['type'] = 'image'
        return out
    if content_type.startswith('audio/'):
        out['type'] = 'audio'
        return out
    if content_type.startswith('video/'):
        out['type'] = 'video'
        return out
    if content_type == 'application/pdf':
        out['type'] = 'document'
        return out

    raise ValueError(f'Unsupported media type: {content_type}')


def to_interaction_role(role: str) -> str:
    """Map a Genkit message role to the Interactions API role."""
    if role == 'user':
        return 'user'
    if role == 'model':
        return 'model'
    if role == 'tool':
        return 'user'
    if role == 'system':
        raise ValueError('System role should be handled as system_instruction, not part of turns.')
    return 'user'


def to_interaction_steps(messages: list[Message]) -> list[Step]:
    """Convert Genkit messages to Interactions API steps."""
    steps: list[Step] = []

    for message in messages:
        normal_content: list[Content] = []
        for part in message.content:
            root = part.root
            if isinstance(root, ToolRequestPart) and root.tool_request:
                tool_request = root.tool_request
                steps.append({
                    'type': 'function_call',
                    'name': tool_request.name,
                    'arguments': tool_request.input if isinstance(tool_request.input, dict) else {},
                    'id': tool_request.ref or '',
                })
            elif isinstance(root, ToolResponsePart) and root.tool_response:
                tool_response = root.tool_response
                output = tool_response.output
                if not isinstance(output, (dict, str)) and output is not None:
                    output = {'result': output}
                steps.append({
                    'type': 'function_result',
                    'name': tool_response.name,
                    'result': output,
                    'call_id': tool_response.ref or '',
                })
            elif isinstance(root, CustomPart):
                custom = root.custom or {}
                metadata = root.metadata or {}
                if 'googleSearchCall' in custom:
                    gs_call = custom['googleSearchCall']
                    steps.append({
                        'type': 'google_search_call',
                        'id': gs_call['id'],
                        'arguments': gs_call['arguments'],
                        'signature': metadata.get('thoughtSignature'),
                    })
                elif 'googleSearchResult' in custom:
                    gs_result = custom['googleSearchResult']
                    steps.append({
                        'type': 'google_search_result',
                        'call_id': gs_result['callId'],
                        'result': gs_result['result'],
                        'signature': metadata.get('thoughtSignature'),
                    })
                elif 'executableCode' in custom:
                    exec_code = custom['executableCode']
                    steps.append({
                        'type': 'code_execution_call',
                        'id': metadata['callId'],
                        'arguments': {
                            'code': exec_code['code'],
                            'language': exec_code.get('language', 'PYTHON'),
                        },
                        'signature': metadata.get('thoughtSignature'),
                    })
                elif 'codeExecutionResult' in custom:
                    exec_result = custom['codeExecutionResult']
                    steps.append({
                        'type': 'code_execution_result',
                        'call_id': metadata['callId'],
                        'result': exec_result['output'],
                        'signature': metadata.get('thoughtSignature'),
                    })
                else:
                    content = to_interaction_content(part)
                    if content is not None:
                        normal_content.append(content)
            elif isinstance(root, ReasoningPart):
                metadata = root.metadata or {}
                steps.append({
                    'type': 'thought',
                    'summary': [{'type': 'text', 'text': root.reasoning}],
                    'signature': metadata.get('thoughtSignature'),
                })
            else:
                content = to_interaction_content(part)
                if content is not None:
                    normal_content.append(content)

        if normal_content:
            if message.role == 'model':
                steps.append({'type': 'model_output', 'content': normal_content})
            else:
                steps.append({'type': 'user_input', 'content': normal_content})

    return steps


def from_interaction_content(content: Content) -> Part:
    """Convert an Interactions content block back to a Genkit part."""
    content_type = content.get('type')
    if content_type == 'text':
        return from_text_content(content)
    if content_type == 'image':
        return from_image_content(content)
    if content_type in ('audio', 'document'):
        return Part(root=from_media_content(content))
    if content_type == 'video':
        return from_video_content(content)
    if content_type == 'thought':
        return from_thought_content(content)
    if content_type == 'function_call':
        return from_function_call_content(content)
    if content_type == 'function_result':
        return from_function_result_content(content)
    return Part(root=CustomPart(custom={'unknownContent': content}))


def _maybe_add_gemini_thought_signature(step: Step, part: Part) -> Part:
    signature = step.get('signature') if isinstance(step, dict) else None
    if signature:
        root = part.root
        metadata = dict(root.metadata or {})
        metadata['thoughtSignature'] = signature
        updated = root.model_copy(update={'metadata': metadata})
        return Part(root=updated)
    return part


def from_google_search_call(step: Step) -> Part:
    """Convert a google_search_call step to a Genkit custom part."""
    part = Part(
        root=CustomPart(
            custom={
                'googleSearchCall': {
                    'id': step['id'],
                    'arguments': step['arguments'],
                }
            }
        )
    )
    return _maybe_add_gemini_thought_signature(step, part)


def from_google_search_result(step: Step) -> Part:
    """Convert a google_search_result step to a Genkit custom part."""
    part = Part(
        root=CustomPart(
            custom={
                'googleSearchResult': {
                    'callId': step['call_id'],
                    'result': step['result'],
                }
            }
        )
    )
    return _maybe_add_gemini_thought_signature(step, part)


def from_code_execution_call(step: Step) -> Part:
    """Convert a code_execution_call step to a Genkit custom part."""
    arguments = step['arguments']
    part = Part(
        root=CustomPart(
            custom={
                'executableCode': {
                    'code': arguments['code'],
                    'language': arguments.get('language', 'PYTHON'),
                }
            },
            metadata={'callId': step['id']},
        )
    )
    return _maybe_add_gemini_thought_signature(step, part)


def from_code_execution_result(step: Step) -> Part:
    """Convert a code_execution_result step to a Genkit custom part."""
    result = step['result']
    part = Part(
        root=CustomPart(
            custom={
                'codeExecutionResult': {
                    'output': result if isinstance(result, str) else json.dumps(result),
                    'outcome': 'OUTCOME_OK',
                }
            },
            metadata={'callId': step['call_id']},
        )
    )
    return _maybe_add_gemini_thought_signature(step, part)


def from_server_function_call(step: Step) -> Part:
    """Convert a standalone function_call step to a Genkit custom part."""
    return Part(
        root=CustomPart(
            custom={
                'serverFunctionCall': {
                    'id': step['id'],
                    'name': step['name'],
                    'arguments': step.get('arguments'),
                }
            }
        )
    )


def from_server_function_result(step: Step) -> Part:
    """Convert a standalone function_result step to a Genkit custom part."""
    return Part(
        root=CustomPart(
            custom={
                'serverFunctionResult': {
                    'callId': step['call_id'],
                    'name': step['name'],
                    'result': step.get('result'),
                    'isError': step.get('is_error'),
                }
            }
        )
    )


def from_interaction_step(step: Step) -> list[Part]:
    """Convert an Interactions step to Genkit parts."""
    step_type = step.get('type')
    if step_type == 'model_output':
        return [from_interaction_content(content) for content in step['content']]
    if step_type == 'user_input':
        # The API echoes our prompt back; including it would duplicate the input.
        return []
    if step_type == 'google_search_call':
        return [from_google_search_call(step)]
    if step_type == 'google_search_result':
        return [from_google_search_result(step)]
    if step_type == 'code_execution_call':
        return [from_code_execution_call(step)]
    if step_type == 'code_execution_result':
        return [from_code_execution_result(step)]
    if step_type == 'thought':
        return [from_thought_content(step)]
    if step_type == 'function_call':
        return [from_server_function_call(step)]
    if step_type == 'function_result':
        return [from_server_function_result(step)]
    return [Part(root=CustomPart(custom={'unknownStep': step}))]


def from_media_content(content: Content) -> MediaPart:
    """Convert wire media content to a Genkit media part."""
    url = content.get('uri')
    if content.get('data') and content.get('mime_type'):
        url = f'data:{content["mime_type"]};base64,{content["data"]}'
    return MediaPart(media=Media(url=url or '', content_type=content.get('mime_type')))


def from_text_content(content: Content) -> Part:
    """Convert wire text content to a Genkit text part."""
    # Empty annotations still show up in metadata so round-trips stay stable.
    return Part(
        root=TextPart(
            text=content.get('text') or '',
            metadata={'annotations': content.get('annotations')},
        )
    )


def from_image_content(content: Content) -> Part:
    """Convert wire image content to a Genkit media part."""
    part = Part(root=from_media_content(content))
    if content.get('resolution') is not None:
        root = part.root
        metadata = dict(root.metadata or {})
        metadata['resolution'] = content['resolution']
        part = Part(root=root.model_copy(update={'metadata': metadata}))
    return part


def from_video_content(content: Content) -> Part:
    """Convert wire video content to a Genkit media part."""
    part = Part(root=from_media_content(content))
    if content.get('resolution') is not None:
        root = part.root
        metadata = dict(root.metadata or {})
        metadata['resolution'] = content['resolution']
        part = Part(root=root.model_copy(update={'metadata': metadata}))
    return part


def from_thought_content(content: Content) -> Part:
    """Convert wire thought content to a Genkit reasoning part."""
    reasoning = ''
    summary = content.get('summary')
    if summary:
        chunks: list[str] = []
        for item in summary:
            if item.get('type') == 'text':
                chunks.append(str(item.get('text') or ''))
            else:
                chunks.append('[Image]')
        reasoning = '\n'.join(chunks)
    return Part(
        root=ReasoningPart(
            reasoning=reasoning,
            metadata={'thoughtSignature': content.get('signature')},
            custom={'thought': content},
        )
    )


def from_function_call_content(content: Content) -> Part:
    """Convert wire function_call content to a Genkit tool request part."""
    from genkit import ToolRequest

    return Part(
        root=ToolRequestPart(
            tool_request=ToolRequest(
                name=content['name'],
                input=content.get('arguments'),
                ref=content['id'],
            )
        )
    )


def from_function_result_content(content: Content) -> Part:
    """Convert wire function_result content to a Genkit tool response part."""
    from genkit import ToolResponse

    return Part(
        root=ToolResponsePart(
            tool_response=ToolResponse(
                name=content['name'],
                output=content.get('result'),
                ref=content['call_id'],
            )
        )
    )


def _interaction_message_metadata(interaction: InteractionDict) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    if interaction.get('id'):
        metadata['interactionId'] = interaction['id']
    if interaction.get('environment_id'):
        metadata['environmentId'] = interaction['environment_id']
    if interaction.get('status'):
        # Preserve the wire status when finish_reason is a coarser Genkit enum.
        metadata['interactionStatus'] = interaction['status']
    return metadata or None


def _usage_from_interaction(usage: dict[str, Any]) -> ModelUsage:
    response_usage = ModelUsage(
        input_tokens=usage.get('total_input_tokens'),
        output_tokens=usage.get('total_output_tokens'),
        total_tokens=usage.get('total_tokens'),
        cached_content_tokens=usage.get('total_cached_tokens'),
        thoughts_tokens=usage.get('total_thought_tokens'),
    )
    for modality_token in usage.get('input_tokens_by_modality') or []:
        match modality_token.get('modality'):
            case 'text':
                response_usage.input_characters = modality_token.get('tokens')
            case 'image':
                response_usage.input_images = modality_token.get('tokens')
            case 'audio':
                response_usage.input_audio_files = modality_token.get('tokens')
            case _:
                pass
    for modality_token in usage.get('output_tokens_by_modality') or []:
        match modality_token.get('modality'):
            case 'text':
                response_usage.output_characters = modality_token.get('tokens')
            case 'image':
                response_usage.output_images = modality_token.get('tokens')
            case 'audio':
                response_usage.output_audio_files = modality_token.get('tokens')
            case _:
                pass
    return response_usage


def _parts_from_steps(steps: list[Step]) -> list[Part]:
    return [
        part
        for part in (item for step in steps for item in from_interaction_step(step))
        if part.model_dump(exclude_none=True)
    ]


def _cancelled_response(interaction: InteractionDict) -> ModelResponse:
    message_metadata = _interaction_message_metadata(interaction)
    message = Message(
        role='model',
        content=[Part(root=TextPart(text='Operation cancelled.'))],
        metadata=message_metadata,
    )
    return ModelResponse.model_construct(
        finish_reason=FinishReason.ABORTED,
        finish_message='Operation cancelled',
        message=message,
    )


def _steps_response(
    interaction: InteractionDict,
    *,
    finish_reason: FinishReason,
    finish_message: str | None = None,
) -> ModelResponse | None:
    """Build a ModelResponse from interaction steps, or None when there are none."""
    steps = interaction.get('steps') or []
    if not steps:
        return None
    content = _parts_from_steps(steps)
    message_metadata = _interaction_message_metadata(interaction)
    response = ModelResponse.model_construct(
        finish_reason=finish_reason,
        finish_message=finish_message,
        message=Message(role='model', content=content, metadata=message_metadata),
        custom=dict(interaction),
        raw=dict(interaction),
    )
    if interaction.get('usage'):
        response.usage = _usage_from_interaction(interaction['usage'])
    return response


def _completed_response(interaction: InteractionDict) -> ModelResponse | None:
    return _steps_response(interaction, finish_reason=FinishReason.STOP)


def from_interaction_sync(interaction: BaseModel | InteractionDict) -> ModelResponse:
    """Convert a completed interaction to a synchronous model response."""
    payload = as_interaction_dict(interaction)
    if payload.get('status') == 'failed':
        raise ValueError('Interaction failed')

    message_metadata = _interaction_message_metadata(payload)
    response = ModelResponse.model_construct(
        finish_reason=FinishReason.STOP,
        message=Message(role='model', content=[], metadata=message_metadata),
        custom=dict(payload),
        raw=dict(payload),
    )

    if payload.get('status') == 'cancelled':
        response.finish_reason = FinishReason.ABORTED
        response.finish_message = 'Operation cancelled'
        response.message = Message(
            role='model',
            content=[Part(root=TextPart(text='Operation cancelled.'))],
            metadata=message_metadata,
        )
        return response

    steps = payload.get('steps')
    if steps:
        response.message = Message(
            role='model',
            content=_parts_from_steps(steps),
            metadata=message_metadata,
        )
        if payload.get('usage'):
            response.usage = _usage_from_interaction(payload['usage'])
    return response


def from_interaction(
    interaction: BaseModel | InteractionDict,
    client_options: ClientOptions | None = None,
) -> Operation:
    """Convert an interaction poll result to a Genkit operation."""
    payload = as_interaction_dict(interaction)
    op = Operation.model_construct(id=payload.get('id') or '')
    if client_options:
        op.metadata = {'clientOptions': client_options}

    status = payload.get('status')
    if status in ('in_progress', 'queued'):
        # Keep polling — still running or waiting on the server.
        op.done = False
    elif status == 'cancelled':
        op.done = True
        op.output = _cancelled_response(payload)
    elif status == 'completed':
        op.done = True
        op.output = _completed_response(payload)
    elif status == 'incomplete':
        # Terminal truncated result (e.g. max_tokens). Surface steps when present.
        op.done = True
        op.output = _steps_response(
            payload,
            finish_reason=FinishReason.LENGTH,
            finish_message='Interaction incomplete (truncated output)',
        )
        if op.output is None:
            op.error = Error(message='Interaction incomplete (truncated output)')
    elif status == 'budget_exceeded':
        # Terminal halt on budget. Prefer partial steps over a bare error.
        op.done = True
        op.output = _steps_response(
            payload,
            finish_reason=FinishReason.ABORTED,
            finish_message='Interaction exceeded its budget',
        )
        if op.output is None:
            op.error = Error(message='Interaction exceeded its budget')
    elif status == 'failed':
        # Always exit the poll loop on failure; leaving done unset hangs forever.
        op.done = True
        error_payload = payload.get('error') or {}
        message = error_payload.get('message') if isinstance(error_payload, dict) else None
        op.error = Error(message=message or 'Interaction failed')
    # requires_action: leave done unset. Resuming that turn is a separate product
    # decision; we don't invent interrupt/resume here.
    return op
