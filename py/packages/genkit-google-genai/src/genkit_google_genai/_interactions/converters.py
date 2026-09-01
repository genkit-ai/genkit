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
from typing import Any, Literal, cast

from genkit_google_genai._interactions.options import ClientOptions
from google.genai.interactions import (
    AudioContent,
    CodeExecutionCallStep,
    CodeExecutionCallStepParam,
    CodeExecutionResultStep,
    CodeExecutionResultStepParam,
    Content,
    ContentParam,
    DocumentContent,
    FunctionCallStep,
    FunctionCallStepParam,
    FunctionParam,
    FunctionResultStep,
    FunctionResultStepParam,
    FunctionResultStepResultUnionParam,
    GoogleSearchCallStep,
    GoogleSearchCallStepParam,
    GoogleSearchResultStep,
    GoogleSearchResultStepParam,
    ImageContent,
    Interaction,
    ModelOutputStep,
    ModelOutputStepParam,
    Step,
    StepParam,
    TextContent,
    TextContentParam,
    ThoughtStep,
    ThoughtStepParam,
    UnknownContent,
    Usage,
    UserInputStep,
    UserInputStepParam,
    VideoContent,
)
from pydantic import BaseModel

from genkit import (
    CustomPart,
    GenkitError,
    Media,
    MediaPart,
    Part,
    ReasoningPart,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)
from genkit.model import Error, FinishReason, Message, ModelResponse, ModelUsage, Operation, ToolDefinition

logger = logging.getLogger(__name__)

InteractionRole = Literal['user', 'model']

# Keys we round-trip through Genkit custom parts and metadata for wire concepts
# Genkit has no first-class part for. Both directions read them, so a typo in
# one place would silently break pairing — keep them defined once.
THOUGHT_SIGNATURE = 'thoughtSignature'
CALL_ID = 'callId'
GOOGLE_SEARCH_CALL = 'googleSearchCall'
GOOGLE_SEARCH_RESULT = 'googleSearchResult'
EXECUTABLE_CODE = 'executableCode'
CODE_EXECUTION_RESULT = 'codeExecutionResult'
UNKNOWN_STEP = 'unknownStep'
SERVER_FUNCTION_RESULT = 'serverFunctionResult'
THOUGHT_CUSTOM = 'thought'
# Wire function_result.content is only text/image blocks, not a JSON array.
CONTENT_BLOCK_TYPES = frozenset({'text', 'image'})
# Code execution only ever runs Python, and the wire rejects any other value.
CODE_LANGUAGE: Literal['python'] = 'python'

FAILED_MESSAGE = 'Interaction failed'

# Interactions splits media into typed content blocks, chosen by mime prefix.
MEDIA_CONTENT_TYPES: tuple[tuple[str, str], ...] = (
    ('image/', 'image'),
    ('audio/', 'audio'),
    ('video/', 'video'),
    ('application/pdf', 'document'),
)

# Turns a Genkit role into the side of the transcript it belongs on. Tool
# results are part of the user's side; system prompts aren't turns at all.
INTERACTION_ROLES: dict[str, InteractionRole] = {
    'user': 'user',
    'model': 'model',
    'tool': 'user',
}

# Terminal statuses that can still carry partial output worth surfacing.
PARTIAL_TERMINAL_STATUSES: dict[str, tuple[FinishReason, str]] = {
    'incomplete': (FinishReason.LENGTH, 'Interaction incomplete (truncated output)'),
    'budget_exceeded': (FinishReason.ABORTED, 'Interaction exceeded its budget'),
}


def interaction_error_message(interaction: Interaction) -> str | None:
    """Read the failure message a failed Interaction reports on its output step."""
    for step in interaction.steps or []:
        if isinstance(step, ModelOutputStep) and step.error is not None and step.error.message:
            return step.error.message
    return None


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
        elif isinstance(value, list):
            if key == 'type':
                out[key] = next((item for item in value if item != 'null'), value[0] if value else value)
            else:
                out[key] = [clean_schema(item) if isinstance(item, dict) else item for item in value]
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


def to_interaction_tool(tool: ToolDefinition) -> FunctionParam:
    """Convert a Genkit tool definition to an Interactions function tool."""
    func: FunctionParam = {
        'type': 'function',
        'name': tool.name,
    }
    if tool.description is not None:
        func['description'] = tool.description
    if tool.input_schema is not None:
        if isinstance(tool.input_schema, dict):
            func['parameters'] = clean_schema(tool.input_schema)
        else:
            func['parameters'] = clean_schema(dict(tool.input_schema))
    return func


def to_interaction_content(part: Part) -> ContentParam | None:
    """Convert a Genkit part to an Interactions content block."""
    root = part.root
    if isinstance(root, TextPart):
        text: TextContentParam = {'type': 'text', 'text': root.text}
        return text
    if isinstance(root, MediaPart) and root.media is not None:
        return to_interaction_media(root)
    logger.warning('Unsupported part type for Interaction input: %s', part.model_dump(by_alias=True))
    return None


def to_interaction_media(part: MediaPart) -> ContentParam:
    """Convert a media part to an Interactions image/audio/video/document block."""
    if part.media is None:
        raise ValueError('Media part missing media')
    content_type = part.media.content_type
    if not content_type:
        raise ValueError('Media part missing contentType')
    block_type = next((name for prefix, name in MEDIA_CONTENT_TYPES if content_type.startswith(prefix)), None)
    if block_type is None:
        raise ValueError(f'Unsupported media type: {content_type}')

    block: dict[str, Any] = {'type': block_type, 'mime_type': content_type}
    # Inline data URLs travel as base64; anything else is a reference the API fetches.
    url = part.media.url
    if url.startswith('data:'):
        if ',' not in url:
            raise ValueError('Malformed data URL for media part: missing payload separator')
        block['data'] = url.split(',', 1)[1]
    else:
        block['uri'] = url
    return cast(ContentParam, block)


def to_interaction_role(role: str) -> InteractionRole:
    """Map a Genkit message role to the Interactions API role."""
    if role == 'system':
        raise ValueError('System role should be handled as system_instruction, not part of turns.')
    return INTERACTION_ROLES.get(role, 'user')


def split_system_instruction(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Lift system turns out of the transcript into the interaction's instruction.

    An interaction carries one system instruction that governs the whole thing
    rather than a turn in the tape, so however many system messages a prompt
    has, they fold into a single block of text ahead of the conversation.
    """
    instructions: list[str] = []
    turns: list[Message] = []
    for message in messages:
        if message.role != 'system':
            turns.append(message)
            continue
        if any(not isinstance(part.root, TextPart) for part in message.content):
            logger.warning('Dropping non-text content from a system message; system instructions are text only.')
        if message.text:
            instructions.append(message.text)
    return '\n\n'.join(instructions) or None, turns


def with_signature(step: StepParam, metadata: dict[str, Any]) -> StepParam:
    """Carry a thought signature from part metadata back onto its step."""
    signature = metadata.get(THOUGHT_SIGNATURE)
    if signature is not None:
        cast(dict[str, Any], step)['signature'] = signature
    return step


def to_function_call_step(request: ToolRequest) -> FunctionCallStepParam:
    """Compile a Genkit tool request into a function_call step.

    Arguments on the wire are a JSON object. A scalar like ``'Paris'`` would
    silently become ``{}`` and the model would see a different call than the
    one the caller made.
    """
    args = request.input
    if args is None:
        args = {}
    elif not isinstance(args, dict):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                'Tool request input must be a JSON object (got '
                f'{type(args).__name__}). Wrap scalars as '
                "{'city': 'Paris'} rather than passing a bare string."
            ),
        )
    return {
        'type': 'function_call',
        'name': request.name,
        'arguments': args,
        'id': request.ref or '',
    }


def is_content_block_list(value: object) -> bool:
    """True when value is a non-empty list of text/image content blocks."""
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        block = cast(dict[str, object], item)
        if block.get('type') not in CONTENT_BLOCK_TYPES:
            return False
    return True


def metadata_flag(metadata: dict[str, Any] | None, *keys: str) -> object:
    """Read the first present metadata key (camelCase or snake_case)."""
    if not metadata:
        return None
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def to_function_result_step(
    response: ToolResponse,
    metadata: dict[str, Any] | None = None,
) -> FunctionResultStepParam:
    """Compile a Genkit tool response into a function_result step.

    Ordinary JSON arrays (``[1, 2]``) are not multimodal content — they
    have to be boxed. Only a list of text/image blocks goes on the wire as
    content. A failed result keeps ``is_error`` so the model sees the miss.
    """
    output = response.output
    if isinstance(output, (str, dict)):
        result = cast(FunctionResultStepResultUnionParam, output)
    elif is_content_block_list(output):
        result = cast(FunctionResultStepResultUnionParam, output)
    elif output is None:
        result = cast(FunctionResultStepResultUnionParam, {})
    else:
        result = cast(FunctionResultStepResultUnionParam, {'result': output})
    step: FunctionResultStepParam = {
        'type': 'function_result',
        'name': response.name,
        'result': result,
        'call_id': response.ref or '',
    }
    if metadata_flag(metadata, 'isError', 'is_error'):
        step['is_error'] = True
    return step


def thought_from_custom(custom: dict[str, Any] | None) -> StepParam | None:
    """Replay a thought step that was stashed on the part, images and all.

    Rebuilding from ``reasoning`` would drop image summaries and the
    signature the next turn needs to continue the same thought.
    """
    if not custom:
        return None
    thought = custom.get(THOUGHT_CUSTOM)
    if isinstance(thought, dict) and thought.get('type') == 'thought':
        return cast(StepParam, copy.deepcopy(thought))
    return None


def to_thought_step(part: ReasoningPart) -> StepParam:
    """Compile a Genkit reasoning part into a thought step."""
    original = thought_from_custom(part.custom)
    if original is not None:
        return original
    thought: ThoughtStepParam = {
        'type': 'thought',
        'summary': [{'type': 'text', 'text': part.reasoning}],
    }
    return with_signature(thought, part.metadata or {})


def to_server_function_result_step(payload: dict[str, Any]) -> FunctionResultStepParam:
    """Replay a function_result that was stashed as custom, including is_error."""
    step: FunctionResultStepParam = {
        'type': 'function_result',
        'name': payload.get('name') or '',
        'result': cast(FunctionResultStepResultUnionParam, payload.get('result')),
        'call_id': payload.get('call_id') or payload.get('callId') or '',
    }
    if payload.get('is_error') or payload.get('isError'):
        step['is_error'] = True
    return step


def to_server_tool_step(custom: dict[str, Any], metadata: dict[str, Any]) -> StepParam | None:
    """Compile the custom parts that stand in for Google-side tool activity.

    Search and code execution run on Google's side, so Genkit has no first-class
    part for them and we round-trip them through custom parts instead.
    """
    if GOOGLE_SEARCH_CALL in custom:
        call = custom[GOOGLE_SEARCH_CALL]
        search_call: GoogleSearchCallStepParam = {
            'type': 'google_search_call',
            'id': call.get('id', ''),
            'arguments': call.get('arguments'),
        }
        return search_call
    if GOOGLE_SEARCH_RESULT in custom:
        result = custom[GOOGLE_SEARCH_RESULT]
        search_result: GoogleSearchResultStepParam = {
            'type': 'google_search_result',
            'call_id': result.get(CALL_ID, ''),
            'result': result.get('result'),
        }
        return search_result
    if EXECUTABLE_CODE in custom:
        code = custom[EXECUTABLE_CODE]
        code_call: CodeExecutionCallStepParam = {
            'type': 'code_execution_call',
            'id': metadata.get(CALL_ID, ''),
            'arguments': {'code': code.get('code'), 'language': CODE_LANGUAGE},
        }
        return code_call
    if CODE_EXECUTION_RESULT in custom:
        code_result: CodeExecutionResultStepParam = {
            'type': 'code_execution_result',
            'call_id': metadata.get(CALL_ID, ''),
            'result': custom[CODE_EXECUTION_RESULT].get('output'),
        }
        return code_result
    return None


def to_standalone_step(part: Part) -> StepParam | None:
    """Return the step a part compiles to on its own, or None if it is inline content."""
    root = part.root
    if isinstance(root, ToolRequestPart) and root.tool_request:
        return to_function_call_step(root.tool_request)
    if isinstance(root, ToolResponsePart) and root.tool_response:
        return to_function_result_step(root.tool_response, root.metadata)
    if isinstance(root, ReasoningPart):
        return to_thought_step(root)
    if isinstance(root, CustomPart):
        custom = root.custom or {}
        original_thought = thought_from_custom(custom)
        if original_thought is not None:
            return original_thought
        unknown = custom.get(UNKNOWN_STEP)
        if isinstance(unknown, dict):
            return cast(StepParam, copy.deepcopy(unknown))
        server_result = custom.get(SERVER_FUNCTION_RESULT)
        if isinstance(server_result, dict):
            return to_server_function_result_step(server_result)
        metadata = root.metadata or {}
        step = to_server_tool_step(custom, metadata)
        return with_signature(step, metadata) if step is not None else None
    return None


def to_content_step(role: InteractionRole, content: list[ContentParam]) -> StepParam:
    """Wrap a turn's inline content in the step for whoever produced it."""
    if role == 'model':
        model_step: ModelOutputStepParam = {'type': 'model_output', 'content': content}
        return model_step
    user_step: UserInputStepParam = {'type': 'user_input', 'content': content}
    return user_step


def to_interaction_steps(messages: list[Message]) -> list[StepParam]:
    """Convert Genkit messages to Interactions API steps.

    Text and media collapse into a content step, but tool calls, thoughts,
    and Google-side tool activity are steps of their own. Flush any buffered
    inline content before those so mixed parts stay in message order — a
    thought sitting between two sentences must not jump the first sentence
    to the end of the turn.
    """
    steps: list[StepParam] = []
    for message in messages:
        role = to_interaction_role(message.role)
        inline: list[ContentParam] = []

        for part in message.content:
            step = to_standalone_step(part)
            if step is not None:
                if inline:
                    steps.append(to_content_step(role, list(inline)))
                    inline.clear()
                steps.append(step)
                continue
            content = to_interaction_content(part)
            if content is not None:
                inline.append(content)
        if inline:
            steps.append(to_content_step(role, inline))
    return steps


def with_metadata(part: Part, key: str, value: object) -> Part:
    """Return a copy of the part carrying one more metadata entry."""
    if not value:
        return part
    root = part.root
    updated = root.model_copy(update={'metadata': {**(root.metadata or {}), key: value}})
    return Part(updated)


def plain(value: object) -> object:
    """Unwrap SDK models into plain data so parts stay JSON-serializable."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode='python')
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def server_tool_part(key: str, payload: dict[str, Any], *, signature: str | None, call_id: str | None = None) -> Part:
    """Wrap Google-side tool activity in the custom part Genkit round-trips."""
    part = Part(CustomPart(custom={key: payload}, metadata={CALL_ID: call_id} if call_id else None))
    return with_metadata(part, THOUGHT_SIGNATURE, signature)


def from_google_search_call(step: GoogleSearchCallStep) -> Part:
    """Convert a google_search_call step to a Genkit custom part."""
    return server_tool_part(
        GOOGLE_SEARCH_CALL,
        {'id': step.id, 'arguments': plain(step.arguments)},
        signature=step.signature,
    )


def from_google_search_result(step: GoogleSearchResultStep) -> Part:
    """Convert a google_search_result step to a Genkit custom part."""
    return server_tool_part(
        GOOGLE_SEARCH_RESULT,
        {CALL_ID: step.call_id, 'result': plain(step.result or [])},
        signature=step.signature,
    )


def from_code_execution_call(step: CodeExecutionCallStep) -> Part:
    """Convert a code_execution_call step to a Genkit custom part."""
    arguments = step.arguments
    return server_tool_part(
        EXECUTABLE_CODE,
        {
            'code': arguments.code if arguments else None,
            'language': (arguments.language if arguments else None) or CODE_LANGUAGE,
        },
        signature=step.signature,
        call_id=step.id,
    )


def from_code_execution_result(step: CodeExecutionResultStep) -> Part:
    """Convert a code_execution_result step to a Genkit custom part."""
    result = step.result
    return server_tool_part(
        CODE_EXECUTION_RESULT,
        {
            'output': result if isinstance(result, str) else json.dumps(result),
            'outcome': 'OUTCOME_OK',
        },
        signature=step.signature,
        call_id=step.call_id,
    )


def from_function_call_step(step: FunctionCallStep) -> Part:
    """Convert a function_call step to a Genkit tool request part.

    Fresh create/get steps with client tool calls are pending work for Genkit's
    tool loop, so they use ToolRequestPart — not an opaque custom blob.
    """
    return Part(
        ToolRequestPart(
            tool_request=ToolRequest(
                name=step.name or '',
                input=plain(step.arguments) if step.arguments is not None else {},
                ref=step.id,
            )
        )
    )


def from_function_result_step(step: FunctionResultStep) -> Part:
    """Stash a function_result as custom — never a ToolResponsePart.

    The API echoes the client's own tool result back on the model tape.
    Putting that on a ToolResponsePart would send it again next turn.
    """
    return Part(
        CustomPart(
            custom={
                SERVER_FUNCTION_RESULT: {
                    'name': step.name or '',
                    'result': plain(step.result),
                    'call_id': step.call_id,
                    'is_error': step.is_error,
                }
            }
        )
    )


def from_media_content(content: ImageContent | AudioContent | DocumentContent | VideoContent) -> MediaPart:
    """Convert wire media content to a Genkit media part."""
    url = content.uri
    if content.data and content.mime_type:
        url = f'data:{content.mime_type};base64,{content.data}'
    return MediaPart(media=Media(url=url or '', content_type=content.mime_type))


def from_text_content(content: TextContent) -> Part:
    """Convert wire text content to a Genkit text part."""
    # Empty annotations still show up in metadata so round-trips stay stable.
    return Part(
        TextPart(
            text=content.text or '',
            metadata={'annotations': content.annotations},
        )
    )


def from_visual_content(content: ImageContent | VideoContent) -> Part:
    """Convert wire image or video content, keeping the resolution it came back at."""
    return with_metadata(Part(from_media_content(content)), 'resolution', content.resolution)


def from_thought_step(step: ThoughtStep) -> Part:
    """Convert a thought step to a Genkit reasoning part."""
    summary = step.summary or []
    reasoning = '\n'.join(item.text or '' if isinstance(item, TextContent) else '[Image]' for item in summary)
    return Part(
        ReasoningPart(
            reasoning=reasoning,
            metadata={THOUGHT_SIGNATURE: step.signature},
            custom={THOUGHT_CUSTOM: step.model_dump(mode='python')},
        )
    )


def from_interaction_content(content: Content) -> Part:
    """Convert an Interactions content block back to a Genkit part."""
    if isinstance(content, TextContent):
        return from_text_content(content)
    if isinstance(content, (ImageContent, VideoContent)):
        return from_visual_content(content)
    if isinstance(content, (AudioContent, DocumentContent)):
        return Part(from_media_content(content))
    if isinstance(content, UnknownContent):
        return Part(CustomPart(custom={'unknownContent': content.model_dump(mode='python')}))
    return Part(CustomPart(custom={'unknownContent': content}))


def from_interaction_step(step: Step) -> list[Part]:
    """Convert an Interactions step to Genkit parts."""
    if isinstance(step, ModelOutputStep):
        return [from_interaction_content(content) for content in (step.content or [])]
    if isinstance(step, UserInputStep):
        # The API echoes our prompt back; including it would duplicate the input.
        return []
    if isinstance(step, GoogleSearchCallStep):
        return [from_google_search_call(step)]
    if isinstance(step, GoogleSearchResultStep):
        return [from_google_search_result(step)]
    if isinstance(step, CodeExecutionCallStep):
        return [from_code_execution_call(step)]
    if isinstance(step, CodeExecutionResultStep):
        return [from_code_execution_result(step)]
    if isinstance(step, ThoughtStep):
        return [from_thought_step(step)]
    if isinstance(step, FunctionCallStep):
        return [from_function_call_step(step)]
    if isinstance(step, FunctionResultStep):
        return [from_function_result_step(step)]
    if isinstance(step, BaseModel):
        return [Part(CustomPart(custom={UNKNOWN_STEP: step.model_dump(mode='python')}))]
    return [Part(CustomPart(custom={UNKNOWN_STEP: step}))]


def interaction_message_metadata(interaction: Interaction) -> dict[str, Any] | None:
    """Build message.metadata fields that identify the source Interaction."""
    metadata: dict[str, Any] = {}
    if interaction.id:
        metadata['interactionId'] = interaction.id
    if interaction.environment_id:
        metadata['environmentId'] = interaction.environment_id
    if interaction.status:
        # Preserve the wire status when finish_reason is a coarser Genkit enum.
        metadata['interactionStatus'] = interaction.status
    return metadata or None


def usage_from_interaction(usage: Usage) -> ModelUsage:
    """Map Interactions Usage onto Genkit ModelUsage."""
    response_usage = ModelUsage(
        input_tokens=usage.total_input_tokens,
        output_tokens=usage.total_output_tokens,
        total_tokens=usage.total_tokens,
        cached_content_tokens=usage.total_cached_tokens,
        thoughts_tokens=usage.total_thought_tokens,
    )
    for modality_token in usage.input_tokens_by_modality or []:
        match modality_token.modality:
            case 'text':
                response_usage.input_characters = modality_token.tokens
            case 'image':
                response_usage.input_images = modality_token.tokens
            case 'audio':
                response_usage.input_audio_files = modality_token.tokens
            case _:
                pass
    for modality_token in usage.output_tokens_by_modality or []:
        match modality_token.modality:
            case 'text':
                response_usage.output_characters = modality_token.tokens
            case 'image':
                response_usage.output_images = modality_token.tokens
            case 'audio':
                response_usage.output_audio_files = modality_token.tokens
            case _:
                pass
    return response_usage


def parts_from_steps(steps: list[Step]) -> list[Part]:
    """Flatten interaction steps into Genkit parts, dropping empty ones."""
    return [
        part
        for part in (item for step in steps for item in from_interaction_step(step))
        if part.model_dump(exclude_none=True)
    ]


def model_response(
    interaction: Interaction,
    *,
    content: list[Part],
    finish_reason: FinishReason,
    finish_message: str | None = None,
) -> ModelResponse:
    """Build a ModelResponse that carries the raw interaction alongside its parts."""
    dumped = interaction.model_dump(mode='python')
    response = ModelResponse.model_construct(
        finish_reason=finish_reason,
        finish_message=finish_message,
        message=Message(role='model', content=content, metadata=interaction_message_metadata(interaction)),
        custom=dumped,
        raw=dumped,
    )
    if interaction.usage is not None:
        response.usage = usage_from_interaction(interaction.usage)
    return response


def cancelled_response(interaction: Interaction) -> ModelResponse:
    """Build the ModelResponse for a cancelled Interaction."""
    return model_response(
        interaction,
        content=[Part(TextPart(text='Operation cancelled.'))],
        finish_reason=FinishReason.ABORTED,
        finish_message='Operation cancelled',
    )


def steps_response(
    interaction: Interaction,
    *,
    finish_reason: FinishReason,
    finish_message: str | None = None,
) -> ModelResponse | None:
    """Build a ModelResponse from interaction steps, or None when there are none."""
    steps = list(interaction.steps or [])
    if not steps:
        return None
    return model_response(
        interaction,
        content=parts_from_steps(steps),
        finish_reason=finish_reason,
        finish_message=finish_message,
    )


def completed_response(interaction: Interaction) -> ModelResponse | None:
    """Build the ModelResponse for a completed Interaction, if it has steps."""
    return steps_response(interaction, finish_reason=FinishReason.STOP)


def from_interaction_sync(interaction: Interaction) -> ModelResponse:
    """Convert a completed interaction to a synchronous model response."""
    if interaction.status == 'failed':
        raise ValueError(interaction_error_message(interaction) or FAILED_MESSAGE)
    if interaction.status == 'cancelled':
        return cancelled_response(interaction)
    return completed_response(interaction) or model_response(interaction, content=[], finish_reason=FinishReason.STOP)


def from_interaction(
    interaction: Interaction,
    client_options: ClientOptions | None = None,
) -> Operation:
    """Convert an interaction poll result to a Genkit operation."""
    op = Operation.model_construct(id=interaction.id or '')
    if client_options is not None:
        dumped = client_options.to_metadata_dict()
        if dumped:
            op.metadata = {'clientOptions': dumped}

    status = interaction.status
    if status in ('in_progress', 'queued'):
        # Keep polling — still running or waiting on the server.
        op.done = False
    elif status == 'cancelled':
        op.done = True
        op.output = cancelled_response(interaction)
    elif status == 'completed':
        op.done = True
        op.output = completed_response(interaction)
    elif partial := PARTIAL_TERMINAL_STATUSES.get(cast(str, status)):
        # Halted for length or budget. Prefer whatever landed over a bare error.
        finish_reason, message = partial
        op.done = True
        op.output = steps_response(interaction, finish_reason=finish_reason, finish_message=message)
        if op.output is None:
            op.error = Error(message=message)
    elif status == 'failed':
        # Always exit the poll loop on failure; leaving done unset hangs forever.
        op.done = True
        op.error = Error(message=interaction_error_message(interaction) or FAILED_MESSAGE)
    # requires_action: leave done unset. Resuming that turn is a separate product
    # decision; we don't invent interrupt/resume here.
    return op
