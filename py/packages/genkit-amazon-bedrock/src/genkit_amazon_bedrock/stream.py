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

"""ConverseStream event reassembly.

Bedrock streams a message as
deltas tagged with a ``contentBlockIndex``; this module accumulates them per
block, emits Genkit chunks as they arrive, and assembles the final message in
block order. Request conversion is shared with the non-streaming path.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from genkit import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    Part,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
)
from genkit.plugin_api import ActionRunContext, GenkitError
from genkit_amazon_bedrock.converters import (
    bedrock_reasoning_part,
    coerce_tool_input,
    map_finish_reason,
    usage_from_response,
    usage_log_fields,
)

logger = structlog.get_logger(__name__)


@dataclass
class _StreamBlock:
    """Accumulated state of one content block across its delta events."""

    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    signature: str | None = None
    redacted: bytearray = field(default_factory=bytearray)
    tool_id: str | None = None
    tool_name: str = ''
    tool_input: list[str] = field(default_factory=list)
    is_tool: bool = False


async def consume_converse_stream(
    events: AsyncIterator[dict[str, Any]],
    request: ModelRequest[Any],
    ctx: ActionRunContext | None = None,
    model_id: str | None = None,
) -> ModelResponse:
    """Reassembles a ConverseStream into a ModelResponse, streaming chunks.

    Chunks are deltas, not snapshots: text and reasoning text are forwarded as
    they arrive, while a tool call is held back until its block closes because
    its input arrives as JSON fragments. The framework assigns each chunk's
    index when it re-wraps them, so the block index is not propagated.

    Args:
        events: Raw ConverseStream event dicts, in wire order.
        request: The originating Genkit request; supplies tool schemas and is
            echoed on the response.
        ctx: Action run context; chunks are sent only when it is streaming.
        model_id: Model the events came from, for the completion log line only.

    Returns:
        The assembled model response.
    """
    blocks: dict[int, _StreamBlock] = {}
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None
    streaming = ctx is not None and ctx.is_streaming

    async for event in events:
        if (block_start := event.get('contentBlockStart')) is not None:
            block = _get_or_init(blocks, _block_index(block_start))
            # Only a toolUse start carries state; other variants are ignored.
            tool_use = (block_start.get('start') or {}).get('toolUse')
            if tool_use is not None:
                block.is_tool = True
                block.tool_id = tool_use.get('toolUseId')
                block.tool_name = tool_use.get('name') or ''
        elif (block_delta := event.get('contentBlockDelta')) is not None:
            delta = block_delta.get('delta')
            if delta is None:
                continue
            block = _get_or_init(blocks, _block_index(block_delta))
            part = _append_delta(block, delta)
            if part is not None and streaming and ctx is not None:
                ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=0, content=[part]))
        elif (block_stop := event.get('contentBlockStop')) is not None:
            index = _block_index(block_stop)
            block = blocks.get(index)
            # A tool call is only complete now, so this is its one chunk.
            if streaming and ctx is not None and block is not None and block.is_tool:
                ctx.send_chunk(
                    ModelResponseChunk(
                        role=Role.MODEL,
                        index=0,
                        content=[_tool_block_to_part(index, block, request.tools)],
                    )
                )
        elif (message_stop := event.get('messageStop')) is not None:
            stop_reason = message_stop.get('stopReason')
        elif (metadata := event.get('metadata')) is not None:
            usage = metadata.get('usage')
        # messageStart and unrecognized events are ignored so new Bedrock
        # event types don't break streaming.

    parts = _blocks_to_parts(blocks, request.tools)
    if not parts:
        parts = [Part(root=TextPart(text=''))]
    # A stream that ends without messageStop stopped normally; the sync path's
    # mapping would call that OTHER.
    finish_reason = map_finish_reason(stop_reason) if stop_reason else FinishReason.STOP
    logger.debug(
        'Bedrock stream complete',
        model=model_id,
        stop_reason=stop_reason,
        blocks=len(blocks),
        tool_blocks=sum(1 for block in blocks.values() if block.is_tool),
        **usage_log_fields(usage),
    )
    return ModelResponse(
        message=Message(role=Role.MODEL, content=parts),
        finish_reason=finish_reason,
        usage=usage_from_response(usage),
        request=request,
    )


def _block_index(event: dict[str, Any]) -> int:
    """Reads contentBlockIndex; an absent index means block 0."""
    index = event.get('contentBlockIndex')
    return int(index) if index is not None else 0


def _get_or_init(blocks: dict[int, _StreamBlock], index: int) -> _StreamBlock:
    block = blocks.get(index)
    if block is None:
        block = _StreamBlock()
        blocks[index] = block
    return block


def _append_delta(block: _StreamBlock, delta: dict[str, Any]) -> Part | None:
    """Accumulates one content delta; returns the part to stream, if any."""
    if (text := delta.get('text')) is not None:
        block.text.append(text)
        return Part(root=TextPart(text=text))
    if (tool_use := delta.get('toolUse')) is not None:
        block.is_tool = True
        block.tool_input.append(tool_use.get('input') or '')
        return None
    if (reasoning := delta.get('reasoningContent')) is not None:
        return _append_reasoning_delta(block, reasoning)
    # Unhandled variants (citation today) are dropped rather than raised on:
    # Bedrock adds delta variants without notice, and failing loud here
    # would kill any stream that used one. The sync path fails loud instead,
    # because dropping a whole content block would lose model output.
    return None


def _append_reasoning_delta(block: _StreamBlock, delta: dict[str, Any]) -> Part | None:
    """Accumulates a reasoning delta; only text is streamed as a chunk.

    The signature and redacted blob are needed whole to replay the reasoning
    on the next turn, so they accumulate silently.
    """
    if (text := delta.get('text')) is not None:
        block.reasoning.append(text)
        return bedrock_reasoning_part(text, None, None)
    if (signature := delta.get('signature')) is not None:
        block.signature = signature
        return None
    if (redacted := delta.get('redactedContent')) is not None:
        block.redacted.extend(redacted)
    # Unhandled reasoning variants are dropped, as above.
    return None


def _blocks_to_parts(blocks: dict[int, _StreamBlock], tools: list[ToolDefinition] | None) -> list[Part]:
    """Assembles accumulated blocks into parts in contentBlockIndex order."""
    parts: list[Part] = []
    for index in sorted(blocks):
        block = blocks[index]
        if block.is_tool:
            parts.append(_tool_block_to_part(index, block, tools))
            continue
        # One block can carry both reasoning and text; reasoning comes first.
        reasoning = ''.join(block.reasoning)
        if reasoning or block.redacted:
            parts.append(bedrock_reasoning_part(reasoning, block.signature, bytes(block.redacted) or None))
        text = ''.join(block.text)
        if text:
            parts.append(Part(root=TextPart(text=text)))
    return parts


def _tool_block_to_part(index: int, block: _StreamBlock, tools: list[ToolDefinition] | None) -> Part:
    tool_input = _decode_tool_input(index, ''.join(block.tool_input))
    if tool_input is None:
        # {} rather than None, matching the sync path: a tool declared with an
        # input model rejects None, so the two paths would dispatch differently.
        tool_input = {}
    if isinstance(tool_input, dict):
        tool_input = coerce_tool_input(block.tool_name, tool_input, tools)
    return Part(
        root=ToolRequestPart(tool_request=ToolRequest(ref=block.tool_id, name=block.tool_name, input=tool_input))
    )


def _decode_tool_input(index: int, raw: str) -> Any:  # noqa: ANN401
    """Decodes accumulated tool-input fragments; None when none were sent.

    ``json.loads`` rejects trailing data after the JSON value, so a doubled or
    truncated fragment raises instead of decoding halfway.
    """
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except ValueError as e:
        raise GenkitError(
            message=f'bedrock: stream tool block {index}: decode tool input: {e}',
            status='INTERNAL',
        ) from e
