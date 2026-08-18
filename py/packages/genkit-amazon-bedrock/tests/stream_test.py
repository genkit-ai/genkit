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

"""Tests for ConverseStream event reassembly (no AWS involved)."""

import base64
from collections.abc import AsyncIterator
from typing import Any

import pytest
from genkit_amazon_bedrock.converters import (
    REASONING_SIGNATURE_METADATA_KEY,
    REDACTED_CONTENT_METADATA_KEY,
    to_model_response,
)
from genkit_amazon_bedrock.stream import consume_converse_stream

from genkit import FinishReason, Message, ModelRequest, Part, Role, TextPart, ToolDefinition
from genkit.plugin_api import ActionRunContext, GenkitError

pytestmark = pytest.mark.asyncio


async def events(*items: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    for item in items:
        yield item


def text_request(text: str = 'hello', **kwargs: Any) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text=text))])],
        **kwargs,
    )


def text_delta(index: int, text: str) -> dict[str, Any]:
    return {'contentBlockDelta': {'contentBlockIndex': index, 'delta': {'text': text}}}


def reasoning_delta(index: int, delta: dict[str, Any]) -> dict[str, Any]:
    return {'contentBlockDelta': {'contentBlockIndex': index, 'delta': {'reasoningContent': delta}}}


def tool_start(index: int, tool_use_id: str, name: str) -> dict[str, Any]:
    return {
        'contentBlockStart': {
            'contentBlockIndex': index,
            'start': {'toolUse': {'toolUseId': tool_use_id, 'name': name}},
        }
    }


def tool_delta(index: int, fragment: str) -> dict[str, Any]:
    return {'contentBlockDelta': {'contentBlockIndex': index, 'delta': {'toolUse': {'input': fragment}}}}


def tool_stop(index: int) -> dict[str, Any]:
    return {'contentBlockStop': {'contentBlockIndex': index}}


def message_stop(stop_reason: str) -> dict[str, Any]:
    return {'messageStop': {'stopReason': stop_reason}}


def metadata(usage: dict[str, Any]) -> dict[str, Any]:
    return {'metadata': {'usage': usage}}


def streaming_ctx() -> tuple[ActionRunContext, list[Any]]:
    chunks: list[Any] = []
    return ActionRunContext(streaming_callback=chunks.append), chunks


async def test_text_deltas_stream_and_accumulate() -> None:
    ctx, chunks = streaming_ctx()
    request = text_request()

    response = await consume_converse_stream(
        events(
            {'messageStart': {'role': 'assistant'}},
            text_delta(0, 'hel'),
            text_delta(0, 'lo'),
            metadata({'inputTokens': 3, 'outputTokens': 2, 'totalTokens': 5}),
            message_stop('end_turn'),
        ),
        request,
        ctx,
    )

    assert [chunk.content[0].root.text for chunk in chunks] == ['hel', 'lo']
    assert all(chunk.role == Role.MODEL for chunk in chunks)
    assert response.message is not None
    assert [part.root.text for part in response.message.content] == ['hello']
    assert response.finish_reason == FinishReason.STOP
    assert response.usage is not None
    assert response.usage.total_tokens == 5
    assert response.request is request


async def test_missing_message_stop_finishes_as_stop() -> None:
    response = await consume_converse_stream(events(text_delta(0, 'hi')), text_request())

    # The sync path maps an empty stop reason to OTHER; a truncated stream did
    # not tell us anything went wrong.
    assert response.finish_reason == FinishReason.STOP


async def test_max_tokens_stop_reason_maps_to_length() -> None:
    response = await consume_converse_stream(
        events(text_delta(0, 'hi'), message_stop('max_tokens')),
        text_request(),
    )

    assert response.finish_reason == FinishReason.LENGTH


async def test_absent_block_index_defaults_to_zero() -> None:
    response = await consume_converse_stream(
        events(
            {'contentBlockDelta': {'delta': {'text': 'a'}}},
            text_delta(0, 'b'),
        ),
        text_request(),
    )

    assert response.message is not None
    assert [part.root.text for part in response.message.content] == ['ab']


async def test_blocks_assemble_in_index_order() -> None:
    response = await consume_converse_stream(
        events(
            text_delta(1, 'second'),
            text_delta(0, 'first'),
            message_stop('end_turn'),
        ),
        text_request(),
    )

    assert response.message is not None
    assert [part.root.text for part in response.message.content] == ['first', 'second']


async def test_empty_stream_returns_empty_text_placeholder() -> None:
    response = await consume_converse_stream(events(), text_request())

    assert response.message is not None
    assert len(response.message.content) == 1
    assert response.message.content[0].root.text == ''


async def test_unknown_events_and_deltas_are_ignored() -> None:
    ctx, chunks = streaming_ctx()

    response = await consume_converse_stream(
        events(
            {'someFutureEvent': {'whatever': True}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'citation': {'title': 'a'}}}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': None}},
            text_delta(0, 'ok'),
            message_stop('end_turn'),
        ),
        text_request(),
        ctx,
    )

    assert [chunk.content[0].root.text for chunk in chunks] == ['ok']
    assert response.message is not None
    assert [part.root.text for part in response.message.content] == ['ok']


async def test_tool_use_fragments_emit_one_chunk_on_block_stop() -> None:
    ctx, chunks = streaming_ctx()
    request = text_request(
        tools=[
            ToolDefinition(
                name='get_weather',
                description='Get the weather',
                input_schema={
                    'type': 'object',
                    'properties': {'location': {'type': 'string'}, 'count': {'type': 'integer'}},
                },
            )
        ]
    )

    response = await consume_converse_stream(
        events(
            tool_start(0, 'call_1', 'get_weather'),
            tool_delta(0, '{"location":"NYC",'),
            tool_delta(0, '"count":"42"}'),
            tool_stop(0),
            message_stop('tool_use'),
        ),
        request,
        ctx,
    )

    assert len(chunks) == 1
    streamed = chunks[0].content[0].root.tool_request
    assert streamed.name == 'get_weather'
    assert streamed.ref == 'call_1'
    # Coerced against the declared schema, like the sync path.
    assert streamed.input == {'location': 'NYC', 'count': 42}
    assert response.finish_reason == FinishReason.STOP
    assert response.message is not None
    final = response.message.content[0].root.tool_request
    assert final is not None
    assert final.input == {'location': 'NYC', 'count': 42}


async def test_tool_block_without_input_matches_the_sync_path() -> None:
    stream_response = await consume_converse_stream(
        events(tool_start(0, 'call_1', 'noop'), tool_stop(0), message_stop('tool_use')),
        text_request(),
    )
    # The same response, unstreamed. Both paths must dispatch identically: a
    # tool declared with an input model accepts {} and rejects None.
    sync_response = to_model_response(
        {
            'output': {
                'message': {'role': 'assistant', 'content': [{'toolUse': {'toolUseId': 'call_1', 'name': 'noop'}}]}
            },
            'stopReason': 'tool_use',
        },
        text_request(),
    )

    assert stream_response.message is not None
    assert sync_response.message is not None
    streamed = stream_response.message.content[0].root.tool_request
    synced = sync_response.message.content[0].root.tool_request
    assert streamed is not None and synced is not None
    assert streamed.input == synced.input == {}


async def test_malformed_tool_input_raises() -> None:
    ctx, chunks = streaming_ctx()

    with pytest.raises(GenkitError) as excinfo:
        await consume_converse_stream(
            events(tool_start(0, 'call_1', 'get_weather'), tool_delta(0, '{not valid'), tool_stop(0)),
            text_request(),
            ctx,
        )

    assert 'stream tool block 0' in excinfo.value.original_message
    assert chunks == []


async def test_trailing_data_after_tool_input_raises() -> None:
    with pytest.raises(GenkitError) as excinfo:
        await consume_converse_stream(
            events(tool_start(0, 'call_1', 'get_weather'), tool_delta(0, '{"ok":true} {"extra":true}'), tool_stop(0)),
            text_request(),
        )

    assert 'stream tool block 0' in excinfo.value.original_message


async def test_content_block_stop_for_unknown_index_is_a_noop() -> None:
    ctx, chunks = streaming_ctx()

    response = await consume_converse_stream(events(tool_stop(7), message_stop('end_turn')), text_request(), ctx)

    assert chunks == []
    assert response.message is not None
    assert response.message.content[0].root.text == ''


async def test_reasoning_text_streams_and_signature_accumulates() -> None:
    ctx, chunks = streaming_ctx()

    response = await consume_converse_stream(
        events(
            reasoning_delta(0, {'text': 'think'}),
            reasoning_delta(0, {'text': 'ing'}),
            reasoning_delta(0, {'signature': 'sig-abc'}),
            text_delta(0, 'answer'),
            message_stop('end_turn'),
        ),
        text_request(),
        ctx,
    )

    # Signature deltas are accumulated only; they carry no user-visible text.
    assert [chunk.content[0].root.reasoning for chunk in chunks[:2]] == ['think', 'ing']
    assert len(chunks) == 3
    assert chunks[2].content[0].root.text == 'answer'

    assert response.message is not None
    reasoning_part, text_part = response.message.content
    assert reasoning_part.root.reasoning == 'thinking'
    assert reasoning_part.root.metadata is not None
    assert reasoning_part.root.metadata[REASONING_SIGNATURE_METADATA_KEY] == 'sig-abc'
    assert text_part.root.text == 'answer'


async def test_redacted_reasoning_accumulates_as_base64_without_chunks() -> None:
    ctx, chunks = streaming_ctx()

    response = await consume_converse_stream(
        events(
            reasoning_delta(0, {'redactedContent': b'\x00\x01'}),
            reasoning_delta(0, {'redactedContent': b'\x02'}),
            message_stop('end_turn'),
        ),
        text_request(),
        ctx,
    )

    assert chunks == []
    assert response.message is not None
    part = response.message.content[0].root
    assert part.reasoning == ''
    assert part.metadata is not None
    assert base64.b64decode(part.metadata[REDACTED_CONTENT_METADATA_KEY]) == b'\x00\x01\x02'


async def test_non_streaming_context_sends_no_chunks() -> None:
    response = await consume_converse_stream(
        events(text_delta(0, 'hi'), tool_start(1, 'call_1', 'noop'), tool_delta(1, '{}'), tool_stop(1)),
        text_request(),
        ActionRunContext(),
    )

    assert response.message is not None
    assert len(response.message.content) == 2


async def test_chunk_callback_error_aborts_the_stream() -> None:
    def explode(_chunk: Any) -> None:
        raise RuntimeError('callback failed')

    with pytest.raises(RuntimeError, match='callback failed'):
        await consume_converse_stream(
            events(text_delta(0, 'hi')),
            text_request(),
            ActionRunContext(streaming_callback=explode),
        )
