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

"""Tests for CohereModel — generate, streaming, config, and model info.

Uses MagicMock for Cohere SDK stream events because the SDK types are
strict Pydantic models that reject SimpleNamespace substitutes.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    from cohere.types import TextAssistantMessageV2ContentItem
except ImportError:
    from cohere.types import TextAssistantMessageV2ContentOneItem as TextAssistantMessageV2ContentItem
from cohere.v2.types.v2chat_stream_response import (
    ContentDeltaV2ChatStreamResponse,
    MessageEndV2ChatStreamResponse,
    ToolCallDeltaV2ChatStreamResponse,
    ToolCallStartV2ChatStreamResponse,
)

from genkit.core.typing import (
    FinishReason,
    GenerateRequest,
    Message,
    OutputConfig,
    Part,
    Role,
    TextPart,
    ToolDefinition,
)
from genkit.plugins.cohere.models import CohereConfig, CohereModel, cohere_name


def _make_mock_response(
    text: str = 'Hello!',
    finish_reason: str = 'COMPLETE',
    tool_calls: list[Any] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Build a mock that mimics V2ChatResponse structure.

    Uses MagicMock instead of real Pydantic models to avoid
    cross-version content-type validation issues in the Cohere SDK.
    """
    mock = MagicMock()
    mock.id = 'resp-1'
    mock.finish_reason = finish_reason

    # Message content — must pass isinstance(TextAssistantMessageV2ContentItem)
    content_item = MagicMock()
    content_item.__class__ = TextAssistantMessageV2ContentItem
    content_item.type = 'text'
    content_item.text = text
    mock.message.content = [content_item]
    mock.message.tool_calls = tool_calls

    # Usage
    mock.usage.billed_units.input_tokens = input_tokens
    mock.usage.billed_units.output_tokens = output_tokens

    return mock


class TestCohereModelInfo:
    """Tests for CohereModel.get_model_info."""

    def test_known_model_info(self) -> None:
        """Known model returns its metadata."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        info = model.get_model_info()
        assert info is not None
        assert 'Command A' in info['name']
        assert info['supports']['multiturn'] is True

    def test_unknown_model_gets_default_info(self) -> None:
        """Unknown model gets default conservative metadata."""
        model = CohereModel(model='future-model-2099', api_key='test-key')
        info = model.get_model_info()
        assert info is not None
        assert 'future-model-2099' in info['name']
        assert info['supports']['tools'] is False

    def test_to_generate_fn(self) -> None:
        """to_generate_fn returns the generate method."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        fn = model.to_generate_fn()
        assert fn == model.generate


class TestCohereModelGenerate:
    """Tests for CohereModel.generate (non-streaming)."""

    @pytest.mark.asyncio
    async def test_basic_text_generation(self) -> None:
        """Basic text generation returns correct response."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=_make_mock_response('Hello, World!'))

        request = GenerateRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])])
        response = await model.generate(request)

        assert response.message is not None
        assert len(response.message.content) > 0
        assert response.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generation_with_dict_config(self) -> None:
        """Dict config is forwarded to the API."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=_make_mock_response())

        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            config={'temperature': 0.5, 'max_tokens': 100, 'top_k': 10, 'top_p': 0.9},
        )
        await model.generate(request)

        call_kwargs = model.client.chat.call_args.kwargs
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 100
        assert call_kwargs['k'] == 10  # top_k → k alias
        assert call_kwargs['p'] == 0.9  # top_p → p alias

    @pytest.mark.asyncio
    async def test_generation_with_tools(self) -> None:
        """Tools are converted and passed to the API."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=_make_mock_response())

        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Weather?'))])],
            tools=[
                ToolDefinition(
                    name='get_weather',
                    description='Get weather',
                    input_schema={'type': 'object', 'properties': {'loc': {'type': 'string'}}},
                )
            ],
        )
        await model.generate(request)

        call_kwargs = model.client.chat.call_args.kwargs
        assert 'tools' in call_kwargs
        assert len(call_kwargs['tools']) == 1

    @pytest.mark.asyncio
    async def test_generation_with_structured_output(self) -> None:
        """Structured output config is forwarded."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=_make_mock_response())

        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='JSON please'))])],
            output=OutputConfig(format='json', schema={'type': 'object'}),
        )
        await model.generate(request)

        call_kwargs = model.client.chat.call_args.kwargs
        assert call_kwargs['response_format']['type'] == 'json_object'

    @pytest.mark.asyncio
    async def test_generation_with_tool_calls_in_response(self) -> None:
        """Tool calls in the response are converted to ToolRequestParts."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()

        tc_mock = MagicMock()
        tc_mock.id = 'tc-1'
        tc_mock.type = 'function'
        tc_mock.function.name = 'get_weather'
        tc_mock.function.arguments = '{"loc": "NYC"}'

        resp = _make_mock_response(text='', finish_reason='TOOL_CALL', tool_calls=[tc_mock])
        model.client.chat = AsyncMock(return_value=resp)

        request = GenerateRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Weather?'))])])
        response = await model.generate(request)

        assert response.message is not None
        has_tool = any(hasattr(p.root, 'tool_request') for p in response.message.content)
        assert has_tool

    @pytest.mark.asyncio
    async def test_generation_with_no_config(self) -> None:
        """Generation works with no config."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=_make_mock_response())

        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            config=None,
        )
        response = await model.generate(request)
        assert response.message is not None

    @pytest.mark.asyncio
    async def test_generation_with_stop_sequences_and_seed(self) -> None:
        """stop_sequences, seed, and penalties are forwarded."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=_make_mock_response())

        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            config={
                'stop_sequences': ['END'],
                'seed': 42,
                'frequency_penalty': 0.5,
                'presence_penalty': 0.3,
            },
        )
        await model.generate(request)

        call_kwargs = model.client.chat.call_args.kwargs
        assert call_kwargs['stop_sequences'] == ['END']
        assert call_kwargs['seed'] == 42
        assert call_kwargs['frequency_penalty'] == 0.5
        assert call_kwargs['presence_penalty'] == 0.3


class TestCohereModelStreaming:
    """Tests for CohereModel._generate_streaming."""

    @pytest.mark.asyncio
    async def test_streaming_text(self) -> None:
        """Streaming text events are accumulated and chunks sent."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()

        events = [
            _mock_content_delta('Hello'),
            _mock_content_delta(', World!'),
            _mock_message_end('COMPLETE'),
        ]
        model.client.chat_stream = lambda **kwargs: _async_iter(events)

        chunks_sent: list[Any] = []
        ctx = MagicMock()
        ctx.send_chunk = lambda c: chunks_sent.append(c)

        request = GenerateRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])])
        response = await model.generate(request, ctx=ctx)

        assert response.message is not None
        text_parts: list[str] = [str(p.root.text) for p in response.message.content if hasattr(p.root, 'text')]
        assert 'Hello, World!' in ''.join(text_parts)
        assert len(chunks_sent) == 2
        assert response.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls(self) -> None:
        """Streaming tool call events are accumulated correctly."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()

        events = [
            _mock_tool_call_start(0, 'tc-1', 'get_weather'),
            _mock_tool_call_delta(0, '{"loc":'),
            _mock_tool_call_delta(0, '"NYC"}'),
            _mock_message_end('TOOL_CALL'),
        ]
        model.client.chat_stream = lambda **kwargs: _async_iter(events)

        chunks_sent: list[Any] = []
        ctx = MagicMock()
        ctx.send_chunk = lambda c: chunks_sent.append(c)

        request = GenerateRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Weather?'))])])
        response = await model.generate(request, ctx=ctx)

        assert response.message is not None
        has_tool = any(hasattr(p.root, 'tool_request') for p in response.message.content)
        assert has_tool
        assert response.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_streaming_tool_call_delta_without_start(self) -> None:
        """Tool call delta without prior start creates default entry."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()

        events = [
            _mock_tool_call_delta(0, '{"a":1}'),
            _mock_message_end('TOOL_CALL'),
        ]
        model.client.chat_stream = lambda **kwargs: _async_iter(events)

        ctx = MagicMock()
        ctx.send_chunk = MagicMock()

        request = GenerateRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])])
        response = await model.generate(request, ctx=ctx)

        assert response.message is not None
        has_tool = any(hasattr(p.root, 'tool_request') for p in response.message.content)
        assert has_tool

    @pytest.mark.asyncio
    async def test_streaming_unknown_finish_reason(self) -> None:
        """Unknown finish reason maps to OTHER."""
        model = CohereModel(model='command-a-03-2025', api_key='test-key')
        model.client = AsyncMock()

        events = [
            _mock_content_delta('Hi'),
            _mock_message_end('UNKNOWN_REASON'),
        ]
        model.client.chat_stream = lambda **kwargs: _async_iter(events)

        ctx = MagicMock()
        ctx.send_chunk = MagicMock()

        request = GenerateRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])])
        response = await model.generate(request, ctx=ctx)
        assert response.finish_reason == FinishReason.OTHER


class TestCohereConfig:
    """Tests for CohereConfig validation."""

    def test_defaults(self) -> None:
        """All defaults are None."""
        config = CohereConfig()
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.stop_sequences is None

    def test_valid_config(self) -> None:
        """Valid config values pass validation."""
        config = CohereConfig(
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            top_k=50,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            seed=42,
            stop_sequences=['END'],
        )
        assert config.temperature == 0.7
        assert config.top_k == 50

    def test_cohere_name(self) -> None:
        """cohere_name produces expected format."""
        assert cohere_name('command-a-03-2025') == 'cohere/command-a-03-2025'
        assert cohere_name('embed-v4.0') == 'cohere/embed-v4.0'


# ── Helpers for streaming mock events ────────────────────────────────
#
# Cohere SDK stream events are strict Pydantic models. Using
# MagicMock(spec=...) restricts attribute access via getattr, which
# breaks the converter helpers. Instead, we use plain MagicMock and
# override __class__ so isinstance checks in models.py still pass.


def _mock_content_delta(text: str) -> MagicMock:
    """Create a mock that passes isinstance(ContentDeltaV2ChatStreamResponse)."""
    mock = MagicMock()
    mock.__class__ = ContentDeltaV2ChatStreamResponse
    mock.delta.message.content.text = text
    return mock


def _mock_tool_call_start(idx: int, tc_id: str, name: str) -> MagicMock:
    """Create a mock that passes isinstance(ToolCallStartV2ChatStreamResponse)."""
    mock = MagicMock()
    mock.__class__ = ToolCallStartV2ChatStreamResponse
    mock.index = idx
    mock.delta.message.tool_calls.id = tc_id
    mock.delta.message.tool_calls.function.name = name
    return mock


def _mock_tool_call_delta(idx: int, args: str) -> MagicMock:
    """Create a mock that passes isinstance(ToolCallDeltaV2ChatStreamResponse)."""
    mock = MagicMock()
    mock.__class__ = ToolCallDeltaV2ChatStreamResponse
    mock.index = idx
    mock.delta.message.tool_calls.function.arguments = args
    return mock


def _mock_message_end(finish_reason: str) -> MagicMock:
    """Create a mock that passes isinstance(MessageEndV2ChatStreamResponse)."""
    mock = MagicMock()
    mock.__class__ = MessageEndV2ChatStreamResponse
    mock.delta.finish_reason = finish_reason
    return mock


async def _async_iter(items: list[Any]) -> AsyncIterator[Any]:
    """Create an async iterator from a list."""
    for item in items:
        yield item
