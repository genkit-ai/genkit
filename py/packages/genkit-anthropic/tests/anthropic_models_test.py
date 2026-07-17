# Copyright 2025 Google LLC
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

"""Tests for Anthropic models."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AsyncAnthropic
from genkit_anthropic import models as anthropic_models
from genkit_anthropic.config import AnthropicConfig
from genkit_anthropic.models import AnthropicModel, _to_anthropic_thinking_config
from genkit_anthropic.utils import maybe_strip_fences, strip_markdown_fences
from pydantic import ValidationError

from genkit import (
    Constrained,
    FinishReason,
    Media,
    MediaPart,
    Message,
    Metadata,
    ModelConfig,
    ModelInfo,
    ModelRequest,
    ModelResponseChunk,
    Part,
    Role,
    Supports,
    TextPart,
    ToolDefinition,
    ToolRequestPart,
)


def _create_sample_request() -> ModelRequest:
    """Create a sample generation request for testing."""
    return ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='Hello, how are you?'))],
            )
        ],
        config=ModelConfig(),
        tools=[
            ToolDefinition(
                name='get_weather',
                description='Get weather for a location',
                input_schema={
                    'type': 'object',
                    'properties': {'location': {'type': 'string', 'description': 'Location name'}},
                    'required': ['location'],
                },
            )
        ],
    )


@pytest.mark.asyncio
async def test_generate_basic() -> None:
    """Test basic generation."""
    sample_request = _create_sample_request()

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type='text', text="Hello! I'm doing well.")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=15)
    mock_response.stop_reason = 'end_turn'

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)
    response = await model.generate(sample_request)

    assert response.message is not None
    assert response.message.content is not None
    assert len(response.message.content) == 1
    part = response.message.content[0]
    actual_part = part.root if isinstance(part, Part) else part
    assert isinstance(actual_part, TextPart)
    assert actual_part.text == "Hello! I'm doing well."
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 15
    assert response.finish_reason == 'stop'


@pytest.mark.asyncio
async def test_generate_with_tools() -> None:
    """Test generation with tool calls."""
    sample_request = _create_sample_request()

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = 'tool_use'
    mock_block.id = 'tool_123'
    mock_block.name = 'get_weather'
    mock_block.input = {'location': 'Paris'}
    mock_response.content = [mock_block]
    mock_response.usage = MagicMock(input_tokens=20, output_tokens=10)
    mock_response.stop_reason = 'tool_use'

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)
    response = await model.generate(sample_request)

    assert response.message is not None
    assert response.message.content is not None
    assert len(response.message.content) == 1
    part = response.message.content[0]
    actual_part = part.root if isinstance(part, Part) else part
    assert isinstance(actual_part, ToolRequestPart)
    assert actual_part.tool_request is not None
    assert actual_part.tool_request.name == 'get_weather'
    assert actual_part.tool_request.ref == 'tool_123'
    assert actual_part.tool_request.input == {'location': 'Paris'}


@pytest.mark.asyncio
async def test_generate_defaults_empty_tool_input_schema() -> None:
    """Test that tools with a missing or empty input schema get a default object schema."""
    populated_schema = {
        'type': 'object',
        'properties': {'location': {'type': 'string', 'description': 'Location name'}},
        'required': ['location'],
    }
    request = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='Hello'))],
            )
        ],
        config=ModelConfig(),
        tools=[
            ToolDefinition(name='no_schema_tool', description='Tool with no input schema', input_schema=None),
            ToolDefinition(name='empty_schema_tool', description='Tool with empty input schema', input_schema={}),
            ToolDefinition(
                name='untyped_schema_tool',
                description='Tool with a schema missing a top-level type',
                input_schema={'properties': {'location': {'type': 'string'}}},
            ),
            ToolDefinition(name='get_weather', description='Get weather for a location', input_schema=populated_schema),
        ],
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type='text', text='Done')]
    mock_response.usage = MagicMock(input_tokens=5, output_tokens=5)
    mock_response.stop_reason = 'end_turn'
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)
    await model.generate(request)

    sent_tools = mock_client.messages.create.call_args.kwargs['tools']
    default_schema = {'type': 'object', 'properties': {}}
    assert sent_tools[0]['input_schema'] == default_schema
    assert sent_tools[1]['input_schema'] == default_schema
    assert sent_tools[2]['input_schema'] == {'properties': {'location': {'type': 'string'}}, 'type': 'object'}
    assert sent_tools[3]['input_schema'] == populated_schema


@pytest.mark.asyncio
async def test_generate_with_config() -> None:
    """Test generation with custom config."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type='text', text='Response')]
    mock_response.usage = MagicMock(input_tokens=5, output_tokens=5)
    mock_response.stop_reason = 'end_turn'

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
        config=ModelConfig(
            temperature=0.7,
            max_output_tokens=100,
            top_p=0.9,
        ),
    )

    await model.generate(request)

    call_args = mock_client.messages.create.call_args
    assert call_args.kwargs['temperature'] == 0.7
    assert call_args.kwargs['max_tokens'] == 100
    assert call_args.kwargs['top_p'] == 0.9


def test_extract_system() -> None:
    """Test system prompt extraction."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='You are helpful.'))]),
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
    ]

    system = model._extract_system(messages)
    assert system == 'You are helpful.'


def test_to_anthropic_messages() -> None:
    """Test message conversion."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
        Message(role=Role.MODEL, content=[Part(root=TextPart(text='Hi there'))]),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)

    assert len(anthropic_messages) == 2
    assert anthropic_messages[0]['role'] == 'user'
    assert anthropic_messages[0]['content'][0]['text'] == 'Hello'
    assert anthropic_messages[1]['role'] == 'assistant'
    assert anthropic_messages[1]['content'][0]['text'] == 'Hi there'


class MockStreamManager:
    """Mock stream manager for testing Anthropic streaming."""

    def __init__(self, chunks: list[Any], final_content: list[Any] | None = None) -> None:
        """Initialize the MockStreamManager."""
        self.chunks = chunks
        self.final_message = MagicMock()
        self.final_message.content = final_content if final_content else []
        self.final_message.usage = MagicMock(input_tokens=10, output_tokens=20)
        self.final_message.stop_reason = 'end_turn'

    async def __aenter__(self) -> 'MockStreamManager':
        """Enter the async context manager."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context manager."""
        pass

    def __aiter__(self) -> 'MockStreamManager':
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Return the next chunk from the stream."""
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    async def get_final_message(self) -> object:
        """Get the final message from the stream."""
        return self.final_message


@pytest.mark.asyncio
async def test_streaming_generation() -> None:
    """Test streaming generation."""
    sample_request = _create_sample_request()

    mock_client = MagicMock()

    chunks = [
        MagicMock(type='content_block_delta', delta=MagicMock(type='text_delta', text='Hello')),
        MagicMock(type='content_block_delta', delta=MagicMock(type='text_delta', text=' world')),
        MagicMock(type='content_block_delta', delta=MagicMock(type='text_delta', text='!')),
    ]

    final_content = [MagicMock(type='text', text='Hello world!')]
    mock_stream = MockStreamManager(chunks, final_content=final_content)
    mock_client.messages.stream.return_value = mock_stream

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    ctx = MagicMock()
    ctx.is_streaming = True
    collected_chunks: list[ModelResponseChunk] = []

    def send_chunk(chunk: ModelResponseChunk) -> None:
        collected_chunks.append(chunk)

    ctx.send_chunk = send_chunk

    response = await model.generate(sample_request, ctx)

    assert len(collected_chunks) == 3
    chunk0_part = collected_chunks[0].content[0]
    chunk0_actual = chunk0_part.root if isinstance(chunk0_part, Part) else chunk0_part
    assert chunk0_actual.text == 'Hello'

    chunk1_part = collected_chunks[1].content[0]
    chunk1_actual = chunk1_part.root if isinstance(chunk1_part, Part) else chunk1_part
    assert chunk1_actual.text == ' world'

    chunk2_part = collected_chunks[2].content[0]
    chunk2_actual = chunk2_part.root if isinstance(chunk2_part, Part) else chunk2_part
    assert chunk2_actual.text == '!'

    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 20

    # Verify final response content is populated
    assert response.message is not None
    assert len(response.message.content) == 1
    final_part = response.message.content[0]
    assert isinstance(final_part, Part)
    assert isinstance(final_part.root, TextPart)
    assert final_part.root.text == 'Hello world!'


@pytest.mark.asyncio
async def test_streaming_tool_request() -> None:
    """Test streaming generation with tool use blocks."""
    sample_request = _create_sample_request()

    mock_client = MagicMock()

    # Simulate: text chunk, then tool_use block (start + json deltas + stop).
    tool_block = MagicMock(type='tool_use', id='tool_abc')
    tool_block.name = 'get_weather'
    chunks = [
        MagicMock(type='content_block_delta', delta=MagicMock(type='text_delta', text='Let me check.')),
        MagicMock(type='content_block_start', index=1, content_block=tool_block),
        MagicMock(
            type='content_block_delta',
            index=1,
            delta=MagicMock(type='input_json_delta', partial_json='{"location"'),
        ),
        MagicMock(
            type='content_block_delta',
            index=1,
            delta=MagicMock(type='input_json_delta', partial_json=': "Paris"}'),
        ),
        MagicMock(type='content_block_stop', index=1),
    ]

    final_tool = MagicMock(type='tool_use', id='tool_abc', input={'location': 'Paris'})
    final_tool.name = 'get_weather'
    final_content = [
        MagicMock(type='text', text='Let me check.'),
        final_tool,
    ]
    mock_stream = MockStreamManager(chunks, final_content=final_content)
    mock_client.messages.stream.return_value = mock_stream

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    ctx = MagicMock()
    ctx.is_streaming = True
    collected_chunks: list[ModelResponseChunk] = []
    ctx.send_chunk = lambda chunk: collected_chunks.append(chunk)

    response = await model.generate(sample_request, ctx)

    # Should have 2 chunks: one text, one tool request.
    assert len(collected_chunks) == 2

    text_part = collected_chunks[0].content[0].root
    assert isinstance(text_part, TextPart)
    assert text_part.text == 'Let me check.'

    tool_part = collected_chunks[1].content[0].root
    assert isinstance(tool_part, ToolRequestPart)
    assert tool_part.tool_request.name == 'get_weather'
    assert tool_part.tool_request.ref == 'tool_abc'
    assert tool_part.tool_request.input == {'location': 'Paris'}

    # Final response should also contain the tool request.
    assert response.message is not None
    assert len(response.message.content) == 2


class TestStripMarkdownFences:
    """Tests for strip_markdown_fences."""

    def test_strips_json_fences(self) -> None:
        """Strips ```json ... ``` fences."""
        text = '```json\n{"name": "John", "age": 30}\n```'
        assert strip_markdown_fences(text) == '{"name": "John", "age": 30}'

    def test_strips_plain_fences(self) -> None:
        """Strips ``` ... ``` fences without language tag."""
        text = '```\n{"name": "John"}\n```'
        assert strip_markdown_fences(text) == '{"name": "John"}'

    def test_strips_fences_with_surrounding_whitespace(self) -> None:
        """Strips fences even with leading/trailing whitespace."""
        text = '  \n```json\n{"a": 1}\n```\n  '
        assert strip_markdown_fences(text) == '{"a": 1}'

    def test_preserves_plain_json(self) -> None:
        """Does not alter valid JSON without fences."""
        text = '{"name": "John", "age": 30}'
        assert strip_markdown_fences(text) == text

    def test_preserves_non_json_text(self) -> None:
        """Does not alter plain text."""
        text = 'Hello, world!'
        assert strip_markdown_fences(text) == text

    def test_strips_multiline_json_in_fences(self) -> None:
        """Strips fences around multiline JSON."""
        text = '```json\n{\n  "name": "John",\n  "age": 30\n}\n```'
        result = strip_markdown_fences(text)
        assert result == '{\n  "name": "John",\n  "age": 30\n}'


class TestMaybeStripFences:
    """Tests for maybe_strip_fences."""

    def test_strips_fences_for_json_output(self) -> None:
        """Strips markdown fences when JSON output is requested."""
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output_format='json',
            output_schema={'type': 'object'},
        )
        parts = [Part(root=TextPart(text='```json\n{"a": 1}\n```'))]
        result = maybe_strip_fences(request, parts)
        assert result[0].root.text == '{"a": 1}'

    def test_no_op_for_text_output(self) -> None:
        """Does not modify responses when output format is not json."""
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output_format='text',
        )
        fenced = '```json\n{"a": 1}\n```'
        parts = [Part(root=TextPart(text=fenced))]
        result = maybe_strip_fences(request, parts)
        assert result[0].root.text == fenced

    def test_no_op_for_no_output(self) -> None:
        """Does not modify responses when no output config is set."""
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
        )
        fenced = '```json\n{"a": 1}\n```'
        parts = [Part(root=TextPart(text=fenced))]
        result = maybe_strip_fences(request, parts)
        assert result[0].root.text == fenced

    def test_no_op_when_no_fences(self) -> None:
        """Does not modify clean JSON responses."""
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output_format='json',
            output_schema={'type': 'object'},
        )
        text = '{"name": "John"}'
        parts = [Part(root=TextPart(text=text))]
        result = maybe_strip_fences(request, parts)
        assert result is parts


def test_cache_control_on_text_block() -> None:
    """Test that cache_control metadata is forwarded to Anthropic blocks."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(root=TextPart(text='Cached context', metadata=Metadata({'cache_control': {'type': 'ephemeral'}}))),
                Part(root=TextPart(text='Question about the context')),
            ],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)

    assert len(anthropic_messages) == 1
    blocks = anthropic_messages[0]['content']
    assert len(blocks) == 2

    # First block should have cache_control.
    assert blocks[0]['type'] == 'text'
    assert blocks[0]['text'] == 'Cached context'
    assert blocks[0]['cache_control'] == {'type': 'ephemeral'}

    # Second block should not have cache_control.
    assert blocks[1]['type'] == 'text'
    assert 'cache_control' not in blocks[1]


def test_cache_control_not_applied_without_metadata() -> None:
    """Test that no cache_control is applied when metadata is absent."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(
            role=Role.USER,
            content=[Part(root=TextPart(text='No cache'))],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)
    blocks = anthropic_messages[0]['content']
    assert 'cache_control' not in blocks[0]


@pytest.mark.asyncio
async def test_cache_token_tracking_in_usage() -> None:
    """Test that cache creation/read tokens are included in usage."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type='text', text='Cached response')]
    mock_response.usage = MagicMock(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=80,
        cache_read_input_tokens=20,
    )
    mock_response.stop_reason = 'end_turn'

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
    )

    response = await model.generate(request)

    assert response.usage is not None
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 50
    assert response.usage.custom is not None
    assert response.usage.custom['cache_creation_input_tokens'] == 80
    assert response.usage.custom['cache_read_input_tokens'] == 20


@pytest.mark.asyncio
async def test_no_cache_tokens_when_caching_not_used() -> None:
    """Test that custom is None when no cache tokens are present."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type='text', text='Response')]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    mock_response.stop_reason = 'end_turn'
    # Simulate no cache attributes.
    del mock_response.usage.cache_creation_input_tokens
    del mock_response.usage.cache_read_input_tokens

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
    )

    response = await model.generate(request)
    assert response.usage is not None
    assert response.usage.custom is None


def test_pdf_base64_becomes_document_block() -> None:
    """Test that a base64 PDF MediaPart converts to Anthropic document block."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    pdf_data = 'data:application/pdf;base64,JVBERi0xLjQ='
    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(root=MediaPart(media=Media(url=pdf_data, content_type='application/pdf'))),
                Part(root=TextPart(text='Summarize this PDF')),
            ],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)
    blocks = anthropic_messages[0]['content']

    assert blocks[0]['type'] == 'document'
    assert blocks[0]['source']['type'] == 'base64'
    assert blocks[0]['source']['media_type'] == 'application/pdf'
    assert blocks[0]['source']['data'] == 'JVBERi0xLjQ='

    assert blocks[1]['type'] == 'text'
    assert blocks[1]['text'] == 'Summarize this PDF'


def test_pdf_url_becomes_document_block() -> None:
    """Test that a URL-based PDF converts to Anthropic document block."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(
                    root=MediaPart(
                        media=Media(
                            url='https://example.com/doc.pdf',
                            content_type='application/pdf',
                        )
                    )
                ),
            ],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)
    blocks = anthropic_messages[0]['content']

    assert blocks[0]['type'] == 'document'
    assert blocks[0]['source']['type'] == 'url'
    assert blocks[0]['source']['url'] == 'https://example.com/doc.pdf'


def test_image_still_works() -> None:
    """Test that non-document images still produce image blocks."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(root=MediaPart(media=Media(url='https://example.com/cat.jpg', content_type='image/jpeg'))),
            ],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)
    blocks = anthropic_messages[0]['content']

    assert blocks[0]['type'] == 'image'
    assert blocks[0]['source']['type'] == 'url'


def test_pdf_with_cache_control() -> None:
    """Test that cache_control can be applied to document blocks."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    pdf_data = 'data:application/pdf;base64,JVBERi0xLjQ='
    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(
                    root=MediaPart(
                        media=Media(url=pdf_data, content_type='application/pdf'),
                        metadata=Metadata({'cache_control': {'type': 'ephemeral'}}),
                    )
                ),
            ],
        ),
    ]

    anthropic_messages = model._to_anthropic_messages(messages)
    blocks = anthropic_messages[0]['content']

    assert blocks[0]['type'] == 'document'
    assert blocks[0]['cache_control'] == {'type': 'ephemeral'}


@pytest.mark.parametrize('model_name', ['claude-opus-4-6', 'claude-opus-4-7', 'claude-opus-4-8'])
def test_structured_output_uses_native_output_config(model_name: str) -> None:
    """Test that JSON schema uses native output_config when model supports it."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name=model_name, client=mock_client)

    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a cat'))])],
        output_format='json',
        output_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
        output_constrained=True,
    )

    params = model._build_params(request)

    assert 'output_config' in params
    assert params['output_config']['format']['type'] == 'json_schema'
    assert params['output_config']['format']['schema']['additionalProperties'] is False


def test_structured_output_uses_native_output_config_for_empty_schema() -> None:
    """Test that an empty, but present, schema enables native structured output."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-opus-4-6', client=mock_client)

    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate JSON'))])],
        output_format='json',
        output_schema={},
        output_constrained=True,
    )

    params = model._build_params(request)

    assert params['output_config']['format'] == {'type': 'json_schema', 'schema': {}}


def test_structured_output_falls_back_to_system_prompt() -> None:
    """Test that JSON without schema falls back to system prompt instruction."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-opus-4-6', client=mock_client)

    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate JSON'))])],
        output_format='json',
        output_constrained=True,
    )

    params = model._build_params(request)

    assert 'output_config' not in params
    assert 'system' in params
    assert 'Output valid JSON' in params['system']


@pytest.mark.parametrize('output_constrained', [None, False])
def test_structured_output_falls_back_when_unconstrained(output_constrained: bool | None) -> None:
    """Test that callers can opt out of native constrained output."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-opus-4-6', client=mock_client)

    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a cat'))])],
        output_format='json',
        output_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
        output_constrained=output_constrained,
    )

    params = model._build_params(request)

    assert 'output_config' not in params
    assert 'Output valid JSON' in params['system']
    assert 'Follow this JSON schema' in params['system']
    assert '"name"' in params['system']


def test_structured_output_falls_back_for_unsupported_models() -> None:
    """Test that JSON with schema falls back to system prompt for unsupported models."""
    mock_client = MagicMock()
    # Unknown models resolve to the generic fallback in model_info.py, whose
    # supports.constrained is unset — no native constrained-output support.
    model = AnthropicModel(model_name='claude-unknown-model', client=mock_client)

    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a cat'))])],
        output_format='json',
        output_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
        output_constrained=True,
    )

    params = model._build_params(request)

    assert 'output_config' not in params
    assert 'system' in params
    assert 'Output valid JSON' in params['system']
    assert 'Follow this JSON schema' in params['system']
    assert '"name"' in params['system']


def test_structured_output_falls_back_when_model_disallows_constraints() -> None:
    """Test that an explicit constrained=none capability disables native output."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-opus-4-6', client=mock_client)
    model._model_info = ModelInfo(label='Test model', supports=Supports(constrained=Constrained.NONE))

    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a cat'))])],
        output_format='json',
        output_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
        output_constrained=True,
    )

    params = model._build_params(request)

    assert 'output_config' not in params
    assert 'Output valid JSON' in params['system']


def test_structured_output_with_no_tools_capability() -> None:
    """Test that no-tools constrained output is disabled only when tools are present."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-opus-4-6', client=mock_client)
    model._model_info = ModelInfo(label='Test model', supports=Supports(constrained=Constrained.NO_TOOLS))

    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a cat'))])],
        output_format='json',
        output_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
        output_constrained=True,
    )
    params_without_tools = model._build_params(request)

    request_with_tools = request.model_copy(
        update={
            'tools': [
                ToolDefinition(
                    name='get_weather',
                    description='Get weather for a location',
                    input_schema={'type': 'object'},
                )
            ]
        }
    )
    params_with_tools = model._build_params(request_with_tools)

    assert 'output_config' in params_without_tools
    assert 'output_config' not in params_with_tools
    assert 'Output valid JSON' in params_with_tools['system']


# --- typed config (AnthropicConfig) ----------------------------------------


def _mock_client_for_generate() -> MagicMock:
    """A client whose messages.create returns a minimal text response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type='text', text='ok')]
    mock_response.usage = MagicMock(input_tokens=1, output_tokens=1)
    mock_response.stop_reason = 'end_turn'
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    mock_client.beta.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


def _text_request(config: Any) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
        config=config,
    )


@pytest.mark.asyncio
async def test_dict_config_unknown_key_reaches_sdk() -> None:
    """Unknown extra keys in a dict config pass through the SDK body escape hatch."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    await model.generate(_text_request({'temperature': 0.3, 'future_option': 'x'}))

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs['temperature'] == 0.3
    assert kwargs['extra_body'] == {'future_option': 'x'}


@pytest.mark.asyncio
async def test_typed_config_thinking_translated_for_sdk() -> None:
    """A typed thinking config is translated to the SDK's snake_case shape."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    config = AnthropicConfig.model_validate({'thinking': {'enabled': True, 'budgetTokens': 2048}})
    await model.generate(_text_request(config))

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs['thinking'] == {'type': 'enabled', 'budget_tokens': 2048}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'thinking, expected',
    [
        ({'adaptive': True, 'display': 'summarized'}, {'type': 'adaptive', 'display': 'summarized'}),
        ({'enabled': False}, {'type': 'disabled'}),
        ({'budgetTokens': 2048}, {'type': 'enabled', 'budget_tokens': 2048}),
    ],
)
async def test_typed_config_thinking_variants_translated_for_sdk(
    thinking: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Advertised thinking variants are translated to the SDK shape."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    config = AnthropicConfig.model_validate({'thinking': thinking})
    await model.generate(_text_request(config))

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs['thinking'] == expected


@pytest.mark.asyncio
async def test_beta_config_uses_beta_sdk_and_sends_betas() -> None:
    """apiVersion / betas route through the beta SDK surface."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    config = AnthropicConfig.model_validate({
        'apiVersion': 'beta',
        'betas': ['token-efficient-tools-2025'],
    })
    await model.generate(_text_request(config))

    mock_client.messages.create.assert_not_called()
    kwargs = mock_client.beta.messages.create.call_args.kwargs
    assert 'api_version' not in kwargs
    assert 'apiVersion' not in kwargs
    assert kwargs['betas'] == ['token-efficient-tools-2025']


@pytest.mark.asyncio
async def test_api_key_does_not_reach_sdk_params() -> None:
    """apiKey is a client override and is never passed as a messages kwarg."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    config = AnthropicConfig.model_validate({'apiKey': 'secret'})
    await model.generate(_text_request(config))

    kwargs = mock_client.messages.create.call_args.kwargs
    assert 'api_key' not in kwargs
    assert 'apiKey' not in kwargs


def test_api_key_config_overrides_real_sdk_client() -> None:
    """apiKey yields a request-scoped copy that keeps client settings and transport."""
    base_client = AsyncAnthropic(api_key='base-key', default_headers={'X-Custom': 'yes'})
    model = AnthropicModel(model_name='claude-sonnet-4', client=base_client)

    client = model._client_for_config(AnthropicConfig.model_validate({'apiKey': 'request-key'}))

    assert client is not base_client
    assert isinstance(client, AsyncAnthropic)
    assert client.api_key == 'request-key'
    assert client.default_headers.get('X-Custom') == 'yes'
    assert client._client is base_client._client


def test_build_params_consumes_client_level_keys_silently() -> None:
    """apiVersion/apiKey are honored elsewhere and must not be logged as ignored."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    with patch.object(anthropic_models, 'logger') as mock_logger:
        params = model._build_params(_text_request({'apiVersion': 'beta', 'apiKey': 'request-key'}))

    mock_logger.warning.assert_not_called()
    assert 'api_version' not in params
    assert 'api_key' not in params


@pytest.mark.asyncio
async def test_invalid_config_raises_from_generate() -> None:
    """An invalid dict config surfaces a validation error from generate()."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    with pytest.raises(ValidationError):
        await model.generate(_text_request({'thinking': {'enabled': True}}))

    mock_client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_config_tool_choice_and_metadata_reach_sdk() -> None:
    """Config-level tool_choice and metadata reach the SDK kwargs."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    config = AnthropicConfig.model_validate({
        'tool_choice': {'type': 'tool', 'name': 'get_weather'},
        'metadata': {'user_id': 'user-123'},
        'tools': [{'name': 'get_weather', 'description': 'Weather', 'input_schema': {'type': 'object'}}],
    })
    await model.generate(_text_request(config))

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs['tool_choice'] == {'type': 'tool', 'name': 'get_weather'}
    assert kwargs['metadata'] == {'user_id': 'user-123'}


@pytest.mark.asyncio
async def test_config_tool_choice_none_reaches_sdk() -> None:
    """Config-level tool_choice none remains valid for dict compatibility."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    await model.generate(
        _text_request({
            'tool_choice': {'type': 'none'},
            'tools': [{'name': 'get_weather', 'description': 'Weather', 'input_schema': {'type': 'object'}}],
        })
    )

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs['tool_choice'] == {'type': 'none'}


@pytest.mark.asyncio
async def test_config_tool_choice_dropped_without_tools() -> None:
    """Config-level tool_choice is dropped when the request carries no tools."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    await model.generate(_text_request({'tool_choice': {'type': 'auto'}}))

    kwargs = mock_client.messages.create.call_args.kwargs
    assert 'tool_choice' not in kwargs


def test_structured_output_merges_existing_output_config() -> None:
    """Native structured output keeps user-supplied output_config options."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-opus-4-6', client=mock_client)
    config: Any = AnthropicConfig.model_validate({'output_config': {'effort': 'high', 'task_budget': {'total': 20000}}})

    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Generate a cat'))])],
        output_format='json',
        output_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
        config=config,
        output_constrained=True,
    )

    params = model._build_params(request)

    assert params['output_config']['effort'] == 'high'
    assert params['output_config']['task_budget'] == {'type': 'tokens', 'total': 20000}
    assert params['output_config']['format']['type'] == 'json_schema'


def test_config_version_overrides_model_name() -> None:
    """Per-request version maps to the Anthropic model parameter."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    params = model._build_params(_text_request({'version': 'claude-sonnet-4-20260101'}))

    assert params['model'] == 'claude-sonnet-4-20260101'


def test_backward_compat_plain_model_config() -> None:
    """A plain ModelConfig still maps to the same SDK params (no behavior change)."""
    mock_client = MagicMock()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    params = model._build_params(_text_request(ModelConfig(temperature=0.7, max_output_tokens=100, top_p=0.9)))

    assert params['temperature'] == 0.7
    assert params['max_tokens'] == 100
    assert params['top_p'] == 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('config', 'kwarg'),
    [
        ({'speed': 'fast'}, 'speed'),
        ({'mcp_servers': [{'type': 'url', 'name': 'x', 'url': 'https://example.com'}]}, 'mcp_servers'),
        ({'context_management': {'edits': []}}, 'context_management'),
    ],
)
async def test_beta_only_params_select_beta_surface(config: dict, kwarg: str) -> None:
    """Beta-only params route to the beta surface instead of crashing the stable one."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    await model.generate(_text_request(config))

    mock_client.messages.create.assert_not_called()
    kwargs = mock_client.beta.messages.create.call_args.kwargs
    assert kwargs[kwarg] == config[kwarg]
    assert 'extra_body' not in kwargs


@pytest.mark.asyncio
async def test_task_budget_selects_beta_surface() -> None:
    """output_config.task_budget is beta-only and must not ship on the stable surface."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    await model.generate(_text_request({'output_config': {'task_budget': {'total': 20000}}}))

    mock_client.messages.create.assert_not_called()
    kwargs = mock_client.beta.messages.create.call_args.kwargs
    assert kwargs['output_config']['task_budget'] == {'type': 'tokens', 'total': 20000}


@pytest.mark.asyncio
async def test_unknown_params_still_route_to_extra_body_on_beta() -> None:
    """The escape hatch keeps working once the beta surface is selected."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    await model.generate(_text_request({'speed': 'fast', 'future_option': 'x'}))

    kwargs = mock_client.beta.messages.create.call_args.kwargs
    assert kwargs['speed'] == 'fast'
    assert kwargs['extra_body'] == {'future_option': 'x'}


@pytest.mark.asyncio
async def test_config_stream_does_not_reach_sdk() -> None:
    """Genkit owns streaming, so a config-level stream flag is dropped."""
    mock_client = _mock_client_for_generate()
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    await model.generate(_text_request({'stream': True}))

    kwargs = mock_client.messages.create.call_args.kwargs
    assert 'stream' not in kwargs
    assert 'stream' not in (kwargs.get('extra_body') or {})


@pytest.mark.parametrize(
    ('stop_reason', 'expected'),
    [
        ('end_turn', FinishReason.STOP),
        ('max_tokens', FinishReason.LENGTH),
        ('model_context_window_exceeded', FinishReason.LENGTH),
        ('refusal', FinishReason.BLOCKED),
        ('pause_turn', FinishReason.OTHER),
        ('compaction', FinishReason.OTHER),
        ('something_new', FinishReason.UNKNOWN),
    ],
)
@pytest.mark.asyncio
async def test_finish_reason_mapping(stop_reason: str, expected: FinishReason) -> None:
    """Anthropic stop reasons map onto Genkit finish reasons."""
    mock_client = _mock_client_for_generate()
    mock_client.messages.create.return_value.stop_reason = stop_reason
    model = AnthropicModel(model_name='claude-sonnet-4', client=mock_client)

    response = await model.generate(_text_request({}))

    assert response.finish_reason == expected


def test_per_request_api_key_ignored_when_client_uses_auth_token() -> None:
    """An auth-token client cannot be re-credentialed by copy(), so the key is ignored."""
    client = AsyncAnthropic(auth_token='corp-bearer')
    model = AnthropicModel(model_name='claude-sonnet-4', client=client)

    assert model._client_for_config(AnthropicConfig.model_validate({'apiKey': 'user-key'})) is client


def test_per_request_api_key_ignored_when_client_pins_api_key_header() -> None:
    """A pinned x-api-key header outranks copy(api_key=...), so the key is ignored."""
    client = AsyncAnthropic(api_key='plugin-key', default_headers={'X-Api-Key': 'pinned'})
    model = AnthropicModel(model_name='claude-sonnet-4', client=client)

    assert model._client_for_config(AnthropicConfig.model_validate({'apiKey': 'user-key'})) is client


def test_per_request_api_key_applied_on_plain_client() -> None:
    """A plain api-key client is re-credentialed for the request."""
    client = AsyncAnthropic(api_key='plugin-key')
    model = AnthropicModel(model_name='claude-sonnet-4', client=client)

    applied = model._client_for_config(AnthropicConfig.model_validate({'apiKey': 'user-key'}))

    assert applied is not client
    assert cast(AsyncAnthropic, applied).api_key == 'user-key'


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        (
            {'enabled': True, 'budgetTokens': 2048, 'display': 'omitted'},
            {'display': 'omitted', 'type': 'enabled', 'budget_tokens': 2048},
        ),
        ({'adaptive': True, 'display': 'summarized'}, {'display': 'summarized', 'type': 'adaptive'}),
        ({'type': 'interleaved'}, {'type': 'interleaved'}),
    ],
)
def test_thinking_preserves_display_and_forward_compatible_keys(raw: dict, expected: dict) -> None:
    """display and unknown thinking keys survive translation to the SDK shape."""
    thinking = AnthropicConfig.model_validate({'thinking': raw}).model_dump(exclude_none=True, by_alias=False)[
        'thinking'
    ]

    assert _to_anthropic_thinking_config(thinking) == expected


@pytest.mark.parametrize('raw', [{'display': 'summarized'}, {}])
def test_thinking_dropped_when_no_mode_is_set(raw: dict) -> None:
    """A thinking config with no mode has no SDK type, so it is dropped rather than sent."""
    thinking = AnthropicConfig.model_validate({'thinking': raw}).model_dump(exclude_none=True, by_alias=False)[
        'thinking'
    ]

    assert _to_anthropic_thinking_config(thinking) is None
