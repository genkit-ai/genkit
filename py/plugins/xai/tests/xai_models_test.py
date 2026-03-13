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

"""Tests for xAI models."""

import asyncio
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from genkit.plugins.xai.models import XAIConfig, XAIModel
from genkit.types import (
    GenerateRequest,
    GenerateResponseChunk,
    Message,
    OutputConfig,
    Part,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequestPart,
)


def _create_sample_request() -> GenerateRequest:
    """Create a sample generation request for testing."""
    return GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='Hello, how are you?'))],
            )
        ],
        config=XAIConfig(),
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

    mock_response = MagicMock()
    mock_response.content = "Hello! I'm doing well."
    mock_response.finish_reason = 'STOP'
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=15,
        total_tokens=25,
    )
    mock_response.tool_calls = None

    mock_chat = MagicMock()
    mock_chat.sample = MagicMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.create = MagicMock(return_value=mock_chat)

    model = XAIModel(model_name='grok-3', client=mock_client)
    response = await model.generate(sample_request)

    assert response.message
    assert response.message.content
    assert len(response.message.content) == 1
    part = response.message.content[0]
    actual_part = part.root if isinstance(part, Part) else part
    assert isinstance(actual_part, TextPart)
    assert actual_part.text == "Hello! I'm doing well."
    assert response.usage
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 15
    assert response.finish_reason == 'stop'


@pytest.mark.asyncio
async def test_generate_with_config() -> None:
    """Test generation with config."""
    mock_response = MagicMock()
    mock_response.content = 'Response'
    mock_response.finish_reason = 'STOP'
    mock_response.usage = MagicMock(
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
    )
    mock_response.tool_calls = None

    mock_chat = MagicMock()
    mock_chat.sample = MagicMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.create = MagicMock(return_value=mock_chat)

    model = XAIModel(model_name='grok-3', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
        config=XAIConfig(
            temperature=0.7,
            max_output_tokens=100,
            top_p=0.9,
        ),
    )

    await model.generate(request)

    call_args = mock_client.chat.create.call_args
    assert call_args.kwargs['temperature'] == 0.7
    assert call_args.kwargs['max_tokens'] == 100
    assert call_args.kwargs['top_p'] == 0.9


def test_to_xai_messages() -> None:
    """Test xAI messages conversion."""
    mock_client = MagicMock()
    model = XAIModel(model_name='grok-3', client=mock_client)

    messages = [
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
        Message(role=Role.MODEL, content=[Part(root=TextPart(text='Hi there'))]),
    ]

    xai_messages = model._to_xai_messages(messages)
    assert len(xai_messages) == 2


def test_to_genkit_content() -> None:
    """Test content conversion."""
    mock_client = MagicMock()
    model = XAIModel(model_name='grok-3', client=mock_client)

    mock_response = MagicMock()
    mock_response.content = 'Test response'
    mock_response.tool_calls = None

    content = model._to_genkit_content(mock_response)
    assert len(content) == 1
    part = content[0]
    actual_part = part.root if isinstance(part, Part) else part
    assert isinstance(actual_part, TextPart)
    assert actual_part.text == 'Test response'


@pytest.mark.asyncio
async def test_streaming_generation() -> None:
    """Test streaming generation."""
    sample_request = _create_sample_request()

    mock_chunk1 = MagicMock()
    mock_chunk1.content = 'Hello'
    mock_chunk1.choices = []

    mock_chunk2 = MagicMock()
    mock_chunk2.content = ' world'
    mock_chunk2.choices = []

    mock_chunk3 = MagicMock()
    mock_chunk3.content = '!'
    mock_chunk3.choices = []

    mock_response = MagicMock()
    mock_response.finish_reason = 'STOP'
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )

    def mock_stream() -> Iterator:
        yield mock_response, mock_chunk1
        yield mock_response, mock_chunk2
        yield mock_response, mock_chunk3

    mock_chat = MagicMock()
    mock_chat.stream = MagicMock(return_value=mock_stream())

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.create = MagicMock(return_value=mock_chat)

    model = XAIModel(model_name='grok-3', client=mock_client)

    ctx = MagicMock()
    ctx.is_streaming = True
    collected_chunks = []

    def send_chunk(chunk: GenerateResponseChunk) -> None:
        collected_chunks.append(chunk)

    ctx.send_chunk = send_chunk

    response = await model.generate(sample_request, ctx)

    # Allow event loop to process call_soon_threadsafe callbacks
    await asyncio.sleep(0.1)

    assert len(collected_chunks) == 3
    assert response.usage
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 20
    assert response.finish_reason == 'stop'

    accumulated_text = ''
    assert response.message
    assert response.message.content
    for part in response.message.content:
        actual_part = part.root if isinstance(part, Part) else part
        if isinstance(actual_part, TextPart):
            accumulated_text += actual_part.text

    assert accumulated_text == 'Hello world!'


@pytest.mark.asyncio
async def test_generate_with_tools() -> None:
    """Test generation with tools."""
    sample_request = _create_sample_request()

    mock_tool_call = MagicMock()
    mock_tool_call.id = 'tool_123'
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = 'get_weather'
    mock_tool_call.function.arguments = '{"location": "Paris"}'

    mock_response = MagicMock()
    mock_response.content = None
    mock_response.finish_reason = 'TOOL_CALLS'
    mock_response.usage = MagicMock(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    )
    mock_response.tool_calls = [mock_tool_call]

    mock_chat = MagicMock()
    mock_chat.sample = MagicMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.create = MagicMock(return_value=mock_chat)

    model = XAIModel(model_name='grok-3', client=mock_client)
    response = await model.generate(sample_request)

    assert response.message
    assert response.message.content
    assert len(response.message.content) == 1
    part = response.message.content[0]
    actual_part = part.root if isinstance(part, Part) else part
    assert isinstance(actual_part, ToolRequestPart)
    assert actual_part.tool_request.name == 'get_weather'
    assert actual_part.tool_request.ref == 'tool_123'


@pytest.mark.asyncio
async def test_build_params_basic() -> None:
    """Test parameters build."""
    mock_client = MagicMock()
    model = XAIModel(model_name='grok-3', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
        config=XAIConfig(),
    )

    params = model._build_params(request)

    assert params['model'] == 'grok-3'
    assert 'messages' in params
    assert params['max_tokens'] == 4096


@pytest.mark.asyncio
async def test_build_params_with_config() -> None:
    """Test parameters build with config."""
    mock_client = MagicMock()
    model = XAIModel(model_name='grok-3', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
        config=XAIConfig(
            temperature=0.5,
            max_output_tokens=200,
            top_p=0.8,
        ),
    )

    params = model._build_params(request)

    assert params['temperature'] == 0.5
    assert params['max_tokens'] == 200
    assert params['top_p'] == 0.8


@pytest.mark.asyncio
async def test_build_params_with_xai_specific_config() -> None:
    """Test parameters build with xAI config."""
    mock_client = MagicMock()
    model = XAIModel(model_name='grok-3', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
        config=XAIConfig(
            temperature=0.7,
            max_output_tokens=300,
            deferred=True,
            reasoning_effort='high',
            web_search_options={'enabled': True},
        ),
    )

    params = model._build_params(request)

    assert params['temperature'] == 0.7
    assert params['max_tokens'] == 300
    assert params['deferred'] is True
    assert params['reasoning_effort'] == 'high'
    assert params['web_search_options'] == {'enabled': True}


@pytest.mark.asyncio
async def test_to_genkit_content_parses_json_arguments() -> None:
    """Test content conversion with JSON arguments."""
    mock_client = MagicMock()
    model = XAIModel(model_name='grok-3', client=mock_client)

    mock_tool_call = MagicMock()
    mock_tool_call.id = 'call_123'
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = 'get_weather'
    mock_tool_call.function.arguments = '{"location": "Paris", "unit": "celsius"}'

    mock_response = MagicMock()
    mock_response.content = 'Some response'
    mock_response.tool_calls = [mock_tool_call]

    content = model._to_genkit_content(mock_response)

    assert len(content) == 2
    part0 = content[0].root
    assert isinstance(part0, TextPart)
    assert part0.text == 'Some response'
    part1 = content[1].root
    assert isinstance(part1, ToolRequestPart)
    assert part1.tool_request.name == 'get_weather'
    assert isinstance(part1.tool_request.input, dict)
    assert part1.tool_request.input['location'] == 'Paris'
    assert part1.tool_request.input['unit'] == 'celsius'


def test_build_params_json_output_with_schema() -> None:
    """Test that _build_params sets FORMAT_TYPE_JSON_SCHEMA when schema is provided."""
    from xai_sdk.proto.v6 import chat_pb2

    mock_client = MagicMock()
    model = XAIModel(model_name='grok-3', client=mock_client)

    schema = {
        'type': 'object',
        'properties': {'name': {'type': 'string'}, 'age': {'type': 'number'}},
        'required': ['name', 'age'],
    }
    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
        config=XAIConfig(),
        output=OutputConfig(format='json', schema=schema),
    )

    params = model._build_params(request)

    response_format = params['response_format']
    assert isinstance(response_format, chat_pb2.ResponseFormat)
    assert response_format.format_type == chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA
    assert response_format.schema  # schema string is set


def test_build_params_json_output_without_schema() -> None:
    """Test that _build_params sets FORMAT_TYPE_JSON_OBJECT when no schema."""
    from xai_sdk.proto.v6 import chat_pb2

    mock_client = MagicMock()
    model = XAIModel(model_name='grok-3', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
        config=XAIConfig(),
        output=OutputConfig(format='json'),
    )

    params = model._build_params(request)

    response_format = params['response_format']
    assert isinstance(response_format, chat_pb2.ResponseFormat)
    assert response_format.format_type == chat_pb2.FormatType.FORMAT_TYPE_JSON_OBJECT


def test_build_params_no_output() -> None:
    """Test that _build_params omits response_format when no output config."""
    mock_client = MagicMock()
    model = XAIModel(model_name='grok-3', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Test'))])],
        config=XAIConfig(),
    )

    params = model._build_params(request)

    assert 'response_format' not in params


@pytest.mark.asyncio
async def test_streaming_tool_calls_emit_chunks() -> None:
    """Test that tool call parts are emitted as chunks during streaming."""
    mock_tool_call = MagicMock()
    mock_tool_call.id = 'call_42'
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = 'get_weather'
    mock_tool_call.function.arguments = '{"city": "NYC"}'

    mock_choice = MagicMock()
    mock_choice.tool_calls = [mock_tool_call]

    mock_chunk = MagicMock()
    mock_chunk.content = None
    mock_chunk.choices = [mock_choice]

    mock_response = MagicMock()
    mock_response.finish_reason = 'TOOL_CALLS'
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    def mock_stream() -> Iterator:
        yield mock_response, mock_chunk

    mock_chat = MagicMock()
    mock_chat.stream = MagicMock(return_value=mock_stream())

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.create = MagicMock(return_value=mock_chat)

    model = XAIModel(model_name='grok-3', client=mock_client)

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Use weather tool'))])],
        config=XAIConfig(),
        tools=[
            ToolDefinition(
                name='get_weather',
                description='Get weather',
                input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
            )
        ],
    )

    ctx = MagicMock()
    ctx.is_streaming = True
    collected_chunks: list[GenerateResponseChunk] = []

    def send_chunk(chunk: GenerateResponseChunk) -> None:
        collected_chunks.append(chunk)

    ctx.send_chunk = send_chunk

    response = await model.generate(request, ctx)

    # Allow event loop to process call_soon_threadsafe callbacks
    await asyncio.sleep(0.1)

    # A chunk should have been emitted for the tool request
    assert len(collected_chunks) == 1
    chunk_part = collected_chunks[0].content[0]
    actual = chunk_part.root if isinstance(chunk_part, Part) else chunk_part
    assert isinstance(actual, ToolRequestPart)
    assert actual.tool_request.name == 'get_weather'
    assert actual.tool_request.input == {'city': 'NYC'}

    # Final response should also contain the tool request
    assert response.message
    assert len(response.message.content) == 1
    final_part = response.message.content[0].root
    assert isinstance(final_part, ToolRequestPart)
    assert final_part.tool_request.ref == 'call_42'
