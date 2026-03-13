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

"""Tests for Mistral AI model implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mistralai.models import (
    AssistantMessage,
    AudioChunk,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ImageURLChunk,
    SystemMessage,
    TextChunk,
    ThinkChunk,
    ToolCall,
    ToolMessage,
    UsageInfo,
    UserMessage,
)
from pydantic import ValidationError

from genkit.core.typing import (
    GenerateRequest,
    Media,
    MediaPart,
    Message,
    Part,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)
from genkit.plugins.mistral.models import MistralConfig, MistralModel, _extract_text


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock Mistral client."""
    return MagicMock()


@pytest.fixture
def model(mock_client: MagicMock) -> MistralModel:
    """Create a MistralModel with mocked client."""
    with patch('genkit.plugins.mistral.models.MistralClient', return_value=mock_client):
        return MistralModel(
            model='mistral-large-latest',
            api_key='test-key',
        )


def test_model_initialization(model: MistralModel) -> None:
    """Test model initialization."""
    assert model.name == 'mistral-large-latest'


def test_get_model_info(model: MistralModel) -> None:
    """Test get_model_info returns expected structure."""
    info = model.get_model_info()
    assert info is not None
    assert 'name' in info
    assert 'supports' in info


def test_convert_messages_text_only(model: MistralModel) -> None:
    """Test converting simple text messages."""
    messages = [
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
        Message(role=Role.MODEL, content=[Part(root=TextPart(text='Hi there!'))]),
    ]

    mistral_messages = model._convert_messages(messages)

    assert len(mistral_messages) == 2
    assert isinstance(mistral_messages[0], UserMessage)
    assert mistral_messages[0].content == 'Hello'
    assert isinstance(mistral_messages[1], AssistantMessage)
    assert mistral_messages[1].content == 'Hi there!'


def test_convert_messages_system_role(model: MistralModel) -> None:
    """Test converting system messages."""
    messages = [
        Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='You are helpful.'))]),
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
    ]

    mistral_messages = model._convert_messages(messages)

    assert len(mistral_messages) == 2
    assert isinstance(mistral_messages[0], SystemMessage)
    assert mistral_messages[0].content == 'You are helpful.'


def test_convert_messages_with_tool_request(model: MistralModel) -> None:
    """Test converting messages with tool requests."""
    messages = [
        Message(
            role=Role.MODEL,
            content=[
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(
                            ref='call_123',
                            name='get_weather',
                            input={'city': 'Paris'},
                        )
                    )
                )
            ],
        ),
    ]

    mistral_messages = model._convert_messages(messages)

    assert len(mistral_messages) == 1
    assert isinstance(mistral_messages[0], AssistantMessage)
    tool_calls = mistral_messages[0].tool_calls
    assert tool_calls is not None
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    assert tool_calls[0].id == 'call_123'
    assert tool_calls[0].function.name == 'get_weather'


def test_convert_messages_with_tool_response(model: MistralModel) -> None:
    """Test converting messages with tool responses."""
    messages = [
        Message(
            role=Role.TOOL,
            content=[
                Part(
                    root=ToolResponsePart(
                        tool_response=ToolResponse(
                            ref='call_123',
                            name='get_weather',
                            output={'temperature': 20},
                        )
                    )
                )
            ],
        ),
    ]

    mistral_messages = model._convert_messages(messages)

    assert len(mistral_messages) == 1
    assert isinstance(mistral_messages[0], ToolMessage)
    assert mistral_messages[0].tool_call_id == 'call_123'


def test_convert_tools(model: MistralModel) -> None:
    """Test converting tool definitions."""
    tools = [
        ToolDefinition(
            name='get_weather',
            description='Get weather for a city',
            input_schema={
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            },
        ),
    ]

    mistral_tools = model._convert_tools(tools)

    assert len(mistral_tools) == 1
    assert mistral_tools[0].type == 'function'
    assert mistral_tools[0].function.name == 'get_weather'
    assert mistral_tools[0].function.description == 'Get weather for a city'
    assert 'additionalProperties' in mistral_tools[0].function.parameters


def test_get_response_format_json(model: MistralModel) -> None:
    """Test response format for JSON output."""
    from genkit.core.typing import OutputConfig

    output = OutputConfig(format='json')
    result = model._get_response_format(output)

    assert result == {'type': 'json_object'}


def test_get_response_format_json_with_schema(model: MistralModel) -> None:
    """Test response format for JSON with schema."""
    from genkit.core.typing import OutputConfig

    schema = {'type': 'object', 'title': 'Person', 'properties': {'name': {'type': 'string'}}}
    output = OutputConfig(format='json', schema=schema)
    result = model._get_response_format(output)

    assert result is not None
    assert result['type'] == 'json_schema'
    assert result['json_schema']['name'] == 'Person'
    assert result['json_schema']['strict'] is True


def test_get_response_format_text(model: MistralModel) -> None:
    """Test response format for text output returns None."""
    from genkit.core.typing import OutputConfig

    output = OutputConfig(format='text')
    result = model._get_response_format(output)

    assert result is None


def test_convert_response_text_only(model: MistralModel) -> None:
    """Test converting simple text response."""
    response = ChatCompletionResponse(
        id='test-id',
        object='chat.completion',
        created=1234567890,
        model='mistral-large-latest',
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(content='Hello, world!'),
                finish_reason='stop',
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    result = model._convert_response(response)

    assert result.message is not None
    assert result.message.role == Role.MODEL
    assert len(result.message.content) == 1
    assert result.message.content[0].root.text == 'Hello, world!'
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_convert_response_with_tool_calls(model: MistralModel) -> None:
    """Test converting response with tool calls."""
    response = ChatCompletionResponse(
        id='test-id',
        object='chat.completion',
        created=1234567890,
        model='mistral-large-latest',
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id='call_abc',
                            type='function',
                            function=FunctionCall(
                                name='get_weather',
                                arguments='{"city": "Paris"}',
                            ),
                        )
                    ],
                ),
                finish_reason='tool_calls',
            )
        ],
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )

    result = model._convert_response(response)

    assert result.message is not None
    assert len(result.message.content) == 1
    tool_part = result.message.content[0].root
    assert isinstance(tool_part, ToolRequestPart)
    assert tool_part.tool_request.name == 'get_weather'
    assert tool_part.tool_request.ref == 'call_abc'


@patch('genkit.plugins.mistral.models.MistralClient')
@pytest.mark.asyncio
async def test_generate_simple_request(mock_client_class: MagicMock) -> None:
    """Test simple generate request."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Mock response
    mock_response = ChatCompletionResponse(
        id='test-id',
        object='chat.completion',
        created=1234567890,
        model='mistral-large-latest',
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(content='Hello, world!'),
                finish_reason='stop',
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    mock_client.chat.complete_async = AsyncMock(return_value=mock_response)

    model = MistralModel(
        model='mistral-large-latest',
        api_key='test-key',
    )

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
    )

    response = await model.generate(request)

    assert response.message is not None
    assert response.message.role == Role.MODEL
    assert len(response.message.content) == 1
    assert response.message.content[0].root.text == 'Hello, world!'


@patch('genkit.plugins.mistral.models.MistralClient')
@pytest.mark.asyncio
async def test_generate_returns_empty_on_none_response(mock_client_class: MagicMock) -> None:
    """Test generate handles None response gracefully."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.chat.complete_async = AsyncMock(return_value=None)

    model = MistralModel(
        model='mistral-large-latest',
        api_key='test-key',
    )

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
    )

    response = await model.generate(request)

    assert response.message is not None
    assert response.message.role == Role.MODEL
    assert len(response.message.content) == 1
    assert response.message.content[0].root.text == ''


def test_to_generate_fn(model: MistralModel) -> None:
    """Test to_generate_fn returns callable."""
    fn = model.to_generate_fn()
    assert callable(fn)
    assert fn == model.generate


def test_convert_messages_with_image_url(model: MistralModel) -> None:
    """Test converting messages with image media parts uses ImageURLChunk."""
    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(
                    root=MediaPart(
                        media=Media(
                            url='https://example.com/photo.jpg',
                            content_type='image/jpeg',
                        )
                    )
                ),
                Part(root=TextPart(text='Describe this image.')),
            ],
        ),
    ]

    mistral_messages = model._convert_messages(messages)

    assert len(mistral_messages) == 1
    msg = mistral_messages[0]
    assert isinstance(msg, UserMessage)
    # Multimodal: content should be a list of chunks, not a string.
    content = msg.content
    assert isinstance(content, list)
    assert len(content) == 2
    assert isinstance(content[0], ImageURLChunk)
    assert content[0].image_url == 'https://example.com/photo.jpg'
    assert isinstance(content[1], TextChunk)
    assert content[1].text == 'Describe this image.'


def test_convert_messages_with_audio_data_uri(model: MistralModel) -> None:
    """Test converting messages with audio data URIs uses AudioChunk."""
    audio_b64 = 'SGVsbG8gV29ybGQ='
    messages = [
        Message(
            role=Role.USER,
            content=[
                Part(
                    root=MediaPart(
                        media=Media(
                            url=f'data:audio/mp3;base64,{audio_b64}',
                            content_type='audio/mp3',
                        )
                    )
                ),
                Part(root=TextPart(text='What is in this audio?')),
            ],
        ),
    ]

    mistral_messages = model._convert_messages(messages)

    assert len(mistral_messages) == 1
    msg = mistral_messages[0]
    assert isinstance(msg, UserMessage)
    content = msg.content
    assert isinstance(content, list)
    assert len(content) == 2
    assert isinstance(content[0], AudioChunk)
    # Data URI prefix should be stripped — only base64 payload goes to Mistral.
    assert content[0].input_audio == audio_b64
    assert isinstance(content[1], TextChunk)
    assert content[1].text == 'What is in this audio?'


def test_convert_messages_text_only_stays_string(model: MistralModel) -> None:
    """Text-only user messages should use a plain string, not a list."""
    messages = [
        Message(
            role=Role.USER,
            content=[Part(root=TextPart(text='Just text, no media.'))],
        ),
    ]

    mistral_messages = model._convert_messages(messages)

    assert len(mistral_messages) == 1
    msg = mistral_messages[0]
    assert isinstance(msg, UserMessage)
    # No media means plain string content.
    assert isinstance(msg.content, str)
    assert msg.content == 'Just text, no media.'


def test_extract_text_from_string() -> None:
    """_extract_text returns plain strings unchanged."""
    assert _extract_text('hello') == 'hello'


def test_extract_text_from_text_chunk() -> None:
    """_extract_text extracts text from a TextChunk."""
    assert _extract_text(TextChunk(text='hello')) == 'hello'


def test_extract_text_from_think_chunk() -> None:
    """_extract_text extracts text from ThinkChunk thinking fragments."""
    chunk = ThinkChunk(thinking=[TextChunk(text='Let '), TextChunk(text='me think')])
    assert _extract_text(chunk) == 'Let me think'


def test_extract_text_from_list() -> None:
    """_extract_text handles mixed lists of TextChunk and ThinkChunk."""
    items = [
        ThinkChunk(thinking=[TextChunk(text='reasoning ')]),
        TextChunk(text='answer'),
    ]
    assert _extract_text(items) == 'reasoning answer'


def test_extract_text_from_unknown_type() -> None:
    """_extract_text returns empty string for unrecognised types."""
    assert _extract_text(42) == ''


def test_convert_response_with_think_chunks(model: MistralModel) -> None:
    """Test _convert_response extracts text from ThinkChunk content.

    Magistral reasoning models return a list of ThinkChunk + TextChunk
    instead of a plain string.
    """
    response = ChatCompletionResponse(
        id='test-id',
        object='chat.completion',
        created=1234567890,
        model='magistral-small-latest',
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(
                    content=[
                        ThinkChunk(thinking=[TextChunk(text='Let me think.')]),
                        TextChunk(text='The answer is 42.'),
                    ],
                ),
                finish_reason='stop',
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    result = model._convert_response(response)

    assert result.message is not None
    assert len(result.message.content) == 2
    assert result.message.content[0].root.text == 'Let me think.'
    assert result.message.content[1].root.text == 'The answer is 42.'


def test_mistral_config_defaults() -> None:
    """All MistralConfig fields should default to None."""
    config = MistralConfig()
    assert config.temperature is None
    assert config.max_tokens is None
    assert config.top_p is None
    assert config.random_seed is None
    assert config.stop is None
    assert config.presence_penalty is None
    assert config.frequency_penalty is None
    assert config.safe_prompt is None


def test_mistral_config_with_all_params() -> None:
    """MistralConfig should accept all supported API parameters."""
    config = MistralConfig(
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        random_seed=42,
        stop=['\n', 'END'],
        presence_penalty=0.5,
        frequency_penalty=0.3,
        safe_prompt=True,
    )
    assert config.temperature == 0.7
    assert config.max_tokens == 1024
    assert config.top_p == 0.9
    assert config.random_seed == 42
    assert config.stop == ['\n', 'END']
    assert config.presence_penalty == 0.5
    assert config.frequency_penalty == 0.3
    assert config.safe_prompt is True


def test_mistral_config_stop_accepts_string() -> None:
    """Stop can be a single string, not just a list."""
    config = MistralConfig(stop='END')
    assert config.stop == 'END'


def test_mistral_config_rejects_negative_temperature() -> None:
    """Temperature must be >= 0.0."""
    with pytest.raises(ValidationError):
        MistralConfig(temperature=-0.1)


def test_mistral_config_rejects_top_p_above_one() -> None:
    """Top_p must be <= 1.0."""
    with pytest.raises(ValidationError):
        MistralConfig(top_p=1.5)


@patch('genkit.plugins.mistral.models.MistralClient')
@pytest.mark.asyncio
async def test_generate_forwards_all_config_params(mock_client_class: MagicMock) -> None:
    """Config params (stop, presence_penalty, etc.) should be forwarded to the API."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = ChatCompletionResponse(
        id='test-id',
        object='chat.completion',
        created=1234567890,
        model='mistral-large-latest',
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(content='ok'),
                finish_reason='stop',
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )
    mock_client.chat.complete_async = AsyncMock(return_value=mock_response)

    model_instance = MistralModel(model='mistral-large-latest', api_key='test-key')

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
        config={
            'temperature': 0.3,
            'max_tokens': 512,
            'top_p': 0.8,
            'random_seed': 99,
            'stop': ['END'],
            'presence_penalty': 0.4,
            'frequency_penalty': 0.2,
            'safe_prompt': True,
        },
    )

    await model_instance.generate(request, None)

    call_kwargs = mock_client.chat.complete_async.call_args[1]
    assert call_kwargs['temperature'] == 0.3
    assert call_kwargs['max_tokens'] == 512
    assert call_kwargs['top_p'] == 0.8
    assert call_kwargs['random_seed'] == 99
    assert call_kwargs['stop'] == ['END']
    assert call_kwargs['presence_penalty'] == 0.4
    assert call_kwargs['frequency_penalty'] == 0.2
    assert call_kwargs['safe_prompt'] is True
