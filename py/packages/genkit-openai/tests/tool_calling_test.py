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

"""Test tool calling."""

import json
from collections.abc import Callable
from functools import reduce
from unittest.mock import AsyncMock, MagicMock

import pytest
from genkit_openai._models._model import OpenAIModel
from openai.types.chat import ChatCompletion

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, Part, Role
from genkit.model import ModelContext, ModelRequest, ToolDefinition, ToolRequest


@pytest.mark.asyncio
async def test_generate_with_tool_calls_executes_tools(
    sample_request: ModelRequest, make_completion: Callable[..., ChatCompletion]
) -> None:
    """Test generate with tool calls executes tools."""
    # First call triggers tool execution
    first_response = make_completion(
        choice={
            'finish_reason': 'tool_calls',
            'message': {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'tool123',
                        'type': 'function',
                        'function': {'name': 'tool_fn', 'arguments': '{"a": 1}'},
                    }
                ],
            },
        }
    )

    # Second call is the model response
    second_response = make_completion(content='final response')

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            first_response,
            second_response,
        ]
    )

    model = OpenAIModel(model='gpt-4', client=mock_client)

    response = await model._generate(sample_request)

    assert response.message is not None
    part = response.message.content[0]

    assert part.tool_request is not None
    assert part.tool_request.input == {'a': 1}
    assert part.tool_request.name == 'tool_fn'
    assert part.tool_request.ref == 'tool123'

    response = await model._generate(sample_request)

    assert response.message is not None
    part = response.message.content[0]

    assert part.text is not None
    assert part.text == 'final response'

    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_generate_unwraps_string_tool_input(make_completion: Callable[..., ChatCompletion]) -> None:
    """A string tool is declared as an object and reaches Genkit as a string."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('Weather in Paris?')])],
        tools=[ToolDefinition(name='weather', description='Get weather', input_schema={'type': 'string'})],
    )
    completion = make_completion(
        choice={
            'finish_reason': 'tool_calls',
            'message': {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'tool_1',
                        'type': 'function',
                        'function': {'name': 'weather', 'arguments': '{"input":"Paris"}'},
                    }
                ],
            },
        }
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)

    result = await OpenAIModel(model='gpt-4', client=client)._generate(request)

    schema = client.chat.completions.create.call_args.kwargs['tools'][0]['function']['parameters']
    assert schema['type'] == 'object'
    assert schema['properties']['input']['type'] == 'string'
    assert result.message is not None
    assert result.message.content[0].tool_request is not None
    assert result.message.content[0].tool_request.input == 'Paris'


@pytest.mark.asyncio
async def test_string_tool_reaches_genkit_dispatch(make_completion: Callable[..., ChatCompletion]) -> None:
    """A provider tool call executes the registered string tool with its bare value."""
    calls: list[str] = []
    ai = Genkit()

    @ai.tool(name='weather', description='Look up weather')
    async def weather(city: str) -> str:
        calls.append(city)
        return 'Sunny'

    first = make_completion(
        choice={
            'finish_reason': 'tool_calls',
            'message': {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'tool_1',
                        'type': 'function',
                        'function': {'name': 'weather', 'arguments': '{"input":"Paris"}'},
                    }
                ],
            },
        }
    )
    second = make_completion(content='It is sunny in Paris.')
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[first, second])
    model = OpenAIModel(model='gpt-4', client=client)

    async def model_fn(request: ModelRequest, _ctx: ModelContext) -> ModelResponse:
        return await model._generate(request)

    ai.define_model(name='openai-mock', fn=model_fn)
    response = await ai.generate(model='openai-mock', prompt='Weather in Paris?', tools=['weather'])

    assert response.text == 'It is sunny in Paris.'
    assert calls == ['Paris']
    assert client.chat.completions.create.call_count == 2
    second_messages = client.chat.completions.create.call_args.kwargs['messages']
    assert json.loads(second_messages[1]['tool_calls'][0]['function']['arguments']) == {'input': 'Paris'}


@pytest.mark.asyncio
async def test_string_tool_history_and_stream_use_original_value() -> None:
    """History is wrapped for OpenAI and a streamed call resolves to a string."""
    tool = ToolDefinition(name='weather', description='Get weather', input_schema={'type': 'string'})
    history = [
        Message(role=Role.USER, content=[Part.from_text('Weather in Paris?')]),
        Message(role=Role.MODEL, content=[Part(tool_request=ToolRequest(name='weather', ref='old', input='Paris'))]),
    ]
    request = ModelRequest(messages=history, tools=[tool])
    call = MagicMock(index=0, id='new')
    call.function.name = 'weather'
    call.function.arguments = '{"input":"London"}'
    delta = MagicMock(content=None, role=None, tool_calls=[call], reasoning_content=None, refusal=None)
    choice = MagicMock(delta=delta, finish_reason='tool_calls', model_extra={})

    async def stream():
        yield MagicMock(choices=[choice], usage=None)

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=stream())
    chunks: list[ModelResponseChunk] = []
    result = await OpenAIModel(model='gpt-4', client=client)._generate_stream(request, chunks.append)

    sent = client.chat.completions.create.call_args.kwargs['messages'][1]['tool_calls'][0]['function']['arguments']
    assert json.loads(sent) == {'input': 'Paris'}
    assert result.message is not None
    assert result.message.content[0].tool_request is not None
    assert result.message.content[0].tool_request.input == 'London'
    assert all(chunk.content and chunk.content[0].tool_request is not None for chunk in chunks)
    assert [chunk.content[0].tool_request.input for chunk in chunks if chunk.content[0].tool_request] == ['London']


@pytest.mark.asyncio
async def test_generate_stream_accepts_tool_call_before_function(sample_request: ModelRequest) -> None:
    """A tool-call id may arrive before its function metadata and arguments."""
    first = MagicMock(index=0, id='tool123', function=None)
    second = MagicMock(index=0, id=None)
    second.function.name = 'tool_fn'
    second.function.arguments = '{"a": '
    third = MagicMock(index=0, id=None)
    third.function.name = None
    third.function.arguments = '1}'

    async def stream():
        for call in (first, second, third):
            delta = MagicMock(content=None, role=None, tool_calls=[call], reasoning_content=None, refusal=None)
            yield MagicMock(choices=[MagicMock(delta=delta, finish_reason=None, model_extra={})], usage=None)

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=stream())
    chunks: list[ModelResponseChunk] = []
    result = await OpenAIModel(model='gpt-4', client=client)._generate_stream(sample_request, chunks.append)

    assert result.message is not None
    assert len(result.message.content) == 1
    request = result.message.content[0].tool_request
    assert request is not None
    assert request.name == 'tool_fn'
    assert request.input == {'a': 1}
    chunk_inputs = []
    for chunk in chunks:
        tool_request = chunk.content[0].tool_request
        assert tool_request is not None
        chunk_inputs.append(tool_request.input)
    assert chunk_inputs == ['{"a": ', '1}']


@pytest.mark.asyncio
async def test_generate_stream_with_tool_calls(sample_request: ModelRequest) -> None:
    """Test generate_stream processes tool calls streamed in chunks correctly."""
    mock_client = MagicMock()

    class MockToolCall:
        def __init__(self, id: str, index: int, name: str, args_chunk: str) -> None:
            self.id = id
            self.index = index
            self.function = MagicMock()
            self.function.name = name
            self.function.arguments = args_chunk

    class MockStream:
        def __init__(self) -> None:
            self._chunks = [
                # Initial chunk - empty args
                self._make_tool_chunk(id='tool123', index=0, name='tool_fn', args_chunk=''),
                # First chunk - partial tool call args
                self._make_tool_chunk(id='tool123', index=0, name='tool_fn', args_chunk='{"a": '),
                # Second chunk - rest of tool call args
                self._make_tool_chunk(id='tool123', index=0, name='tool_fn', args_chunk='1}'),
            ]
            self._current = 0

        def _make_tool_chunk(self, id: str, index: int, name: str, args_chunk: str) -> object:
            delta_mock = MagicMock()
            delta_mock.content = None
            delta_mock.role = None
            delta_mock.tool_calls = [MockToolCall(id, index, name, args_chunk)]
            delta_mock.reasoning_content = None
            delta_mock.refusal = None

            choice_mock = MagicMock()
            choice_mock.delta = delta_mock

            return MagicMock(choices=[choice_mock], usage=None)

        def __aiter__(self) -> 'MockStream':
            return self

        async def __anext__(self) -> object:
            if self._current >= len(self._chunks):
                raise StopAsyncIteration
            chunk = self._chunks[self._current]
            self._current += 1
            return chunk

    mock_client.chat.completions.create = AsyncMock(return_value=MockStream())

    model = OpenAIModel(model='gpt-4', client=mock_client)
    collected_chunks = []

    def callback(chunk: ModelResponseChunk) -> None:
        collected_chunks.append(chunk.content[0])

    await model._generate_stream(sample_request, callback)

    assert len(collected_chunks) == 3
    assert all(part.tool_request is not None for part in collected_chunks)

    tool_part = collected_chunks[0]
    assert tool_part.tool_request is not None
    tool_request = tool_part.tool_request
    assert tool_request.name == 'tool_fn'
    assert tool_request.ref == 'tool123'

    accumulated_output = reduce(lambda res, tool_call: res + tool_call.tool_request.input, collected_chunks, '')
    assert json.loads(accumulated_output) == {'a': 1}
