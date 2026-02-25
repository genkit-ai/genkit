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
from functools import reduce
from unittest.mock import AsyncMock, MagicMock

import pytest

from genkit.plugins.compat_oai.models import OpenAIModel
from genkit.types import GenerateRequest, GenerateResponseChunk, TextPart, ToolRequestPart


@pytest.mark.asyncio
async def test_generate_with_tool_calls_executes_tools(sample_request: GenerateRequest) -> None:
    """Test generate with tool calls executes tools."""
    mock_tool_call = MagicMock()
    mock_tool_call.id = 'tool123'
    mock_tool_call.function.name = 'tool_fn'
    mock_tool_call.function.arguments = '{"a": 1}'

    # First call triggers tool execution
    first_message = MagicMock()
    first_message.role = 'assistant'
    first_message.tool_calls = [mock_tool_call]
    first_message.content = None
    first_message.reasoning_content = None

    first_response = MagicMock()
    first_response.choices = [MagicMock(finish_reason='tool_calls', message=first_message)]

    # Second call is the model response
    second_message = MagicMock()
    second_message.role = 'model'
    second_message.tool_calls = None
    second_message.content = 'final response'
    second_message.reasoning_content = None

    second_response = MagicMock()
    second_response.choices = [MagicMock(finish_reason='stop', message=second_message)]

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
    part = response.message.content[0].root

    assert isinstance(part, ToolRequestPart)
    assert part.tool_request.input == {'a': 1}
    assert part.tool_request.name == 'tool_fn'
    assert part.tool_request.ref == 'tool123'

    response = await model._generate(sample_request)

    assert response.message is not None
    part = response.message.content[0].root

    assert isinstance(part, TextPart)
    assert part.text == 'final response'

    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_generate_stream_with_tool_calls(sample_request: GenerateRequest) -> None:
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

            choice_mock = MagicMock()
            choice_mock.delta = delta_mock

            return MagicMock(choices=[choice_mock])

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

    def callback(chunk: GenerateResponseChunk) -> None:
        collected_chunks.append(chunk.content[0].root)

    await model._generate_stream(sample_request, callback)

    assert len(collected_chunks) == 3
    assert all(isinstance(part, ToolRequestPart) for part in collected_chunks)

    tool_part = collected_chunks[0]
    assert isinstance(tool_part, ToolRequestPart)
    assert tool_part.tool_request is not None
    tool_request = tool_part.tool_request
    assert tool_request.name == 'tool_fn'
    assert tool_request.ref == 'tool123'

    accumulated_output = reduce(lambda res, tool_call: res + tool_call.tool_request.input, collected_chunks, '')
    assert json.loads(accumulated_output) == {'a': 1}
