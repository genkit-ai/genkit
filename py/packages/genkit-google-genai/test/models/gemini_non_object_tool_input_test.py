# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A tool whose input isn't an object reaches Gemini as `{"input": ...}` and comes back unwrapped."""

from collections.abc import Mapping
from typing import Any

import pytest
from genkit_google_genai._models._gemini import GeminiModel
from google.genai import types as genai_types
from pytest_mock import MockerFixture

from genkit import ActionRunContext, Message, Part, Role
from genkit.model import ModelRequest, ToolDefinition, ToolRequest

STRING_INPUT = {'type': 'string'}
CITY_OBJECT_INPUT = {
    'properties': {'city': {'title': 'City', 'type': 'string'}},
    'required': ['city'],
    'title': 'WeatherInput',
    'type': 'object',
}


def _tool(input_schema: Mapping[str, object], name: str = 'weather') -> ToolDefinition:
    return ToolDefinition(name=name, description='Weather for a city', input_schema=dict(input_schema))


def _request(*tools: ToolDefinition, messages: list[Message] | None = None) -> ModelRequest:
    return ModelRequest(
        messages=messages or [Message(role=Role.USER, content=[Part.from_text('Weather in Paris?')])],
        tools=list(tools),
    )


def _gemini_calls(name: str, args: dict[str, Any]) -> genai_types.GenerateContentResponse:
    """Gemini's reply: one function call to ``name`` with ``args``."""
    part = genai_types.Part(function_call=genai_types.FunctionCall(name=name, args=args, id='c1'))
    return genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content=genai_types.Content(role='model', parts=[part]))]
    )


async def _generate(mocker: MockerFixture, request: ModelRequest, reply: genai_types.GenerateContentResponse) -> Any:  # noqa: ANN401
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = reply
    response = await GeminiModel('gemini-2.5-flash', client).generate(request, ActionRunContext())
    return response, client.aio.models.generate_content.call_args.kwargs


def _declared_parameters(sent: dict[str, Any]) -> genai_types.Schema:
    return sent['config'].tools[0].function_declarations[0].parameters


def _tool_request(response: Any) -> ToolRequest:  # noqa: ANN401
    part = response.message.content[0]
    assert part.tool_request is not None
    return part.tool_request


def _no_call() -> genai_types.GenerateContentResponse:
    part = genai_types.Part(text='ok')
    return genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content=genai_types.Content(role='model', parts=[part]))]
    )


@pytest.mark.asyncio
async def test_gemini_str_tool_declares_object_with_required_input_string(mocker: MockerFixture) -> None:
    """`weather(city: str)` is declared to Gemini as an object with one required `input` string."""
    _, sent = await _generate(mocker, _request(_tool(STRING_INPUT)), _no_call())

    assert _declared_parameters(sent) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={'input': genai_types.Schema(type=genai_types.Type.STRING)},
        required=['input'],
    )


@pytest.mark.asyncio
async def test_gemini_list_tool_declares_input_array(mocker: MockerFixture) -> None:
    """`compare(cities: list[str])` is declared with `input` as an array of strings."""
    schema = {'items': {'type': 'string'}, 'type': 'array'}

    _, sent = await _generate(mocker, _request(_tool(schema, name='compare')), _no_call())

    assert _declared_parameters(sent) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            'input': genai_types.Schema(
                type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING)
            )
        },
        required=['input'],
    )


@pytest.mark.asyncio
async def test_gemini_nullable_str_tool_declares_nullable_input(mocker: MockerFixture) -> None:
    """A `{"type": ["string", "null"]}` input is declared as a nullable `input` string."""
    _, sent = await _generate(mocker, _request(_tool({'type': ['string', 'null']})), _no_call())

    assert _declared_parameters(sent) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={'input': genai_types.Schema(type=genai_types.Type.STRING, nullable=True)},
        required=['input'],
    )


@pytest.mark.asyncio
async def test_gemini_object_tool_declaration_unchanged(mocker: MockerFixture) -> None:
    """A tool taking a Pydantic model declares its own fields, with no `input` wrapper."""
    _, sent = await _generate(mocker, _request(_tool(CITY_OBJECT_INPUT)), _no_call())

    assert _declared_parameters(sent) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={'city': genai_types.Schema(type=genai_types.Type.STRING)},
        required=['city'],
    )


@pytest.mark.asyncio
async def test_gemini_no_input_tool_declares_empty_object(mocker: MockerFixture) -> None:
    """A tool with no input (schema `{}`) is still declared as an empty object."""
    _, sent = await _generate(mocker, _request(_tool({}, name='ping')), _no_call())

    assert _declared_parameters(sent) == genai_types.Schema(type=genai_types.Type.OBJECT, properties={})


@pytest.mark.asyncio
async def test_gemini_calls_str_tool_with_input_tool_receives_string(mocker: MockerFixture) -> None:
    """Gemini calling `weather` with `{"input": "Paris"}` gives a tool request whose input is `'Paris'`."""
    response, _ = await _generate(mocker, _request(_tool(STRING_INPUT)), _gemini_calls('weather', {'input': 'Paris'}))

    assert _tool_request(response) == ToolRequest(name='weather', input='Paris', ref='c1')


@pytest.mark.asyncio
async def test_gemini_calls_object_tool_with_input_field_keeps_whole_args(mocker: MockerFixture) -> None:
    """An object tool with a real `input` field gets Gemini's whole args, not just `input`."""
    schema = {'properties': {'input': {'type': 'string'}, 'mode': {'type': 'string'}}, 'type': 'object'}

    response, _ = await _generate(
        mocker, _request(_tool(schema, name='run')), _gemini_calls('run', {'input': 'x', 'mode': 'fast'})
    )

    assert _tool_request(response) == ToolRequest(name='run', input={'input': 'x', 'mode': 'fast'}, ref='c1')


@pytest.mark.asyncio
async def test_gemini_calls_str_tool_without_input_passes_args_through(mocker: MockerFixture) -> None:
    """Gemini calling `weather` with `{}` gives a tool request with input `{}`, so the tool's own check rejects it."""
    response, _ = await _generate(mocker, _request(_tool(STRING_INPUT)), _gemini_calls('weather', {}))

    assert _tool_request(response) == ToolRequest(name='weather', input={}, ref='c1')


@pytest.mark.asyncio
async def test_gemini_history_str_tool_request_sends_input_field(mocker: MockerFixture) -> None:
    """A previous `weather` call with input `'Paris'` is sent back to Gemini as `args={"input": "Paris"}`."""
    history = [
        Message(role=Role.USER, content=[Part.from_text('Weather in Paris?')]),
        Message(role=Role.MODEL, content=[Part(tool_request=ToolRequest(name='weather', input='Paris', ref='c1'))]),
    ]

    _, sent = await _generate(mocker, _request(_tool(STRING_INPUT), messages=history), _no_call())

    call = sent['contents'][1].parts[0].function_call
    assert call == genai_types.FunctionCall(name='weather', args={'input': 'Paris'}, id='c1')


@pytest.mark.asyncio
async def test_gemini_history_object_tool_request_sends_args_unchanged(mocker: MockerFixture) -> None:
    """A previous `weather` call with input `{"city": "Paris"}` is sent back unchanged."""
    history = [
        Message(role=Role.USER, content=[Part.from_text('Weather in Paris?')]),
        Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name='weather', input={'city': 'Paris'}, ref='c1'))],
        ),
    ]

    _, sent = await _generate(mocker, _request(_tool(CITY_OBJECT_INPUT), messages=history), _no_call())

    call = sent['contents'][1].parts[0].function_call
    assert call == genai_types.FunctionCall(name='weather', args={'city': 'Paris'}, id='c1')


@pytest.mark.asyncio
async def test_gemini_streaming_str_tool_call_receives_string(mocker: MockerFixture) -> None:
    """A streamed `{"input": "Paris"}` call gives `'Paris'` in both the chunk and the final tool request."""
    client = mocker.AsyncMock()

    async def stream() -> Any:  # noqa: ANN401
        yield _gemini_calls('weather', {'input': 'Paris'})

    client.aio.models.generate_content_stream.return_value = stream()
    chunks: list[Any] = []
    ctx = ActionRunContext(streaming_callback=chunks.append)

    response = await GeminiModel('gemini-2.5-flash', client).generate(_request(_tool(STRING_INPUT)), ctx)

    assert chunks[0].content[0].tool_request == ToolRequest(name='weather', input='Paris', ref='c1')
    assert _tool_request(response) == ToolRequest(name='weather', input='Paris', ref='c1')
