# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A tool takes one input; the context arrives only on a ToolRunContext-annotated parameter."""

from typing import Any

import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, Part, ToolRunContext, tool
from genkit._ai._testing import ProgrammableModel, define_programmable_model
from genkit._core._typing import FinishReason, Role, ToolRequest


class WeatherInput(BaseModel):
    city: str
    unit: str = 'C'


WEATHER_SCHEMA = {
    'properties': {
        'city': {'title': 'City', 'type': 'string'},
        'unit': {'default': 'C', 'title': 'Unit', 'type': 'string'},
    },
    'required': ['city'],
    'title': 'WeatherInput',
    'type': 'object',
}


def _app() -> tuple[Genkit, ProgrammableModel]:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    return ai, pm


async def _model_calls_tool(
    ai: Genkit,
    pm: ProgrammableModel,
    *,
    name: str,
    tool_input: object,
    context: dict[str, object] | None = None,
) -> ModelResponse:
    """The model calls ``name`` once with ``tool_input``, then answers 'done'."""
    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name=name, input=tool_input, ref='r1'))],
            )
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    )
    return await ai.generate(model='programmableModel', prompt='hi', tools=[name], context=context)


def _tool_output(response: ModelResponse) -> object:
    part = response.messages[2].content[0]
    assert part.tool_response is not None
    return part.tool_response.output


def _advertised_schema(pm: ProgrammableModel) -> object:
    assert pm.last_request is not None and pm.last_request.tools
    return pm.last_request.tools[0].input_schema


@pytest.mark.asyncio
async def test_tool_with_one_str_input_model_sees_string_schema_and_gets_value() -> None:
    """A lone `city: str` tool still advertises a string schema and receives the string the model sent."""
    ai, pm = _app()
    seen: list[object] = []

    @ai.tool()
    async def weather(city: str) -> str:
        seen.append(city)
        return f'Sunny in {city}'

    response = await _model_calls_tool(ai, pm, name='weather', tool_input='Paris')

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == {'type': 'string'}
    assert seen == ['Paris']
    assert _tool_output(response) == 'Sunny in Paris'


@pytest.mark.asyncio
async def test_tool_with_pydantic_input_gets_model_instance() -> None:
    """A tool taking one Pydantic model advertises its fields and receives a model instance."""
    ai, pm = _app()
    seen: list[object] = []

    @ai.tool()
    async def weather(input: WeatherInput) -> str:
        seen.append(input)
        return f'22{input.unit} in {input.city}'

    response = await _model_calls_tool(ai, pm, name='weather', tool_input={'city': 'Paris', 'unit': 'F'})

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == WEATHER_SCHEMA
    assert seen == [WeatherInput(city='Paris', unit='F')]
    assert _tool_output(response) == '22F in Paris'


@pytest.mark.asyncio
async def test_tool_with_input_then_tool_run_context_gets_both() -> None:
    """`(input, ctx: ToolRunContext)` receives the input and a ToolRunContext."""
    ai, pm = _app()
    seen: list[object] = []

    @ai.tool()
    async def weather(input: WeatherInput, ctx: ToolRunContext) -> str:
        seen.extend([input, type(ctx), ctx.context])
        return f'22{input.unit} in {input.city}'

    response = await _model_calls_tool(ai, pm, name='weather', tool_input={'city': 'Paris'}, context={'user': 'u1'})

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == WEATHER_SCHEMA
    assert seen == [WeatherInput(city='Paris'), ToolRunContext, {'user': 'u1'}]
    assert _tool_output(response) == '22C in Paris'


@pytest.mark.asyncio
async def test_tool_with_tool_run_context_first_then_input_gets_both() -> None:
    """`(ctx: ToolRunContext, input)` receives the context and the input, and the advertised schema is the input's."""
    ai, pm = _app()
    seen: list[object] = []

    @ai.tool()
    async def weather(ctx: ToolRunContext, input: WeatherInput) -> str:
        seen.extend([input, type(ctx), ctx.context])
        return f'22{input.unit} in {input.city}'

    response = await _model_calls_tool(ai, pm, name='weather', tool_input={'city': 'Paris'}, context={'user': 'u1'})

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == WEATHER_SCHEMA
    assert seen == [WeatherInput(city='Paris'), ToolRunContext, {'user': 'u1'}]
    assert _tool_output(response) == '22C in Paris'


@pytest.mark.asyncio
async def test_tool_with_only_tool_run_context_runs_with_context_and_no_input() -> None:
    """`(ctx: ToolRunContext)` runs with the context instead of receiving the model's arguments in `ctx`."""
    ai, pm = _app()
    seen: list[object] = []

    @ai.tool()
    async def whoami(ctx: ToolRunContext) -> str:
        seen.extend([type(ctx), ctx.context])
        return f'you are {ctx.context["user"]}'

    response = await _model_calls_tool(ai, pm, name='whoami', tool_input={}, context={'user': 'u1'})

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == {}
    assert seen == [ToolRunContext, {'user': 'u1'}]
    assert _tool_output(response) == 'you are u1'


@pytest.mark.asyncio
async def test_tool_with_no_parameters_runs() -> None:
    """A zero-parameter tool still runs when the model calls it."""
    ai, pm = _app()

    @ai.tool()
    async def ping() -> str:
        return 'pong'

    response = await _model_calls_tool(ai, pm, name='ping', tool_input={})

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == {}
    assert _tool_output(response) == 'pong'


def test_tool_with_two_plain_parameters_raises_type_error_at_definition() -> None:
    """`weather(city: str, unit: str)` raises TypeError when decorated, naming `unit`."""
    ai, _ = _app()

    async def weather(city: str, unit: str) -> str:
        return f'{city} {unit}'

    with pytest.raises(TypeError, match="tool 'weather' takes one input, but 'unit' is a second parameter") as exc:
        ai.tool()(weather)
    assert "annotate 'unit' as ToolRunContext" in str(exc.value)


def test_tool_with_unannotated_ctx_parameter_raises_type_error_at_definition() -> None:
    """`(input, ctx)` with no annotation raises TypeError telling them to annotate ToolRunContext."""
    ai, _ = _app()

    async def weather(input: WeatherInput, ctx) -> str:  # noqa: ANN001
        return input.city

    with pytest.raises(TypeError, match="'ctx' is a second parameter") as exc:
        ai.tool()(weather)
    assert "annotate 'ctx' as ToolRunContext" in str(exc.value)


def test_tool_with_two_tool_run_context_parameters_raises_type_error_at_definition() -> None:
    """Two ToolRunContext parameters raise TypeError instead of picking one."""
    ai, _ = _app()

    async def weather(a: ToolRunContext, b: ToolRunContext) -> str:
        return ''

    with pytest.raises(TypeError, match="tool 'weather' has two ToolRunContext parameters, 'a' and 'b'"):
        ai.tool()(weather)


@pytest.mark.asyncio
async def test_tool_with_postponed_annotations_finds_tool_run_context() -> None:
    """Under `from __future__ import annotations`, the string `'ToolRunContext'` still marks the context parameter."""
    ai, pm = _app()
    seen: list[object] = []

    # Postponed annotations are stored as these strings.
    @ai.tool()
    async def weather(ctx: 'ToolRunContext', input: 'WeatherInput') -> 'str':
        seen.extend([input, type(ctx), ctx.context])
        return f'22{input.unit} in {input.city}'

    response = await _model_calls_tool(ai, pm, name='weather', tool_input={'city': 'Paris'}, context={'user': 'u1'})

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == WEATHER_SCHEMA
    assert seen == [WeatherInput(city='Paris'), ToolRunContext, {'user': 'u1'}]
    assert _tool_output(response) == '22C in Paris'


def test_ephemeral_tool_with_two_plain_parameters_raises_type_error() -> None:
    """`genkit.tool(fn)` follows the same one-input rule as `@ai.tool()`."""

    async def weather(city: str, unit: str) -> str:
        return f'{city} {unit}'

    with pytest.raises(TypeError, match="tool 'weather' takes one input, but 'unit' is a second parameter"):
        tool(weather)


@pytest.mark.asyncio
async def test_generate_context_reaches_tool_with_context_first() -> None:
    """`ai.generate(context={'user': 'u1'}, tools=[t])` shows `ctx.context['user'] == 'u1'` inside a ctx-first tool."""
    ai, pm = _app()
    seen: list[Any] = []

    @ai.tool()
    async def lookup(ctx: ToolRunContext, sku: str) -> str:
        seen.append(ctx.context['user'])
        return f'{sku} for {ctx.context["user"]}'

    response = await _model_calls_tool(ai, pm, name='lookup', tool_input='SKU-1', context={'user': 'u1'})

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == {'type': 'string'}
    assert seen == ['u1']
    assert _tool_output(response) == 'SKU-1 for u1'


@pytest.mark.asyncio
async def test_await_tool_with_input_and_context_called_directly_returns_result() -> None:
    """Calling `await weather(WeatherInput(...))` on a `(input, ctx)` tool runs it and returns the output."""
    ai, _ = _app()

    @ai.tool()
    async def weather(input: WeatherInput, ctx: ToolRunContext) -> str:
        return f'22{input.unit} in {input.city} ({type(ctx).__name__})'

    result = await weather(WeatherInput(city='Paris'))

    assert result.output == '22C in Paris (ToolRunContext)'
    assert result.content is None
