# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A tool takes one typed input; the context arrives only on a ToolRunContext-annotated parameter."""

from dataclasses import dataclass
from typing import Any, TypedDict

import pytest
from pydantic import BaseModel

from genkit import Genkit, GenkitError, Message, ModelResponse, Part, ToolRunContext, tool
from genkit._ai._testing import ProgrammableModel, define_programmable_model
from genkit._core._typing import FinishReason, Role, ToolRequest


class WeatherInput(BaseModel):
    city: str
    unit: str = 'C'


@dataclass
class WeatherQuery:
    city: str
    days: int


class Forecast(TypedDict):
    city: str
    days: int


class Thermometer:
    """A plain class with no schema, so it can't be a tool's input."""


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


def test_tool_with_unannotated_input_raises_type_error_at_definition() -> None:
    """`search(query)` with no annotation raises TypeError naming `query` and the `Any` option."""
    ai, _ = _app()

    async def search(query) -> str:  # noqa: ANN001
        return str(query)

    with pytest.raises(TypeError, match="tool 'search' input 'query' has no type annotation") as exc:
        ai.tool()(search)
    assert 'or use Any to accept anything' in str(exc.value)


def test_ephemeral_tool_with_unannotated_input_raises_type_error() -> None:
    """`genkit.tool(search)` with an unannotated input raises the same TypeError as `@ai.tool()`."""

    async def search(query) -> str:  # noqa: ANN001
        return str(query)

    with pytest.raises(TypeError, match="tool 'search' input 'query' has no type annotation"):
        tool(search)


@pytest.mark.asyncio
async def test_tool_with_any_input_accepts_any_value() -> None:
    """`search(query: Any)` advertises an empty schema and receives whatever the model sent."""
    ai, pm = _app()
    seen: list[object] = []

    @ai.tool()
    async def search(query: Any) -> str:  # noqa: ANN401
        seen.append(query)
        return 'found'

    response = await _model_calls_tool(ai, pm, name='search', tool_input=5)

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == {}
    assert seen == [5]
    assert _tool_output(response) == 'found'


def test_tool_with_plain_class_input_raises_type_error_naming_input() -> None:
    """`read(t: Thermometer)` raises TypeError naming the tool, the input, and the kinds of types it can be."""
    ai, _ = _app()

    async def read(t: Thermometer) -> str:
        return ''

    with pytest.raises(TypeError, match="tool 'read' input 't' has type Thermometer, which has no JSON schema") as exc:
        ai.tool()(read)
    assert 'Use a Pydantic model, dataclass, TypedDict, or a basic type' in str(exc.value)


@pytest.mark.asyncio
async def test_tool_with_int_input_rejects_string_with_invalid_argument() -> None:
    """`await days_until(n: int)` called with 'abc' raises INVALID_ARGUMENT, and the tool advertises an integer."""
    ai, _ = _app()

    @ai.tool()
    async def days_until(n: int) -> str:
        return f'{n} days'

    with pytest.raises(GenkitError, match="Invalid input for action 'days_until'") as exc:
        await days_until('abc')
    assert exc.value.status == 'INVALID_ARGUMENT'
    assert days_until.input_schema == {'type': 'integer'}
    assert (await days_until(3)).output == '3 days'


@pytest.mark.asyncio
async def test_tool_with_list_input_receives_list() -> None:
    """`compare(cities: list[str])` advertises an array of strings and receives the list."""
    ai, pm = _app()
    seen: list[object] = []

    @ai.tool()
    async def compare(cities: list[str]) -> str:
        seen.append(cities)
        return ' vs '.join(cities)

    response = await _model_calls_tool(ai, pm, name='compare', tool_input=['Paris', 'Rome'])

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == {'items': {'type': 'string'}, 'type': 'array'}
    assert seen == [['Paris', 'Rome']]
    assert _tool_output(response) == 'Paris vs Rome'


@pytest.mark.asyncio
async def test_tool_with_dataclass_input_receives_dataclass_instance() -> None:
    """`forecast(q: WeatherQuery)` on a dataclass receives a WeatherQuery instance, not a dict."""
    ai, pm = _app()
    seen: list[object] = []

    @ai.tool()
    async def forecast(q: WeatherQuery) -> str:
        seen.append(q)
        return f'{q.days} days in {q.city}'

    response = await _model_calls_tool(ai, pm, name='forecast', tool_input={'city': 'Paris', 'days': 3})

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _advertised_schema(pm) == {
        'properties': {'city': {'title': 'City', 'type': 'string'}, 'days': {'title': 'Days', 'type': 'integer'}},
        'required': ['city', 'days'],
        'title': 'WeatherQuery',
        'type': 'object',
    }
    assert seen == [WeatherQuery(city='Paris', days=3)]
    assert _tool_output(response) == '3 days in Paris'


@pytest.mark.asyncio
async def test_tool_with_typed_dict_input_receives_dict() -> None:
    """`forecast(f: Forecast)` on a TypedDict receives a plain dict, and a missing field raises INVALID_ARGUMENT."""
    ai, pm = _app()
    seen: list[object] = []

    @ai.tool()
    async def forecast(f: Forecast) -> str:
        seen.append(f)
        return f'{f["days"]} days in {f["city"]}'

    response = await _model_calls_tool(ai, pm, name='forecast', tool_input={'city': 'Paris', 'days': 3})

    assert response.text == 'done'
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert seen == [{'city': 'Paris', 'days': 3}]
    assert _tool_output(response) == '3 days in Paris'
    with pytest.raises(GenkitError) as exc:
        await forecast({'city': 'Paris'})
    assert exc.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_tool_with_default_input_called_without_input_gets_default() -> None:
    """`await weather()` on `weather(city: str = 'Paris')` runs with the default."""
    ai, _ = _app()

    @ai.tool()
    async def weather(city: str = 'Paris') -> str:
        return f'Sunny in {city}'

    assert (await weather()).output == 'Sunny in Paris'
    assert (await weather('Rome')).output == 'Sunny in Rome'


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
