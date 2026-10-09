# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The return annotation is the schema the model sees. ``await tool(...)`` returns what the function returned."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import (
    FinishReason,
    Genkit,
    GenkitError,
    Message,
    ModelResponse,
    MultipartToolResponse,
    Part,
    Role,
    Tool,
    ToolRunContext,
    response,
)
from genkit._core._action import Action
from genkit._core._schema import to_json_schema
from genkit.model import ToolRequest, ToolResponse
from genkit.plugin_api import ActionKind
from genkit.testing import ScriptedModel, define_scripted_model


class ShotOut(BaseModel):
    ok: bool
    label: str


class Pair(BaseModel):
    a: int
    b: int


class Camera:
    """A plain object with no JSON form."""


SHOT = {'ok': True, 'label': 'lab'}
WIRE_PNG = {'media': {'contentType': 'image/png', 'url': 'data:image/png;base64,abc'}}
WIRE_CAPTION = {'text': 'lab camera'}


def _png() -> Part:
    return Part.from_media('data:image/png;base64,abc', content_type='image/png')


def _caption() -> Part:
    return Part.from_text('lab camera')


def _model_calls_tool(*, name: str, ref: str, tool_input: object | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[
                Part(tool_request=ToolRequest(name=name, input=tool_input if tool_input is not None else {}, ref=ref))
            ],
        ),
    )


def _ok() -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
    )


def _tool_response(generated: ModelResponse) -> tuple[ToolResponse, object | None]:
    tool_msg = next(message for message in generated.messages if message.role == Role.TOOL)
    part = tool_msg.content[0]
    assert part.tool_response is not None
    return part.tool_response, part.metadata


def _assert_closed_tool_round(generated: ModelResponse) -> None:
    assert generated.finish_reason == FinishReason.STOP
    assert generated.message is not None
    assert generated.messages[-1] == generated.message
    assert [message.role for message in generated.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]


async def _generate_tool_turn(
    ai: Genkit, pm: ScriptedModel, *, name: str, tool_input: object | None = None
) -> ModelResponse:
    pm.responses = [_model_calls_tool(name=name, ref='t1', tool_input=tool_input), _ok()]
    return await ai.generate(prompt='go', tools=[name])


@pytest.mark.asyncio
async def test_await_tool_built_from_raw_action_returning_response_returns_multipart_tool_response() -> None:
    """`await Tool(Action(fn=raw_shot))()` returns raw_shot's `response(ShotOut(...), parts=[png])` as built."""

    async def raw_shot() -> MultipartToolResponse[ShotOut]:
        return response(ShotOut(ok=True, label='lab'), parts=[_png()], metadata={'src': 'cam'})

    handle = Tool(Action(name='shot', kind=ActionKind.TOOL, fn=raw_shot))
    out = await handle()
    assert isinstance(out, MultipartToolResponse)
    assert out.output == ShotOut(ok=True, label='lab')
    assert out.content == [_png()]
    assert out.metadata == {'src': 'cam'}


@pytest.mark.asyncio
async def test_await_int_tool_returns_int() -> None:
    """`await add_pair(Pair(a=2, b=3))` on a `-> int` tool returns 5."""
    ai = Genkit()

    @ai.tool()
    async def add_pair(pair: Pair) -> int:
        return pair.a + pair.b

    assert await add_pair(Pair(a=2, b=3)) == 5


@pytest.mark.asyncio
async def test_await_str_tool_returns_string() -> None:
    """`await weather('Austin')` on a `-> str` tool returns 'Sunny in Austin'."""
    ai = Genkit()

    @ai.tool(name='weather')
    async def weather(city: str) -> str:
        return f'Sunny in {city}'

    assert await weather('Austin') == 'Sunny in Austin'
    assert weather.output_schema == {'type': 'string'}


@pytest.mark.asyncio
async def test_await_tool_returning_none_returns_none() -> None:
    """`await log_visit('Austin')` on a `-> None` tool returns None."""
    ai = Genkit()

    @ai.tool()
    async def log_visit(city: str) -> None:
        return None

    assert await log_visit('Austin') is None


@pytest.mark.asyncio
async def test_await_dict_tool_returns_the_dict() -> None:
    """`await stamp()` on a `-> dict` tool returns the dict as built, datetimes still datetimes."""
    ai = Genkit()
    when = datetime(2026, 8, 25, 12, 0)

    @ai.tool()
    async def stamp() -> dict[str, Any]:
        return {'when': when, 'ok': True}

    out = await stamp()
    assert out == {'when': when, 'ok': True}
    assert isinstance(out['when'], datetime)


@pytest.mark.asyncio
async def test_generate_str_tool_message_has_string_and_no_media() -> None:
    """Generate copies that string onto the tool message and tells the model string."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='weather')
    async def weather(city: str) -> str:
        return f'Sunny in {city}'

    generated = await _generate_tool_turn(ai, pm, name='weather', tool_input='Austin')
    _assert_closed_tool_round(generated)
    tool_response, metadata = _tool_response(generated)
    assert tool_response.name == 'weather'
    assert tool_response.output == 'Sunny in Austin'
    assert tool_response.content is None
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == {'type': 'string'}
    assert weather.output_schema == {'type': 'string'}


@pytest.mark.asyncio
async def test_action_run_str_tool_still_returns_multipart_tool_response() -> None:
    """`await weather.action().run('Austin')`, what the Dev UI runs, still returns a MultipartToolResponse."""
    ai = Genkit()

    @ai.tool(name='weather')
    async def weather(city: str) -> str:
        return f'Sunny in {city}'

    ran = await weather.action().run('Austin')
    assert isinstance(ran.response, MultipartToolResponse)
    assert ran.response.output == 'Sunny in Austin'
    assert ran.response.content is None
    assert ran.response.metadata is None


@pytest.mark.asyncio
async def test_await_pydantic_tool_returns_the_model_instance() -> None:
    """`await shot()` on a `-> ShotOut` tool returns the ShotOut instance, not a dict."""
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> ShotOut:
        return ShotOut(ok=True, label='lab')

    out = await shot()
    assert isinstance(out, ShotOut)
    assert out == ShotOut(ok=True, label='lab')
    assert shot.output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_generate_pydantic_tool_message_still_has_json_dump() -> None:
    """`ai.generate(tools=['shot'])` on a `-> ShotOut` tool still sends the model the {ok, label} JSON dump."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='shot')
    async def shot() -> ShotOut:
        return ShotOut(ok=True, label='lab')

    generated = await _generate_tool_turn(ai, pm, name='shot')
    _assert_closed_tool_round(generated)
    tool_response, metadata = _tool_response(generated)
    assert tool_response.name == 'shot'
    assert tool_response.output == SHOT
    assert tool_response.content is None
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)
    assert shot.output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_await_tool_returning_response_with_png_returns_multipart_tool_response_with_png() -> None:
    """`await screenshot()` returning `response(ShotOut(...), parts=[png])` gives back the ShotOut and the PNG."""
    ai = Genkit()

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse[ShotOut]:
        return response(ShotOut(ok=True, label='lab'), parts=[_png()], metadata={'src': 'cam'})

    out = await screenshot()
    assert isinstance(out, MultipartToolResponse)
    assert isinstance(out.output, ShotOut)
    assert out.output == ShotOut(ok=True, label='lab')
    assert out.content == [_png()]
    assert out.metadata == {'src': 'cam'}
    assert screenshot.output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_generate_multipart_shotout_puts_png_on_the_tool_message() -> None:
    """Generate copies dump and PNG onto the tool message; the model is still told ShotOut."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse[ShotOut]:
        return response(ShotOut(ok=True, label='lab'), parts=[_png()], metadata={'src': 'cam'})

    generated = await _generate_tool_turn(ai, pm, name='screenshot')
    _assert_closed_tool_round(generated)
    tool_response, metadata = _tool_response(generated)
    assert tool_response.name == 'screenshot'
    assert tool_response.output == SHOT
    assert tool_response.content == [WIRE_PNG]
    assert metadata == {'src': 'cam'}
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)
    assert screenshot.output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_await_bare_multipart_returns_png_and_tells_model_no_schema() -> None:
    """Bare -> MultipartToolResponse still returns the PNG; output_schema is None."""
    ai = Genkit()

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse:
        return response({'ok': True, 'label': 'lab'}, parts=[_png()], metadata={'src': 'cam'})

    out = await screenshot()
    assert isinstance(out, MultipartToolResponse)
    assert out.output == SHOT
    assert out.content == [_png()]
    assert out.metadata == {'src': 'cam'}
    assert screenshot.output_schema is None


@pytest.mark.asyncio
async def test_generate_bare_multipart_puts_png_on_the_tool_message_with_no_schema() -> None:
    """Generate still puts the PNG on the tool message; the model is told no schema."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse:
        return response({'ok': True, 'label': 'lab'}, parts=[_png()], metadata={'src': 'cam'})

    generated = await _generate_tool_turn(ai, pm, name='screenshot')
    _assert_closed_tool_round(generated)
    tool_response, metadata = _tool_response(generated)
    assert tool_response.name == 'screenshot'
    assert tool_response.output == SHOT
    assert tool_response.content == [WIRE_PNG]
    assert metadata == {'src': 'cam'}
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema is None
    assert screenshot.output_schema is None


@pytest.mark.asyncio
async def test_await_multipart_annotated_tool_with_bare_return_returns_the_bare_value() -> None:
    """`await shot()` on a `-> MultipartToolResponse[ShotOut]` tool that returns a bare ShotOut returns that ShotOut."""
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse[ShotOut]:
        return ShotOut(ok=True, label='lab')  # type: ignore[return-value]

    out: object = await shot()
    assert isinstance(out, ShotOut)
    assert out == ShotOut(ok=True, label='lab')
    assert shot.output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_generate_multipart_shotout_with_bare_return_has_dump_and_no_media() -> None:
    """Generate copies that dump with no media; the model is still told ShotOut."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse[ShotOut]:
        return ShotOut(ok=True, label='lab')  # type: ignore[return-value]

    generated = await _generate_tool_turn(ai, pm, name='shot')
    _assert_closed_tool_round(generated)
    tool_response, metadata = _tool_response(generated)
    assert tool_response.name == 'shot'
    assert tool_response.output == SHOT
    assert tool_response.content is None
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_await_tool_returning_response_without_parts_returns_multipart_tool_response_with_no_media() -> None:
    """`await shot()` returning `response(ShotOut(...))` gives back the ShotOut as output and no media."""
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse[ShotOut]:
        return response(ShotOut(ok=True, label='lab'))

    out = await shot()
    assert isinstance(out, MultipartToolResponse)
    assert out.output == ShotOut(ok=True, label='lab')
    assert out.content is None
    assert out.metadata is None
    assert shot.output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_generate_response_without_parts_has_dump_and_no_media() -> None:
    """Generate copies that dump with no media; the model is still told ShotOut."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse[ShotOut]:
        return response(ShotOut(ok=True, label='lab'))

    generated = await _generate_tool_turn(ai, pm, name='shot')
    _assert_closed_tool_round(generated)
    tool_response, metadata = _tool_response(generated)
    assert tool_response.name == 'shot'
    assert tool_response.output == SHOT
    assert tool_response.content is None
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_action_run_multipart_shotout_returns_dump_and_png() -> None:
    """`await screenshot.action().run()` still returns a MultipartToolResponse with the {ok, label} dump and the PNG."""
    ai = Genkit()

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse[ShotOut]:
        return response(ShotOut(ok=True, label='lab'), parts=[_png()], metadata={'src': 'cam'})

    ran = await screenshot.action().run()
    assert isinstance(ran.response, MultipartToolResponse)
    assert ran.response.output == SHOT
    assert ran.response.content == [_png()]
    assert ran.response.metadata == {'src': 'cam'}


@pytest.mark.asyncio
async def test_generate_response_with_png_and_text_puts_both_on_the_tool_message() -> None:
    """Two parts on response() both land on the tool message."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse[ShotOut]:
        return response(ShotOut(ok=True, label='lab'), parts=[_png(), _caption()])

    generated = await _generate_tool_turn(ai, pm, name='screenshot')
    _assert_closed_tool_round(generated)
    tool_response, metadata = _tool_response(generated)
    assert tool_response.name == 'screenshot'
    assert tool_response.output == SHOT
    assert tool_response.content == [WIRE_PNG, WIRE_CAPTION]
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_await_str_tool_returning_response_returns_multipart_tool_response() -> None:
    """`await weather('Austin')` on a `-> str` tool returning `response(..., parts=[png])` gives back that response."""
    ai = Genkit()

    @ai.tool(name='weather')
    async def weather(city: str) -> str:
        return response(f'Sunny in {city}', parts=[_png()])  # type: ignore[return-value]

    out: object = await weather('Austin')
    assert isinstance(out, MultipartToolResponse)
    assert out.output == 'Sunny in Austin'
    assert out.content == [_png()]
    assert out.metadata is None
    assert weather.output_schema == {'type': 'string'}


@pytest.mark.asyncio
async def test_generate_str_tool_may_put_png_on_the_tool_message_without_changing_schema() -> None:
    """Generate puts that PNG on the tool message; the model is still told string."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='weather')
    async def weather(city: str) -> str:
        return response(f'Sunny in {city}', parts=[_png()])  # type: ignore[return-value]

    generated = await _generate_tool_turn(ai, pm, name='weather', tool_input='Austin')
    _assert_closed_tool_round(generated)
    tool_response, metadata = _tool_response(generated)
    assert tool_response.name == 'weather'
    assert tool_response.output == 'Sunny in Austin'
    assert tool_response.content == [WIRE_PNG]
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == {'type': 'string'}
    assert weather.output_schema == {'type': 'string'}


@pytest.mark.asyncio
async def test_optional_multipart_shotout_annotation_tells_model_shotout() -> None:
    """-> MultipartToolResponse[ShotOut] | None still tells the model ShotOut."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse[ShotOut] | None:
        return response(ShotOut(ok=True, label='lab'), parts=[_png()])

    generated = await _generate_tool_turn(ai, pm, name='screenshot')
    _assert_closed_tool_round(generated)
    tool_response, metadata = _tool_response(generated)
    assert tool_response.name == 'screenshot'
    assert tool_response.output == SHOT
    assert tool_response.content == [WIRE_PNG]
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)
    assert screenshot.output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_await_tool_return_types_match_annotation() -> None:
    """Type checkers see `await tool(...)` as the function's own return type, context tools included."""
    ai = Genkit()

    @ai.tool()
    async def add_pair(pair: Pair) -> int:
        return pair.a + pair.b

    @ai.tool()
    async def shot() -> ShotOut:
        return ShotOut(ok=True, label='lab')

    @ai.tool()
    async def log_visit(city: str) -> None:
        return None

    @ai.tool()
    async def screenshot() -> MultipartToolResponse[ShotOut]:
        return response(ShotOut(ok=True, label='lab'), parts=[_png()])

    @ai.tool()
    async def whoami(ctx: ToolRunContext) -> str:
        return str(ctx.context.get('user'))

    assert_type(await add_pair(Pair(a=2, b=3)), int)
    assert_type(await shot(), ShotOut)
    assert_type(await log_visit('Austin'), None)
    assert_type(await screenshot(), MultipartToolResponse[ShotOut])
    assert_type(await whoami(context={'user': 'u1'}), str)
    assert await whoami(context={'user': 'u1'}) == 'u1'


@pytest.mark.asyncio
async def test_await_tool_with_unserializable_output_raises_invalid_argument() -> None:
    """`await snap()` returning something a model can't receive raises INVALID_ARGUMENT, same message as generate."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='snap')
    async def snap() -> dict[str, Any]:
        return {'cam': Camera()}

    with pytest.raises(GenkitError) as exc:
        await snap()
    assert exc.value.status == 'INVALID_ARGUMENT'
    assert exc.value.original_message == "Tool 'snap' output is not JSON-serializable."

    generated = await _generate_tool_turn(ai, pm, name='snap')
    assert generated.finish_reason == FinishReason.FAILED
    assert generated.finish_message is not None
    assert exc.value.original_message in generated.finish_message


@pytest.mark.asyncio
async def test_generate_after_direct_call_sends_model_the_same_tool_output() -> None:
    """`await shot()` then `ai.generate(tools=['shot'])`: the caller got ShotOut, the model still gets the dump."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='shot')
    async def shot() -> ShotOut:
        return ShotOut(ok=True, label='lab')

    direct = await shot()
    generated = await _generate_tool_turn(ai, pm, name='shot')
    _assert_closed_tool_round(generated)
    tool_response, _ = _tool_response(generated)
    assert direct == ShotOut(ok=True, label='lab')
    assert tool_response.output == SHOT
