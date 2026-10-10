# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The return annotation is the schema. Callers get a MultipartToolResponse box."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from genkit import (
    FinishReason,
    Genkit,
    GenkitError,
    Message,
    ModelResponse,
    MultipartToolResponse,
    Part,
    Role,
    RuntimeErrorReason,
    Tool,
)
from genkit._core._action import Action
from genkit._core._schema import to_json_schema
from genkit.model import ToolRequest, ToolResponse
from genkit.plugin_api import ActionKind
from genkit.testing import ScriptedModel, define_scripted_model


class ShotOut(BaseModel):
    ok: bool
    label: str


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
async def test_await_tool_from_raw_action_returns_multipart_envelope() -> None:
    """Awaiting a Tool built from a raw Action returns the same multipart envelope generate already speaks."""

    async def raw_shot() -> MultipartToolResponse[ShotOut]:
        return MultipartToolResponse(output=ShotOut(ok=True, label='lab'), content=[_png()], metadata={'src': 'cam'})

    handle = Tool(Action(name='shot', kind=ActionKind.TOOL, fn=raw_shot))
    out = await handle()
    assert isinstance(out, MultipartToolResponse)
    assert out.output == SHOT
    assert out.content == [_png()]
    assert out.metadata == {'src': 'cam'}


@pytest.mark.asyncio
async def test_await_str_tool_returns_box_with_string_and_no_media() -> None:
    """await weather('Austin') after -> str is a box: the string, no media."""
    ai = Genkit()

    @ai.tool(name='weather')
    async def weather(city: str) -> str:
        return f'Sunny in {city}'

    out = await weather('Austin')
    assert isinstance(out, MultipartToolResponse)
    assert out.output == 'Sunny in Austin'
    assert out.content is None
    assert out.metadata is None
    assert weather.output_schema == {'type': 'string'}


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
    sent, metadata = _tool_response(generated)
    assert sent.name == 'weather'
    assert sent.output == 'Sunny in Austin'
    assert sent.content is None
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == {'type': 'string'}
    assert weather.output_schema == {'type': 'string'}


@pytest.mark.asyncio
async def test_action_run_str_tool_returns_the_same_box() -> None:
    """Action.run / Dev UI Run is the same box as await weather()."""
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
async def test_await_pydantic_tool_returns_box_with_dump_and_no_media() -> None:
    """await shot() after -> ShotOut dumps {ok, label} and has no media."""
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> ShotOut:
        return ShotOut(ok=True, label='lab')

    out = await shot()
    assert isinstance(out, MultipartToolResponse)
    assert out.output == SHOT
    assert out.content is None
    assert out.metadata is None
    assert shot.output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_generate_pydantic_tool_message_has_dump_and_no_media() -> None:
    """Generate copies that dump onto the tool message and tells the model ShotOut."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='shot')
    async def shot() -> ShotOut:
        return ShotOut(ok=True, label='lab')

    generated = await _generate_tool_turn(ai, pm, name='shot')
    _assert_closed_tool_round(generated)
    sent, metadata = _tool_response(generated)
    assert sent.name == 'shot'
    assert sent.output == SHOT
    assert sent.content is None
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)
    assert shot.output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_await_multipart_shotout_returns_dump_and_png() -> None:
    """-> MultipartToolResponse[ShotOut] returning MultipartToolResponse(..., content=[png]) is dump and PNG."""
    ai = Genkit()

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse[ShotOut]:
        return MultipartToolResponse(output=ShotOut(ok=True, label='lab'), content=[_png()], metadata={'src': 'cam'})

    out = await screenshot()
    assert isinstance(out, MultipartToolResponse)
    assert out.output == SHOT
    assert out.content == [_png()]
    assert out.metadata == {'src': 'cam'}
    assert screenshot.output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_generate_tool_returning_tool_response_with_image_gives_model_output_and_image_part() -> None:
    """A tool returning MultipartToolResponse(output=ShotOut(...), content=[png]) sends the dump plus the PNG."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse[ShotOut]:
        return MultipartToolResponse(output=ShotOut(ok=True, label='lab'), content=[_png()], metadata={'src': 'cam'})

    generated = await _generate_tool_turn(ai, pm, name='screenshot')
    _assert_closed_tool_round(generated)
    sent, metadata = _tool_response(generated)
    assert sent.name == 'screenshot'
    assert sent.output == SHOT
    assert sent.content == [WIRE_PNG]
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
        return MultipartToolResponse(output={'ok': True, 'label': 'lab'}, content=[_png()], metadata={'src': 'cam'})

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
        return MultipartToolResponse(output={'ok': True, 'label': 'lab'}, content=[_png()], metadata={'src': 'cam'})

    generated = await _generate_tool_turn(ai, pm, name='screenshot')
    _assert_closed_tool_round(generated)
    sent, metadata = _tool_response(generated)
    assert sent.name == 'screenshot'
    assert sent.output == SHOT
    assert sent.content == [WIRE_PNG]
    assert metadata == {'src': 'cam'}
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema is None
    assert screenshot.output_schema is None


@pytest.mark.asyncio
async def test_await_multipart_shotout_with_bare_return_has_dump_and_no_media() -> None:
    """Annotated MultipartToolResponse[ShotOut] with return ShotOut(...) is dump, no media."""
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse[ShotOut]:
        return ShotOut(ok=True, label='lab')  # type: ignore[return-value]

    out = await shot()
    assert isinstance(out, MultipartToolResponse)
    assert out.output == SHOT
    assert out.content is None
    assert out.metadata is None
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
    sent, metadata = _tool_response(generated)
    assert sent.name == 'shot'
    assert sent.output == SHOT
    assert sent.content is None
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_await_response_without_parts_has_dump_and_no_media() -> None:
    """MultipartToolResponse with no content is dump and no media."""
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse[ShotOut]:
        return MultipartToolResponse(output=ShotOut(ok=True, label='lab'))

    out = await shot()
    assert isinstance(out, MultipartToolResponse)
    assert out.output == SHOT
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
        return MultipartToolResponse(output=ShotOut(ok=True, label='lab'))

    generated = await _generate_tool_turn(ai, pm, name='shot')
    _assert_closed_tool_round(generated)
    sent, metadata = _tool_response(generated)
    assert sent.name == 'shot'
    assert sent.output == SHOT
    assert sent.content is None
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_action_run_multipart_shotout_returns_dump_and_png() -> None:
    """Action.run of a screenshot tool is the same dump-and-PNG box as await."""
    ai = Genkit()

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse[ShotOut]:
        return MultipartToolResponse(output=ShotOut(ok=True, label='lab'), content=[_png()], metadata={'src': 'cam'})

    ran = await screenshot.action().run()
    assert isinstance(ran.response, MultipartToolResponse)
    assert ran.response.output == SHOT
    assert ran.response.content == [_png()]
    assert ran.response.metadata == {'src': 'cam'}


@pytest.mark.asyncio
async def test_generate_response_with_png_and_text_puts_both_on_the_tool_message() -> None:
    """Two content parts on MultipartToolResponse both land on the tool message."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='screenshot')
    async def screenshot() -> MultipartToolResponse[ShotOut]:
        return MultipartToolResponse(output=ShotOut(ok=True, label='lab'), content=[_png(), _caption()])

    generated = await _generate_tool_turn(ai, pm, name='screenshot')
    _assert_closed_tool_round(generated)
    sent, metadata = _tool_response(generated)
    assert sent.name == 'screenshot'
    assert sent.output == SHOT
    assert sent.content == [WIRE_PNG, WIRE_CAPTION]
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_await_str_tool_may_return_png_without_changing_schema() -> None:
    """-> str returning MultipartToolResponse(output='Sunny', content=[png]) keeps the string schema and the PNG."""
    ai = Genkit()

    @ai.tool(name='weather')
    async def weather(city: str) -> str:
        return MultipartToolResponse(output=f'Sunny in {city}', content=[_png()])  # type: ignore[return-value]

    out = await weather('Austin')
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
        return MultipartToolResponse(output=f'Sunny in {city}', content=[_png()])  # type: ignore[return-value]

    generated = await _generate_tool_turn(ai, pm, name='weather', tool_input='Austin')
    _assert_closed_tool_round(generated)
    sent, metadata = _tool_response(generated)
    assert sent.name == 'weather'
    assert sent.output == 'Sunny in Austin'
    assert sent.content == [WIRE_PNG]
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
        return MultipartToolResponse(output=ShotOut(ok=True, label='lab'), content=[_png()])

    generated = await _generate_tool_turn(ai, pm, name='screenshot')
    _assert_closed_tool_round(generated)
    sent, metadata = _tool_response(generated)
    assert sent.name == 'screenshot'
    assert sent.output == SHOT
    assert sent.content == [WIRE_PNG]
    assert metadata is None
    assert pm.last_request is not None
    assert pm.last_request.tools is not None
    assert pm.last_request.tools[0].output_schema == to_json_schema(ShotOut)
    assert screenshot.output_schema == to_json_schema(ShotOut)


async def _generate_with_failing_tool(ai: Genkit, pm: ScriptedModel, *, name: str) -> ModelResponse:
    generated = await _generate_tool_turn(ai, pm, name=name)
    assert generated.finish_reason == FinishReason.FAILED
    assert generated.error is not None
    return generated


@pytest.mark.asyncio
async def test_tool_response_with_non_dict_metadata_raises_invalid_argument_naming_tool_response() -> None:
    """MultipartToolResponse(metadata='nope') raises INVALID_ARGUMENT naming the class; generate reports the same."""
    with pytest.raises(GenkitError) as raised:
        MultipartToolResponse(output=ShotOut(ok=True, label='lab'), metadata='nope')  # type: ignore[arg-type]
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert raised.value.original_message == 'MultipartToolResponse metadata must be a dict, got str.'

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse[ShotOut]:
        return MultipartToolResponse(output=ShotOut(ok=True, label='lab'), metadata='nope')  # type: ignore[arg-type]

    generated = await _generate_with_failing_tool(ai, pm, name='shot')
    assert generated.error is not None
    assert generated.error.status == 'INVALID_ARGUMENT'
    assert generated.finish_message == 'MultipartToolResponse metadata must be a dict, got str.'


@pytest.mark.asyncio
async def test_tool_response_with_non_part_in_parts_raises_invalid_part_naming_tool_response() -> None:
    """MultipartToolResponse(content=[object()]) raises INVALID_PART naming the class; generate reports the same."""
    with pytest.raises(GenkitError) as raised:
        MultipartToolResponse(output=ShotOut(ok=True, label='lab'), content=[object()])  # type: ignore[list-item]
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.reason is RuntimeErrorReason.INVALID_PART
    assert (
        raised.value.original_message
        == 'MultipartToolResponse content must be a list of Parts, got object in the list.'
    )

    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse[ShotOut]:
        return MultipartToolResponse(output=ShotOut(ok=True, label='lab'), content=[object()])  # type: ignore[list-item]

    generated = await _generate_with_failing_tool(ai, pm, name='shot')
    assert generated.error is not None
    assert generated.error.status == 'INVALID_ARGUMENT'
    assert generated.finish_message == 'MultipartToolResponse content must be a list of Parts, got object in the list.'
