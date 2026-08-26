# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tool envelope: ``response()``, direct calls, schemas, and ``tool.v2`` keys."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

import pytest
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit import ActionKind, Genkit, Message, MiddlewareRef, ModelResponse
from genkit._ai._generate import generate_action
from genkit._ai._testing import define_programmable_model
from genkit._ai._tools import parts_to_wire, response
from genkit._core._action import create_action_key, parse_action_key
from genkit._core._error import GenkitError
from genkit._core._model import GenerateActionOptions
from genkit._core._schema import to_json_schema
from genkit._core._typing import (
    DataPart,
    FinishReason,
    Media,
    MediaPart,
    MultipartToolResponse,
    Part,
    ReasoningPart,
    Resource,
    ResourcePart,
    TextPart,
)
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ToolHookParams


def _png() -> Part:
    return Part(root=MediaPart(media=Media(content_type='image/png', url='data:image/png;base64,abc')))


WIRE_PNG = {'media': {'contentType': 'image/png', 'url': 'data:image/png;base64,abc'}}


def test_response_text_part_is_live() -> None:
    env = response({'ok': True}, parts=TextPart(text='lab camera'))
    assert parts_to_wire(env.content) == [{'text': 'lab camera'}]


def test_response_data_and_reasoning_parts_are_live() -> None:
    data = response({'ok': True}, parts=DataPart(data={'rows': [1]}))
    assert parts_to_wire(data.content) == [{'data': {'rows': [1]}}]
    thought = response({'ok': True}, parts=ReasoningPart(reasoning='checking the label'))
    assert parts_to_wire(thought.content) == [{'reasoning': 'checking the label'}]
    res = response({'ok': True}, parts=ResourcePart(resource=Resource(uri='file://shot.png')))
    assert parts_to_wire(res.content) == [{'resource': {'uri': 'file://shot.png'}}]


def test_response_rejects_hollow_parts() -> None:
    with pytest.raises(GenkitError) as ei:
        response({'ok': True}, parts=Part(root=DataPart()))
    assert ei.value.status == 'INVALID_ARGUMENT'
    assert 'no live payload' in ei.value.original_message


def test_response_builds_the_envelope() -> None:
    env = response({'ok': True}, parts=[_png()], metadata={'src': 'test'})
    assert isinstance(env, MultipartToolResponse)
    assert env.output == {'ok': True}
    assert env.content is not None
    assert env.metadata == {'src': 'test'}


def test_response_without_parts_is_output_only() -> None:
    env = response({'ok': True})
    assert env.output == {'ok': True}
    assert env.content is None


def test_response_wraps_a_bare_media_part() -> None:
    env = response({'ok': True}, parts=_png().root)
    assert env.content is not None
    assert len(env.content) == 1
    assert parts_to_wire(env.content) == [WIRE_PNG]


def test_response_rejects_non_part_parts() -> None:
    with pytest.raises(GenkitError) as ei:
        response({'ok': True}, parts='not-a-part')  # type: ignore[arg-type]
    assert ei.value.status == 'INVALID_ARGUMENT'
    assert 'parts' in ei.value.original_message


def test_response_rejects_non_dict_metadata() -> None:
    with pytest.raises(GenkitError) as ei:
        response(1, metadata='nope')  # type: ignore[arg-type]
    assert ei.value.status == 'INVALID_ARGUMENT'
    assert 'metadata' in ei.value.original_message


class CamelOut(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    content_type: str
    taken_at: datetime


class Camera:
    pass


def test_unserializable_part_data_is_invalid_argument() -> None:
    with pytest.raises(GenkitError) as ei:
        response({'ok': True}, parts=DataPart(data={'cam': Camera()}))
    assert ei.value.status == 'INVALID_ARGUMENT'
    assert 'response()' in ei.value.original_message
    assert 'content' in ei.value.original_message


@pytest.mark.asyncio
async def test_direct_tool_call_returns_envelope() -> None:
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot(label: str) -> MultipartToolResponse:
        return response({'ok': True, 'label': label}, parts=[_png()])

    out = await shot('lab')
    assert isinstance(out, MultipartToolResponse)
    assert out.output == {'ok': True, 'label': 'lab'}
    assert out.content is not None


@pytest.mark.asyncio
async def test_bare_return_is_still_output_only() -> None:
    ai = Genkit()

    @ai.tool(name='weather')
    async def weather(city: str) -> str:
        return f'Sunny in {city}'

    out = await weather('Austin')
    assert isinstance(out, MultipartToolResponse)
    assert out.output == 'Sunny in Austin'
    assert out.content is None


@pytest.mark.asyncio
async def test_pydantic_return_dumps_json_aliases() -> None:
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> CamelOut:
        return CamelOut(content_type='image/png', taken_at=datetime(2026, 8, 25, 12, 0))

    out = await shot()
    assert out.output == {'contentType': 'image/png', 'takenAt': '2026-08-25T12:00:00'}


@pytest.mark.asyncio
async def test_response_metadata_datetime_is_json() -> None:
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse:
        return response({'ok': True}, metadata={'when': datetime(2026, 8, 25, 12, 0)})

    out = await shot()
    assert out.metadata == {'when': '2026-08-25T12:00:00'}


@pytest.mark.asyncio
async def test_response_pydantic_output_plus_media() -> None:
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse:
        return response(
            CamelOut(content_type='image/png', taken_at=datetime(2026, 8, 25, 12, 0)),
            parts=[_png()],
        )

    out = await shot()
    assert out.output == {'contentType': 'image/png', 'takenAt': '2026-08-25T12:00:00'}
    assert parts_to_wire(out.content) == [WIRE_PNG]


@pytest.mark.asyncio
async def test_unserializable_metadata_is_invalid_argument() -> None:
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse:
        return response({'ok': True}, metadata={'cam': Camera()})

    with pytest.raises(GenkitError) as ei:
        await shot()
    assert ei.value.status == 'INVALID_ARGUMENT'
    assert 'shot' in ei.value.original_message
    assert 'metadata' in ei.value.original_message


async def _tools_sent_to_model(ai: Genkit, tool_name: str, *, tool_input: dict | None = None) -> list:
    pm, _ = define_programmable_model(ai)
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [{'toolRequest': {'ref': 't1', 'name': tool_name, 'input': tool_input or {}}}],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'ok'}]}),
        )
    )
    await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'go'}]})],
            tools=[tool_name],
        ),
    )
    assert pm.last_request is not None
    assert pm.last_request.tools
    return pm.last_request.tools


@pytest.mark.asyncio
async def test_envelope_annotation_does_not_advertise_wrapper_to_model() -> None:
    ai = Genkit()

    @ai.tool(name='shot')
    async def shot() -> MultipartToolResponse:
        return response({'ok': True}, parts=[_png()])

    tools = await _tools_sent_to_model(ai, 'shot')
    assert tools[0].output_schema is None


class ShotIn(BaseModel):
    city: str


class ShotOut(BaseModel):
    ok: bool
    label: str


@pytest.mark.asyncio
async def test_input_schema_override_is_what_the_model_sees() -> None:
    ai = Genkit()

    @ai.tool(name='shot', input_schema=ShotIn)
    async def shot(inp: dict) -> dict:  # noqa: ARG001
        return {'ok': True}

    tools = await _tools_sent_to_model(ai, 'shot', tool_input={'city': 'Austin'})
    assert tools[0].input_schema == to_json_schema(ShotIn)


@pytest.mark.asyncio
async def test_output_schema_pydantic_model_is_what_the_model_sees() -> None:
    ai = Genkit()

    @ai.tool(name='shot', output_schema=ShotOut)
    async def shot() -> ShotOut:
        return ShotOut(ok=True, label='lab')

    tools = await _tools_sent_to_model(ai, 'shot')
    assert tools[0].output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_output_schema_override_with_response() -> None:
    ai = Genkit()

    @ai.tool(name='shot', output_schema=ShotOut)
    async def shot() -> MultipartToolResponse:
        return response(ShotOut(ok=True, label='lab'), parts=[_png()])

    tools = await _tools_sent_to_model(ai, 'shot')
    assert tools[0].output_schema == to_json_schema(ShotOut)


@pytest.mark.asyncio
async def test_output_schema_plain_type_is_invalid_argument() -> None:
    ai = Genkit()

    with pytest.raises(GenkitError) as ei:

        @ai.tool(name='shot', output_schema=str)  # type: ignore[arg-type]
        async def shot() -> str:
            return 'ok'

    assert ei.value.status == 'INVALID_ARGUMENT'
    assert 'shot' in ei.value.original_message
    assert repr(str) in ei.value.original_message


@pytest.mark.asyncio
async def test_generate_copies_parts_onto_tool_response() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='screenshot')
    async def screenshot(_: dict) -> MultipartToolResponse:  # noqa: ARG001
        return response({'ok': True}, parts=[_png()], metadata={'src': 'cam'})

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [{'toolRequest': {'ref': 's1', 'name': 'screenshot', 'input': {}}}],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'nice photo'}]}),
        )
    )

    res = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'snap'}]})],
            tools=['screenshot'],
        ),
    )
    tool_msg = next(m for m in res.messages if m.role == 'tool')
    tr = tool_msg.content[0].root.tool_response
    assert tr is not None
    assert tr.output == {'ok': True}
    assert tr.content == [WIRE_PNG]
    assert tool_msg.content[0].root.metadata == {'src': 'cam'}
    assert pm.last_request is not None
    assert pm.last_request.tools
    assert pm.last_request.tools[0].output_schema is None


@pytest.mark.asyncio
async def test_unserializable_tool_output_is_invalid_argument() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    ran = False

    @ai.tool(name='screenshot')
    async def screenshot(_: dict) -> object:  # noqa: ARG001
        nonlocal ran
        ran = True
        return Camera()

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [{'toolRequest': {'ref': 's1', 'name': 'screenshot', 'input': {}}}],
            }),
        )
    )

    with pytest.raises(GenkitError) as ei:
        await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'snap'}]})],
                tools=['screenshot'],
            ),
        )
    assert ran
    assert ei.value.status == 'INVALID_ARGUMENT'
    assert 'screenshot' in ei.value.original_message


@pytest.mark.asyncio
async def test_tool_registers_under_tool_v2_only() -> None:
    ai = Genkit()

    @ai.tool(name='weather')
    async def weather(city: str) -> str:
        return f'Sunny in {city}'

    assert create_action_key(ActionKind.TOOL, 'weather') == '/tool.v2/weather'
    assert parse_action_key('/tool.v2/weather') == (ActionKind.TOOL, 'weather')

    by_name = await ai.registry.resolve_action(ActionKind.TOOL, 'weather')
    by_new_key = await ai.registry.resolve_action_by_key('/tool.v2/weather')
    assert by_name is weather.action()
    assert by_new_key is weather.action()
    with pytest.raises(ValueError, match='Invalid action kind'):
        await ai.registry.resolve_action_by_key('/tool/weather')


@pytest.mark.asyncio
async def test_generate_sends_declared_output_schema_on_the_model_request() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='weather')
    async def weather(_: dict) -> str:  # noqa: ARG001
        return 'Sunny'

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [{'toolRequest': {'ref': 'w1', 'name': 'weather', 'input': {}}}],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'ok'}]}),
        )
    )

    await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'wx'}]})],
            tools=['weather'],
        ),
    )
    assert pm.last_request is not None
    assert pm.last_request.tools
    assert pm.last_request.tools[0].output_schema == {'type': 'string'}


@pytest.mark.asyncio
async def test_wrap_tool_can_substitute_a_response() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.middleware(name='deny_mw')
    class DenyMW(BaseMiddleware):
        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            return response('denied')

    @ai.tool(name='weather')
    async def weather(_: dict) -> str:  # noqa: ARG001
        return 'Sunny'

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({
                'role': 'model',
                'content': [{'toolRequest': {'ref': 'w1', 'name': 'weather', 'input': {}}}],
            }),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message.model_validate({'role': 'model', 'content': [{'text': 'ok'}]}),
        )
    )

    res = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message.model_validate({'role': 'user', 'content': [{'text': 'wx'}]})],
            tools=['weather'],
            use=[MiddlewareRef(name='deny_mw')],
        ),
    )
    tool_msg = next(m for m in res.messages if m.role == 'tool')
    denied = tool_msg.content[0].root.tool_response
    assert denied is not None
    assert denied.output == 'denied'
