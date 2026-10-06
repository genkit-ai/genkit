#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Passing define_model's action to generate is the same call as passing its name."""

from typing import Any

import pytest

from genkit import Genkit
from genkit._ai._model import ModelConfig
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import Message, ModelRequest, ModelResponse, Part
from genkit._core._typing import Operation, Role
from genkit.model import model_ref
from genkit.testing import define_echo_model


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


async def _chunk_texts(stream: Any) -> list[str]:
    return [chunk.text async for chunk in stream.stream]


@pytest.mark.asyncio
async def test_generate_with_define_model_action_returns_the_same_reply_as_its_name() -> None:
    """define_model still returns an action; generate(model=that action) matches model=the name."""
    ai = Genkit()
    echo, action = define_echo_model(ai, name='local/echo')

    by_action = await ai.generate(model=action, prompt='hi')
    by_name = await ai.generate(model='local/echo', prompt='hi')

    assert type(action).__name__ == 'Action'
    assert by_action.text == by_name.text
    assert by_action.message == by_name.message
    assert echo.last_request is not None


@pytest.mark.asyncio
async def test_generate_with_define_model_action_sends_the_call_temperature() -> None:
    """generate(model=action, config={'temperature': 0.2}) sends that temperature, same as the name."""
    ai = Genkit()
    echo, action = define_echo_model(ai, name='local/echo')

    await ai.generate(model=action, prompt='hi', config={'temperature': 0.2})
    action_config = echo.last_request.config if echo.last_request else None
    await ai.generate(model='local/echo', prompt='hi', config={'temperature': 0.2})
    name_config = echo.last_request.config if echo.last_request else None

    assert _config_value(action_config, 'temperature') == 0.2
    assert _config_value(action_config, 'temperature') == _config_value(name_config, 'temperature')


@pytest.mark.asyncio
async def test_generate_with_define_model_action_keeps_max_output_tokens_key() -> None:
    """generate(model=action, config={'maxOutputTokens': 5}) sends maxOutputTokens, same as the name."""
    ai = Genkit()
    echo, action = define_echo_model(ai, name='local/echo')

    await ai.generate(model=action, prompt='hi', config={'maxOutputTokens': 5})
    action_config = echo.last_request.config if echo.last_request else None
    await ai.generate(model='local/echo', prompt='hi', config={'maxOutputTokens': 5})
    name_config = echo.last_request.config if echo.last_request else None

    assert _config_value(action_config, 'maxOutputTokens') == 5
    assert _config_value(action_config, 'maxOutputTokens') == _config_value(name_config, 'maxOutputTokens')


@pytest.mark.asyncio
async def test_generate_with_define_model_action_omits_null_temperature() -> None:
    """generate(model=action, config={'temperature': None, 'top_k': 1}) omits temperature, same as the name."""
    ai = Genkit()
    echo, action = define_echo_model(ai, name='local/echo')

    await ai.generate(model=action, prompt='hi', config={'temperature': None, 'top_k': 1})
    action_config = echo.last_request.config if echo.last_request else None
    await ai.generate(model='local/echo', prompt='hi', config={'temperature': None, 'top_k': 1})
    name_config = echo.last_request.config if echo.last_request else None

    assert _config_value(action_config, 'temperature') is None
    assert _config_value(action_config, 'top_k') == 1
    assert _config_value(action_config, 'temperature') == _config_value(name_config, 'temperature')
    assert _config_value(action_config, 'top_k') == _config_value(name_config, 'top_k')


@pytest.mark.asyncio
async def test_generate_with_define_model_action_does_not_call_the_constructor_model() -> None:
    """With a different constructor default, generate(model=echo_action) returns echo and does not call the default."""
    default_hits: list[str] = []

    async def default_fn(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        default_hits.append('called')
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('default')]))

    ai = Genkit(model='default/model')
    ai.define_model(name='default/model', fn=default_fn)
    echo, action = define_echo_model(ai, name='local/echo')

    response = await ai.generate(model=action, prompt='hi')

    assert '[ECHO]' in response.text
    assert default_hits == []
    assert echo.last_request is not None


@pytest.mark.asyncio
async def test_generate_with_define_model_action_same_name_does_not_send_constructor_temperature() -> None:
    """generate(model=action) does not send constructor temperature when the action shares that model's name."""
    flash = model_ref('flash', config_schema=ModelConfig, config=ModelConfig(temperature=0.7))
    ai = Genkit(model=flash)
    echo, action = define_echo_model(ai, name='flash')

    await ai.generate(model=action, prompt='hi')
    action_config = echo.last_request.config if echo.last_request else None
    await ai.generate(model='flash', prompt='hi')
    name_config = echo.last_request.config if echo.last_request else None

    assert _config_value(action_config, 'temperature') is None
    assert _config_value(action_config, 'temperature') == _config_value(name_config, 'temperature')


@pytest.mark.asyncio
async def test_generate_stream_with_define_model_action_yields_the_same_chunks_and_reply() -> None:
    """generate_stream(model=action) yields the same chunk texts and final reply as the name."""
    ai = Genkit()
    echo, action = define_echo_model(ai, name='local/echo', stream_countdown=True)

    action_stream = ai.generate_stream(model=action, prompt='hi')
    action_chunks = await _chunk_texts(action_stream)
    action_reply = await action_stream.response

    name_stream = ai.generate_stream(model='local/echo', prompt='hi')
    name_chunks = await _chunk_texts(name_stream)
    name_reply = await name_stream.response

    assert action_chunks == name_chunks == ['3', '2', '1']
    assert action_reply.text == name_reply.text
    assert echo.last_request is not None


@pytest.mark.asyncio
async def test_prompt_stream_with_define_model_action_yields_the_same_chunks_and_reply() -> None:
    """prompt.stream(model=action) yields the same chunk texts and final reply as the name."""
    ai = Genkit()
    echo, action = define_echo_model(ai, name='local/echo', stream_countdown=True)
    prompt = ai.define_prompt(name='p', prompt='hi', model='other/model')

    action_stream = prompt.stream(model=action)
    action_chunks = await _chunk_texts(action_stream)
    action_reply = await action_stream.response

    name_stream = prompt.stream(model='local/echo')
    name_chunks = await _chunk_texts(name_stream)
    name_reply = await name_stream.response

    assert action_chunks == name_chunks == ['3', '2', '1']
    assert action_reply.text == name_reply.text
    assert echo.last_request is not None


@pytest.mark.asyncio
async def test_prompt_with_define_model_action_returns_the_same_reply_as_its_name() -> None:
    """A prompt stored with another model, called as prompt(model=action), matches prompt(model=the name)."""
    ai = Genkit()
    echo, action = define_echo_model(ai, name='local/echo')
    prompt = ai.define_prompt(name='p', prompt='hi', model='other/model')

    by_action = await prompt(model=action)
    by_name = await prompt(model='local/echo')

    assert by_action.text == by_name.text
    assert echo.last_request is not None


@pytest.mark.asyncio
async def test_define_prompt_with_define_model_action_returns_the_same_reply_as_its_name() -> None:
    """define_prompt(model=action) then calling the prompt matches define_prompt(model=the name)."""
    ai = Genkit()
    echo, action = define_echo_model(ai, name='local/echo')

    by_action = await ai.define_prompt(name='p-action', prompt='hi', model=action)()
    by_name = await ai.define_prompt(name='p-name', prompt='hi', model='local/echo')()

    assert by_action.text == by_name.text
    assert echo.last_request is not None


@pytest.mark.asyncio
async def test_generate_with_define_model_action_returns_the_same_failed_response_as_its_name() -> None:
    """When the model function raises, generate(model=action) returns the same failed response as the name."""

    async def boom(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        raise RuntimeError('model down')

    ai = Genkit()
    action = ai.define_model(name='boom', fn=boom)

    by_action = await ai.generate(model=action, prompt='hi')
    by_name = await ai.generate(model='boom', prompt='hi')

    assert by_action.finish_reason == by_name.finish_reason
    assert by_action.error == by_name.error
    assert by_action.message is None
    assert by_name.message is None


@pytest.mark.asyncio
async def test_generate_stream_with_define_model_action_returns_the_same_failed_response_as_its_name() -> None:
    """When the model function raises, generate_stream(model=action).response matches the name."""

    async def boom(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        raise RuntimeError('model down')

    ai = Genkit()
    action = ai.define_model(name='boom', fn=boom)

    by_action = await ai.generate_stream(model=action, prompt='hi').response
    by_name = await ai.generate_stream(model='boom', prompt='hi').response

    assert by_action.finish_reason == by_name.finish_reason
    assert by_action.error == by_name.error
    assert by_action.message is None
    assert by_name.message is None


@pytest.mark.asyncio
async def test_generate_with_background_model_object_raises_and_does_not_call_the_default() -> None:
    """generate(model=the object define_background_model returns) raises INVALID_ARGUMENT; the default is not called."""
    default_hits: list[str] = []

    async def default_fn(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        default_hits.append('called')
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('default')]))

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='op', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai = Genkit(model='default/model')
    ai.define_model(name='default/model', fn=default_fn)
    background = ai.define_background_model(name='bg', start=start, check=check)

    with pytest.raises(GenkitError, match='model is BackgroundAction') as raised:
        await ai.generate(model=background, prompt='hi')  # type: ignore[arg-type]

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert default_hits == []


@pytest.mark.asyncio
async def test_generate_operation_with_define_model_action_raises_the_same_long_running_error() -> None:
    """generate_operation(model=the define_model action) raises the same long-running error as the name."""
    ai = Genkit()
    _echo, action = define_echo_model(ai, name='local/echo')

    with pytest.raises(GenkitError, match='does not support long running') as by_action:
        await ai.generate_operation(model=action, prompt='hi')
    with pytest.raises(GenkitError, match='does not support long running') as by_name:
        await ai.generate_operation(model='local/echo', prompt='hi')

    assert str(by_action.value) == str(by_name.value)
