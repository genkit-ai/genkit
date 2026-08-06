#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for generate/prompt with ModelRef."""

from typing import Any

import pytest
from pydantic import BaseModel

from genkit import Genkit
from genkit._ai._model import ModelConfig
from genkit._ai._testing import EchoModel, define_echo_model
from genkit._core._model import Message, ModelRequest, ModelResponse
from genkit._core._typing import ModelInfo, Operation, Role, Supports, TextPart
from genkit.model import model_ref


@pytest.fixture
def ai_with_echo() -> tuple[Genkit, EchoModel]:
    ai = Genkit()
    echo, _ = define_echo_model(ai, name='testEcho')
    return ai, echo


@pytest.mark.asyncio
async def test_generate_with_model_ref(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """generate accepts a ModelRef and resolves its wire name."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=ModelConfig)

    response = await ai.generate(model=ref, prompt='Hello')

    assert '[ECHO]' in response.text
    assert echo.last_request is not None


@pytest.mark.asyncio
async def test_generate_model_ref_default_config(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """Default config on the ref is used when the call omits config."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=ModelConfig, config=ModelConfig(temperature=0.1))

    await ai.generate(model=ref, prompt='Hello')

    assert echo.last_request is not None
    assert echo.last_request.config is not None
    cfg = echo.last_request.config
    assert (cfg['temperature'] if isinstance(cfg, dict) else getattr(cfg, 'temperature', None)) == 0.1


@pytest.mark.asyncio
async def test_generate_string_model_config_dict_unchanged(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """Bare string model path still accepts dict config."""
    ai, echo = ai_with_echo

    response = await ai.generate(model='testEcho', prompt='Hello', config={'temperature': 0.1})

    assert '0.1' in response.text
    assert echo.last_request is not None


class CustomConfig(ModelConfig):
    custom_setting: str | None = None


@pytest.mark.asyncio
async def test_generate_with_model_ref_accepts_matching_dict(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """generate(model=ModelRef[CustomConfig]) accepts matching dictionary literals."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=CustomConfig, config=CustomConfig(temperature=0.9))

    # Passing a dict matching CustomConfig fields
    await ai.generate(model=ref, prompt='Hello', config={'temperature': 0.3, 'stop_sequences': ['\n']})

    assert echo.last_request is not None
    cfg = echo.last_request.config
    assert isinstance(cfg, dict)
    assert cfg['temperature'] == 0.3
    assert cfg['stopSequences'] == ['\n']


@pytest.mark.asyncio
async def test_generate_stream_with_model_ref(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """generate_stream accepts a ModelRef."""
    ai, _ = ai_with_echo
    ref = model_ref('testEcho', config_schema=ModelConfig)

    stream = ai.generate_stream(model=ref, prompt='Hello')
    response = await stream.response

    assert '[ECHO]' in response.text


@pytest.mark.asyncio
async def test_define_prompt_with_model_ref(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """define_prompt stores a ModelRef and unwraps it at execution time."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=ModelConfig, config=ModelConfig(temperature=0.2))

    prompt = ai.define_prompt(
        name='echoPrompt',
        model=ref,
        prompt='Hello',
    )
    response = await prompt()

    assert '0.2' in response.text
    assert echo.last_request is not None


@pytest.mark.asyncio
async def test_generate_operation_with_model_ref(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """generate_operation accepts a ModelRef with typed output_schema and returns Operation."""
    ai, echo = ai_with_echo

    async def lro_model_fn(request: ModelRequest, ctx: Any) -> ModelResponse:
        echo.last_request = request
        return ModelResponse(
            message=Message(role=Role.MODEL, content=[TextPart(text='Started')]),
            operation=Operation(id='op-123', done=False),
        )

    ai.define_model(
        name='lroEcho',
        fn=lro_model_fn,
        info=ModelInfo(supports=Supports(long_running=True)),
    )

    ref = model_ref('lroEcho', config_schema=ModelConfig, config=ModelConfig(temperature=0.3))
    op = await ai.generate_operation(model=ref, prompt='Hello')

    assert isinstance(op, Operation)
    assert op.id == 'op-123'
    assert echo.last_request is not None
    cfg = echo.last_request.config
    assert (cfg['temperature'] if isinstance(cfg, dict) else getattr(cfg, 'temperature', None)) == 0.3
