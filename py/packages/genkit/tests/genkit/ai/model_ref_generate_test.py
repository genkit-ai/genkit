#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for generate/prompt with ModelRef."""

import pytest

from genkit import Genkit
from genkit._ai._testing import EchoModel, define_echo_model
from genkit.model import model_ref
from genkit.plugin_api import ModelConfig


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

    response = await ai.generate(model=ref, prompt='Hello')

    assert '0.1' in response.text
    assert echo.last_request is not None
    assert echo.last_request.config is not None
    assert echo.last_request.config.temperature == 0.1


@pytest.mark.asyncio
async def test_generate_string_model_config_dict_unchanged(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """Bare string model path still accepts dict config."""
    ai, echo = ai_with_echo

    response = await ai.generate(model='testEcho', prompt='Hello', config={'temperature': 0.1})

    assert '0.1' in response.text
    assert echo.last_request is not None


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
