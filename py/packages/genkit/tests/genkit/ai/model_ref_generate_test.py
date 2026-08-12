#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for generate/prompt/agent/operation with ModelRef."""

from typing import Any

import pytest
from pydantic import BaseModel

from genkit import Genkit
from genkit._ai._model import ModelConfig
from genkit._ai._testing import EchoModel, define_echo_model
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import Message, ModelRequest, ModelResponse
from genkit._core._typing import ModelInfo, Operation, Part, Role, Supports, TextPart
from genkit.model import model_ref


class CustomConfig(BaseModel):
    """Plugin-style config used for merge tests."""

    temperature: float | None = None
    top_k: float | None = None
    safety_settings: dict[str, str] | None = None


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


@pytest.fixture
def ai() -> Genkit:
    return Genkit()


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
    assert _config_value(echo.last_request.config, 'temperature') == 0.1


@pytest.mark.asyncio
async def test_generate_model_ref_merges_call_time_dict(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Call-time dict config merges over ModelRef defaults (JS parity)."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        config=ModelConfig(temperature=0.2),
    )

    await ai.generate(model=ref, config={'top_k': 0.9}, prompt='Hello')

    assert echo.last_request is not None
    assert echo.last_request.config is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.2
    assert _config_value(echo.last_request.config, 'top_k') == 0.9


@pytest.mark.asyncio
async def test_generate_model_ref_same_key_override(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Call-time config wins when the same key exists on the ModelRef default."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        config=ModelConfig(temperature=0.2),
    )

    await ai.generate(model=ref, config={'temperature': 0.9}, prompt='Hello')

    assert echo.last_request is not None
    assert echo.last_request.config is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.9


@pytest.mark.asyncio
async def test_generate_string_model_config_dict_unchanged(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
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
    assert _config_value(echo.last_request.config, 'temperature') == 0.2


@pytest.mark.asyncio
async def test_generate_operation_with_model_ref(ai: Genkit) -> None:
    """generate_operation accepts a ModelRef and uses its wire name."""
    expected_operation = Operation(
        id='ref-op-123',
        done=False,
        action='/background-model/lro-model',
    )

    async def model_fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='Started'))],
            ),
            operation=expected_operation,
        )

    ai.define_model(
        name='lro-model',
        fn=model_fn,
        info=ModelInfo(supports=Supports(long_running=True)),
    )
    ref = model_ref('lro-model', config_schema=ModelConfig, config=ModelConfig(temperature=0.4))

    operation = await ai.generate_operation(model=ref, prompt='Generate video')

    assert operation.id == 'ref-op-123'


@pytest.mark.asyncio
async def test_generate_operation_model_ref_rejects_non_lro(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """generate_operation still rejects ModelRefs whose model lacks LRO support."""
    ai, _ = ai_with_echo
    ref = model_ref('testEcho', config_schema=ModelConfig)

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate_operation(model=ref, prompt='Hi')

    assert 'does not support long running' in str(exc_info.value)


@pytest.mark.asyncio
async def test_define_agent_with_model_ref(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """define_agent accepts a ModelRef and uses resolved name/config on turns."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        config=ModelConfig(temperature=0.3),
    )

    agent = ai.define_agent(name='echoAgent', model=ref, system='Reply briefly.')
    chat = agent.chat()
    out = await chat.send('Hello')

    assert '[ECHO]' in out.text
    assert echo.last_request is not None
    assert echo.last_request.config is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.3


@pytest.mark.asyncio
async def test_model_ref_version_seeds_config(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """ref.version flows into config at lowest precedence (JS generate.ts parity)."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=ModelConfig, version='001')

    await ai.generate(model=ref, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'version') == '001'


@pytest.mark.asyncio
async def test_model_ref_version_overridden_by_call_config(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Call-time config version beats ref.version."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=ModelConfig, version='001')

    await ai.generate(model=ref, config={'version': '002'}, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'version') == '002'


@pytest.mark.asyncio
async def test_unknown_config_keys_pass_through_to_plugin(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Escape hatch: keys outside the ref schema reach the plugin untouched."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=CustomConfig, config=CustomConfig(temperature=0.7))

    await ai.generate(model=ref, config={'thinking_config': {'budget': 8192}}, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.7
    assert _config_value(echo.last_request.config, 'thinking_config') == {'budget': 8192}


@pytest.mark.asyncio
async def test_explicit_none_clears_ref_default_via_generate(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Explicitly-set None clears the ref default; plugin sees absence."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=CustomConfig, config=CustomConfig(temperature=0.7))

    await ai.generate(model=ref, config=CustomConfig(temperature=None), prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') is None


@pytest.mark.asyncio
async def test_explicit_none_clears_default_via_prompt(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """The prompt path honors the same clearing rule as generate."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=CustomConfig)

    prompt = ai.define_prompt(
        name='clearingPrompt',
        model=ref,
        prompt='Hello',
        config=CustomConfig(temperature=0.7),
    )
    await prompt(config=CustomConfig(temperature=None))

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') is None


@pytest.mark.asyncio
async def test_unset_fields_do_not_clobber_ref_defaults(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Unset != None: untouched fields on a typed config cannot clear defaults."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=CustomConfig, config=CustomConfig(temperature=0.7))

    await ai.generate(model=ref, config=CustomConfig(top_k=40), prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.7
    assert _config_value(echo.last_request.config, 'top_k') == 40
