#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for generate/prompt/agent/operation with ModelRef."""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from genkit import Genkit
from genkit._ai._model import ModelConfig
from genkit._ai._prompt import PromptConfig, to_generate_action_options
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


class ExcludedKeyConfig(ModelConfig):
    """ModelConfig whose api_key is omitted from model_dump."""

    api_key: str | None = Field(default=None, exclude=True)


class OtherFamilyConfig(BaseModel):
    """A second family's knobs, used to pin leftover keys across a hop."""

    temperature: float | None = None
    frequency_penalty: float | None = None


class NestedConfig(BaseModel):
    """Nested bag used to pin shallow replace."""

    thinking: dict[str, object] | None = None


class AliasedNestedConfig(BaseModel):
    """Schema with a generated camel alias, used to pin alias overlay."""

    model_config = ConfigDict(alias_generator=to_camel, extra='allow', populate_by_name=True)
    thinking_config: dict[str, object] | None = None


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
    """Call-time dict config merges over ModelRef defaults."""
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
    """A name has no ref to overlay. The call dict is the whole config."""
    ai, echo = ai_with_echo

    response = await ai.generate(model='testEcho', prompt='Hello', config={'temperature': 0.1})

    assert '0.1' in response.text
    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.1


@pytest.mark.asyncio
async def test_generate_stream_with_model_ref(ai_with_echo: tuple[Genkit, EchoModel]) -> None:
    """generate_stream applies the ref's version and config, not just the name."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(temperature=0.4),
    )

    stream = ai.generate_stream(model=ref, prompt='Hello')
    response = await stream.response

    assert '[ECHO]' in response.text
    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.4
    assert _config_value(echo.last_request.config, 'version') == '001'


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
async def test_define_prompt_dict_none_clear_and_extra(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """define_prompt dicts accept None-clears and extra keys the same as generate()."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=ModelConfig, config=ModelConfig(temperature=0.7))

    prompt = ai.define_prompt(
        name='echoPrompt',
        model=ref,
        prompt='Hello',
        config={'temperature': None, 'banana': True},
    )
    await prompt()

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') is None
    assert _config_value(echo.last_request.config, 'banana') is True


@pytest.mark.asyncio
async def test_generate_operation_with_model_ref(ai: Genkit) -> None:
    """generate_operation applies the ref's version and config, not just the name."""
    expected_operation = Operation(
        id='ref-op-123',
        done=False,
        action='/background-model/lro-model',
    )
    seen: list[ModelRequest] = []

    async def model_fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        seen.append(request)
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
    ref = model_ref(
        'lro-model',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(temperature=0.4),
    )

    operation = await ai.generate_operation(model=ref, prompt='Generate video')

    assert operation.id == 'ref-op-123'
    assert seen
    assert _config_value(seen[0].config, 'temperature') == 0.4
    assert _config_value(seen[0].config, 'version') == '001'


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
    """ref.version flows into config at lowest precedence."""
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


@pytest.mark.asyncio
async def test_model_config_none_clears_ref_default_via_generate(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """ModelConfig(temperature=None) clears a ref default on the generate path."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=ModelConfig, config=ModelConfig(temperature=0.7))

    await ai.generate(model=ref, config=ModelConfig(temperature=None), prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') is None


@pytest.mark.asyncio
async def test_model_config_aliased_field_same_key_override(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Call-time max_output_tokens replaces the ref default instead of adding maxOutputTokens."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        config=ModelConfig(max_output_tokens=100),
    )

    await ai.generate(model=ref, config={'max_output_tokens': 200}, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'max_output_tokens') == 200
    assert _config_value(echo.last_request.config, 'maxOutputTokens') is None


@pytest.mark.asyncio
async def test_excluded_api_key_reaches_plugin(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Per-request api_key still lands on the plugin request after veneer dump."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=ExcludedKeyConfig)

    await ai.generate(model=ref, config=ExcludedKeyConfig(api_key='secret'), prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'api_key') == 'secret'


@pytest.mark.asyncio
async def test_dict_none_clears_ref_default_via_generate(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """A dict ``None`` is the same clear as a typed ``temperature=None``."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=CustomConfig, config=CustomConfig(temperature=0.7))

    await ai.generate(model=ref, config={'temperature': None}, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') is None


@pytest.mark.asyncio
async def test_string_model_none_omits_key(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """None means omit on a name too, same as on a ref."""
    ai, echo = ai_with_echo

    await ai.generate(model='testEcho', config={'temperature': None}, prompt='Hello')

    assert echo.last_request is not None
    cfg = echo.last_request.config
    if isinstance(cfg, dict):
        assert 'temperature' not in cfg
    else:
        assert cfg is not None
        assert 'temperature' not in cfg.model_dump(exclude_unset=True)


@pytest.mark.asyncio
async def test_empty_values_stay_on_generate(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """``0``, ``''``, ``[]``, ``{}``, and ``False`` are values the caller set."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        config=ModelConfig(temperature=0.7, stop_sequences=['STOP'], version='001'),
    )

    await ai.generate(
        model=ref,
        config={
            'temperature': 0,
            'max_output_tokens': 0,
            'version': '',
            'stop_sequences': [],
            'thinking': {},
            'stream': False,
        },
        prompt='Hello',
    )

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0
    assert _config_value(echo.last_request.config, 'max_output_tokens') == 0
    assert _config_value(echo.last_request.config, 'version') == ''
    assert _config_value(echo.last_request.config, 'stop_sequences') == []
    assert _config_value(echo.last_request.config, 'thinking') == {}
    assert _config_value(echo.last_request.config, 'stream') is False


@pytest.mark.asyncio
async def test_nested_bag_replaces_whole_on_generate(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Call-time ``thinking={'budget': 256}`` drops the ref's ``level``."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=NestedConfig,
        config=NestedConfig(thinking={'budget': 1024, 'level': 'low'}),
    )

    await ai.generate(model=ref, config={'thinking': {'budget': 256}}, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'thinking') == {'budget': 256}


@pytest.mark.asyncio
async def test_generate_family_hop_keeps_leftover_keys() -> None:
    """Call-time keys the new schema does not know still reach the plugin."""
    ai = Genkit()
    gpt_echo, _ = define_echo_model(ai, name='gpt')
    gpt = model_ref(
        'gpt',
        config_schema=OtherFamilyConfig,
        config=OtherFamilyConfig(frequency_penalty=0.5),
    )

    await ai.generate(
        model=gpt,
        config={'temperature': 0.7, 'safety_settings': {'HARM': 'BLOCK'}},
        prompt='hi',
    )

    assert gpt_echo.last_request is not None
    assert _config_value(gpt_echo.last_request.config, 'frequency_penalty') == 0.5
    assert _config_value(gpt_echo.last_request.config, 'temperature') == 0.7
    assert _config_value(gpt_echo.last_request.config, 'safety_settings') == {'HARM': 'BLOCK'}


@pytest.mark.asyncio
async def test_prompt_family_hop_keeps_leftover_keys() -> None:
    """A prompt can still be holding a Gemini bag when the call picks another model."""
    ai = Genkit()
    _, _ = define_echo_model(ai, name='flash')
    gpt_echo, _ = define_echo_model(ai, name='gpt')
    flash = model_ref('flash', config_schema=CustomConfig)
    gpt = model_ref(
        'gpt',
        config_schema=OtherFamilyConfig,
        config=OtherFamilyConfig(frequency_penalty=0.5),
    )

    joke = ai.define_prompt(
        name='joke',
        model=flash,
        config=CustomConfig(temperature=0.7, safety_settings={'HARM': 'BLOCK'}),
        prompt='hi',
    )
    await joke(model=gpt)

    assert gpt_echo.last_request is not None
    assert _config_value(gpt_echo.last_request.config, 'frequency_penalty') == 0.5
    assert _config_value(gpt_echo.last_request.config, 'temperature') == 0.7
    assert _config_value(gpt_echo.last_request.config, 'safety_settings') == {'HARM': 'BLOCK'}


@pytest.mark.asyncio
async def test_explicit_string_does_not_leak_constructor_ref() -> None:
    """An explicit name is a different model; constructor knobs stay off it."""
    flash = model_ref(
        'flash',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(temperature=0.7),
    )
    ai = Genkit(model=flash)
    define_echo_model(ai, name='flash')
    gpt_echo, _ = define_echo_model(ai, name='gpt')

    await ai.generate(model='gpt', prompt='hi')

    assert gpt_echo.last_request is not None
    assert _config_value(gpt_echo.last_request.config, 'temperature') != 0.7
    assert _config_value(gpt_echo.last_request.config, 'version') != '001'


@pytest.mark.asyncio
async def test_empty_string_model_uses_constructor_ref() -> None:
    """``model=''`` is omitted, same as an unset ``MODEL=`` env var."""
    flash = model_ref(
        'flash',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(temperature=0.7),
    )
    ai = Genkit(model=flash)
    echo, _ = define_echo_model(ai, name='flash')

    await ai.generate(model='', prompt='hi')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.7
    assert _config_value(echo.last_request.config, 'version') == '001'


@pytest.mark.asyncio
async def test_non_name_model_is_hard_error_not_default() -> None:
    """A leftover int must not silently run the constructor default."""
    flash = model_ref('flash', config_schema=ModelConfig, config=ModelConfig(temperature=0.7))
    ai = Genkit(model=flash)
    echo, _ = define_echo_model(ai, name='flash')

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model=123, prompt='hi')  # type: ignore[arg-type]

    assert 'model is int, expected str or ModelRef' in str(exc_info.value)
    assert echo.last_request is None


@pytest.mark.asyncio
async def test_non_name_config_is_hard_error() -> None:
    """A leftover int on config= is INVALID_ARGUMENT, same as model=123."""
    ai = Genkit()
    echo, _ = define_echo_model(ai, name='echo')

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model='echo', prompt='hi', config=123)  # type: ignore[arg-type]

    assert 'config is int, expected Mapping or BaseModel' in str(exc_info.value)
    assert echo.last_request is None


@pytest.mark.asyncio
async def test_version_survives_temperature_overlay(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Overlaying temperature does not drop ``ref.version``."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(temperature=0.7),
    )

    await ai.generate(model=ref, config={'temperature': 0.2}, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'version') == '001'
    assert _config_value(echo.last_request.config, 'temperature') == 0.2


@pytest.mark.asyncio
async def test_ref_config_version_beats_constructor_version(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """``ref.config.version`` overlays the constructor ``version=`` field."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(version='002'),
    )

    await ai.generate(model=ref, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'version') == '002'


@pytest.mark.asyncio
async def test_generate_uses_constructor_model_ref() -> None:
    """Genkit(model=ref) then generate() with no model= applies name, version, and config."""
    flash = model_ref(
        'flash',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(temperature=0.7),
    )
    ai = Genkit(model=flash)
    echo, _ = define_echo_model(ai, name='flash')

    response = await ai.generate(prompt='hi')

    assert '[ECHO]' in response.text
    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.7
    assert _config_value(echo.last_request.config, 'version') == '001'


@pytest.mark.asyncio
async def test_generate_constructor_model_ref_call_time_config_wins() -> None:
    """Call-time config overlays the constructor ref per key."""
    flash = model_ref(
        'flash',
        config_schema=ModelConfig,
        config=ModelConfig(temperature=0.7),
    )
    ai = Genkit(model=flash)
    echo, _ = define_echo_model(ai, name='flash')

    await ai.generate(prompt='hi', config={'temperature': 0.2})

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.2


@pytest.mark.asyncio
async def test_prompt_uses_constructor_model_ref_config() -> None:
    """A prompt with no model= still picks up the constructor ref's config."""
    flash = model_ref(
        'flash',
        config_schema=ModelConfig,
        config=ModelConfig(temperature=0.7),
    )
    ai = Genkit(model=flash)
    echo, _ = define_echo_model(ai, name='flash')
    hello = ai.define_prompt(name='hello', prompt='hi')

    await hello()

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') == 0.7


@pytest.mark.asyncio
async def test_to_generate_action_options_uses_constructor_ref() -> None:
    """A stored ModelRef default still resolves when PromptConfig.model is omitted."""
    flash = model_ref(
        'flash',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(temperature=0.7),
    )
    ai = Genkit(model=flash)
    define_echo_model(ai, name='flash')

    options = await to_generate_action_options(ai.registry, PromptConfig(prompt='hi'))

    assert options.model == 'flash'
    assert _config_value(options.config, 'temperature') == 0.7
    assert _config_value(options.config, 'version') == '001'


def _leftover_alias(config: Any, field: str, alias: str) -> None:
    """The other spelling must not sit beside the field on the plugin object."""
    if isinstance(config, dict):
        assert alias not in config
        return
    extra = getattr(config, 'model_extra', None) or {}
    assert extra.get(field) is None
    assert extra.get(alias) is None


@pytest.mark.asyncio
async def test_generate_alias_dict_overrides_ref_field(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Call-time maxOutputTokens replaces the ref's max_output_tokens."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        config=ModelConfig(max_output_tokens=100),
    )

    await ai.generate(model=ref, config={'maxOutputTokens': 5}, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'max_output_tokens') == 5
    _leftover_alias(echo.last_request.config, 'max_output_tokens', 'maxOutputTokens')


@pytest.mark.asyncio
async def test_generate_alias_none_clears_ref_field(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Call-time maxOutputTokens=None clears the ref cap."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        config=ModelConfig(max_output_tokens=100),
    )

    await ai.generate(model=ref, config={'maxOutputTokens': None}, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'max_output_tokens') is None
    _leftover_alias(echo.last_request.config, 'max_output_tokens', 'maxOutputTokens')


@pytest.mark.asyncio
async def test_generate_alias_replaces_nested_bag(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """thinkingConfig is the thinking_config slot. The call's bag replaces it whole.

    ``thinking_level`` and ``include_thoughts`` from the ref are gone. Inner
    keys stay as written.
    """
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=AliasedNestedConfig,
        config=AliasedNestedConfig(
            thinking_config={
                'include_thoughts': True,
                'thinking_budget': 1024,
                'thinking_level': 'low',
            }
        ),
    )

    await ai.generate(model=ref, config={'thinkingConfig': {'thinkingBudget': 256}}, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'thinking_config') == {'thinkingBudget': 256}
    if isinstance(echo.last_request.config, dict):
        assert 'thinkingConfig' not in echo.last_request.config
    else:
        extra = getattr(echo.last_request.config, 'model_extra', None) or {}
        assert extra.get('thinkingConfig') is None


@pytest.mark.asyncio
async def test_generate_alias_none_clears_nested_bag(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """thinkingConfig=None clears the whole thinking_config slot."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=AliasedNestedConfig,
        config=AliasedNestedConfig(thinking_config={'thinking_budget': 1024, 'thinking_level': 'low'}),
    )

    await ai.generate(model=ref, config={'thinkingConfig': None}, prompt='Hello')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'thinking_config') is None
    _leftover_alias(echo.last_request.config, 'thinking_config', 'thinkingConfig')


@pytest.mark.asyncio
async def test_generate_does_not_fold_inner_keys(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Both spellings inside a nested bag stay. Only the outer key folds."""
    ai, echo = ai_with_echo
    ref = model_ref('testEcho', config_schema=AliasedNestedConfig)

    await ai.generate(
        model=ref,
        config={'thinkingConfig': {'thinking_budget': 1, 'thinkingBudget': 256}},
        prompt='Hello',
    )

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'thinking_config') == {
        'thinking_budget': 1,
        'thinkingBudget': 256,
    }
    if isinstance(echo.last_request.config, dict):
        assert 'thinkingConfig' not in echo.last_request.config
    else:
        extra = getattr(echo.last_request.config, 'model_extra', None) or {}
        assert extra.get('thinkingConfig') is None


@pytest.mark.asyncio
async def test_generate_both_spellings_last_write_wins(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Both keys in one call dict fold onto one field; later key wins."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        config=ModelConfig(max_output_tokens=100),
    )

    await ai.generate(
        model=ref,
        config={'max_output_tokens': 1, 'maxOutputTokens': 5},
        prompt='Hello',
    )

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'max_output_tokens') == 5
    _leftover_alias(echo.last_request.config, 'max_output_tokens', 'maxOutputTokens')

    await ai.generate(
        model=ref,
        config={'maxOutputTokens': 5, 'max_output_tokens': 1},
        prompt='Hello',
    )

    assert _config_value(echo.last_request.config, 'max_output_tokens') == 1
    _leftover_alias(echo.last_request.config, 'max_output_tokens', 'maxOutputTokens')


@pytest.mark.asyncio
async def test_constructor_ref_alias_dict_overrides() -> None:
    """Constructor ModelRef + camel call dict is the same overlay."""
    flash = model_ref(
        'flash',
        config_schema=ModelConfig,
        config=ModelConfig(max_output_tokens=100),
    )
    ai = Genkit(model=flash)
    echo, _ = define_echo_model(ai, name='flash')

    await ai.generate(prompt='hi', config={'maxOutputTokens': 5})

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'max_output_tokens') == 5
    _leftover_alias(echo.last_request.config, 'max_output_tokens', 'maxOutputTokens')


@pytest.mark.asyncio
async def test_prompt_alias_dict_overrides_ref_field(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Prompt call-time maxOutputTokens replaces the ref default."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        config=ModelConfig(max_output_tokens=100),
    )
    joke = ai.define_prompt(name='jokeAlias', model=ref, prompt='hi')

    await joke(config={'maxOutputTokens': 5})

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'max_output_tokens') == 5
    _leftover_alias(echo.last_request.config, 'max_output_tokens', 'maxOutputTokens')


@pytest.mark.asyncio
async def test_prompt_alias_none_clears_ref_field(
    ai_with_echo: tuple[Genkit, EchoModel],
) -> None:
    """Prompt call-time maxOutputTokens=None is the same clear as generate."""
    ai, echo = ai_with_echo
    ref = model_ref(
        'testEcho',
        config_schema=ModelConfig,
        config=ModelConfig(max_output_tokens=100),
    )
    joke = ai.define_prompt(name='jokeAliasClear', model=ref, prompt='hi')

    await joke(config={'maxOutputTokens': None})

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'max_output_tokens') is None
    _leftover_alias(echo.last_request.config, 'max_output_tokens', 'maxOutputTokens')


@pytest.mark.asyncio
async def test_same_name_string_does_not_use_constructor_ref_bag() -> None:
    """``model='flash'`` is a name, even when the constructor ref is also flash."""
    flash = model_ref(
        'flash',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(temperature=0.7),
    )
    ai = Genkit(model=flash)
    echo, _ = define_echo_model(ai, name='flash')

    await ai.generate(model='flash', prompt='hi')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') != 0.7
    assert _config_value(echo.last_request.config, 'version') != '001'


@pytest.mark.asyncio
async def test_constructor_string_default_has_no_ref_bag() -> None:
    """``Genkit(model='flash')`` is a name. Omit does not invent a config bag."""
    ai = Genkit(model='flash')
    echo, _ = define_echo_model(ai, name='flash')

    await ai.generate(prompt='hi')

    assert echo.last_request is not None
    assert _config_value(echo.last_request.config, 'temperature') is None
    assert _config_value(echo.last_request.config, 'version') is None
