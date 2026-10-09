#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A config key the model doesn't declare fails the call before the model runs.

Every failing case asserts the same three things: the status, that the model
was never called, and the message (model name plus the offending key). A
per-request API key in config fails the same way, with a message that points
at ``context.secrets``.
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from genkit import Genkit, Message, ModelResponse, Part
from genkit._core._action import ActionKind, ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import FinishReason, ModelRef, ModelRequest
from genkit._core._typing import GenerationCommonConfig, Operation, Role
from genkit.model import ModelConfig, model_ref

KEY = 'sk-tenant'
SECRETS_HINT = "context={'secrets': {'api_key': ...}}"


class StrictConfig(ModelConfig):
    """A plugin class that declares one provider setting of its own."""

    safe_prompt: bool | None = None


class OtherConfig(ModelConfig):
    """Some other plugin's class."""


class TaskBudget(BaseModel):
    """A nested setting that is not useful until `total` is set."""

    model_config = ConfigDict(extra='forbid')
    total: int


class OutputSetting(BaseModel):
    """A nested object sent whole, not merged field-by-field."""

    model_config = ConfigDict(extra='forbid')
    task_budget: TaskBudget


class NestedConfig(ModelConfig):
    """A Claude-shaped class with a required nested field."""

    output_config: OutputSetting | None = None


class RequiredTopConfig(ModelConfig):
    """A class with a required top-level field another layer may supply."""

    must: int


class GeminiLikeConfig(ModelConfig):
    """Declares Gemini settings the shared ModelConfig class does not."""

    safety_settings: list[dict[str, str]] | None = Field(default=None, alias='safety_settings')
    thinking_config: dict[str, Any] | None = Field(default=None, alias='thinkingConfig')


class OwnStrictConfig(BaseModel):
    """A plugin class that isn't built on ModelConfig and forbids unknown keys."""

    model_config = ConfigDict(extra='forbid')

    num_ctx: int | None = None


class LegacyConfig(ModelConfig):
    """A plugin class that still declares its own ``api_key`` setting."""

    api_key: str | None = None


def _config_value(config: Any, key: str) -> Any:  # noqa: ANN401
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


class _Model:
    """A model that records every request it gets and answers 'ok'."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def define(self, ai: Genkit, *, name: str, config_schema: type[BaseModel] | None) -> None:
        async def fn(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
            self.requests.append(request)
            return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

        ai.define_model(name=name, fn=fn, config_schema=config_schema)


def _ai_with_model(
    *, config_schema: type[BaseModel] | None = StrictConfig, name: str = 'strict'
) -> tuple[Genkit, _Model]:
    ai = Genkit()
    model = _Model()
    model.define(ai, name=name, config_schema=config_schema)
    return ai, model


def _assert_points_to_secrets(err: pytest.ExceptionInfo[GenkitError], fn: _Model) -> None:
    assert err.value.status == 'INVALID_ARGUMENT'
    assert fn.requests == []
    assert SECRETS_HINT in str(err.value)
    assert 'unknown config key' not in str(err.value)
    assert KEY not in str(err.value)
    assert KEY not in repr(err.value)


def _assert_rejected(err: pytest.ExceptionInfo[GenkitError], fn: _Model, *needles: str) -> None:
    assert err.value.status == 'INVALID_ARGUMENT'
    assert fn.requests == []
    for needle in needles:
        assert needle in str(err.value)


@pytest.mark.asyncio
async def test_generate_unknown_config_key_raises_invalid_argument_naming_model_and_key() -> None:
    """`config={'temprature': 0.2}` raises INVALID_ARGUMENT naming the model and the key; the model never runs."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='strict', prompt='hi', config={'temprature': 0.2})

    _assert_rejected(err, fn, "strict: unknown config key 'temprature'")


@pytest.mark.asyncio
async def test_generate_unknown_config_key_error_says_where_provider_settings_go() -> None:
    """The unknown-key error points at `config['extra']` for settings the plugin doesn't declare."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='strict', prompt='hi', config={'labels': {'team': 'search'}})

    _assert_rejected(err, fn, "unknown config key 'labels'", "config['extra']")


@pytest.mark.asyncio
async def test_generate_two_unknown_config_keys_names_both() -> None:
    """Two typos in one dict are both named in one error."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='strict', prompt='hi', config={'temprature': 0.2, 'top_kk': 3})

    _assert_rejected(err, fn, "unknown config keys 'temprature', 'top_kk'")


@pytest.mark.asyncio
async def test_generate_unknown_config_key_error_has_validation_error_as_cause() -> None:
    """The raised error's `cause` is the class's own validation error."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='strict', prompt='hi', config={'temprature': 0.2})

    _assert_rejected(err, fn, 'temprature')
    assert isinstance(err.value.cause, ValidationError)


@pytest.mark.asyncio
async def test_generate_wrong_type_config_value_raises_invalid_argument() -> None:
    """`{'temperature': 'hot'}` raises INVALID_ARGUMENT naming `temperature`, before the model runs."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='strict', prompt='hi', config={'temperature': 'hot'})

    _assert_rejected(err, fn, "strict: config 'temperature'")


@pytest.mark.asyncio
async def test_generate_camel_case_key_alone_reaches_model() -> None:
    """`{'maxOutputTokens': 5}` is the same declared setting; it reaches the model under the key the class accepts."""
    ai, fn = _ai_with_model()

    await ai.generate(model='strict', prompt='hi', config={'maxOutputTokens': 5})

    assert fn.requests[-1].config == {'max_output_tokens': 5}


@pytest.mark.asyncio
async def test_generate_snake_case_key_alone_reaches_model() -> None:
    """`{'max_output_tokens': 5}` reaches the model unchanged."""
    ai, fn = _ai_with_model()

    await ai.generate(model='strict', prompt='hi', config={'max_output_tokens': 5})

    assert _config_value(fn.requests[-1].config, 'max_output_tokens') == 5


@pytest.mark.asyncio
async def test_generate_declared_plugin_setting_reaches_model() -> None:
    """`{'safe_prompt': True}` is declared by the model's class, so it's accepted."""
    ai, fn = _ai_with_model()

    await ai.generate(model='strict', prompt='hi', config={'safe_prompt': True})

    assert _config_value(fn.requests[-1].config, 'safe_prompt') is True


@pytest.mark.asyncio
async def test_generate_none_value_clears_instead_of_raising() -> None:
    """`{'temperature': None}` is accepted: None means "unset", not a bad value."""
    ai, fn = _ai_with_model()

    await ai.generate(model='strict', prompt='hi', config={'temperature': None})

    assert _config_value(fn.requests[-1].config, 'temperature') is None


@pytest.mark.asyncio
async def test_generate_extra_reaches_model_as_given() -> None:
    """`{'extra': {'labels': {'a': 'b'}}}` arrives on the model's config as `extra`, unchanged."""
    ai, fn = _ai_with_model()

    await ai.generate(model='strict', prompt='hi', config={'extra': {'labels': {'a': 'b'}}})

    assert _config_value(fn.requests[-1].config, 'extra') == {'labels': {'a': 'b'}}


@pytest.mark.asyncio
async def test_generate_extra_contents_are_not_checked() -> None:
    """`{'extra': {'temprature': 1}}` is accepted; what's inside `extra` is the caller's."""
    ai, fn = _ai_with_model()

    await ai.generate(model='strict', prompt='hi', config={'extra': {'temprature': 1}})

    assert _config_value(fn.requests[-1].config, 'extra') == {'temprature': 1}


@pytest.mark.asyncio
async def test_generate_model_config_extra_reaches_model() -> None:
    """`config=StrictConfig(extra={...})` arrives with `extra` intact."""
    ai, fn = _ai_with_model()

    await ai.generate(model='strict', prompt='hi', config=StrictConfig(extra={'labels': {'a': 'b'}}))

    assert _config_value(fn.requests[-1].config, 'extra') == {'labels': {'a': 'b'}}


def test_model_config_with_unknown_keyword_raises_validation_error() -> None:
    """`ModelConfig(temprature=0.2)` fails at construction."""
    with pytest.raises(ValidationError, match='temprature'):
        ModelConfig(temprature=0.2)  # type: ignore[call-arg]


def test_model_config_extra_takes_any_provider_settings() -> None:
    """`ModelConfig(extra={'labels': {...}})` builds; `extra` is a plain dict."""
    config = ModelConfig(extra={'labels': {'team': 'search'}, 'anything': 1})

    assert config.extra == {'labels': {'team': 'search'}, 'anything': 1}


@pytest.mark.asyncio
async def test_generate_other_plugin_config_class_raises_with_public_name() -> None:
    """`config=OtherConfig()` on a StrictConfig model raises INVALID_ARGUMENT naming the class."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='strict', prompt='hi', config=OtherConfig(temperature=0.2))

    _assert_rejected(err, fn, 'OtherConfig')


@pytest.mark.asyncio
async def test_generate_model_without_config_class_accepts_any_dict() -> None:
    """A model defined with no `config_schema` still accepts `{'anything': 1}`."""
    ai, fn = _ai_with_model(config_schema=None, name='loose')

    await ai.generate(model='loose', prompt='hi', config={'anything': 1})

    assert _config_value(fn.requests[-1].config, 'anything') == 1


@pytest.mark.asyncio
async def test_model_ref_for_model_without_config_class_accepts_any_dict() -> None:
    """A ModelRef for a model with no config class accepts `{'anything': 1}`, same as the name."""
    ai, fn = _ai_with_model(config_schema=None, name='loose')
    ref = ModelRef(name='loose', config_schema=GenerationCommonConfig)

    await ai.generate(model=ref, prompt='hi', config={'anything': 1})

    assert _config_value(fn.requests[-1].config, 'anything') == 1


@pytest.mark.asyncio
async def test_model_ref_with_declared_schema_config_typo_raises() -> None:
    """A ModelRef with StrictConfig plus `config={'temprature': 0.2}` raises like the string name."""
    ai, fn = _ai_with_model()
    ref = model_ref('strict', config_schema=StrictConfig)

    with pytest.raises(GenkitError) as err:
        await ai.generate(model=ref, prompt='hi', config={'temprature': 0.2})

    _assert_rejected(err, fn, "strict: unknown config key 'temprature'")


@pytest.mark.asyncio
async def test_generate_model_ref_config_typo_raises() -> None:
    """A typo in the call-site dict next to `model_ref(..., config=...)` raises like a string model name."""
    ai, fn = _ai_with_model()
    ref = model_ref('strict', config_schema=StrictConfig, config=StrictConfig(temperature=0.5))

    with pytest.raises(GenkitError) as err:
        await ai.generate(model=ref, prompt='hi', config={'temprature': 0.2})

    _assert_rejected(err, fn, "strict: unknown config key 'temprature'")


@pytest.mark.asyncio
async def test_generate_stream_unknown_config_key_raises_before_first_chunk() -> None:
    """`generate_stream` raises before any chunk is produced."""
    ai, fn = _ai_with_model()
    chunks: list[Any] = []

    stream = ai.generate_stream(model='strict', prompt='hi', config={'temprature': 0.2})
    with pytest.raises(GenkitError) as err:
        async for chunk in stream.stream:
            chunks.append(chunk)

    _assert_rejected(err, fn, "strict: unknown config key 'temprature'")
    assert chunks == []


@pytest.mark.asyncio
async def test_generate_operation_unknown_config_key_raises() -> None:
    """`generate_operation` on a background model with a strict class raises the same error and starts no job."""
    ai = Genkit()
    started: list[ModelRequest] = []

    async def start(request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        started.append(request)
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai.define_background_model(name='bg', start=start, check=check, config_schema=StrictConfig)

    with pytest.raises(GenkitError) as err:
        await ai.generate_operation(model='bg', prompt='hi', config={'temprature': 0.2})

    assert err.value.status == 'INVALID_ARGUMENT'
    assert "bg: unknown config key 'temprature'" in str(err.value)
    assert started == []


@pytest.mark.asyncio
async def test_prompt_with_config_typo_defines_and_raises_when_called() -> None:
    """`define_prompt(config={'temprature': 0.2})` succeeds; calling it raises INVALID_ARGUMENT."""
    ai, fn = _ai_with_model()
    prompt = ai.define_prompt(name='typo', model='strict', prompt='hi', config={'temprature': 0.2})

    with pytest.raises(GenkitError) as err:
        await prompt()

    _assert_rejected(err, fn, "strict: unknown config key 'temprature'")


@pytest.mark.asyncio
async def test_prompt_call_config_typo_raises() -> None:
    """A typo in the call-time `config=` of a prompt raises."""
    ai, fn = _ai_with_model()
    prompt = ai.define_prompt(name='ok', model='strict', prompt='hi', config={'temperature': 0.2})

    with pytest.raises(GenkitError) as err:
        await prompt(config={'temprature': 0.5})

    _assert_rejected(err, fn, "strict: unknown config key 'temprature'")


@pytest.mark.asyncio
async def test_prompt_camel_case_definition_with_snake_case_call_reaches_model() -> None:
    """Prompt `maxOutputTokens: 5` called with `max_output_tokens: 7` sends only `max_output_tokens: 7`."""
    ai, fn = _ai_with_model()
    prompt = ai.define_prompt(name='spell', model='strict', prompt='hi', config={'maxOutputTokens': 5})

    await prompt(config={'max_output_tokens': 7})

    assert fn.requests[-1].config == {'max_output_tokens': 7}


@pytest.mark.asyncio
async def test_prompt_called_with_other_model_checks_config_against_that_model() -> None:
    """A prompt's `safe_prompt` raises when the call targets a model whose class doesn't declare it."""
    ai, strict = _ai_with_model()
    plain = _Model()
    plain.define(ai, name='plain', config_schema=ModelConfig)
    prompt = ai.define_prompt(name='hop', model='strict', prompt='hi', config={'safe_prompt': True})

    with pytest.raises(GenkitError) as err:
        await prompt(model='plain')

    _assert_rejected(err, plain, "plain: unknown config key 'safe_prompt'")
    assert strict.requests == []


@pytest.mark.asyncio
async def test_dotprompt_file_config_typo_raises_when_called(tmp_path: Path) -> None:
    """A `.prompt` frontmatter typo loads fine and raises at the first call."""
    (tmp_path / 'typo.prompt').write_text('---\nconfig:\n  temprature: 0.2\n---\nhi\n')
    ai = Genkit(prompt_dir=tmp_path, model='strict')
    fn = _Model()
    fn.define(ai, name='strict', config_schema=StrictConfig)
    prompt = ai.prompt('typo')

    with pytest.raises(GenkitError) as err:
        await prompt()

    _assert_rejected(err, fn, "strict: unknown config key 'temprature'")


@pytest.mark.asyncio
async def test_generate_valid_config_runs_unchanged() -> None:
    """A correct config produces the request and response it did before."""
    ai, fn = _ai_with_model()

    response = await ai.generate(
        model='strict',
        prompt='hi',
        config={'temperature': 0.2, 'max_output_tokens': 10, 'safe_prompt': True},
    )

    assert response.text == 'ok'
    config = fn.requests[-1].config
    assert _config_value(config, 'temperature') == 0.2
    assert _config_value(config, 'max_output_tokens') == 10
    assert _config_value(config, 'safe_prompt') is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'config',
    [
        {'maxOutputTokens': 5, 'max_output_tokens': 5},
        {'maxOutputTokens': 5, 'max_output_tokens': 6},
        {'max_output_tokens': 5, 'maxOutputTokens': 5},
    ],
    ids=['same_value', 'different_value', 'snake_case_first'],
)
async def test_generate_both_spellings_of_one_setting_raises_same_setting(config: dict[str, int]) -> None:
    """Both spellings raise "are the same setting; pass one" with the field name first, whatever the dict order."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='strict', prompt='hi', config=config)

    _assert_rejected(err, fn)
    assert err.value.original_message == 'strict: max_output_tokens and maxOutputTokens are the same setting; pass one'


@pytest.mark.asyncio
async def test_generate_two_settings_each_in_both_spellings_names_both_pairs() -> None:
    """`maxOutputTokens`/`max_output_tokens` plus `topK`/`top_k` name both pairs in one error."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(
            model='strict',
            prompt='hi',
            config={'maxOutputTokens': 5, 'max_output_tokens': 5, 'topK': 3, 'top_k': 3},
        )

    _assert_rejected(err, fn)
    assert err.value.original_message == (
        'strict: max_output_tokens and maxOutputTokens are the same setting; pass one; '
        'top_k and topK are the same setting; pass one'
    )


@pytest.mark.asyncio
async def test_generate_both_spellings_of_plugin_declared_alias_raises_same_setting() -> None:
    """A plugin class's own alias pair (`thinking_config`/`thinkingConfig`) gets the same message."""
    ai, fn = _ai_with_model(config_schema=GeminiLikeConfig, name='gem')

    with pytest.raises(GenkitError) as err:
        await ai.generate(
            model='gem',
            prompt='hi',
            config={'thinkingConfig': {'thinkingBudget': 0}, 'thinking_config': {'thinkingBudget': 0}},
        )

    _assert_rejected(err, fn)
    assert err.value.original_message == 'gem: thinking_config and thinkingConfig are the same setting; pass one'


@pytest.mark.asyncio
async def test_generate_three_spellings_of_one_setting_lists_all_three() -> None:
    """`AliasChoices('seed_value', 'seedValue', 'seed')` with all three set names all three in one list."""

    class SeedConfig(ModelConfig):
        seed_value: int | None = Field(default=None, validation_alias=AliasChoices('seed_value', 'seedValue', 'seed'))

    ai, fn = _ai_with_model(config_schema=SeedConfig, name='seeded')

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='seeded', prompt='hi', config={'seedValue': 1, 'seed': 1, 'seed_value': 1})

    _assert_rejected(err, fn)
    assert err.value.original_message == 'seeded: seed_value, seedValue, and seed are the same setting; pass one'


@pytest.mark.asyncio
async def test_generate_none_on_one_spelling_is_still_a_second_spelling() -> None:
    """`{'maxOutputTokens': 5, 'max_output_tokens': None}` raises: an explicit None is a value."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='strict', prompt='hi', config={'maxOutputTokens': 5, 'max_output_tokens': None})

    _assert_rejected(err, fn, 'max_output_tokens and maxOutputTokens are the same setting')


@pytest.mark.asyncio
async def test_generate_both_spellings_typo_and_bad_value_all_in_one_error() -> None:
    """Both spellings, a typo, and a bad value raise one error: same setting, then unknown key, then the value."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(
            model='strict',
            prompt='hi',
            config={'maxOutputTokens': 5, 'max_output_tokens': 5, 'temprature': 0.2, 'top_p': 'high'},
        )

    _assert_rejected(err, fn)
    assert err.value.original_message.startswith(
        'strict: max_output_tokens and maxOutputTokens are the same setting; pass one; '
        "unknown config key 'temprature'; put provider-only settings in config['extra']; "
        "config 'top_p': "
    )


@pytest.mark.asyncio
async def test_prompt_call_other_model_with_none_for_a_prompt_only_key_raises() -> None:
    """`model='other', config={'thinkingConfig': None}` raises: None is a value, and other has no thinkingConfig."""
    ai = Genkit()
    gem = _Model()
    other = _Model()
    gem.define(ai, name='gem', config_schema=GeminiLikeConfig)
    other.define(ai, name='other', config_schema=ModelConfig)
    prompt = ai.define_prompt(
        name='hop',
        model='gem',
        prompt='hi',
        config={'thinkingConfig': {'thinkingBudget': 0}},
    )

    with pytest.raises(GenkitError) as err:
        await prompt(model='other', config={'thinkingConfig': None})

    _assert_rejected(err, other, "unknown config key 'thinkingConfig'")
    assert gem.requests == []


@pytest.mark.asyncio
async def test_prompt_call_other_model_keeping_prompt_only_key_raises() -> None:
    """The same prompt called with `model='other'` and no override raises INVALID_ARGUMENT naming thinkingConfig."""
    ai = Genkit()
    gem = _Model()
    other = _Model()
    gem.define(ai, name='gem', config_schema=GeminiLikeConfig)
    other.define(ai, name='other', config_schema=ModelConfig)
    prompt = ai.define_prompt(
        name='hop',
        model='gem',
        prompt='hi',
        config={'thinkingConfig': {'thinkingBudget': 0}},
    )

    with pytest.raises(GenkitError) as err:
        await prompt(model='other')

    _assert_rejected(err, other, 'thinkingConfig')
    assert gem.requests == []


@pytest.mark.asyncio
async def test_generate_incomplete_nested_setting_raises_naming_the_missing_field() -> None:
    """An incomplete nested `output_config` raises INVALID_ARGUMENT naming output_config.task_budget.total."""
    ai, fn = _ai_with_model(config_schema=NestedConfig, name='anth')

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='anth', prompt='hi', config={'output_config': {'task_budget': {}}})

    _assert_rejected(err, fn, 'output_config.task_budget.total')


@pytest.mark.asyncio
async def test_generate_missing_required_top_level_field_fails_at_the_model() -> None:
    """The call check lets a missing required field through; the model's boundary does not."""
    ai, fn = _ai_with_model(config_schema=RequiredTopConfig, name='needs')

    response = await ai.generate(model='needs', prompt='hi', config={'temperature': 0.2})

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "needs: config 'must': Field required" in response.finish_message
    assert fn.requests == []


@pytest.mark.asyncio
async def test_generate_ref_with_shared_config_class_accepts_model_declared_setting() -> None:
    """`model_ref(..., config_schema=ModelConfig)` with the model's `safety_settings` runs."""
    ai, fn = _ai_with_model(config_schema=GeminiLikeConfig, name='gem')
    ref = model_ref('gem', config_schema=ModelConfig)

    response = await ai.generate(
        model=ref,
        prompt='hi',
        config={'safety_settings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_LOW_AND_ABOVE'}]},
    )

    assert response.text == 'ok'
    assert _config_value(fn.requests[-1].config, 'safety_settings') == [
        {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_LOW_AND_ABOVE'}
    ]


@pytest.mark.asyncio
async def test_generate_ref_with_shared_config_class_rejects_typo_via_model_class() -> None:
    """The same ModelConfig ref with `{'temprature': 0.2}` raises INVALID_ARGUMENT naming temprature."""
    ai, fn = _ai_with_model(config_schema=GeminiLikeConfig, name='gem')
    ref = model_ref('gem', config_schema=ModelConfig)

    with pytest.raises(GenkitError) as err:
        await ai.generate(model=ref, prompt='hi', config={'temprature': 0.2})

    _assert_rejected(err, fn, "gem: unknown config key 'temprature'")


@pytest.mark.asyncio
async def test_generate_ref_with_another_plugins_class_raises() -> None:
    """A ref's config_schema must name the model's class; the model's definition owns it."""
    ai, fn = _ai_with_model(config_schema=GeminiLikeConfig, name='gem')
    ref = model_ref('gem', config_schema=OtherConfig)

    with pytest.raises(GenkitError) as err:
        await ai.generate(model=ref, prompt='hi', config={'safety_settings': [{'category': 'HARM'}]})

    _assert_rejected(
        err, fn, "model 'gem' takes config", 'GeminiLikeConfig', "the ref's config_schema is", 'OtherConfig'
    )


# -- api key ------------------------------------------------------------------


@pytest.mark.parametrize(
    'config',
    [{'api_key': KEY}, {'apiKey': KEY}, {'temperature': 0.2, 'api_key': KEY}],
    ids=['snake_case', 'camelCase', 'next-to-valid-settings'],
)
@pytest.mark.asyncio
async def test_generate_config_api_key_raises_pointing_to_secrets(config: dict[str, Any]) -> None:
    """A key in config raises INVALID_ARGUMENT naming `context.secrets`, without echoing the key."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='strict', prompt='hi', config=config)

    _assert_points_to_secrets(err, fn)


def test_model_config_with_api_key_raises_validation_error() -> None:
    """`ModelConfig(api_key=k)` fails where it's typed; there's no such setting."""
    with pytest.raises(ValidationError, match='api_key'):
        ModelConfig(api_key=KEY)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_generate_config_api_key_on_model_without_config_class_raises() -> None:
    """A model defined with no `config_schema` still rejects a key in config, though it takes any other key."""
    ai, fn = _ai_with_model(config_schema=None, name='loose')

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='loose', prompt='hi', config={'api_key': KEY})

    _assert_points_to_secrets(err, fn)


@pytest.mark.asyncio
async def test_generate_config_api_key_on_plugin_class_not_built_on_model_config_raises() -> None:
    """A plugin's own strict class (`extra='forbid'`, not a ModelConfig) gets the secrets message, not "unknown key"."""
    ai, fn = _ai_with_model(config_schema=OwnStrictConfig, name='own')

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='own', prompt='hi', config={'num_ctx': 2048, 'api_key': KEY})

    _assert_points_to_secrets(err, fn)


@pytest.mark.asyncio
async def test_generate_plugin_class_that_declares_api_key_raises() -> None:
    """`config=LegacyConfig(api_key=k)` on a plugin class with its own `api_key` field raises."""
    ai, fn = _ai_with_model(config_schema=LegacyConfig, name='legacy')

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='legacy', prompt='hi', config=LegacyConfig(api_key=KEY))

    _assert_points_to_secrets(err, fn)


@pytest.mark.asyncio
async def test_generate_model_ref_with_api_key_in_its_config_raises() -> None:
    """`model_ref('legacy', config=LegacyConfig(api_key=k))` raises when generate uses it, with no call-site config."""
    ai, fn = _ai_with_model(config_schema=LegacyConfig, name='legacy')
    ref = model_ref('legacy', config_schema=LegacyConfig, config=LegacyConfig(api_key=KEY))

    with pytest.raises(GenkitError) as err:
        await ai.generate(model=ref, prompt='hi')

    _assert_points_to_secrets(err, fn)


@pytest.mark.asyncio
async def test_generate_config_api_key_none_is_accepted() -> None:
    """`config={'api_key': None}` isn't a key, so the call runs."""
    ai, fn = _ai_with_model(config_schema=None, name='loose')

    await ai.generate(model='loose', prompt='hi', config={'api_key': None})

    assert len(fn.requests) == 1


@pytest.mark.asyncio
async def test_generate_config_api_key_never_reaches_model_or_trace(exporter: Any) -> None:  # noqa: ANN401
    """The model never runs and no recorded span contains the key."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError):
        await ai.generate(model='strict', prompt='hi', config={'api_key': KEY})

    assert fn.requests == []
    for span in exporter.get_finished_spans():
        for value in dict(span.attributes or {}).values():
            assert KEY not in str(value)


@pytest.mark.asyncio
async def test_generate_stream_config_api_key_raises() -> None:
    """`generate_stream` raises the same error before any chunk."""
    ai, fn = _ai_with_model()
    chunks: list[Any] = []

    stream = ai.generate_stream(model='strict', prompt='hi', config={'api_key': KEY})
    with pytest.raises(GenkitError) as err:
        async for chunk in stream.stream:
            chunks.append(chunk)

    _assert_points_to_secrets(err, fn)
    assert chunks == []


@pytest.mark.asyncio
async def test_generate_operation_config_api_key_raises() -> None:
    """`generate_operation` raises the same error and starts no job."""
    ai = Genkit()
    started: list[ModelRequest] = []

    async def start(request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        started.append(request)
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai.define_background_model(name='bg', start=start, check=check)

    with pytest.raises(GenkitError) as err:
        await ai.generate_operation(model='bg', prompt='hi', config={'api_key': KEY})

    assert err.value.status == 'INVALID_ARGUMENT'
    assert SECRETS_HINT in str(err.value)
    assert KEY not in str(err.value)
    assert started == []


@pytest.mark.asyncio
async def test_prompt_config_api_key_raises_when_called() -> None:
    """`define_prompt(config={'api_key': k})` defines; calling the prompt raises the same error."""
    ai, fn = _ai_with_model()
    prompt = ai.define_prompt(name='p', model='strict', prompt='hi', config={'api_key': KEY})

    with pytest.raises(GenkitError) as err:
        await prompt()

    _assert_points_to_secrets(err, fn)


@pytest.mark.asyncio
async def test_prompt_call_config_api_key_raises() -> None:
    """`prompt(config={'apiKey': k})` on a prompt with no key of its own raises the same error."""
    ai, fn = _ai_with_model()
    prompt = ai.define_prompt(name='p', model='strict', prompt='hi', config={'temperature': 0.2})

    with pytest.raises(GenkitError) as err:
        await prompt(config={'apiKey': KEY})

    _assert_points_to_secrets(err, fn)


@pytest.mark.parametrize('name', ['strict', 'loose'])
@pytest.mark.asyncio
async def test_util_generate_action_config_api_key_raises(name: str) -> None:
    """The registered `/util/generate` action (Dev UI, reflection) raises the same error; the model never runs."""
    ai, fn = _ai_with_model(config_schema=StrictConfig if name == 'strict' else None, name=name)
    action = await ai.registry.resolve_action(ActionKind.UTIL, 'generate')
    assert action is not None

    with pytest.raises(GenkitError) as err:
        await action.run({
            'model': name,
            'messages': [{'role': 'user', 'content': [{'text': 'hi'}]}],
            'config': {'apiKey': KEY},
        })

    _assert_points_to_secrets(err, fn)


@pytest.mark.parametrize(
    'request_input',
    [
        {'messages': [], 'config': {'api_key': KEY}},
        ModelRequest(messages=[], config={'apiKey': KEY}),
        ModelRequest(messages=[], config=LegacyConfig(api_key=KEY)),
    ],
    ids=['dict', 'request-with-dict-config', 'request-with-config-object'],
)
@pytest.mark.asyncio
async def test_model_action_run_directly_with_config_api_key_raises(request_input: object) -> None:
    """Running the model action itself, outside generate, raises the same error; the model fn never runs."""
    ai, fn = _ai_with_model(config_schema=None, name='loose')
    action = await ai.registry.resolve_action(ActionKind.MODEL, 'loose')
    assert action is not None

    with pytest.raises(GenkitError) as err:
        await action.run(request_input)

    _assert_points_to_secrets(err, fn)


@pytest.mark.asyncio
async def test_background_model_action_run_directly_with_config_api_key_raises() -> None:
    """Running a background model's action outside generate_operation raises the same error and starts no job."""
    ai = Genkit()
    started: list[ModelRequest] = []

    async def start(request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        started.append(request)
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai.define_background_model(name='bg', start=start, check=check)
    action = await ai.registry.resolve_action(ActionKind.BACKGROUND_MODEL, 'bg')
    assert action is not None

    with pytest.raises(GenkitError) as err:
        await action.run({'messages': [], 'config': {'api_key': KEY}})

    assert err.value.status == 'INVALID_ARGUMENT'
    assert SECRETS_HINT in str(err.value)
    assert started == []


@pytest.mark.parametrize('spelling', ['api_key', 'apiKey'])
@pytest.mark.asyncio
async def test_generate_extra_api_key_raises_pointing_to_secrets(spelling: str) -> None:
    """`config={'extra': {'api_key': k}}` raises the same error; `extra` goes on the wire and into traces."""
    ai, fn = _ai_with_model()

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='strict', prompt='hi', config={'extra': {spelling: KEY}})

    _assert_points_to_secrets(err, fn)
