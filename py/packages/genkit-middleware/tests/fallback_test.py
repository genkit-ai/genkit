# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Fallback middleware."""

from typing import Any, NoReturn

import pytest
from genkit_middleware import Fallback
from pydantic import Field, ValidationError

from genkit import (
    ActionRunContext,
    FinishReason,
    Genkit,
    GenkitError,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    Role,
)
from genkit.middleware import GenerateMiddlewareContext, ModelHookParams
from genkit.model import ModelConfig, ModelRequest, model_ref
from genkit.testing import define_scripted_model


def _make_params() -> ModelHookParams:
    return ModelHookParams(request=ModelRequest(messages=[]))


def _make_fallback(**kwargs) -> Fallback:
    return Fallback(**kwargs)


@pytest.mark.asyncio
async def test_fallback_success_on_first_model(ctx) -> None:
    """Test that successful primary model calls pass through."""
    fallback = _make_fallback(models=['model2', 'model3'])

    async def next_fn(params, ctx):
        return ModelResponse(message=None)

    result = await fallback.wrap_model(_make_params(), ctx, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_fallback_on_retryable_error(ctx) -> None:
    """Test that retryable errors are classified correctly."""
    fallback = _make_fallback(models=['model2'])

    async def next_fn(params, ctx) -> NoReturn:
        raise GenkitError(message='Service unavailable', status='UNAVAILABLE')

    with pytest.raises(GenkitError):
        await fallback.wrap_model(_make_params(), ctx, next_fn)


@pytest.mark.asyncio
async def test_fallback_non_retryable_error(ctx) -> None:
    """Test that non-retryable errors fail immediately."""
    fallback = _make_fallback(models=['model2'])

    async def next_fn(params, ctx) -> NoReturn:
        raise GenkitError(message='Invalid argument', status='INVALID_ARGUMENT')

    with pytest.raises(GenkitError):
        await fallback.wrap_model(_make_params(), ctx, next_fn)


@pytest.mark.asyncio
async def test_fallback_non_genkit_error_raises_without_trying_next_model(ctx) -> None:
    """A raw TypeError (a bug in another middleware, say) propagates without fallback."""
    fallback = _make_fallback(models=['model2'])

    async def next_fn(params, ctx) -> NoReturn:
        raise TypeError("'NoneType' object is not subscriptable")

    with pytest.raises(TypeError, match='not subscriptable'):
        await fallback.wrap_model(_make_params(), ctx, next_fn)


def _config_value(config: Any, key: str) -> Any:  # noqa: ANN401
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


class ThinkingConfig(ModelConfig):
    """A Gemini-shaped class so the failed call can carry thinkingConfig."""

    thinking_config: dict[str, Any] | None = Field(default=None, alias='thinkingConfig')


@pytest.mark.asyncio
async def test_fallback_to_other_model_sends_only_the_entry_config() -> None:
    """When the primary fails, a string backup runs with that model's defaults — no thinkingConfig."""
    ai = Genkit()
    seen: list[object] = []

    async def primary(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='down')

    async def backup(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        seen.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='gem', fn=primary, config_schema=ThinkingConfig)
    ai.define_model(name='oai', fn=backup, config_schema=ModelConfig)

    response = await ai.generate(
        model='gem',
        prompt='hi',
        config={
            'temperature': 0.2,
            'thinkingConfig': {'thinkingBudget': 0},
            'version': 'gemini-2.5-flash-001',
            'extra': {'labels': {'team': 'search'}},
        },
        use=[Fallback(models=['oai'])],
    )

    assert response.text == 'ok'
    assert len(seen) == 1
    assert _config_value(seen[0], 'thinkingConfig') is None
    assert _config_value(seen[0], 'thinking_config') is None
    assert _config_value(seen[0], 'temperature') is None
    assert _config_value(seen[0], 'version') is None
    assert _config_value(seen[0], 'extra') is None


@pytest.mark.asyncio
async def test_fallback_to_other_provider_does_not_pass_thinking_config() -> None:
    """Fallback to another provider does not pass the failed Gemini call's thinkingConfig."""
    ai = Genkit()
    seen: list[object] = []

    async def primary(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='down')

    async def backup(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        seen.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='gem', fn=primary, config_schema=ThinkingConfig)
    ai.define_model(name='oai', fn=backup, config_schema=ModelConfig)

    await ai.generate(
        model='gem',
        prompt='hi',
        config={'thinkingConfig': {'thinkingBudget': 0}},
        use=[Fallback(models=['oai'])],
    )

    assert _config_value(seen[0], 'thinkingConfig') is None
    assert _config_value(seen[0], 'thinking_config') is None


@pytest.mark.asyncio
async def test_fallback_ref_entry_config_reaches_fallback_model() -> None:
    """A ModelRef backup entry sends only that ref's config to the fallback model."""
    ai = Genkit()
    seen: list[object] = []

    async def primary(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='down')

    async def backup(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        seen.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='gem', fn=primary, config_schema=ThinkingConfig)
    ai.define_model(name='oai', fn=backup, config_schema=ModelConfig)
    ref = model_ref('oai', config_schema=ModelConfig, config=ModelConfig(temperature=0.2))

    response = await ai.generate(
        model='gem',
        prompt='hi',
        config={'temperature': 0.9, 'thinkingConfig': {'thinkingBudget': 0}},
        use=[Fallback(models=[ref])],
    )

    assert response.text == 'ok'
    assert _config_value(seen[0], 'temperature') == 0.2
    assert _config_value(seen[0], 'thinkingConfig') is None
    assert _config_value(seen[0], 'thinking_config') is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('entry', 'temperature'),
    [
        (model_ref('backup', config_schema=ThinkingConfig, config=ThinkingConfig(temperature=0.2)), 0.2),
        ('backup', None),
    ],
    ids=['ref_entry', 'string_entry'],
)
async def test_fallback_backup_with_same_config_class_gets_that_class(entry: object, temperature: float | None) -> None:
    """A backup that shares the primary's config class gets a ThinkingConfig, not a dict or None."""
    ai = Genkit()
    seen: list[object] = []

    async def primary(_request: ModelRequest[ThinkingConfig], _ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='down')

    async def backup(request: ModelRequest[ThinkingConfig], _ctx: ActionRunContext) -> ModelResponse:
        seen.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='primary', fn=primary, config_schema=ThinkingConfig)
    ai.define_model(name='backup', fn=backup, config_schema=ThinkingConfig)

    response = await ai.generate(
        model='primary',
        prompt='hi',
        config={'temperature': 0.9},
        use=[Fallback(models=[entry])],  # type: ignore[list-item]
    )

    assert response.text == 'ok'
    assert isinstance(seen[0], ThinkingConfig)
    assert seen[0].temperature == temperature


@pytest.mark.asyncio
async def test_fallback_switches_to_backup_model_found_by_name() -> None:
    """With `Fallback(models=['backup'])`, a model raising UNAVAILABLE falls back to the app's `backup` model."""
    ai = Genkit()

    async def down(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='provider is down')

    async def backup(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('from backup')]),
        )

    ai.define_model(name='primary', fn=down)
    ai.define_model(name='backup', fn=backup)

    response = await ai.generate(model='primary', prompt='hi', use=[Fallback(models=['backup'])])

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'from backup'
    assert response.error is None


@pytest.mark.asyncio
async def test_fallback_unknown_backup_model_is_not_found() -> None:
    """A backup name nothing on the app answers to fails the call with NOT_FOUND naming that model."""
    ai = Genkit()

    async def down(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='provider is down')

    ai.define_model(name='primary', fn=down)

    response = await ai.generate(model='primary', prompt='hi', use=[Fallback(models=['nope'])])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'No model named "nope" is registered on this app.'
    assert response.error is not None
    assert response.error.status == 'NOT_FOUND'
    assert response.message is None


@pytest.mark.asyncio
async def test_generate_with_model_raising_connection_error_and_fallback_keeps_the_failure() -> None:
    """An unclassified ConnectionError from the model fails the call without trying `backup`."""
    ai = Genkit()
    backup_calls = 0

    async def down(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        raise ConnectionError('connection refused')

    async def backup(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        nonlocal backup_calls
        backup_calls += 1
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('from backup')]))

    ai.define_model(name='primary', fn=down)
    ai.define_model(name='backup', fn=backup)

    response = await ai.generate(model='primary', prompt='hi', use=[Fallback(models=['backup'])])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.message is None
    assert backup_calls == 0


@pytest.mark.asyncio
async def test_prompt_with_failing_on_chunk_and_fallback_does_not_call_backup() -> None:
    """`await prompt(on_chunk=raises, use=[Fallback(...)])` fails with the callback's message and never calls backup."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial')])]]
    backup_calls = 0

    async def backup(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        nonlocal backup_calls
        backup_calls += 1
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('from backup')]),
        )

    ai.define_model(name='backup', fn=backup)
    prompt = ai.define_prompt(model='scriptedModel', prompt='hi')

    def on_chunk(_: object) -> None:
        raise RuntimeError('model sink closed')

    response = await prompt(on_chunk=on_chunk, use=[Fallback(models=['backup'])])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'model sink closed'
    assert backup_calls == 0


@pytest.mark.asyncio
async def test_fallback_stops_when_aborted() -> None:
    """Test that fallback halts and re-raises when the abort signal is already set."""
    ai = Genkit()
    ran: list[str] = []

    async def backup1(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        ran.append('backup1')
        raise GenkitError(status='UNAVAILABLE', message='backup1 down')

    async def backup2(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        ran.append('backup2')
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('backup2')]),
        )

    ai.define_model(name='backup1', fn=backup1)
    ai.define_model(name='backup2', fn=backup2)

    ctx = GenerateMiddlewareContext(ai=ai)
    ctx.abort_signal.set()
    fallback = Fallback(models=['backup1', 'backup2'])

    async def next_fn(_params: ModelHookParams, _ctx: GenerateMiddlewareContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='primary down')

    with pytest.raises(GenkitError, match='primary down'):
        await fallback.wrap_model(_make_params(), ctx, next_fn)

    assert ran == []


@pytest.mark.asyncio
async def test_fallback_halts_subsequent_models_on_abort() -> None:
    """Test that fallback stops calling subsequent backup models when aborted mid-sequence."""
    ai = Genkit()
    ran: list[str] = []

    async def backup1(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        ran.append('backup1')
        ctx.abort_signal.set()
        raise GenkitError(status='UNAVAILABLE', message='backup1 down')

    async def backup2(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        ran.append('backup2')
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('backup2')]),
        )

    ai.define_model(name='backup1', fn=backup1)
    ai.define_model(name='backup2', fn=backup2)

    ctx = GenerateMiddlewareContext(ai=ai)
    fallback = Fallback(models=['backup1', 'backup2'])

    async def next_fn(_params: ModelHookParams, _ctx: GenerateMiddlewareContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='primary down')

    with pytest.raises(GenkitError, match='backup1 down'):
        await fallback.wrap_model(_make_params(), ctx, next_fn)

    assert ran == ['backup1']


@pytest.mark.asyncio
async def test_fallback_streams_chunks_from_the_fallback_model() -> None:
    """Test that fallback streams chunks emitted by the fallback model."""
    ai = Genkit()

    async def fail(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='primary down')

    async def backup(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('from-backup')]))
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )

    ai.define_model(name='primary', fn=fail)
    ai.define_model(name='backup', fn=backup)

    stream = ai.generate_stream(model='primary', prompt='hi', use=[Fallback(models=['backup'])])
    texts: list[str] = []
    async for chunk in stream.stream:
        texts.append(chunk.text)
    final = await stream.response

    assert 'from-backup' in ''.join(texts)
    assert final.text == 'done'


def test_fallback_given_a_model_action_names_the_string_to_pass() -> None:
    """Fallback config is JSON, so a define_model action is refused with its name in the message."""

    async def backup(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    backup_model = Genkit().define_model(name='backup', fn=backup)

    with pytest.raises(ValidationError, match="pass 'backup', not the action"):
        Fallback(models=[backup_model])
