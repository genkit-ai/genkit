#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""One config contract for every model, however its config is declared.

Two checkpoints:

- **Before the turn**: only the values the call sets. A bad one raises
  ``INVALID_ARGUMENT``; a missing top-level field passes, because
  ``wrap_model`` middleware can still fill it.
- **Inside the model action** (a background model's start action): the
  whole config. A required field nothing filled fails here. ``generate``
  returns it as the turn's error; ``generate_operation`` has no response
  to put it in, so it raises.

The contract tests run over every pair of model kind (regular, background)
and declaration (``ModelRequest[...]`` annotation, ``config_schema=`` class
with or without the annotation, ``config_schema=`` JSON-schema dict). A new
case added here covers all of them, so the two model kinds can't drift.
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict

from genkit import FinishReason, Genkit, Part
from genkit._core._error import GenkitError
from genkit._core._model import Message, ModelRequest, ModelResponse
from genkit._core._operation import Operation
from genkit._core._typing import Role
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams
from genkit.model import ModelContext, model_ref

NextModel = Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]]

Kind = Literal['model', 'background']
Declared = Literal['annotation', 'config_schema', 'class_only', 'json_schema']
EVERY_KIND: list[Kind] = ['model', 'background']
EVERY_DECLARATION: list[Declared] = ['annotation', 'config_schema', 'class_only', 'json_schema']
# The two declarations that go through genkit._core._config_check (the
# annotation paths fail in the action's input validation instead).
SCHEMA_DECLARATIONS: list[Declared] = ['class_only', 'json_schema']
PROMPT = 'Your table is ready.'


class VoiceConfig(BaseModel):
    """A text-to-speech config whose voice has no default."""

    voice: Literal['alloy', 'echo']
    speed: float = 1.0


class StrictVoice(BaseModel):
    """A user config class that forbids keys it doesn't declare, version included."""

    model_config = ConfigDict(extra='forbid')
    voice: Literal['alloy', 'echo']


class Tts:
    """A TTS model registered as ``kind`` with its config declared as ``declared``.

    ``seen`` holds the configs the model function (or ``start``) received.
    """

    def __init__(self, ai: Genkit, *, kind: Kind, declared: Declared, config_class: type[BaseModel] = VoiceConfig):
        self.ai = ai
        self.kind = kind
        self.seen: list[object] = []
        typed = declared in ('annotation', 'config_schema')
        schema: type[BaseModel] | dict[str, object] | None = {
            'annotation': None,
            'config_schema': config_class,
            'class_only': config_class,
            'json_schema': config_class.model_json_schema(),
        }[declared]
        # One function serves both kinds; define_model and define_background_model each type it narrowly.
        fn: Any = self._typed_fn(config_class) if typed else self._plain_fn()
        if kind == 'model':
            ai.define_model('acme/tts', fn, config_schema=schema)
        else:

            async def check(operation: Operation, ctx: ModelContext) -> Operation:
                return operation

            ai.define_background_model('acme/tts', start=fn, check=check, config_schema=schema)

    def _reply(self, config: object) -> ModelResponse | Operation:
        self.seen.append(config)
        if self.kind == 'model':
            return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('audio')]))
        return Operation(id='op-1', done=False)

    def _typed_fn(self, config_class: type[BaseModel]) -> Callable[..., Awaitable[object]]:
        async def tts(request: ModelRequest[config_class], ctx: ModelContext):  # type: ignore[valid-type]  # noqa: ANN202
            return self._reply(request.config)

        return tts

    def _plain_fn(self) -> Callable[..., Awaitable[object]]:
        async def tts(request: ModelRequest, ctx: ModelContext):  # noqa: ANN202
            return self._reply(request.config)

        return tts

    async def call(self, *, model: object = 'acme/tts', **kwargs: object) -> object:
        """generate for a regular model, generate_operation for a background model."""
        if self.kind == 'model':
            return await self.ai.generate(model=model, prompt=PROMPT, **kwargs)  # type: ignore[arg-type]
        return await self.ai.generate_operation(model=model, prompt=PROMPT, **kwargs)  # type: ignore[arg-type]

    async def failure(self, **kwargs: object) -> tuple[str, str, str] | None:
        """How the call failed, as (where, status, text), or None if it ran.

        ``where`` is ``'raised'`` or ``'turn'`` (a failed generate response).
        ``text`` is the message plus details plus the chained cause: on the
        annotation paths the action's input validation names the field in
        details (failed turn) or only in the cause (raised).
        """
        try:
            result = await self.call(**kwargs)
        except GenkitError as e:
            return 'raised', e.status, f'{e} {e.details} {e.cause}'
        error = getattr(result, 'error', None)
        if error is not None:
            assert getattr(result, 'finish_reason', None) == FinishReason.FAILED
            return 'turn', error.status, f'{error.message} {error.details}'
        return None


def voice_of(config: object) -> object:
    """The voice a model saw, whether its config arrived typed or as a dict."""
    return config.voice if isinstance(config, BaseModel) else dict(config)['voice']  # type: ignore[arg-type,attr-defined]


class DefaultVoice(Middleware):
    """Fills a house voice when the caller left it out."""

    async def wrap_model(self, params: ModelHookParams, ctx: MiddlewareContext, next_fn: NextModel) -> ModelResponse:
        config = params.request.config
        if isinstance(config, dict) and 'voice' not in config:
            params.request.config = {**config, 'voice': 'echo'}
        return await next_fn(params, ctx)


every_kind = pytest.mark.parametrize('kind', EVERY_KIND)
every_declaration = pytest.mark.parametrize('declared', EVERY_DECLARATION)
schema_declarations = pytest.mark.parametrize('declared', SCHEMA_DECLARATIONS)


# Contract table: every kind x every declaration.


@pytest.mark.asyncio
@every_kind
@every_declaration
async def test_complete_config_runs(kind: Kind, declared: Declared) -> None:
    """A complete config reaches the model unchanged."""
    tts = Tts(Genkit(), kind=kind, declared=declared)

    assert await tts.failure(config={'voice': 'echo'}) is None
    assert [voice_of(config) for config in tts.seen] == ['echo']


@pytest.mark.asyncio
@every_kind
@every_declaration
async def test_bad_value_raises_before_the_model_runs(kind: Kind, declared: Declared) -> None:
    """A value the schema rejects raises INVALID_ARGUMENT before the turn."""
    tts = Tts(Genkit(), kind=kind, declared=declared)

    failure = await tts.failure(config={'voice': 'opera'})

    assert failure is not None
    where, status, text = failure
    assert (where, status) == ('raised', 'INVALID_ARGUMENT')
    assert "'voice'" in text
    assert tts.seen == []


@pytest.mark.asyncio
@every_kind
@every_declaration
async def test_missing_required_field_fails_inside_the_action(kind: Kind, declared: Declared) -> None:
    """Nothing fills voice, so the action rejects it: a failed turn from generate, a raise from generate_operation."""
    tts = Tts(Genkit(), kind=kind, declared=declared)

    failure = await tts.failure(config={'speed': 1.2})

    assert failure is not None
    where, status, text = failure
    assert where == ('turn' if kind == 'model' else 'raised')
    assert status == 'INVALID_ARGUMENT'
    assert 'voice' in text
    assert tts.seen == []


@pytest.mark.asyncio
@every_kind
@every_declaration
async def test_middleware_can_fill_missing_required_field(kind: Kind, declared: Declared) -> None:
    """A wrap_model middleware that supplies voice runs after the pre-turn check, so the call works."""
    ai = Genkit()
    tts = Tts(ai, kind=kind, declared=declared)
    ai.define_middleware(DefaultVoice, name='default_voice')

    assert await tts.failure(config={'speed': 1.2}, use=[DefaultVoice()]) is None
    assert [voice_of(config) for config in tts.seen] == ['echo']


# Paths that go through genkit._core._config_check: every kind x class / dict.


@pytest.mark.asyncio
@every_kind
@schema_declarations
async def test_missing_field_message_is_the_same_for_class_and_dict(kind: Kind, declared: Declared) -> None:
    """Class and dict schemas name a missing field in the same words."""
    tts = Tts(Genkit(), kind=kind, declared=declared)

    failure = await tts.failure(config={'speed': 1.2})

    assert failure is not None
    assert "acme/tts: config 'voice': Field required" in failure[2]


@pytest.mark.asyncio
@every_kind
@schema_declarations
async def test_model_ref_version_passes_a_schema_that_does_not_name_it(kind: Kind, declared: Declared) -> None:
    """ModelRef.version rides in config; a strict schema without a version field still runs."""
    tts = Tts(Genkit(), kind=kind, declared=declared, config_class=StrictVoice)
    ref = model_ref('acme/tts', config_schema=StrictVoice, version='v2')

    assert await tts.failure(model=ref, config={'voice': 'echo'}) is None
    assert [dict(config) for config in tts.seen] == [{'voice': 'echo', 'version': 'v2'}]  # type: ignore[call-overload]


@every_kind
@schema_declarations
def test_sync_fn_fails_at_define(kind: Kind, declared: Declared) -> None:
    """The config check wraps the function, but a sync one is still rejected when the model is defined."""
    ai = Genkit()
    schema = VoiceConfig if declared == 'class_only' else VoiceConfig.model_json_schema()

    def tts(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('audio')]))

    async def check(operation: Operation, ctx: ModelContext) -> Operation:
        return operation

    with pytest.raises(TypeError, match='must be async'):
        if kind == 'model':
            ai.define_model('acme/tts', tts, config_schema=schema)  # type: ignore[arg-type]
        else:
            ai.define_background_model('acme/tts', start=tts, check=check, config_schema=schema)  # type: ignore[arg-type]


@every_kind
def test_invalid_json_schema_fails_at_define(kind: Kind) -> None:
    """A config_schema dict that isn't a JSON schema is the model author's bug; it raises at define."""
    ai = Genkit()

    async def tts(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('audio')]))

    async def check(operation: Operation, ctx: ModelContext) -> Operation:
        return operation

    with pytest.raises(GenkitError, match='not a valid JSON schema'):
        if kind == 'model':
            ai.define_model('acme/tts', tts, config_schema={'type': 'bogus'})
        else:
            ai.define_background_model('acme/tts', start=tts, check=check, config_schema={'type': 'bogus'})  # type: ignore[arg-type]


# Single-path cases.


@pytest.mark.asyncio
async def test_required_field_from_prompt_config_satisfies_call_override() -> None:
    """The prompt supplies voice, the call overrides speed: the merged bag is complete."""
    ai = Genkit()
    tts = Tts(ai, kind='model', declared='annotation')
    announce = ai.define_prompt(model='acme/tts', prompt=PROMPT, config={'voice': 'echo'})

    await announce(config={'speed': 1.2})

    assert tts.seen == [VoiceConfig(voice='echo', speed=1.2)]


@pytest.mark.asyncio
async def test_call_override_clearing_required_field_fails_the_turn() -> None:
    """A call that clears the prompt's voice with None leaves it missing for the model action."""
    ai = Genkit()
    tts = Tts(ai, kind='model', declared='annotation')
    announce = ai.define_prompt(model='acme/tts', prompt=PROMPT, config={'voice': 'echo'})

    response = await announce(config={'voice': None})

    assert response.finish_reason == FinishReason.FAILED
    assert tts.seen == []


@pytest.mark.asyncio
async def test_json_schema_nested_missing_field_raises_before_the_turn() -> None:
    """A nested object is sent whole, so a field missing inside it raises before the turn, as with a class."""

    class Pronunciation(BaseModel):
        lexicon: str

    class LexiconVoiceConfig(VoiceConfig):
        pronunciation: Pronunciation | None = None

    tts = Tts(Genkit(), kind='model', declared='json_schema', config_class=LexiconVoiceConfig)

    failure = await tts.failure(config={'pronunciation': {}})

    assert failure is not None
    where, status, text = failure
    assert (where, status) == ('raised', 'INVALID_ARGUMENT')
    assert "config 'pronunciation.lexicon': Field required" in text
    assert tts.seen == []


@pytest.mark.asyncio
async def test_direct_run_with_unannotated_request_checks_config() -> None:
    """A direct action run (Dev UI) hands an unannotated fn the raw dict; the check reads config from it."""
    ai = Genkit()

    async def tts(request, ctx: ModelContext) -> ModelResponse:  # noqa: ANN001
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('audio')]))

    action = ai.define_model('acme/tts', tts, config_schema=VoiceConfig)

    with pytest.raises(GenkitError) as err:
        await action.run({'messages': [], 'config': {'speed': 1.2}})

    assert err.value.status == 'INVALID_ARGUMENT'
    assert "config 'voice': Field required" in str(err.value)
