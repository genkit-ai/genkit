#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A per-request API key goes in context.secrets; a key in config raises before the model runs.

Every failing case asserts the status, that the model was never called, and
that the message points at ``context={'secrets': {'api_key': ...}}``.
"""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from genkit import Genkit, Message, ModelResponse, Part
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import ModelRequest
from genkit._core._schema import to_json_schema
from genkit._core._typing import Operation, Role
from genkit.model import ModelConfig, model_ref

KEY = 'sk-tenant'
SECRETS_HINT = "context={'secrets': {'api_key': ...}}"


class StrictConfig(ModelConfig):
    """A plugin class built on ModelConfig that declares one setting of its own."""

    safe_prompt: bool | None = None


class OwnStrictConfig(BaseModel):
    """A plugin class that isn't built on ModelConfig and forbids unknown keys."""

    model_config = ConfigDict(extra='forbid')

    num_ctx: int | None = None


class LegacyConfig(ModelConfig):
    """A plugin class that still declares its own ``api_key`` setting."""

    api_key: str | None = None


class _Model:
    """A model that records every request and context it gets and answers 'ok'."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.contexts: list[dict[str, Any]] = []

    def define(self, ai: Genkit, *, name: str, config_schema: type[BaseModel] | None) -> None:
        async def fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            self.requests.append(request)
            self.contexts.append(dict(ctx.context))
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
async def test_generate_plugin_class_that_still_declares_api_key_raises() -> None:
    """`config=LegacyConfig(api_key=k)` on a plugin class that kept an `api_key` field still raises."""
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


@pytest.mark.asyncio
async def test_generate_extra_api_key_is_not_checked() -> None:
    """`config={'extra': {'api_key': k}}` isn't rejected; `extra` is sent as-is and isn't checked."""
    ai, fn = _ai_with_model()

    await ai.generate(model='strict', prompt='hi', config={'extra': {'api_key': KEY}})

    config = fn.requests[-1].config
    extra = config.get('extra') if isinstance(config, dict) else getattr(config, 'extra', None)
    assert extra == {'api_key': KEY}


def test_model_config_json_schema_has_no_api_key() -> None:
    """The Dev UI config form built from `ModelConfig` no longer offers `apiKey`."""
    properties = to_json_schema(ModelConfig)['properties']

    assert 'apiKey' not in properties
    assert 'api_key' not in properties
    assert 'temperature' in properties
