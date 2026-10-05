#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Genkit extra API methods."""

from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from genkit import Genkit
from genkit._ai._testing import define_echo_model
from genkit._core._action import ActionRunContext, _action_context
from genkit._core._error import GenkitError, RuntimeErrorReason
from genkit._core._model import Message, ModelRef, ModelRequest, ModelResponse, Part
from genkit._core._registry import Registry
from genkit._core._telemetry._instrumentation import (
    SpanMetadata,
    SpanNext,
    reset_instrumentation,
)
from genkit._core._typing import FinishReason, Operation, Role
from genkit.exp import Genkit as ExpGenkit
from genkit.middleware import BaseMiddleware, GenerateHookParams, GenerateMiddlewareContext
from genkit.model import ModelInfo, Supports, model
from genkit.plugin_api import Action, ActionKind
from genkit.telemetry import configure_instrumentation


@pytest.mark.asyncio
async def test_genkit_run() -> None:
    """Test Genkit.run method."""
    ai = Genkit()

    async def async_fn() -> str:
        return 'world'

    res1 = await ai.run(name='test1', fn=async_fn)
    assert res1 == 'world'

    # Test with metadata
    res2 = await ai.run(name='test2', fn=async_fn, metadata={'foo': 'bar'})
    assert res2 == 'world'

    # Test that sync functions raise TypeError
    def sync_fn() -> str:
        return 'hello'

    with pytest.raises(TypeError, match='fn must be a coroutine function'):
        await ai.run(name='test3', fn=sync_fn)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_genkit_run_tags_flow_step_action_type() -> None:
    """ai.run tells the provider its span is a flow step, so traces can label it."""

    class Recording:
        last: SpanMetadata | None = None

        async def run_in_new_span(self, metadata: SpanMetadata, next: SpanNext[str]) -> str:
            self.last = metadata
            return await next()

    recording = Recording()
    reset_instrumentation()
    configure_instrumentation(recording)
    try:
        ai = Genkit()

        async def step() -> str:
            return 'ok'

        assert await ai.run(name='lookup_account', fn=step) == 'ok'
        assert recording.last is not None
        assert recording.last.name == 'lookup_account'
        assert recording.last.action_type == 'flowStep'
    finally:
        reset_instrumentation()


@pytest.mark.asyncio
async def test_genkit_check_operation() -> None:
    """Test Genkit.check_operation method."""
    ai = Genkit()

    op = Operation(id='123', done=False, action='/background-model/test_action')

    # Create mock background action with check method
    mock_background_action = MagicMock()
    mock_background_action.check = AsyncMock(return_value=Operation(id='123', done=True, output='result'))

    # Patch lookup_background_action to return our mock
    with mock.patch(
        'genkit._core._background.lookup_background_action',
        new=AsyncMock(return_value=mock_background_action),
    ) as mock_lookup:
        updated_op = await ai.check_operation(op)

        assert updated_op.done is True
        assert updated_op.output == 'result'
        mock_lookup.assert_called_once()


@pytest.mark.asyncio
async def test_genkit_check_operation_no_action() -> None:
    """Test Genkit.check_operation method with no action."""
    ai = Genkit()
    op = Operation(id='123', done=False)  # action is None

    with pytest.raises(GenkitError, match='Provided operation is missing original request information') as exc_info:
        await ai.check_operation(op)
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_genkit_check_operation_malformed_key_is_invalid_argument() -> None:
    """A mangled action key on a reloaded handle is the caller's bad argument."""
    ai = Genkit()
    op = Operation(id='123', done=False, action='missing')

    with pytest.raises(
        GenkitError, match='Failed to resolve background action from original request: missing'
    ) as exc_info:
        await ai.check_operation(op)
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_genkit_check_operation_not_found() -> None:
    """Test Genkit.check_operation method with action not found."""
    ai = Genkit()
    op = Operation(id='123', done=False, action='/background-model/nope')

    with pytest.raises(
        GenkitError, match='Failed to resolve background action from original request: /background-model/nope'
    ) as exc_info:
        await ai.check_operation(op)
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_check_operation_round_trips_persisted_dump() -> None:
    """model_dump(by_alias=True) -> model_validate is the supported save/reload path."""
    ai = Genkit()

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-rt', start=start, check=check)
    op = Operation(id='job-1', done=False, action='/background-model/bg-rt')

    reloaded = Operation.model_validate(op.model_dump(by_alias=True))
    updated = await ai.check_operation(reloaded)

    assert updated.done is True


@pytest.mark.asyncio
async def test_check_operation_dump_is_invalid_argument() -> None:
    """A saved dict is not an Operation until model_validate."""
    ai = Genkit()
    dumped = {
        'id': '123',
        'done': False,
        'action': '/background-model/test_action',
    }

    with pytest.raises(GenkitError, match='got a dump; pass Operation.model_validate') as exc_info:
        await ai.check_operation(dumped)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_check_operation_boxed_response_is_invalid_argument() -> None:
    """generate() returns a ModelResponse; the handle is response.operation."""
    ai = Genkit()
    boxed = ModelResponse(operation=Operation(id='123', action='/background-model/test_action'))

    with pytest.raises(GenkitError, match='got ModelResponse; pass response.operation') as exc_info:
        await ai.check_operation(boxed)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_check_operation_str_is_invalid_argument() -> None:
    ai = Genkit()

    with pytest.raises(GenkitError, match='got str, expected Operation') as exc_info:
        await ai.check_operation('not-an-op')  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_cancel_operation_round_trips_persisted_dump() -> None:
    """Cancel accepts the same save/reload path as check."""
    ai = Genkit()

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    async def cancel(op: Operation, _ctx: ActionRunContext) -> Operation:
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-cancel-rt', start=start, check=check, cancel=cancel)
    op = Operation(id='job-1', done=False, action='/background-model/bg-cancel-rt')

    reloaded = Operation.model_validate(op.model_dump(by_alias=True))
    updated = await ai.cancel_operation(reloaded)

    assert updated.done is True


@pytest.mark.asyncio
async def test_cancel_operation_dump_is_invalid_argument() -> None:
    ai = Genkit()
    dumped = {
        'id': '123',
        'done': False,
        'action': '/background-model/test_action',
    }

    with pytest.raises(GenkitError, match='got a dump; pass Operation.model_validate') as exc_info:
        await ai.cancel_operation(dumped)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_cancel_operation_without_cancel_is_unimplemented() -> None:
    """The wrapper's UNIMPLEMENTED propagates through the veneer unchanged."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='123', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai = Genkit()
    ai.define_background_model(name='veneer-no-cancel', start=start, check=check)
    op = Operation(id='123', done=False, action='/background-model/veneer-no-cancel')

    with pytest.raises(GenkitError, match='does not support cancellation') as exc_info:
        await ai.cancel_operation(op)
    assert exc_info.value.status == 'UNIMPLEMENTED'
    assert exc_info.value.reason is RuntimeErrorReason.UNSUPPORTED_BY_MODEL
    assert 'UNSUPPORTED_BY_MODEL' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_background_action_cancel_without_fn_is_unimplemented() -> None:
    """A real no-cancel BackgroundAction raises UNIMPLEMENTED from .cancel."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai = Genkit()
    action = ai.define_background_model(name='no-cancel', start=start, check=check)
    op = Operation(id='1', action='/background-model/no-cancel')

    with pytest.raises(GenkitError, match='does not support cancellation') as exc_info:
        await action.cancel(op)
    assert exc_info.value.status == 'UNIMPLEMENTED'
    assert exc_info.value.reason is RuntimeErrorReason.UNSUPPORTED_BY_MODEL
    assert 'UNSUPPORTED_BY_MODEL' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_background_action_check_rejects_non_operation() -> None:
    """BackgroundAction.check uses the same require_operation gate as the veneer."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai = Genkit()
    action = ai.define_background_model(name='bg-check', start=start, check=check)
    dumped = {'id': '1', 'action': '/background-model/bg-check'}
    boxed = ModelResponse(operation=Operation(id='1', action='/background-model/bg-check'))

    with pytest.raises(GenkitError, match='got a dump; pass Operation.model_validate') as dump_exc:
        await action.check(dumped)  # type: ignore[arg-type]
    assert dump_exc.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError, match='got ModelResponse; pass response.operation') as box_exc:
        await action.check(boxed)  # type: ignore[arg-type]
    assert box_exc.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError, match='got str, expected Operation') as str_exc:
        await action.check('not-an-op')  # type: ignore[arg-type]
    assert str_exc.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_background_action_cancel_rejects_non_operation() -> None:
    """A dump must not AttributeError on .action before UNIMPLEMENTED."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai = Genkit()
    action = ai.define_background_model(name='no-cancel', start=start, check=check)

    with pytest.raises(GenkitError, match='got a dump; pass Operation.model_validate') as exc_info:
        await action.cancel({'id': '1', 'action': '/background-model/no-cancel'})  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_current_context() -> None:
    """Test Genkit.current_context method."""
    # current_context is a static method
    assert Genkit.current_context() is None

    context: dict[str, object] = {'auth': {'uid': '123'}}

    # Simulate being inside an action run using ActionRunContext internal mechanism
    token = _action_context.set(context)
    try:
        assert Genkit.current_context() == context
    finally:
        _action_context.reset(token)

    assert Genkit.current_context() is None


@pytest.mark.asyncio
async def test_lookup_model_returns_a_model_ref_that_generate_accepts() -> None:
    ai = Genkit()
    define_echo_model(ai, name='echo')

    ref = await ai.lookup_model('echo')

    assert isinstance(ref, ModelRef)
    response = await ai.generate(model=ref, prompt='hi')
    assert '[ECHO]' in response.text


@pytest.mark.asyncio
async def test_lookup_model_unknown_name_returns_none() -> None:
    ai = Genkit()
    assert await ai.lookup_model('ghost') is None


class ShopCfg(BaseModel):
    aisle: str


@pytest.mark.asyncio
async def test_lookup_model_then_generate_accepts_same_config_as_the_name() -> None:
    ai = Genkit()
    define_echo_model(ai, name='echo')

    named = await ai.generate(model='echo', config=ShopCfg(aisle='12'), prompt='hi')
    ref = await ai.lookup_model('echo')
    assert ref is not None
    via_ref = await ai.generate(model=ref, config=ShopCfg(aisle='12'), prompt='hi')

    assert named.text == via_ref.text
    assert 'aisle' in via_ref.text


@pytest.mark.asyncio
async def test_lookup_model_dap_qualified_name_generate_finds_the_same_model() -> None:
    ai = Genkit()

    async def echo(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('[ECHO] dap')]),
        )

    child = model('foo', echo)

    async def dap_fn():
        return {'model': [child]}

    ai.define_dynamic_action_provider('mcp', dap_fn)

    ref = await ai.lookup_model('mcp:model/foo')

    assert ref is not None
    assert ref.name == 'mcp:model/foo'
    response = await ai.generate(model=ref, prompt='hi')
    assert '[ECHO] dap' in response.text


@pytest.mark.asyncio
async def test_lookup_model_copies_supports_onto_info() -> None:
    ai = Genkit()

    async def echo(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )

    supports = Supports(tools=True, multiturn=True)
    ai.define_model(name='echo', fn=echo, info=ModelInfo(supports=supports))

    ref = await ai.lookup_model('echo')

    assert ref is not None
    assert ref.info is not None
    assert ref.info.supports is not None
    assert ref.info.supports.tools is True
    assert ref.info.supports.multiturn is True


@pytest.mark.asyncio
async def test_lookup_agent_found_returns_agent() -> None:
    ai = ExpGenkit()
    define_echo_model(ai, name='echo')
    defined = ai.define_agent(name='shop', model='echo')

    found = await ai.lookup_agent('shop')

    assert found is defined


@pytest.mark.asyncio
async def test_lookup_agent_unknown_returns_none() -> None:
    ai = ExpGenkit()
    assert await ai.lookup_agent('ghost') is None


@pytest.mark.asyncio
async def test_lookup_agent_non_agent_slot_raises() -> None:
    ai = ExpGenkit()

    async def dummy(_inp: str) -> str:
        return _inp

    ai.registry.register_action_from_instance(Action(kind=ActionKind.AGENT, name='occupied', fn=dummy))

    with pytest.raises(GenkitError, match="Registry entry 'occupied' is not an Agent.") as exc_info:
        await ai.lookup_agent('occupied')
    assert exc_info.value.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_lookup_background_model_returns_a_ref_that_generate_operation_accepts() -> None:
    ai = Genkit()

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai.define_background_model(name='bg', start=start, check=check)
    ref = await ai.lookup_background_model('bg')

    assert isinstance(ref, ModelRef)
    operation = await ai.generate_operation(model=ref, prompt='hi')
    assert operation.id == 'job-1'


@pytest.mark.asyncio
async def test_lookup_background_model_unknown_name_returns_none() -> None:
    ai = Genkit()
    assert await ai.lookup_background_model('ghost') is None


@pytest.mark.asyncio
async def test_resolve_action_returns_the_registered_action() -> None:
    ai = Genkit()
    _echo, defined = define_echo_model(ai, name='echo')

    found = await ai.registry.resolve_action(ActionKind.MODEL, 'echo')

    assert found is defined


@pytest.mark.asyncio
async def test_resolve_action_unknown_name_returns_none() -> None:
    ai = Genkit()
    assert await ai.registry.resolve_action(ActionKind.MODEL, 'ghost') is None


@pytest.mark.asyncio
async def test_resolve_action_on_ctx_ai_finds_a_per_call_model() -> None:
    parent_ai = Genkit()

    async def echo(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )

    parent_ai.define_model(name='echo', fn=echo)
    seen: dict[str, object] = {}

    class RegisterPerCall(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn,
        ) -> ModelResponse:
            async def per_call(_request: ModelRequest, _run_ctx: ActionRunContext) -> ModelResponse:
                return ModelResponse(
                    finish_reason=FinishReason.STOP,
                    message=Message(role=Role.MODEL, content=[Part.from_text('child')]),
                )

            ctx.ai.registry.register_action_from_instance(model('per-call', per_call))
            seen['child'] = await ctx.ai.registry.resolve_action(ActionKind.MODEL, 'per-call')
            seen['parent'] = await parent_ai.registry.resolve_action(ActionKind.MODEL, 'per-call')
            return await next_fn(params, ctx)

    await parent_ai.generate(model='echo', prompt='hi', use=[RegisterPerCall()])

    assert seen['child'] is not None
    assert seen['parent'] is None


def test_ai_registry_is_accessible_and_registers_values() -> None:
    ai = Genkit()
    assert isinstance(ai.registry, Registry)
    v = {'id': 'shop'}
    ai.registry.register_value('a2ui-catalog', 'shop', v)
    assert ai.registry.lookup_value('a2ui-catalog', 'shop') is v
    assert ai.registry.lookup_value('a2ui-catalog', 'ghost') is None


def test_ai_registry_duplicate_registration_raises() -> None:
    ai = Genkit()
    first = {'id': 'shop'}
    ai.registry.register_value('a2ui-catalog', 'shop', first)
    with pytest.raises(ValueError, match='already registered'):
        ai.registry.register_value('a2ui-catalog', 'shop', {'id': 'other'})
    assert ai.registry.lookup_value('a2ui-catalog', 'shop') is first


@pytest.mark.asyncio
async def test_middleware_ctx_ai_registry_sees_app_values_and_per_call_isolation() -> None:
    ai = Genkit()
    define_echo_model(ai, name='echo')
    app_val = {'id': 'app-scope'}
    ai.registry.register_value('custom', 'app', app_val)
    seen: dict[str, object] = {}

    class InspectRegistry(BaseMiddleware):
        async def wrap_generate(self, params, ctx: GenerateMiddlewareContext, next_fn):
            seen['app_from_ctx'] = ctx.ai.registry.lookup_value('custom', 'app')
            ctx.ai.registry.register_value('custom', 'call', 'call-scope')
            seen['call_from_ctx'] = ctx.ai.registry.lookup_value('custom', 'call')
            return await next_fn(params, ctx)

    await ai.generate(model='echo', prompt='hi', use=[InspectRegistry()])
    assert seen['app_from_ctx'] is app_val
    assert seen['call_from_ctx'] == 'call-scope'
    assert ai.registry.lookup_value('custom', 'call') is None
