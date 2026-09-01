#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Genkit extra API methods."""

from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from genkit import Genkit
from genkit._core._action import ActionRunContext, _action_context
from genkit._core._error import GenkitError
from genkit._core._model import ModelRequest, ModelResponse
from genkit._core._typing import Operation


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

    async def check(op: Operation) -> Operation:
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


@pytest.mark.asyncio
async def test_check_operation_boxed_response_is_invalid_argument() -> None:
    """generate() returns a ModelResponse; the handle is response.operation."""
    ai = Genkit()
    boxed = ModelResponse(operation=Operation(id='123', action='/background-model/test_action'))

    with pytest.raises(GenkitError, match='got ModelResponse; pass response.operation') as exc_info:
        await ai.check_operation(boxed)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_check_operation_str_is_invalid_argument() -> None:
    ai = Genkit()

    with pytest.raises(GenkitError, match='got str, expected Operation') as exc_info:
        await ai.check_operation('not-an-op')  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_cancel_operation_round_trips_persisted_dump() -> None:
    """Cancel accepts the same save/reload path as check."""
    ai = Genkit()

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='job-1', done=False)

    async def check(op: Operation) -> Operation:
        return op

    async def cancel(op: Operation) -> Operation:
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


@pytest.mark.asyncio
async def test_cancel_operation_without_cancel_is_unimplemented() -> None:
    """The wrapper's UNIMPLEMENTED propagates through the veneer unchanged."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='123', done=False)

    async def check(op: Operation) -> Operation:
        return op

    ai = Genkit()
    ai.define_background_model(name='veneer-no-cancel', start=start, check=check)
    op = Operation(id='123', done=False, action='/background-model/veneer-no-cancel')

    with pytest.raises(GenkitError, match='does not support cancellation') as exc_info:
        await ai.cancel_operation(op)
    assert exc_info.value.status == 'UNIMPLEMENTED'


@pytest.mark.asyncio
async def test_background_action_cancel_without_fn_is_unimplemented() -> None:
    """A real no-cancel BackgroundAction raises UNIMPLEMENTED from .cancel."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='1', done=False)

    async def check(op: Operation) -> Operation:
        return op

    ai = Genkit()
    action = ai.define_background_model(name='no-cancel', start=start, check=check)
    op = Operation(id='1', action='/background-model/no-cancel')

    with pytest.raises(GenkitError, match='does not support cancellation') as exc_info:
        await action.cancel(op)
    assert exc_info.value.status == 'UNIMPLEMENTED'


@pytest.mark.asyncio
async def test_background_action_check_rejects_non_operation() -> None:
    """BackgroundAction.check uses the same require_operation gate as the veneer."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='1', done=False)

    async def check(op: Operation) -> Operation:
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

    async def check(op: Operation) -> Operation:
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
