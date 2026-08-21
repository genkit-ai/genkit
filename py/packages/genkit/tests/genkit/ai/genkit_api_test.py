#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Genkit extra API methods."""

from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from genkit import Genkit
from genkit._core._action import _action_context
from genkit._core._error import GenkitError
from genkit._core._model import ModelResponse
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
async def test_genkit_check_operation_not_found() -> None:
    """Test Genkit.check_operation method with action not found."""
    ai = Genkit()
    op = Operation(id='123', done=False, action='missing')
    ai.registry.resolve_action_by_key = AsyncMock(return_value=None)  # type: ignore[assignment]

    with pytest.raises(
        GenkitError, match='Failed to resolve background action from original request: missing'
    ) as exc_info:
        await ai.check_operation(op)
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_check_operation_rejects_boxed_response() -> None:
    """A generate() ModelResponse is not an Operation dump."""
    ai = Genkit()
    op = Operation(id='123', done=False, action='/background-model/test_action')

    with pytest.raises(GenkitError, match='not a valid Operation') as exc_info:
        await ai.check_operation(ModelResponse(operation=op))  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_check_operation_rejects_empty_response() -> None:
    ai = Genkit()
    with pytest.raises(GenkitError, match='not a valid Operation') as exc_info:
        await ai.check_operation(ModelResponse())  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_check_operation_accepts_dumped_operation() -> None:
    """A persisted dump still polls, even with leftover keys like latencyMs."""
    ai = Genkit()
    dumped = {
        'id': '123',
        'done': False,
        'action': '/background-model/test_action',
        'latencyMs': 42,
    }
    mock_background_action = MagicMock()
    mock_background_action.check = AsyncMock(return_value=Operation(id='123', done=True))

    with mock.patch(
        'genkit._core._background.lookup_background_action',
        new=AsyncMock(return_value=mock_background_action),
    ):
        updated = await ai.check_operation(dumped)

    assert updated.done is True


@pytest.mark.asyncio
async def test_check_operation_rejects_dumped_response() -> None:
    """A dumped generate() envelope is not an Operation (no id)."""
    ai = Genkit()
    dumped = {
        'operation': {
            'id': '123',
            'done': False,
            'action': '/background-model/test_action',
        },
        'finishReason': 'stop',
    }

    with pytest.raises(GenkitError, match='not a valid Operation') as exc_info:
        await ai.check_operation(dumped)
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_check_operation_rejects_unreadable_handle() -> None:
    ai = Genkit()
    with pytest.raises(GenkitError, match='not a valid Operation') as exc_info:
        await ai.check_operation('not-an-op')  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_cancel_operation_accepts_dumped_operation() -> None:
    """Cancel accepts the same persisted Operation dump as check."""
    ai = Genkit()
    dumped = {
        'id': '123',
        'done': False,
        'action': '/background-model/test_action',
        'latencyMs': 42,
    }
    mock_background_action = MagicMock()
    mock_background_action.cancel = AsyncMock(return_value=Operation(id='123', done=True))

    with mock.patch(
        'genkit._core._background.lookup_background_action',
        new=AsyncMock(return_value=mock_background_action),
    ):
        updated = await ai.cancel_operation(dumped)

    assert updated.done is True


@pytest.mark.asyncio
async def test_cancel_operation_rejects_boxed_response() -> None:
    ai = Genkit()
    op = Operation(id='123', done=False, action='/background-model/test_action')

    with pytest.raises(GenkitError, match='not a valid Operation') as exc_info:
        await ai.cancel_operation(ModelResponse(operation=op))  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_cancel_operation_without_cancel_is_unimplemented() -> None:
    ai = Genkit()
    op = Operation(id='123', done=False, action='/background-model/test_action')
    mock_background_action = MagicMock()
    mock_background_action.supports_cancel = False
    mock_background_action.cancel = AsyncMock()

    with mock.patch(
        'genkit._core._background.lookup_background_action',
        new=AsyncMock(return_value=mock_background_action),
    ):
        with pytest.raises(GenkitError, match='does not support cancellation') as exc_info:
            await ai.cancel_operation(op)
    assert exc_info.value.status == 'UNIMPLEMENTED'
    mock_background_action.cancel.assert_not_awaited()


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
