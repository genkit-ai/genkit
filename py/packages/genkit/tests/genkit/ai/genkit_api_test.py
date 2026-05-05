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

    with pytest.raises(ValueError, match='Provided operation is missing original request information'):
        await ai.check_operation(op)


@pytest.mark.asyncio
async def test_genkit_check_operation_not_found() -> None:
    """Test Genkit.check_operation method with action not found."""
    ai = Genkit()
    op = Operation(id='123', done=False, action='missing')
    ai.registry.resolve_action_by_key = AsyncMock(return_value=None)  # type: ignore[assignment]

    with pytest.raises(ValueError, match='Failed to resolve background action from original request: missing'):
        await ai.check_operation(op)


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
