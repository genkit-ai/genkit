#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for latency tracking in actions."""

import asyncio
from typing import cast

import pytest
from pydantic import BaseModel

from genkit._core._action import Action, ActionKind


class MockResponse(BaseModel):
    """A mock response object for testing latency tracking."""

    latency_ms: float | None = None
    value: str


@pytest.mark.asyncio
async def test_action_latency_ms_population() -> None:
    """Verify that latency_ms is automatically populated for actions returning supporting objects."""

    async def async_model_fn(input: str) -> MockResponse:
        # Simulate some work
        await asyncio.sleep(0.1)
        return MockResponse(value=f'hello {input}')

    # We need asyncio for sleep in the actual test, but for simplicity we can use time.sleep
    # if we want to test sync wrapper or just await a task.
    action = cast(Action[str, MockResponse], Action(name='testModel', kind=ActionKind.MODEL, fn=async_model_fn))

    response = await action.run('world')

    assert response.response.value == 'hello world'
    assert response.response.latency_ms is not None
    assert response.response.latency_ms >= 100  # Should be at least 100ms due to sleep


class ImmutableMockResponse(BaseModel):
    """A mock response object for testing latency tracking with frozen models."""

    model_config = {'frozen': True}
    latency_ms: float | None = None
    value: str


@pytest.mark.asyncio
async def test_immutable_action_latency_ms_population() -> None:
    """Verify that latency_ms is populated even for frozen Pydantic models."""

    async def async_model_fn(input: str) -> ImmutableMockResponse:
        return ImmutableMockResponse(value=f'hello {input}')

    action = cast(
        Action[str, ImmutableMockResponse],
        Action(name='testImmutableModel', kind=ActionKind.MODEL, fn=async_model_fn),
    )

    response = await action.run('world')

    assert response.response.value == 'hello world'
    assert response.response.latency_ms is not None
    assert isinstance(response.response, ImmutableMockResponse)


class ReadOnlyMockResponse(BaseModel):
    """A mock response object with a read-only latency_ms property."""

    _latency_ms: float | None = None
    value: str

    @property
    def latency_ms(self) -> float | None:
        """The latency in milliseconds."""
        return self._latency_ms


@pytest.mark.asyncio
async def test_readonly_action_latency_ms_population() -> None:
    """Verify that latency_ms is handled correctly for read-only properties."""

    async def async_model_fn(input: str) -> ReadOnlyMockResponse:
        return ReadOnlyMockResponse(value=f'hello {input}')

    action = cast(
        Action[str, ReadOnlyMockResponse],
        Action(name='testReadOnlyModel', kind=ActionKind.MODEL, fn=async_model_fn),
    )

    # In this case, it should NOT be updated because model_copy on a non-frozen model
    # will still try to use setattr if the field exists, or it won't have latency_ms in its fields.
    # Actually, Pydantic's model_copy only updates fields. latency_ms is a property here.

    response = await action.run('world')

    assert response.response.value == 'hello world'
    # Since it's a property without a setter AND not a Pydantic field,
    # _record_latency will catch AttributeError and try model_copy.
    # However, model_copy(update={'latency_ms': ...}) will only work if 'latency_ms'
    # is a field in the model.
    # If it's just a property, it might not be updated unless we handle it specifically.
    # But the goal is to NOT crash.
