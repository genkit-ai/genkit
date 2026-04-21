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

"""Tests for genkit.aio.Channel."""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

import pytest

from genkit._core._channel import Channel

T = TypeVar('T')


@pytest.mark.asyncio
async def test_channel_send_and_receive() -> None:
    """Tests sending a value and receiving it from the channel."""
    channel: Channel[str] = Channel[str]()
    channel.send('hello')
    received = await channel.__anext__()
    assert received == 'hello'


@pytest.mark.asyncio
async def test_channel_empty() -> None:
    """Tests that __anext__ waits for a value when the channel is empty."""
    close_future: asyncio.Future[Any] = asyncio.Future()
    channel: Channel[Any] = Channel()
    channel.set_close_future(close_future)

    async def async_send() -> None:
        channel.send('world')

    send_task = asyncio.create_task(async_send())
    receive_task = asyncio.create_task(channel.__anext__())
    assert not receive_task.done()
    await send_task
    received = await receive_task
    assert received == 'world'


@pytest.mark.asyncio
async def test_channel_close() -> None:
    """Tests that the channel closes correctly."""
    channel: Channel[Any] = Channel()
    close_future: asyncio.Future[Any] = asyncio.Future()
    channel.set_close_future(close_future)
    close_future.set_result(None)
    with pytest.raises(StopAsyncIteration):
        await channel.__anext__()


@pytest.mark.asyncio
async def test_channel_multiple_send_receive() -> None:
    """Tests sending and receiving multiple values."""
    channel: Channel[Any] = Channel()
    values = ['one', 'two', 'three']
    for value in values:
        channel.send(value)
    received_values = [await channel.__anext__() for _ in range(len(values))]
    assert received_values == values


@pytest.mark.asyncio
async def test_channel_aiter_anext() -> None:
    """Tests the asynchronous iterator functionality."""
    close_future: asyncio.Future[Any] = asyncio.Future()
    channel: Channel[Any] = Channel()
    channel.set_close_future(close_future)
    values = ['a', 'b', 'c']
    for value in values:
        channel.send(value)
    close_future.set_result('done')
    received_values = []
    async for item in channel:
        received_values.append(item)
    assert received_values == values
    assert (await channel.closed) == 'done'


@pytest.mark.asyncio
async def test_channel_invalid_timeout() -> None:
    """Tests that an invalid timeout value raises ValueError."""
    with pytest.raises(ValueError):
        Channel(timeout=-0.1)


@pytest.mark.asyncio
async def test_channel_timeout() -> None:
    """Tests that the channel raises TimeoutError when timeout is reached."""
    channel: Channel[Any] = Channel(timeout=0.1)
    with pytest.raises(TimeoutError):
        await channel.__anext__()


@pytest.mark.asyncio
async def test_channel_no_timeout() -> None:
    """Tests that the channel doesn't timeout when timeout=None."""
    channel: Channel[Any] = Channel(timeout=None)
    anext_task = asyncio.create_task(channel.__anext__())
    await asyncio.sleep(0.1)
    assert not anext_task.done()
    channel.send('value')
    result = await anext_task
    assert result == 'value'


@pytest.mark.asyncio
async def test_channel_timeout_with_close_future() -> None:
    """Tests timeout with an active close_future."""
    channel: Channel[Any] = Channel(timeout=0.1)
    close_future: asyncio.Future[Any] = asyncio.Future()
    channel.set_close_future(close_future)
    with pytest.raises(TimeoutError):
        await channel.__anext__()
    close_future.set_result(None)
    with pytest.raises(StopAsyncIteration):
        await channel.__anext__()


@pytest.mark.asyncio
async def test_channel_invalid_timeout_negative() -> None:
    """Tests that negative timeout values raise ValueError."""
    with pytest.raises(ValueError) as excinfo:
        Channel(timeout=-1.0)
    assert 'Timeout must be non-negative' in str(excinfo.value)


@pytest.mark.asyncio
async def test_channel_timeout_race_condition() -> None:
    """Tests the behavior when a value arrives just as the timeout occurs."""
    channel: Channel[Any] = Channel(timeout=0.2)

    async def delayed_send() -> None:
        await asyncio.sleep(0.15)
        channel.send('just in time')

    send_task = asyncio.create_task(delayed_send())
    result = await channel.__anext__()
    assert result == 'just in time'
    await send_task


@pytest.mark.asyncio
async def test_channel_close_future_with_exception() -> None:
    """Tests that exceptions from close_future are propagated to channel.closed."""
    channel: Channel[Any] = Channel()

    async def failing_task() -> str:
        raise ValueError('Task failed!')

    task = asyncio.create_task(failing_task())
    channel.set_close_future(task)

    # Wait for the task to complete
    await asyncio.sleep(0.01)

    # The channel.closed future should have the exception
    with pytest.raises(ValueError, match='Task failed!'):
        await channel.closed


@pytest.mark.asyncio
async def test_channel_close_future_cancelled() -> None:
    """Tests that cancellation of close_future is propagated to channel.closed."""
    channel: Channel[Any] = Channel()

    async def long_running_task() -> str:
        await asyncio.sleep(10)
        return 'done'

    task = asyncio.create_task(long_running_task())
    channel.set_close_future(task)

    # Cancel the task
    task.cancel()

    # Wait for cancellation to propagate
    await asyncio.sleep(0.01)

    # The channel.closed future should be cancelled
    assert channel.closed.cancelled()


@pytest.mark.asyncio
async def test_channel_close_future_success_propagates_result() -> None:
    """Tests that successful close_future result is propagated to channel.closed."""
    channel: Channel[Any] = Channel()

    async def successful_task() -> str:
        return 'success_result'

    task = asyncio.create_task(successful_task())
    channel.set_close_future(task)

    # Wait for the task to complete
    result = await channel.closed

    assert result == 'success_result'
