# Copyright 2026 Google LLC
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

"""Region resolution and event-pump tests; neither needs credentials."""

import asyncio
import threading
from typing import Any

import boto3.session
import pytest
from botocore.exceptions import EventStreamError
from genkit_amazon_bedrock.transport import BedrockTransport

from genkit.plugin_api import GenkitError

REGION_ENV_VARS = ('AWS_REGION', 'AWS_DEFAULT_REGION', 'AWS_PROFILE', 'AWS_CONFIG_FILE')


@pytest.fixture(autouse=True)
def _isolate_aws_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Drops ambient AWS config so the tests see only what they set."""
    for name in REGION_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Points botocore at an empty config file rather than the developer's own.
    empty_config = tmp_path / 'aws-config'
    empty_config.write_text('')
    monkeypatch.setenv('AWS_CONFIG_FILE', str(empty_config))


def make_transport(**kwargs) -> BedrockTransport:
    defaults = {
        'max_retries': 3,
        'read_timeout': 3600.0,
        'connect_timeout': 60.0,
        'max_pool_connections': 50,
    }
    return BedrockTransport(**{**defaults, **kwargs})


def test_explicit_region_wins() -> None:
    client = make_transport(region='eu-west-1').client()
    assert client.meta.region_name == 'eu-west-1'


def test_aws_region_env_var_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    # botocore below 1.41 reads only AWS_DEFAULT_REGION, so the plugin resolves
    # AWS_REGION itself; without that this raises FAILED_PRECONDITION.
    monkeypatch.setenv('AWS_REGION', 'us-east-2')
    assert make_transport().client().meta.region_name == 'us-east-2'


def test_aws_default_region_env_var_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'ap-south-1')
    assert make_transport().client().meta.region_name == 'ap-south-1'


def test_aws_region_beats_aws_default_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AWS_REGION', 'us-east-2')
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'ap-south-1')
    assert make_transport().client().meta.region_name == 'us-east-2'


def test_supplied_session_region_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A caller who configured a session chose that region deliberately.
    monkeypatch.setenv('AWS_REGION', 'us-east-2')
    session = boto3.session.Session(region_name='sa-east-1')
    assert make_transport(session=session).client().meta.region_name == 'sa-east-1'


def test_missing_region_fails_loudly() -> None:
    with pytest.raises(GenkitError, match='no AWS region resolved') as excinfo:
        make_transport().client()
    assert excinfo.value.status == 'FAILED_PRECONDITION'


def test_client_is_built_once() -> None:
    transport = make_transport(region='eu-west-1')
    assert transport.client() is transport.client()


def test_botocore_config_carries_the_timeouts() -> None:
    config = make_transport(region='eu-west-1', read_timeout=1800.0).client().meta.config
    assert config.read_timeout == 1800.0
    assert config.connect_timeout == 60.0
    assert config.max_pool_connections == 50
    # botocore normalizes max_attempts to total attempts: 3 retries plus the first call.
    assert config.retries['total_max_attempts'] == 4
    assert config.retries['mode'] == 'standard'


class FakeEventStream:
    """Stands in for botocore's blocking EventStream.

    ``__iter__`` is a generator, as botocore's is, and each ``next()`` blocks
    on an event so the test fails if the pump stops crossing the thread bridge.
    """

    def __init__(self, events: list[dict[str, Any]], error: Exception | None = None) -> None:
        self._events = events
        self._error = error
        self.close_calls = 0
        self.iter_threads: list[int] = []
        self._resume = threading.Event()
        self._resume.set()

    def __iter__(self):  # noqa: ANN204
        for event in self._events:
            # Bounded so a transport that stops closing the stream fails the
            # test instead of parking this worker for the whole run.
            self._resume.wait(timeout=5)
            self.iter_threads.append(threading.get_ident())
            yield event
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.close_calls += 1
        # Releases a worker parked in next(), which is why the real close is
        # called inline rather than through the bridge.
        self._resume.set()

    def block(self) -> None:
        self._resume.clear()


class FakeClient:
    """Minimal bedrock-runtime stand-in returning a FakeEventStream.

    ``gate``, when given, holds the call open so a test can cancel while it is
    still in flight on its worker thread.
    """

    def __init__(self, stream: FakeEventStream | None, gate: threading.Event | None = None) -> None:
        self._stream = stream
        self._gate = gate
        self.calls: list[dict[str, Any]] = []

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        if self._gate is not None:
            self._gate.wait(timeout=5)
        self.calls.append(kwargs)
        return {'stream': self._stream} if self._stream is not None else {}


def streaming_transport(
    events: list[dict[str, Any]] | None = None,
    error: Exception | None = None,
    with_stream: bool = True,
    gate: threading.Event | None = None,
) -> tuple[BedrockTransport, FakeClient, FakeEventStream | None]:
    fake_stream = FakeEventStream(events or [], error) if with_stream else None
    client = FakeClient(fake_stream, gate)
    transport = make_transport(region='eu-west-1')
    transport._client = client  # noqa: SLF001
    return transport, client, fake_stream


TEXT_EVENT = {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'hi'}}}
STOP_EVENT = {'messageStop': {'stopReason': 'end_turn'}}


@pytest.mark.asyncio
async def test_converse_stream_yields_events_in_order_and_closes() -> None:
    transport, client, stream = streaming_transport([TEXT_EVENT, STOP_EVENT])

    received = [event async for event in transport.converse_stream(modelId='m', messages=[])]

    assert received == [TEXT_EVENT, STOP_EVENT]
    assert client.calls == [{'modelId': 'm', 'messages': []}]
    assert stream is not None and stream.close_calls == 1


@pytest.mark.asyncio
async def test_converse_stream_runs_the_iterator_off_the_event_loop() -> None:
    transport, _client, stream = streaming_transport([TEXT_EVENT, STOP_EVENT])

    async for _event in transport.converse_stream(modelId='m'):
        pass

    # One missed bridge freezes the loop for the minutes a generation can take.
    assert stream is not None
    assert stream.iter_threads and threading.get_ident() not in stream.iter_threads


@pytest.mark.asyncio
async def test_converse_stream_propagates_mid_stream_errors_and_closes() -> None:
    error = EventStreamError({'Error': {'Code': 'modelStreamErrorException', 'Message': 'boom'}}, 'ConverseStream')
    transport, _client, stream = streaming_transport([TEXT_EVENT], error=error)
    received = []

    with pytest.raises(EventStreamError) as excinfo:
        async for event in transport.converse_stream(modelId='m'):
            received.append(event)

    assert excinfo.value is error
    # Events delivered before the failure stand.
    assert received == [TEXT_EVENT]
    assert stream is not None and stream.close_calls == 1


@pytest.mark.asyncio
async def test_converse_stream_closes_when_the_consumer_stops_early() -> None:
    transport, _client, stream = streaming_transport([TEXT_EVENT, STOP_EVENT])

    generator = transport.converse_stream(modelId='m')
    async for _event in generator:
        break
    await generator.aclose()

    assert stream is not None and stream.close_calls == 1


@pytest.mark.asyncio
async def test_converse_stream_closes_when_cancelled_mid_pump() -> None:
    transport, _client, stream = streaming_transport([TEXT_EVENT, STOP_EVENT])
    assert stream is not None
    first_event = asyncio.Event()

    async def consume() -> None:
        async for _event in transport.converse_stream(modelId='m'):
            # Parks the next worker in next() so the cancel lands mid-pump.
            stream.block()
            first_event.set()

    task = asyncio.ensure_future(consume())
    await asyncio.wait_for(first_event.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_converse_stream_closes_when_cancelled_during_the_initial_call() -> None:
    # A worker thread cannot be interrupted, so the call still opens a stream
    # after the consumer is gone; it has to be closed on their behalf.
    released = threading.Event()
    transport, _client, stream = streaming_transport([TEXT_EVENT], gate=released)
    assert stream is not None

    async def consume() -> None:
        async for _event in transport.converse_stream(modelId='m'):
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.close_calls == 0, 'the call has not returned yet'

    released.set()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if stream.close_calls:
            break

    assert stream.close_calls == 1
    assert stream.iter_threads == [], 'no events should be consumed after cancellation'


@pytest.mark.asyncio
async def test_converse_stream_without_a_stream_member_fails_loudly() -> None:
    transport, _client, _stream = streaming_transport(with_stream=False)

    with pytest.raises(GenkitError, match='no stream') as excinfo:
        async for _event in transport.converse_stream(modelId='m'):
            pass

    assert excinfo.value.status == 'INTERNAL'
