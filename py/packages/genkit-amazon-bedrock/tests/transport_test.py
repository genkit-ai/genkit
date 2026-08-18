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

"""Transport tests: region resolution, client config, and the boto3 bridge.

Client construction needs no credentials, and the bridge tests stand a fake
client in for boto3 so the real ``converse`` and ``converse_stream`` paths run
without AWS.
"""

import asyncio
import threading
from typing import Any

import boto3.session
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError, EventStreamError
from genkit_amazon_bedrock.transport import BedrockTransport

from genkit.plugin_api import GenkitError

AWS_ENV_VARS = (
    'AWS_REGION',
    'AWS_DEFAULT_REGION',
    'AWS_PROFILE',
    'AWS_CONFIG_FILE',
    'AWS_MAX_ATTEMPTS',
    'AWS_RETRY_MODE',
)


@pytest.fixture(autouse=True)
def _isolate_aws_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Drops ambient AWS config so the tests see only what they set."""
    for name in AWS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Points botocore at an empty config file rather than the developer's own.
    empty_config = tmp_path / 'aws-config'
    empty_config.write_text('')
    monkeypatch.setenv('AWS_CONFIG_FILE', str(empty_config))


def make_transport(**kwargs) -> BedrockTransport:
    return BedrockTransport(**kwargs)


class FakeClient:
    """Stands in for the boto3 bedrock-runtime client."""

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
        before_return: Any = None,  # noqa: ANN401
    ) -> None:
        self.response = response if response is not None else {'stopReason': 'end_turn'}
        self.error = error
        self.before_return = before_return
        self.calls: list[dict[str, Any]] = []
        self.thread_idents: list[int] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        self.thread_idents.append(threading.get_ident())
        if self.before_return is not None:
            self.before_return()
        if self.error is not None:
            raise self.error
        return self.response


def stub_transport(monkeypatch: pytest.MonkeyPatch, client: FakeClient, **kwargs) -> BedrockTransport:
    """Builds a transport whose client() hands back ``client``."""
    transport = make_transport(region='eu-west-1', **kwargs)
    monkeypatch.setattr(transport, '_build_client', lambda: client)
    return transport


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


# --- Deferring to the caller's AWS configuration ----------------------------


def test_retry_env_vars_are_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sending our own retries block would win outright and silently drop these.
    monkeypatch.setenv('AWS_MAX_ATTEMPTS', '10')
    monkeypatch.setenv('AWS_RETRY_MODE', 'adaptive')
    config = make_transport(region='eu-west-1').client().meta.config
    assert config.retries == {'total_max_attempts': 10, 'mode': 'adaptive'}


def test_max_attempts_env_alone_keeps_the_default_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deferring on the whole block would hand back botocore's legacy mode, a
    # downgrade nobody asked for; only the key the env sets is left alone.
    monkeypatch.setenv('AWS_MAX_ATTEMPTS', '10')
    config = make_transport(region='eu-west-1').client().meta.config
    assert config.retries == {'total_max_attempts': 10, 'mode': 'standard'}


def test_retry_mode_env_alone_keeps_the_default_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AWS_RETRY_MODE', 'adaptive')
    config = make_transport(region='eu-west-1').client().meta.config
    assert config.retries == {'total_max_attempts': 4, 'mode': 'adaptive'}


def test_config_file_retry_settings_are_honored(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config_file = tmp_path / 'aws-config-retries'
    config_file.write_text('[default]\nmax_attempts = 7\n')
    monkeypatch.setenv('AWS_CONFIG_FILE', str(config_file))
    config = make_transport(region='eu-west-1').client().meta.config
    assert config.retries == {'total_max_attempts': 7, 'mode': 'standard'}


def test_explicit_max_retries_beats_the_env_without_forcing_the_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tuning the retry count must not drag the caller off adaptive, which is
    # the mode that matters under Bedrock throttling.
    monkeypatch.setenv('AWS_MAX_ATTEMPTS', '10')
    monkeypatch.setenv('AWS_RETRY_MODE', 'adaptive')
    config = make_transport(region='eu-west-1', max_retries=1).client().meta.config
    assert config.retries == {'total_max_attempts': 2, 'mode': 'adaptive'}


def test_session_retry_block_survives_and_explicit_attempts_still_win() -> None:
    session = boto3.session.Session(region_name='eu-west-1')
    session._session.set_default_client_config(Config(retries={'mode': 'adaptive', 'max_attempts': 9}))
    assert make_transport(session=session).client().meta.config.retries['mode'] == 'adaptive'
    config = make_transport(session=session, max_retries=1).client().meta.config
    assert config.retries == {'total_max_attempts': 2, 'mode': 'adaptive'}


def test_session_client_config_is_honored() -> None:
    session = boto3.session.Session(region_name='eu-west-1')
    session._session.set_default_client_config(Config(read_timeout=11.0, max_pool_connections=7))
    config = make_transport(session=session).client().meta.config
    assert config.read_timeout == 11.0
    assert config.max_pool_connections == 7
    # Keys the caller left alone still get the package default.
    assert config.connect_timeout == 60.0


def test_explicit_timeout_beats_the_session_client_config() -> None:
    session = boto3.session.Session(region_name='eu-west-1')
    session._session.set_default_client_config(Config(read_timeout=11.0))
    config = make_transport(session=session, read_timeout=22.0).client().meta.config
    assert config.read_timeout == 22.0


# --- The boto3 bridge -------------------------------------------------------


@pytest.mark.asyncio
async def test_converse_forwards_kwargs_and_returns_the_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(response={'output': {'message': {'role': 'assistant', 'content': [{'text': 'hi'}]}}})
    transport = stub_transport(monkeypatch, client)

    response = await transport.converse(modelId='amazon.nova-lite-v1:0', messages=[{'role': 'user'}])

    assert response == client.response
    assert client.calls == [{'modelId': 'amazon.nova-lite-v1:0', 'messages': [{'role': 'user'}]}]


@pytest.mark.asyncio
async def test_converse_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    transport = stub_transport(monkeypatch, client)

    await transport.converse(modelId='amazon.nova-lite-v1:0')

    assert client.thread_idents[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_converse_reuses_the_one_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    transport = stub_transport(monkeypatch, client)
    builds = 0

    def build() -> FakeClient:
        nonlocal builds
        builds += 1
        return client

    monkeypatch.setattr(transport, '_build_client', build)

    await asyncio.gather(*(transport.converse(modelId='amazon.nova-lite-v1:0') for _ in range(4)))

    assert builds == 1
    assert len(client.calls) == 4


@pytest.mark.asyncio
async def test_converse_propagates_boto3_errors_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mapping AWS failures to Genkit statuses belongs to models.py, not here.
    error = ClientError({'Error': {'Code': 'ValidationException', 'Message': 'nope'}}, 'Converse')
    transport = stub_transport(monkeypatch, FakeClient(error=error))

    with pytest.raises(ClientError) as excinfo:
        await transport.converse(modelId='amazon.nova-lite-v1:0')

    assert excinfo.value is error


@pytest.mark.asyncio
async def test_converse_gives_up_at_the_total_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    transport = stub_transport(monkeypatch, FakeClient(before_return=lambda: release.wait(30)), total_timeout=0.05)

    try:
        with pytest.raises(GenkitError, match='total timeout') as excinfo:
            await transport.converse(modelId='amazon.nova-lite-v1:0')
        assert excinfo.value.status == 'DEADLINE_EXCEEDED'
    finally:
        # The worker thread outlives the deadline; let it finish before teardown.
        release.set()


@pytest.mark.asyncio
async def test_no_total_timeout_waits_for_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # Blocks on the worker thread, long enough that a deadline would have bitten.
    client = FakeClient(before_return=lambda: threading.Event().wait(0.05))
    transport = stub_transport(monkeypatch, client, total_timeout=None)

    assert await transport.converse(modelId='amazon.nova-lite-v1:0') == client.response


@pytest.mark.asyncio
async def test_ensure_client_builds_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    idents: list[int] = []
    transport = make_transport(region='eu-west-1')
    monkeypatch.setattr(transport, '_build_client', lambda: idents.append(threading.get_ident()) or FakeClient())

    await transport.ensure_client()

    assert idents and idents[0] != threading.get_ident()


# --- The ConverseStream event pump ------------------------------------------


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


class FakeStreamClient:
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
    **kwargs,
) -> tuple[BedrockTransport, FakeStreamClient, FakeEventStream | None]:
    fake_stream = FakeEventStream(events or [], error) if with_stream else None
    client = FakeStreamClient(fake_stream, gate)
    transport = make_transport(region='eu-west-1', **kwargs)
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
async def test_converse_stream_gives_up_at_the_total_timeout_before_any_event() -> None:
    # The deadline covers the initial call, and the stream that call goes on
    # to open after the caller is gone still gets closed on their behalf.
    released = threading.Event()
    transport, _client, stream = streaming_transport([TEXT_EVENT], gate=released, total_timeout=0.05)
    assert stream is not None

    with pytest.raises(GenkitError, match='total timeout') as excinfo:
        async for _event in transport.converse_stream(modelId='m'):
            pass
    assert excinfo.value.status == 'DEADLINE_EXCEEDED'
    assert stream.close_calls == 0, 'the call has not returned yet'

    released.set()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if stream.close_calls:
            break

    assert stream.close_calls == 1
    assert stream.iter_threads == [], 'no events should be consumed after the deadline'


@pytest.mark.asyncio
async def test_converse_stream_gives_up_at_the_total_timeout_mid_stream() -> None:
    # One deadline across the whole stream, not a per-event allowance: the
    # budget keeps counting down between events, so a stalled stream ends.
    transport, _client, stream = streaming_transport([TEXT_EVENT, STOP_EVENT], total_timeout=0.1)
    assert stream is not None
    received = []

    with pytest.raises(GenkitError, match='total timeout') as excinfo:
        async for event in transport.converse_stream(modelId='m'):
            received.append(event)
            # Parks the next worker in next() until close() releases it.
            stream.block()

    assert excinfo.value.status == 'DEADLINE_EXCEEDED'
    assert received == [TEXT_EVENT]
    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_converse_stream_without_a_total_timeout_streams_freely() -> None:
    transport, _client, stream = streaming_transport([TEXT_EVENT, STOP_EVENT], total_timeout=None)

    received = [event async for event in transport.converse_stream(modelId='m')]

    assert received == [TEXT_EVENT, STOP_EVENT]
    assert stream is not None and stream.close_calls == 1


@pytest.mark.asyncio
async def test_converse_stream_without_a_stream_member_fails_loudly() -> None:
    transport, _client, _stream = streaming_transport(with_stream=False)

    with pytest.raises(GenkitError, match='no stream') as excinfo:
        async for _event in transport.converse_stream(modelId='m'):
            pass

    assert excinfo.value.status == 'INTERNAL'


class FakeStreamingBody:
    """Stands in for botocore's StreamingBody; ``read`` is a blocking call."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.read_threads: list[int] = []

    def read(self) -> bytes:
        self.read_threads.append(threading.get_ident())
        return self._payload


class FakeInvokeClient:
    """Minimal bedrock-runtime stand-in for InvokeModel."""

    def __init__(self, body: FakeStreamingBody | None) -> None:
        self.body = body
        self.calls: list[dict[str, Any]] = []

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {'body': self.body} if self.body is not None else {}


def invoking_transport(payload: bytes | None) -> tuple[BedrockTransport, FakeInvokeClient]:
    client = FakeInvokeClient(FakeStreamingBody(payload) if payload is not None else None)
    transport = make_transport(region='eu-west-1')
    transport._client = client  # noqa: SLF001
    return transport, client


@pytest.mark.asyncio
async def test_invoke_model_parses_the_response_body() -> None:
    transport, client = invoking_transport(b'{"embedding": [1.0, 2.0]}')

    result = await transport.invoke_model(modelId='m', body='{}')

    assert result == {'embedding': [1.0, 2.0]}
    assert client.calls == [{'modelId': 'm', 'body': '{}'}]


@pytest.mark.asyncio
async def test_invoke_model_reads_the_body_off_the_event_loop() -> None:
    transport, client = invoking_transport(b'{}')

    await transport.invoke_model(modelId='m')

    # StreamingBody.read() is a blocking socket read; on the loop it stalls
    # every other in-flight call.
    assert client.body is not None
    assert client.body.read_threads and threading.get_ident() not in client.body.read_threads


@pytest.mark.asyncio
async def test_invoke_model_rejects_a_non_json_body() -> None:
    transport, _client = invoking_transport(b'<html>gateway timeout</html>')

    with pytest.raises(GenkitError, match='not JSON') as excinfo:
        await transport.invoke_model(modelId='m')

    assert excinfo.value.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_invoke_model_rejects_a_json_body_that_is_not_an_object() -> None:
    # Callers read the result with .get(); a list would raise AttributeError there.
    transport, _client = invoking_transport(b'[1.0, 2.0]')

    with pytest.raises(GenkitError, match='not a JSON object') as excinfo:
        await transport.invoke_model(modelId='m')

    assert excinfo.value.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_invoke_model_without_a_body_member_fails_loudly() -> None:
    transport, _client = invoking_transport(None)

    with pytest.raises(GenkitError, match='no body') as excinfo:
        await transport.invoke_model(modelId='m')

    assert excinfo.value.status == 'INTERNAL'
