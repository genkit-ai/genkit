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
client in for boto3 so the real ``converse`` path runs without AWS.
"""

import asyncio
import threading
from typing import Any

import boto3.session
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
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
