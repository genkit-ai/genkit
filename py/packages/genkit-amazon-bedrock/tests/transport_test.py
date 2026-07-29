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

"""Region resolution tests. Client construction needs no credentials."""

import boto3.session
import pytest
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
