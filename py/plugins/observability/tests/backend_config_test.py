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

"""Tests for observability backend configuration.

Tests cover all 6 backends (Sentry, Honeycomb, Datadog, Grafana, Axiom, Custom),
verifying endpoint construction, header generation, credential encoding,
environment variable fallbacks, and error handling for missing configuration.
"""

import base64
import os
from unittest import mock

import pytest

from genkit.plugins.observability import Backend, _get_backend_config


# ---------------------------------------------------------------------------
# Sentry Backend
# ---------------------------------------------------------------------------
class TestSentryBackend:
    """Tests for SentryBackend."""

    def test_valid_dsn_extracts_endpoint_and_auth(self) -> None:
        """Valid dsn extracts endpoint and auth."""
        endpoint, headers = _get_backend_config(
            Backend.SENTRY,
            sentry_dsn='https://abc123@o123456.ingest.us.sentry.io/4507654321',
        )
        assert endpoint == 'https://o123456.ingest.us.sentry.io/api/4507654321/otlp/v1/traces'
        assert headers == {'x-sentry-auth': 'sentry sentry_key=abc123'}

    def test_dsn_from_env_var(self) -> None:
        """Dsn from env var."""
        with mock.patch.dict(os.environ, {'SENTRY_DSN': 'https://key@host.sentry.io/123'}):
            endpoint, headers = _get_backend_config(Backend.SENTRY)
            assert endpoint == 'https://host.sentry.io/api/123/otlp/v1/traces'
            assert 'key' in headers['x-sentry-auth']

    def test_missing_dsn_raises(self) -> None:
        """Missing dsn raises."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='Sentry DSN is required'):
                _get_backend_config(Backend.SENTRY)

    def test_invalid_dsn_format_raises(self) -> None:
        """Invalid dsn format raises."""
        with pytest.raises(ValueError, match='Invalid Sentry DSN format'):
            _get_backend_config(Backend.SENTRY, sentry_dsn='http://not-valid')

    def test_string_backend_coerced(self) -> None:
        """Verify string 'sentry' is accepted as backend."""
        endpoint, _ = _get_backend_config(
            'sentry',
            sentry_dsn='https://abc@org.ingest.sentry.io/999',
        )
        assert '/otlp/v1/traces' in endpoint


# ---------------------------------------------------------------------------
# Honeycomb Backend
# ---------------------------------------------------------------------------
class TestHoneycombBackend:
    """Tests for HoneycombBackend."""

    def test_basic_config(self) -> None:
        """Basic config."""
        endpoint, headers = _get_backend_config(
            Backend.HONEYCOMB,
            honeycomb_api_key='hcaik_test123',
        )
        assert endpoint == 'https://api.honeycomb.io/v1/traces'
        assert headers == {'x-honeycomb-team': 'hcaik_test123'}

    def test_custom_endpoint(self) -> None:
        """Custom endpoint."""
        endpoint, _ = _get_backend_config(
            Backend.HONEYCOMB,
            honeycomb_api_key='key',
            honeycomb_api_endpoint='https://api.eu1.honeycomb.io/',
        )
        assert endpoint == 'https://api.eu1.honeycomb.io/v1/traces'

    def test_dataset_header_added_for_classic(self) -> None:
        """Dataset header added for classic."""
        _, headers = _get_backend_config(
            Backend.HONEYCOMB,
            honeycomb_api_key='key',
            honeycomb_dataset='my-dataset',
        )
        assert headers['x-honeycomb-dataset'] == 'my-dataset'

    def test_no_dataset_header_without_dataset(self) -> None:
        """No dataset header without dataset."""
        _, headers = _get_backend_config(
            Backend.HONEYCOMB,
            honeycomb_api_key='key',
        )
        assert 'x-honeycomb-dataset' not in headers

    def test_missing_api_key_raises(self) -> None:
        """Missing api key raises."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='Honeycomb API key is required'):
                _get_backend_config(Backend.HONEYCOMB)

    def test_api_key_from_env_var(self) -> None:
        """Api key from env var."""
        with mock.patch.dict(os.environ, {'HONEYCOMB_API_KEY': 'env_key'}):
            _, headers = _get_backend_config(Backend.HONEYCOMB)
            assert headers['x-honeycomb-team'] == 'env_key'


# ---------------------------------------------------------------------------
# Datadog Backend
# ---------------------------------------------------------------------------
class TestDatadogBackend:
    """Tests for DatadogBackend."""

    def test_default_site(self) -> None:
        """Default site."""
        endpoint, headers = _get_backend_config(
            Backend.DATADOG,
            datadog_api_key='dd_api_key_123',
        )
        assert endpoint == 'https://otlp.datadoghq.com/v1/traces'
        assert headers == {'DD-API-KEY': 'dd_api_key_123'}

    def test_custom_site(self) -> None:
        """Custom site."""
        endpoint, _ = _get_backend_config(
            Backend.DATADOG,
            datadog_api_key='key',
            datadog_site='datadoghq.eu',
        )
        assert endpoint == 'https://otlp.datadoghq.eu/v1/traces'

    def test_missing_api_key_raises(self) -> None:
        """Missing api key raises."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='Datadog API key is required'):
                _get_backend_config(Backend.DATADOG)

    def test_api_key_from_env_var(self) -> None:
        """Api key from env var."""
        with mock.patch.dict(os.environ, {'DD_API_KEY': 'from_env'}):
            _, headers = _get_backend_config(Backend.DATADOG)
            assert headers['DD-API-KEY'] == 'from_env'


# ---------------------------------------------------------------------------
# Grafana Backend
# ---------------------------------------------------------------------------
class TestGrafanaBackend:
    """Tests for GrafanaBackend."""

    def test_basic_config_with_basic_auth(self) -> None:
        """Basic config with basic auth."""
        endpoint, headers = _get_backend_config(
            Backend.GRAFANA,
            grafana_endpoint='https://otlp-gateway.grafana.net/otlp',
            grafana_user_id='12345',
            grafana_api_key='glc_test_key',
        )
        # Endpoint should have /v1/traces appended
        assert endpoint == 'https://otlp-gateway.grafana.net/otlp/v1/traces'

        # Verify Basic auth encoding
        expected_creds = base64.b64encode(b'12345:glc_test_key').decode()
        assert headers['Authorization'] == f'Basic {expected_creds}'

    def test_endpoint_already_has_v1_traces(self) -> None:
        """Endpoint already has v1 traces."""
        endpoint, _ = _get_backend_config(
            Backend.GRAFANA,
            grafana_endpoint='https://example.com/v1/traces',
            grafana_user_id='u',
            grafana_api_key='k',
        )
        # Should NOT double-append
        assert endpoint == 'https://example.com/v1/traces'

    def test_missing_endpoint_raises(self) -> None:
        """Missing endpoint raises."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='Grafana endpoint is required'):
                _get_backend_config(
                    Backend.GRAFANA,
                    grafana_user_id='u',
                    grafana_api_key='k',
                )

    def test_missing_user_id_raises(self) -> None:
        """Missing user id raises."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='Grafana user ID is required'):
                _get_backend_config(
                    Backend.GRAFANA,
                    grafana_endpoint='https://example.com',
                    grafana_api_key='k',
                )

    def test_missing_api_key_raises(self) -> None:
        """Missing api key raises."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='Grafana API key is required'):
                _get_backend_config(
                    Backend.GRAFANA,
                    grafana_endpoint='https://example.com',
                    grafana_user_id='u',
                )


# ---------------------------------------------------------------------------
# Axiom Backend
# ---------------------------------------------------------------------------
class TestAxiomBackend:
    """Tests for AxiomBackend."""

    def test_basic_config(self) -> None:
        """Basic config."""
        endpoint, headers = _get_backend_config(
            Backend.AXIOM,
            axiom_api_token='xaat-test-token',
        )
        assert endpoint == 'https://api.axiom.co/v1/traces'
        assert headers['Authorization'] == 'Bearer xaat-test-token'
        assert headers['X-Axiom-Dataset'] == 'genkit'  # Default dataset

    def test_custom_dataset(self) -> None:
        """Custom dataset."""
        _, headers = _get_backend_config(
            Backend.AXIOM,
            axiom_api_token='token',
            axiom_dataset='my-traces',
        )
        assert headers['X-Axiom-Dataset'] == 'my-traces'

    def test_missing_token_raises(self) -> None:
        """Missing token raises."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='Axiom API token is required'):
                _get_backend_config(Backend.AXIOM)

    def test_token_from_env_var(self) -> None:
        """Token from env var."""
        with mock.patch.dict(os.environ, {'AXIOM_TOKEN': 'env_token'}):
            _, headers = _get_backend_config(Backend.AXIOM)
            assert headers['Authorization'] == 'Bearer env_token'


# ---------------------------------------------------------------------------
# Custom Backend
# ---------------------------------------------------------------------------
class TestCustomBackend:
    """Tests for CustomBackend."""

    def test_basic_config(self) -> None:
        """Basic config."""
        endpoint, headers = _get_backend_config(
            Backend.CUSTOM,
            endpoint='https://my-collector/v1/traces',
            headers={'X-Custom': 'value'},
        )
        assert endpoint == 'https://my-collector/v1/traces'
        assert headers == {'X-Custom': 'value'}

    def test_no_headers_defaults_to_empty(self) -> None:
        """No headers defaults to empty."""
        _, headers = _get_backend_config(
            Backend.CUSTOM,
            endpoint='https://my-collector',
        )
        assert headers == {}

    def test_missing_endpoint_raises(self) -> None:
        """Missing endpoint raises."""
        with pytest.raises(ValueError, match='Custom endpoint is required'):
            _get_backend_config(Backend.CUSTOM)


# ---------------------------------------------------------------------------
# Backend Enum
# ---------------------------------------------------------------------------
class TestBackendEnum:
    """Tests for BackendEnum."""

    def test_all_backends_present(self) -> None:
        """All backends present."""
        expected = {'sentry', 'honeycomb', 'datadog', 'grafana', 'axiom', 'custom'}
        actual = {b.value for b in Backend}
        assert actual == expected

    def test_string_coercion(self) -> None:
        """String coercion."""
        assert Backend('sentry') == Backend.SENTRY
        assert Backend('honeycomb') == Backend.HONEYCOMB

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises."""
        with pytest.raises(ValueError):
            Backend('nonexistent')
