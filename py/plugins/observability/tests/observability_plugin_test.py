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

"""Tests for Observability plugin."""

import os
from unittest.mock import MagicMock, patch

import pytest

from genkit.plugins.observability import Backend, configure_telemetry, package_name


def test_package_name() -> None:
    """Test package_name returns correct value."""
    assert package_name() == 'genkit.plugins.observability'


def test_backend_enum_values() -> None:
    """Test Backend enum has all expected values."""
    assert Backend.SENTRY == 'sentry'
    assert Backend.HONEYCOMB == 'honeycomb'
    assert Backend.DATADOG == 'datadog'
    assert Backend.GRAFANA == 'grafana'
    assert Backend.AXIOM == 'axiom'
    assert Backend.CUSTOM == 'custom'


def test_configure_telemetry_callable() -> None:
    """Test configure_telemetry is callable."""
    assert callable(configure_telemetry)


def test_configure_telemetry_sentry_requires_dsn() -> None:
    """Test Sentry backend requires DSN."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            configure_telemetry(backend='sentry')
        assert 'Sentry DSN' in str(exc_info.value)


def test_configure_telemetry_honeycomb_requires_api_key() -> None:
    """Test Honeycomb backend requires API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            configure_telemetry(backend='honeycomb')
        assert 'Honeycomb API key' in str(exc_info.value)


def test_configure_telemetry_datadog_requires_api_key() -> None:
    """Test Datadog backend requires API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            configure_telemetry(backend='datadog')
        assert 'Datadog API key' in str(exc_info.value)


def test_configure_telemetry_grafana_requires_endpoint() -> None:
    """Test Grafana backend requires endpoint."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            configure_telemetry(backend='grafana')
        assert 'Grafana endpoint' in str(exc_info.value)


def test_configure_telemetry_axiom_requires_token() -> None:
    """Test Axiom backend requires token."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            configure_telemetry(backend='axiom')
        assert 'Axiom API token' in str(exc_info.value)


def test_configure_telemetry_custom_requires_endpoint() -> None:
    """Test Custom backend requires endpoint."""
    with pytest.raises(ValueError) as exc_info:
        configure_telemetry(backend='custom')
    assert 'Custom endpoint' in str(exc_info.value)


@patch('genkit.plugins.observability.add_custom_exporter')
@patch('genkit.plugins.observability.trace.set_tracer_provider')
def test_configure_telemetry_with_sentry_dsn(mock_set_provider: MagicMock, mock_add_exporter: MagicMock) -> None:
    """Test Sentry configuration with valid DSN."""
    configure_telemetry(
        backend='sentry',
        sentry_dsn='https://abc123@o123456.ingest.us.sentry.io/4507654321',
    )
    mock_set_provider.assert_called_once()
    mock_add_exporter.assert_called_once()


@patch('genkit.plugins.observability.add_custom_exporter')
@patch('genkit.plugins.observability.trace.set_tracer_provider')
def test_configure_telemetry_with_honeycomb(mock_set_provider: MagicMock, mock_add_exporter: MagicMock) -> None:
    """Test Honeycomb configuration with valid API key."""
    configure_telemetry(
        backend='honeycomb',
        honeycomb_api_key='test-api-key',
    )
    mock_set_provider.assert_called_once()
    mock_add_exporter.assert_called_once()


@patch('genkit.plugins.observability.add_custom_exporter')
@patch('genkit.plugins.observability.trace.set_tracer_provider')
def test_configure_telemetry_disable_traces(mock_set_provider: MagicMock, mock_add_exporter: MagicMock) -> None:
    """Test disabling traces."""
    configure_telemetry(
        backend='custom',
        endpoint='https://example.com/v1/traces',
        disable_traces=True,
    )
    mock_set_provider.assert_not_called()
    mock_add_exporter.assert_not_called()
