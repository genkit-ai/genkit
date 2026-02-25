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

"""Tests for Firebase plugin."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from genkit.plugins.firebase import (
    FirebaseTelemetryConfig,
    add_firebase_telemetry,
    define_firestore_vector_store,
    package_name,
)


def test_package_name() -> None:
    """Test package_name returns correct value."""
    assert package_name() == 'genkit.plugins.firebase'


@patch('genkit.plugins.firebase.telemetry.GcpTelemetry')
def test_add_firebase_telemetry_calls_gcp_telemetry(mock_gcp_telemetry_cls: MagicMock) -> None:
    """Test add_firebase_telemetry delegates to GCP telemetry."""
    mock_manager = MagicMock()
    mock_gcp_telemetry_cls.return_value = mock_manager

    add_firebase_telemetry()

    mock_gcp_telemetry_cls.assert_called_once()
    mock_manager.initialize.assert_called_once()


@patch('genkit.plugins.firebase.telemetry.GcpTelemetry')
def test_add_firebase_telemetry_with_config(mock_gcp_telemetry_cls: MagicMock) -> None:
    """Test add_firebase_telemetry accepts config object."""
    mock_manager = MagicMock()
    mock_gcp_telemetry_cls.return_value = mock_manager

    config = FirebaseTelemetryConfig(
        project_id='test-project',
        log_input_and_output=True,
        force_dev_export=True,
    )
    add_firebase_telemetry(config)

    mock_gcp_telemetry_cls.assert_called_once_with(
        project_id='test-project',
        credentials=None,
        sampler=None,
        log_input_and_output=True,
        force_dev_export=True,
        disable_metrics=False,
        disable_traces=False,
        metric_export_interval_ms=None,
        metric_export_timeout_ms=None,
    )
    mock_manager.initialize.assert_called_once()


def test_define_firestore_vector_store_exported() -> None:
    """Test define_firestore_vector_store is exported."""
    # Just verify the function is importable and callable
    assert callable(define_firestore_vector_store)


def test_add_firebase_telemetry_raises_on_missing_deps() -> None:
    """Test that an informative ImportError is raised if telemetry deps are missing."""
    # Temporarily remove the module from sys.modules to simulate it not being installed.
    with patch.dict(sys.modules, {'genkit.plugins.firebase.telemetry': None}):
        with pytest.raises(ImportError, match='Firebase telemetry requires the Google Cloud telemetry exporter'):
            add_firebase_telemetry()


def test_firebase_telemetry_config_validation() -> None:
    """Test Pydantic validation on FirebaseTelemetryConfig."""
    # Valid config
    config = FirebaseTelemetryConfig(project_id='test-project')
    assert config.project_id == 'test-project'
    assert config.log_input_and_output is False

    # Invalid metric interval (< 1000ms) should raise ValidationError
    with pytest.raises(ValueError):
        FirebaseTelemetryConfig(metric_export_interval_ms=500)
