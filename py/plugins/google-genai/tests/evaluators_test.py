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

"""Tests for Vertex AI Evaluators."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genkit.plugins.google_genai.evaluators import (
    VertexAIEvaluationMetricType,
    create_vertex_evaluators,
)
from genkit.plugins.google_genai.evaluators.evaluation import (
    EvaluatorFactory,
    VertexAIEvaluationMetricConfig,
    _is_config,
    _stringify,
)


def test_vertex_ai_evaluation_metric_type_values() -> None:
    """Test that VertexAIEvaluationMetricType has expected values."""
    assert VertexAIEvaluationMetricType.BLEU == 'BLEU'
    assert VertexAIEvaluationMetricType.ROUGE == 'ROUGE'
    assert VertexAIEvaluationMetricType.FLUENCY == 'FLUENCY'
    assert VertexAIEvaluationMetricType.SAFETY == 'SAFETY'
    assert VertexAIEvaluationMetricType.GROUNDEDNESS == 'GROUNDEDNESS'
    assert VertexAIEvaluationMetricType.SUMMARIZATION_QUALITY == 'SUMMARIZATION_QUALITY'
    assert VertexAIEvaluationMetricType.SUMMARIZATION_HELPFULNESS == 'SUMMARIZATION_HELPFULNESS'
    assert VertexAIEvaluationMetricType.SUMMARIZATION_VERBOSITY == 'SUMMARIZATION_VERBOSITY'


def test_vertex_ai_evaluation_metric_type_is_str_enum() -> None:
    """Test that metric types can be used as strings."""
    metric = VertexAIEvaluationMetricType.FLUENCY
    assert isinstance(metric, str)
    assert metric == 'FLUENCY'


def test_vertex_ai_evaluation_metric_config_basic() -> None:
    """Test VertexAIEvaluationMetricConfig model."""
    config = VertexAIEvaluationMetricConfig(
        type=VertexAIEvaluationMetricType.BLEU,
        metric_spec={'use_sentence_level': True},
    )
    assert config.type == VertexAIEvaluationMetricType.BLEU
    assert config.metric_spec == {'use_sentence_level': True}


def test_vertex_ai_evaluation_metric_config_defaults() -> None:
    """Test VertexAIEvaluationMetricConfig default values."""
    config = VertexAIEvaluationMetricConfig(type=VertexAIEvaluationMetricType.SAFETY)
    assert config.type == VertexAIEvaluationMetricType.SAFETY
    assert config.metric_spec is None


def test_stringify_string_input() -> None:
    """Test _stringify with string input returns unchanged."""
    result = _stringify('hello world')
    assert result == 'hello world'


def test_stringify_dict_input() -> None:
    """Test _stringify with dict input returns JSON."""
    result = _stringify({'key': 'value'})
    assert result == '{"key": "value"}'


def test_stringify_list_input() -> None:
    """Test _stringify with list input returns JSON."""
    result = _stringify(['a', 'b', 'c'])
    assert result == '["a", "b", "c"]'


def test_stringify_number_input() -> None:
    """Test _stringify with number input returns JSON."""
    result = _stringify(42)
    assert result == '42'


def test_is_config_with_metric_type() -> None:
    """Test _is_config returns False for metric type."""
    metric = VertexAIEvaluationMetricType.FLUENCY
    assert _is_config(metric) is False


def test_is_config_with_metric_config() -> None:
    """Test _is_config returns True for metric config."""
    config = VertexAIEvaluationMetricConfig(type=VertexAIEvaluationMetricType.FLUENCY)
    assert _is_config(config) is True


def test_evaluator_factory_initialization() -> None:
    """Test EvaluatorFactory can be initialized."""
    factory = EvaluatorFactory(
        project_id='test-project',
        location='us-central1',
    )
    assert factory.project_id == 'test-project'
    assert factory.location == 'us-central1'


@pytest.mark.asyncio
async def test_evaluator_factory_evaluate_instances_structure() -> None:
    """Test that evaluate_instances makes correct API call structure."""
    factory = EvaluatorFactory(
        project_id='test-project',
        location='us-central1',
    )

    mock_credentials = MagicMock()
    mock_credentials.token = 'mock-token'
    mock_credentials.expired = False

    mock_response_data = {
        'fluencyResult': {
            'score': 4.5,
            'explanation': 'Very fluent text',
        }
    }

    with patch('genkit.plugins.google_genai.evaluators.evaluation.google_auth_default') as mock_auth:
        mock_auth.return_value = (mock_credentials, 'test-project')

        # Mock get_cached_client to return a mock client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch('genkit.plugins.google_genai.evaluators.evaluation.get_cached_client', return_value=mock_client):
            result = await factory.evaluate_instances({'fluencyInput': {'prediction': 'Test'}})

            assert result == mock_response_data
            mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_evaluator_factory_evaluate_instances_error_handling() -> None:
    """Test that evaluate_instances raises GenkitError on API failure."""
    from genkit.core.error import GenkitError

    factory = EvaluatorFactory(
        project_id='test-project',
        location='us-central1',
    )

    mock_credentials = MagicMock()
    mock_credentials.token = 'mock-token'
    mock_credentials.expired = False

    with patch('genkit.plugins.google_genai.evaluators.evaluation.google_auth_default') as mock_auth:
        mock_auth.return_value = (mock_credentials, 'test-project')

        # Mock get_cached_client to return a mock client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch('genkit.plugins.google_genai.evaluators.evaluation.get_cached_client', return_value=mock_client):
            with pytest.raises(GenkitError) as exc_info:
                await factory.evaluate_instances({'input': 'test'})

            assert exc_info.value.status == 'INTERNAL'


def test_create_vertex_evaluators_with_metric_types() -> None:
    """Test create_vertex_evaluators with simple metric types."""
    mock_registry = MagicMock()
    mock_registry.define_evaluator = MagicMock()

    metrics = [
        VertexAIEvaluationMetricType.FLUENCY,
        VertexAIEvaluationMetricType.SAFETY,
    ]

    create_vertex_evaluators(
        registry=mock_registry,
        metrics=metrics,
        project_id='test-project',
        location='us-central1',
    )

    assert mock_registry.define_evaluator.call_count == 2


def test_create_vertex_evaluators_with_metric_configs() -> None:
    """Test create_vertex_evaluators with metric configs."""
    mock_registry = MagicMock()
    mock_registry.define_evaluator = MagicMock()

    metrics = [
        VertexAIEvaluationMetricConfig(
            type=VertexAIEvaluationMetricType.BLEU,
            metric_spec={'use_sentence_level': True},
        ),
    ]

    create_vertex_evaluators(
        registry=mock_registry,
        metrics=metrics,
        project_id='test-project',
        location='us-central1',
    )

    mock_registry.define_evaluator.assert_called_once()


def test_create_vertex_evaluators_names_format() -> None:
    """Test that evaluator names follow vertexai/{metric} format."""
    mock_registry = MagicMock()
    evaluator_names: list[str] = []

    def capture_name(*args: object, **kwargs: object) -> None:
        if 'name' in kwargs:
            name = kwargs['name']
            if isinstance(name, str):
                evaluator_names.append(name)

    mock_registry.define_evaluator = capture_name

    metrics = [
        VertexAIEvaluationMetricType.FLUENCY,
        VertexAIEvaluationMetricType.GROUNDEDNESS,
    ]

    create_vertex_evaluators(
        registry=mock_registry,
        metrics=metrics,
        project_id='test-project',
        location='us-central1',
    )

    assert 'vertexai/fluency' in evaluator_names
    assert 'vertexai/groundedness' in evaluator_names


def test_create_vertex_evaluators_empty_metrics() -> None:
    """Test create_vertex_evaluators with empty metrics list."""
    mock_registry = MagicMock()
    mock_registry.define_evaluator = MagicMock()

    create_vertex_evaluators(
        registry=mock_registry,
        metrics=[],
        project_id='test-project',
        location='us-central1',
    )

    mock_registry.define_evaluator.assert_not_called()


def test_all_metric_types_supported() -> None:
    """Test that all metric types are supported by create_vertex_evaluators."""
    mock_registry = MagicMock()
    mock_registry.define_evaluator = MagicMock()

    all_metrics = list(VertexAIEvaluationMetricType)

    create_vertex_evaluators(
        registry=mock_registry,
        metrics=all_metrics,
        project_id='test-project',
        location='us-central1',
    )

    assert mock_registry.define_evaluator.call_count == len(all_metrics)
