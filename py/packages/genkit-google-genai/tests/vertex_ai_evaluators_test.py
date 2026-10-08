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

import httpx
import pytest
from genkit_google_genai._evaluators import (
    VertexAIEvaluationMetricType,
    create_vertex_evaluators,
)
from genkit_google_genai._evaluators._evaluation import (
    EvaluatorFactory,
    _stringify,
)
from google.auth.exceptions import DefaultCredentialsError, RefreshError

from genkit import BaseDataPoint, Genkit, GenkitError


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

    with patch('genkit_google_genai._evaluators._evaluation.google_auth_default') as mock_auth:
        mock_auth.return_value = (mock_credentials, 'test-project')

        # Mock get_cached_client to return a mock client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch('genkit_google_genai._evaluators._evaluation.get_cached_client', return_value=mock_client):
            result = await factory.evaluate_instances({'fluencyInput': {'prediction': 'Test'}})

            assert result == mock_response_data
            mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_evaluator_factory_evaluate_instances_error_handling() -> None:
    """Test that evaluate_instances raises GenkitError on API failure."""
    factory = EvaluatorFactory(
        project_id='test-project',
        location='us-central1',
    )

    mock_credentials = MagicMock()
    mock_credentials.token = 'mock-token'
    mock_credentials.expired = False

    with patch('genkit_google_genai._evaluators._evaluation.google_auth_default') as mock_auth:
        mock_auth.return_value = (mock_credentials, 'test-project')

        # Mock get_cached_client to return a mock client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch('genkit_google_genai._evaluators._evaluation.get_cached_client', return_value=mock_client):
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


@pytest.mark.asyncio
async def test_evaluator_request_sends_empty_metric_spec() -> None:
    """Fluency evaluator sends fluencyInput with an empty metricSpec to Vertex."""
    mock_registry = MagicMock()

    create_vertex_evaluators(
        registry=mock_registry,
        metrics=[VertexAIEvaluationMetricType.FLUENCY],
        project_id='test-project',
        location='us-central1',
    )
    evaluator_fn = mock_registry.define_evaluator.call_args.kwargs['fn']

    with patch.object(
        EvaluatorFactory,
        'evaluate_instances',
        AsyncMock(return_value={'fluencyResult': {'score': 4.0}}),
    ) as mock_evaluate:
        await evaluator_fn(BaseDataPoint(output='The soup is ready.'))

    mock_evaluate.assert_awaited_once_with({
        'fluencyInput': {
            'metricSpec': {},
            'instance': {'prediction': 'The soup is ready.'},
        }
    })


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


@pytest.mark.asyncio
async def test_vertexai_evaluator_row_evaluation_is_a_list() -> None:
    """ai.evaluate with vertexai/fluency returns a list of rows read as results[0].evaluation[0].score."""
    ai = Genkit()
    create_vertex_evaluators(
        ai, [VertexAIEvaluationMetricType.FLUENCY], project_id='test-project', location='us-central1'
    )

    with patch.object(
        EvaluatorFactory,
        'evaluate_instances',
        AsyncMock(return_value={'fluencyResult': {'score': 4.5, 'explanation': 'Very fluent text'}}),
    ):
        results = await ai.evaluate(
            evaluator='vertexai/fluency',
            dataset=[BaseDataPoint(input='Write about AI.', output='AI helps.', test_case_id='case1')],
        )

    assert type(results) is list
    assert [row.test_case_id for row in results] == ['case1']
    assert [score.score for score in results[0].evaluation] == [4.5]


def _factory() -> EvaluatorFactory:
    return EvaluatorFactory(project_id='menu-prod', location='us-central1')


def _http_client(response: httpx.Response | Exception) -> AsyncMock:
    client = AsyncMock()
    if isinstance(response, Exception):
        client.post = AsyncMock(side_effect=response)
    else:
        client.post = AsyncMock(return_value=response)
    return client


def _creds() -> MagicMock:
    credentials = MagicMock()
    credentials.token = 'mock-token'
    return credentials


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('code', 'status'),
    [(400, 'INVALID_ARGUMENT'), (403, 'PERMISSION_DENIED'), (429, 'RESOURCE_EXHAUSTED'), (503, 'UNAVAILABLE')],
)
async def test_evaluate_instances_http_status_is_classified(code: int, status: str) -> None:
    """A 429 from the eval service is retryable and a 400 is not."""
    response = httpx.Response(
        code,
        json={'error': {'message': 'eval call failed'}},
        headers={'Retry-After': '2'},
        request=httpx.Request('POST', 'https://aiplatform.googleapis.com/v1beta1:evaluateInstances'),
    )
    with (
        patch('genkit_google_genai._evaluators._evaluation.google_auth_default', return_value=(_creds(), 'menu-prod')),
        patch('genkit_google_genai._evaluators._evaluation.get_cached_client', return_value=_http_client(response)),
        pytest.raises(GenkitError, match='eval call failed') as raised,
    ):
        await _factory().evaluate_instances({'fluencyInput': {}})

    assert raised.value.status == status
    assert raised.value.response_metadata == {'retry_after_ms': 2000.0}


@pytest.mark.asyncio
async def test_evaluate_instances_transport_failure_stays_raw() -> None:
    """A refused connection has no known status, so it reaches the caller unchanged."""
    refused = httpx.ConnectError('connection refused')
    with (
        patch('genkit_google_genai._evaluators._evaluation.google_auth_default', return_value=(_creds(), 'menu-prod')),
        patch('genkit_google_genai._evaluators._evaluation.get_cached_client', return_value=_http_client(refused)),
        pytest.raises(httpx.ConnectError) as raised,
    ):
        await _factory().evaluate_instances({'fluencyInput': {}})

    assert raised.value is refused


@pytest.mark.asyncio
async def test_evaluate_instances_non_json_success_is_internal() -> None:
    response = httpx.Response(200, text='<html>proxy page</html>')
    with (
        patch('genkit_google_genai._evaluators._evaluation.google_auth_default', return_value=(_creds(), 'menu-prod')),
        patch('genkit_google_genai._evaluators._evaluation.get_cached_client', return_value=_http_client(response)),
        pytest.raises(GenkitError) as raised,
    ):
        await _factory().evaluate_instances({'fluencyInput': {}})

    assert raised.value.status == 'INTERNAL'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'auth_error',
    [DefaultCredentialsError('Your default credentials were not found.'), RefreshError('invalid_grant')],
)
async def test_evaluate_instances_credential_failure_is_unauthenticated(auth_error: Exception) -> None:
    with (
        patch('genkit_google_genai._evaluators._evaluation.google_auth_default', side_effect=auth_error),
        pytest.raises(GenkitError) as raised,
    ):
        await _factory().evaluate_instances({'fluencyInput': {}})

    assert raised.value.status == 'UNAUTHENTICATED'
    assert raised.value.cause is auth_error


@pytest.mark.asyncio
async def test_evaluator_fn_malformed_result_is_internal() -> None:
    """A 200 body missing the metric's result field is a malformed provider response."""
    factory = _factory()
    evaluator_fn = factory.create_evaluator_fn(
        VertexAIEvaluationMetricType.FLUENCY,
        lambda datapoint: {'fluencyInput': {}},
        lambda r: r['fluencyResult']['score'],
    )
    with (
        patch.object(factory, 'evaluate_instances', AsyncMock(return_value={'unexpected': {}})),
        pytest.raises(GenkitError) as raised,
    ):
        await evaluator_fn(BaseDataPoint(input='Describe the tartine', output='Smoked salmon on rye'))

    assert raised.value.status == 'INTERNAL'
    assert isinstance(raised.value.cause, KeyError)
