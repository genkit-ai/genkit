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

"""Vertex AI Evaluation implementation.

This module implements the Vertex AI Evaluation API for evaluating model outputs
using built-in metrics like BLEU, ROUGE, fluency, safety, and more.

Architecture::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    Vertex AI Evaluators Module                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Types & Configuration                                                  │
    │  ├── VertexAIEvaluationMetricType (enum) - Available metrics            │
    │  └── VertexAIEvaluationMetricConfig - Per-metric configuration          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  EvaluatorFactory                                                       │
    │  ├── evaluate_instances() - Async API call to evaluateInstances         │
    │  └── create_evaluator_fn() - Creates evaluator function for metric      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Evaluator Configurations (per metric)                                  │
    │  ├── BLEU - to_request(), response_handler()                            │
    │  ├── ROUGE - to_request(), response_handler()                           │
    │  ├── FLUENCY - to_request(), response_handler()                         │
    │  ├── SAFETY - to_request(), response_handler()                          │
    │  ├── GROUNDEDNESS - to_request(), response_handler()                    │
    │  ├── SUMMARIZATION_QUALITY - to_request(), response_handler()           │
    │  ├── SUMMARIZATION_HELPFULNESS - to_request(), response_handler()       │
    │  └── SUMMARIZATION_VERBOSITY - to_request(), response_handler()         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Plugin Integration                                                     │
    │  └── create_vertex_evaluators() - Register evaluators with Genkit       │
    └─────────────────────────────────────────────────────────────────────────┘

Implementation Notes:
    - Uses Google Cloud Application Default Credentials (ADC) for auth
    - Calls the Vertex AI Platform evaluateInstances v1beta1 endpoint
    - Each metric has a specific request format and response handler
    - Supports custom metric_spec for fine-tuning metric behavior
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum

from google.auth import default as google_auth_default
from google.auth.transport.requests import Request
from pydantic import BaseModel, ConfigDict

from genkit.ai import GENKIT_CLIENT_HEADER
from genkit.blocks.evaluator import EvalFnResponse
from genkit.core.action import Action
from genkit.core.error import GenkitError
from genkit.core.http_client import get_cached_client
from genkit.core.typing import BaseDataPoint, Details, Score

if TYPE_CHECKING:
    from genkit.ai._registry import GenkitRegistry


class VertexAIEvaluationMetricType(StrEnum):
    """Vertex AI Evaluation metric types.

    See API documentation for more information:
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/evaluation#parameter-list
    """

    BLEU = 'BLEU'
    ROUGE = 'ROUGE'
    FLUENCY = 'FLUENCY'
    SAFETY = 'SAFETY'
    GROUNDEDNESS = 'GROUNDEDNESS'
    SUMMARIZATION_QUALITY = 'SUMMARIZATION_QUALITY'
    SUMMARIZATION_HELPFULNESS = 'SUMMARIZATION_HELPFULNESS'
    SUMMARIZATION_VERBOSITY = 'SUMMARIZATION_VERBOSITY'


class VertexAIEvaluationMetricConfig(BaseModel):
    """Configuration for a Vertex AI evaluation metric.

    Attributes:
        type: The metric type.
        metric_spec: Additional metric-specific configuration.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra='allow',
        populate_by_name=True,
    )

    type: VertexAIEvaluationMetricType
    metric_spec: dict[str, Any] | None = None


def _create_list_based_score_handler(results_key: str, values_key: str) -> Callable[[dict[str, Any]], Score]:
    """Create a response handler for metrics that return a list of scored values.

    This is used for BLEU and ROUGE metrics which have similar response structures.

    Args:
        results_key: The key for the results object (e.g., 'bleuResults').
        values_key: The key for the metrics list (e.g., 'bleuMetricValues').

    Returns:
        A function that extracts a Score from the response.
    """

    def handler(response: dict[str, Any]) -> Score:
        metrics = response.get(results_key, {}).get(values_key, [])
        score = metrics[0].get('score') if metrics else None
        return Score(score=score)

    return handler


# Union type for metric specification
VertexAIEvaluationMetric = VertexAIEvaluationMetricType | VertexAIEvaluationMetricConfig


def _stringify(value: Any) -> str:  # noqa: ANN401
    """Convert a value to string for the API."""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _is_config(metric: VertexAIEvaluationMetric) -> bool:
    """Check if metric is a config object."""
    return isinstance(metric, VertexAIEvaluationMetricConfig)


class EvaluatorFactory:
    """Factory for creating Vertex AI evaluator actions."""

    def __init__(self, project_id: str, location: str) -> None:
        """Initialize the factory.

        Args:
            project_id: Google Cloud project ID.
            location: Google Cloud location.
        """
        self.project_id = project_id
        self.location = location

    async def evaluate_instances(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """Call the Vertex AI evaluateInstances API.

        Args:
            request_body: The request body for the API.

        Returns:
            The API response.

        Raises:
            GenkitError: If the API call fails.
        """
        location_name = f'projects/{self.project_id}/locations/{self.location}'
        url = f'https://{self.location}-aiplatform.googleapis.com/v1beta1/{location_name}:evaluateInstances'

        # Get authentication token
        # Use asyncio.to_thread to avoid blocking the event loop during token refresh
        credentials, _ = google_auth_default()
        await asyncio.to_thread(credentials.refresh, Request())
        token = credentials.token

        if not token:
            raise GenkitError(
                message='Unable to authenticate your request. '
                'Please ensure you have valid Google Cloud credentials configured.',
                status='UNAUTHENTICATED',
            )

        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'X-Goog-Api-Client': GENKIT_CLIENT_HEADER,
        }

        request = {
            'location': location_name,
            **request_body,
        }

        # Use cached client for better connection reuse.
        # Note: Auth headers are passed per-request since tokens may expire.
        client = get_cached_client(
            cache_key='vertex-ai-evaluator',
            timeout=60.0,
        )

        try:
            response = await client.post(
                url,
                headers=headers,
                json=request,
            )

            if response.status_code != 200:
                error_message = response.text
                try:
                    error_json = response.json()
                    if 'error' in error_json and 'message' in error_json['error']:
                        error_message = error_json['error']['message']
                except json.JSONDecodeError:  # noqa: S110
                    pass

                raise GenkitError(
                    message=f'Error calling Vertex AI Evaluation API: [{response.status_code}] {error_message}',
                    status='INTERNAL',
                )

            return response.json()

        except Exception as e:
            if isinstance(e, GenkitError):
                raise
            raise GenkitError(
                message=f'Failed to call Vertex AI Evaluation API: {e}',
                status='UNAVAILABLE',
            ) from e

    def create_evaluator_fn(
        self,
        metric_type: VertexAIEvaluationMetricType,
        metric_spec: dict[str, Any] | None,
        to_request: Any,  # noqa: ANN401
        response_handler: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Create an evaluator function.

        Args:
            metric_type: The metric type.
            metric_spec: Optional metric specification.
            to_request: Function to convert datapoint to request.
            response_handler: Function to extract score from response.

        Returns:
            An async evaluator function.
        """

        async def evaluator_fn(
            datapoint: BaseDataPoint,
            options: dict[str, Any] | None = None,
        ) -> EvalFnResponse:
            """Evaluate a single datapoint.

            Args:
                datapoint: The evaluation data point.
                options: Optional evaluation options.

            Returns:
                The evaluation response with score.
            """
            request_body = to_request(datapoint, metric_spec or {})
            response = await self.evaluate_instances(request_body)
            score = response_handler(response)

            return EvalFnResponse(
                evaluation=score,
                test_case_id=datapoint.test_case_id or '',
            )

        return evaluator_fn


def create_vertex_evaluators(
    registry: GenkitRegistry,
    metrics: list[VertexAIEvaluationMetric],
    project_id: str,
    location: str,
) -> list[Action]:
    """Create Vertex AI evaluator actions.

    Args:
        registry: The Genkit registry.
        metrics: List of metrics to create evaluators for.
        project_id: Google Cloud project ID.
        location: Google Cloud location.

    Returns:
        List of created evaluator actions.
    """
    factory = EvaluatorFactory(project_id, location)
    actions = []

    for metric in metrics:
        if isinstance(metric, VertexAIEvaluationMetricConfig):
            metric_type: VertexAIEvaluationMetricType = metric.type
            metric_spec: dict[str, Any] | None = metric.metric_spec
        else:
            metric_type = metric
            metric_spec = None

        action = _create_evaluator_for_metric(registry, factory, metric_type, metric_spec or {})
        if action:
            actions.append(action)

    return actions


def _create_evaluator_for_metric(
    registry: GenkitRegistry,
    factory: EvaluatorFactory,
    metric_type: VertexAIEvaluationMetricType,
    metric_spec: dict[str, Any],
) -> Action | None:
    """Create an evaluator action for a specific metric.

    Args:
        registry: The Genkit registry.
        factory: The evaluator factory.
        metric_type: The metric type.
        metric_spec: The metric specification.

    Returns:
        The created action, or None if metric is not supported.
    """
    evaluator_configs = {
        VertexAIEvaluationMetricType.BLEU: {
            'display_name': 'BLEU',
            'definition': 'Computes the BLEU score by comparing the output against the ground truth',
            'to_request': lambda dp, spec: {
                'bleuInput': {
                    'metricSpec': spec,
                    'instances': [
                        {
                            'prediction': _stringify(dp.output),
                            'reference': dp.reference,
                        }
                    ],
                }
            },
            'response_handler': _create_list_based_score_handler('bleuResults', 'bleuMetricValues'),
        },
        VertexAIEvaluationMetricType.ROUGE: {
            'display_name': 'ROUGE',
            'definition': 'Computes the ROUGE score by comparing the output against the ground truth',
            'to_request': lambda dp, spec: {
                'rougeInput': {
                    'metricSpec': spec,
                    'instances': [
                        {
                            'prediction': _stringify(dp.output),
                            'reference': dp.reference,
                        }
                    ],
                }
            },
            'response_handler': _create_list_based_score_handler('rougeResults', 'rougeMetricValues'),
        },
        VertexAIEvaluationMetricType.FLUENCY: {
            'display_name': 'Fluency',
            'definition': 'Assesses the language mastery of an output',
            'to_request': lambda dp, spec: {
                'fluencyInput': {
                    'metricSpec': spec,
                    'instance': {
                        'prediction': _stringify(dp.output),
                    },
                }
            },
            'response_handler': lambda r: Score(
                score=r.get('fluencyResult', {}).get('score'),
                details=Details(reasoning=r.get('fluencyResult', {}).get('explanation')),
            ),
        },
        VertexAIEvaluationMetricType.SAFETY: {
            'display_name': 'Safety',
            'definition': 'Assesses the level of safety of an output',
            'to_request': lambda dp, spec: {
                'safetyInput': {
                    'metricSpec': spec,
                    'instance': {
                        'prediction': _stringify(dp.output),
                    },
                }
            },
            'response_handler': lambda r: Score(
                score=r.get('safetyResult', {}).get('score'),
                details=Details(reasoning=r.get('safetyResult', {}).get('explanation')),
            ),
        },
        VertexAIEvaluationMetricType.GROUNDEDNESS: {
            'display_name': 'Groundedness',
            'definition': 'Assesses the ability to provide or reference information included only in the context',
            'to_request': lambda dp, spec: {
                'groundednessInput': {
                    'metricSpec': spec,
                    'instance': {
                        'prediction': _stringify(dp.output),
                        'context': '. '.join(dp.context) if dp.context else None,
                    },
                }
            },
            'response_handler': lambda r: Score(
                score=r.get('groundednessResult', {}).get('score'),
                details=Details(reasoning=r.get('groundednessResult', {}).get('explanation')),
            ),
        },
        VertexAIEvaluationMetricType.SUMMARIZATION_QUALITY: {
            'display_name': 'Summarization quality',
            'definition': 'Assesses the overall ability to summarize text',
            'to_request': lambda dp, spec: {
                'summarizationQualityInput': {
                    'metricSpec': spec,
                    'instance': {
                        'prediction': _stringify(dp.output),
                        'instruction': _stringify(dp.input),
                        'context': '. '.join(dp.context) if dp.context else None,
                    },
                }
            },
            'response_handler': lambda r: Score(
                score=r.get('summarizationQualityResult', {}).get('score'),
                details=Details(reasoning=r.get('summarizationQualityResult', {}).get('explanation')),
            ),
        },
        VertexAIEvaluationMetricType.SUMMARIZATION_HELPFULNESS: {
            'display_name': 'Summarization helpfulness',
            'definition': 'Assesses ability to provide a summarization with details to substitute the original',
            'to_request': lambda dp, spec: {
                'summarizationHelpfulnessInput': {
                    'metricSpec': spec,
                    'instance': {
                        'prediction': _stringify(dp.output),
                        'instruction': _stringify(dp.input),
                        'context': '. '.join(dp.context) if dp.context else None,
                    },
                }
            },
            'response_handler': lambda r: Score(
                score=r.get('summarizationHelpfulnessResult', {}).get('score'),
                details=Details(reasoning=r.get('summarizationHelpfulnessResult', {}).get('explanation')),
            ),
        },
        VertexAIEvaluationMetricType.SUMMARIZATION_VERBOSITY: {
            'display_name': 'Summarization verbosity',
            'definition': 'Assesses the ability to provide a succinct summarization',
            'to_request': lambda dp, spec: {
                'summarizationVerbosityInput': {
                    'metricSpec': spec,
                    'instance': {
                        'prediction': _stringify(dp.output),
                        'instruction': _stringify(dp.input),
                        'context': '. '.join(dp.context) if dp.context else None,
                    },
                }
            },
            'response_handler': lambda r: Score(
                score=r.get('summarizationVerbosityResult', {}).get('score'),
                details=Details(reasoning=r.get('summarizationVerbosityResult', {}).get('explanation')),
            ),
        },
    }

    config = evaluator_configs.get(metric_type)
    if not config:
        return None

    evaluator_name = f'vertexai/{metric_type.lower()}'
    display_name: str = config['display_name']  # type: ignore[assignment]
    definition: str = config['definition']  # type: ignore[assignment]
    evaluator_fn = factory.create_evaluator_fn(
        metric_type,
        metric_spec,
        config['to_request'],
        config['response_handler'],
    )

    return registry.define_evaluator(
        name=evaluator_name,
        display_name=display_name,
        definition=definition,
        fn=evaluator_fn,
        is_billed=True,  # These use Vertex AI API which is billed
    )
