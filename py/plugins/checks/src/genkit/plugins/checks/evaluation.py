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

"""Evaluator registration for the Checks AI Safety plugin.

Creates a single ``checks/guardrails`` Genkit evaluator that classifies
content against all configured Checks safety policies in one API call,
matching the JS canonical implementation.

The JS plugin (``js/plugins/checks/src/evaluation.ts``) registers a single
``checks/guardrails`` evaluator that sends all policies to the API and
returns per-policy results. This Python implementation mirrors that design.

See Also:
    - JS reference: ``js/plugins/checks/src/evaluation.ts``
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from genkit.core.logging import get_logger
from genkit.core.typing import BaseDataPoint, EvalFnResponse, EvalStatusEnum, Score
from genkit.plugins.checks.guardrails import GuardrailsClient
from genkit.plugins.checks.metrics import (
    ChecksEvaluationMetric,
    ChecksEvaluationMetricConfig,
    ChecksEvaluationMetricType,
)

if TYPE_CHECKING:
    from genkit.ai._registry import GenkitRegistry

logger = get_logger(__name__)

CHECKS_EVALUATOR_NAME = 'checks/guardrails'


def _metric_type(metric: ChecksEvaluationMetric) -> ChecksEvaluationMetricType:
    """Extract the metric type from a metric or config."""
    if isinstance(metric, ChecksEvaluationMetricConfig):
        return metric.type
    return metric


def create_checks_evaluators(
    registry: GenkitRegistry,
    guardrails: GuardrailsClient,
    metrics: list[ChecksEvaluationMetric],
) -> None:
    """Register a single Checks safety evaluator with the Genkit registry.

    Matches the JS implementation: one ``checks/guardrails`` evaluator that
    sends all configured policies to the API in a single call and returns
    per-policy results.

    Args:
        registry: The Genkit registry to register evaluators with.
        guardrails: A configured ``GuardrailsClient`` instance.
        metrics: List of safety policies to evaluate against.
    """
    policy_types = [str(_metric_type(m)) for m in metrics]

    async def evaluator_fn(
        datapoint: BaseDataPoint,
        options: object | None = None,
    ) -> EvalFnResponse:
        """Evaluate a single datapoint against all configured Checks policies.

        Mirrors the JS evaluator callback in ``createPolicyEvaluator()``.

        Args:
            datapoint: The evaluation datapoint containing output text.
            options: Optional evaluation options (unused).

        Returns:
            The evaluation response with per-policy scores.
        """
        if datapoint.output is None:
            return EvalFnResponse(
                test_case_id=datapoint.test_case_id or '',
                evaluation=Score(
                    error='Output is required for Checks evaluation',
                    status=EvalStatusEnum.FAIL,
                ),
            )

        output_text = datapoint.output if isinstance(datapoint.output, str) else json.dumps(datapoint.output)

        response = await guardrails.classify_content(output_text, metrics)

        # Return per-policy results matching the JS response format:
        # { evaluation: [{ id, score, details: { reasoning } }] }
        evaluation_results: list[Score] = []
        for result in response.policy_results:
            is_violative = result.violation_result == 'VIOLATIVE'
            status = EvalStatusEnum.FAIL if is_violative else EvalStatusEnum.PASS_
            evaluation_results.append(
                Score(
                    id=result.policy_type,
                    score=result.score,
                    status=status,
                    details={  # type: ignore[arg-type]
                        'reasoning': f'Status {result.violation_result}',
                    },
                ),
            )

        # Return all per-policy results as a list, matching the JS format:
        # { evaluation: [{ id, score, details: { reasoning } }], testCaseId }
        if evaluation_results:
            return EvalFnResponse(
                test_case_id=datapoint.test_case_id or '',
                evaluation=evaluation_results,
            )

        return EvalFnResponse(
            test_case_id=datapoint.test_case_id or '',
            evaluation=Score(
                error='No policy results returned from Checks API',
                status=EvalStatusEnum.FAIL,
            ),
        )

    registry.define_evaluator(
        name=CHECKS_EVALUATOR_NAME,
        display_name=CHECKS_EVALUATOR_NAME,
        definition=f'Evaluates input text against the Checks {policy_types} policies.',
        fn=evaluator_fn,
        is_billed=True,
    )
