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

"""Tests for the Checks AI Safety plugin.

These tests verify the plugin's metric types, guardrails client request
building, and middleware behavior using mocked API responses.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import genkit.plugins.checks as checks_module
from genkit.core.registry import ActionKind
from genkit.plugins.checks import package_name
from genkit.plugins.checks.guardrails import (
    ClassifyContentResponse,
    GuardrailsClient,
    PolicyResult,
)
from genkit.plugins.checks.metrics import (
    ChecksEvaluationMetricConfig,
    ChecksEvaluationMetricType,
)
from genkit.plugins.checks.middleware import _get_violated_policies
from genkit.plugins.checks.plugin import Checks, define_checks_evaluators


class TestChecksEvaluationMetricType:
    """Tests for the ChecksEvaluationMetricType enum."""

    def test_all_metric_types_exist(self) -> None:
        """All JS-parity metric types are defined."""
        expected = [
            'DANGEROUS_CONTENT',
            'PII_SOLICITING_RECITING',
            'HARASSMENT',
            'SEXUALLY_EXPLICIT',
            'HATE_SPEECH',
            'MEDICAL_INFO',
            'VIOLENCE_AND_GORE',
            'OBSCENITY_AND_PROFANITY',
        ]
        for name in expected:
            metric = ChecksEvaluationMetricType(name)
            if str(metric) != name:
                pytest.fail(f'Metric {name} has value {metric}, expected {name}')

    def test_metric_type_is_str(self) -> None:
        """Metric types are string-compatible for JSON serialization."""
        metric = ChecksEvaluationMetricType.DANGEROUS_CONTENT
        if str(metric) != 'DANGEROUS_CONTENT':
            pytest.fail(f'Expected "DANGEROUS_CONTENT", got "{metric}"')


class TestChecksEvaluationMetricConfig:
    """Tests for the ChecksEvaluationMetricConfig model."""

    def test_config_with_threshold(self) -> None:
        """Config with explicit threshold."""
        config = ChecksEvaluationMetricConfig(
            type=ChecksEvaluationMetricType.HATE_SPEECH,
            threshold=0.5,
        )
        if config.type != ChecksEvaluationMetricType.HATE_SPEECH:
            pytest.fail(f'Expected HATE_SPEECH, got {config.type}')
        if config.threshold != 0.5:
            pytest.fail(f'Expected 0.5, got {config.threshold}')

    def test_config_without_threshold(self) -> None:
        """Config without threshold defaults to None."""
        config = ChecksEvaluationMetricConfig(
            type=ChecksEvaluationMetricType.HARASSMENT,
        )
        if config.threshold is not None:
            pytest.fail(f'Expected None, got {config.threshold}')


class TestGuardrailsClient:
    """Tests for the GuardrailsClient request building."""

    def test_build_request_plain_metrics(self) -> None:
        """Request body is built correctly for plain metric types."""
        request = GuardrailsClient._build_request(
            content='Hello world',
            policies=[
                ChecksEvaluationMetricType.DANGEROUS_CONTENT,
                ChecksEvaluationMetricType.HARASSMENT,
            ],
        )
        expected: dict[str, Any] = {
            'input': {
                'text_input': {
                    'content': 'Hello world',
                },
            },
            'policies': [
                {'policy_type': 'DANGEROUS_CONTENT'},
                {'policy_type': 'HARASSMENT'},
            ],
        }
        if request != expected:
            pytest.fail(f'Request mismatch:\n  got:  {request}\n  want: {expected}')

    def test_build_request_with_threshold(self) -> None:
        """Request body includes threshold when specified."""
        request = GuardrailsClient._build_request(
            content='Test content',
            policies=[
                ChecksEvaluationMetricConfig(
                    type=ChecksEvaluationMetricType.HATE_SPEECH,
                    threshold=0.3,
                ),
            ],
        )
        policies = request['policies']
        if len(policies) != 1:
            pytest.fail(f'Expected 1 policy, got {len(policies)}')
        policy = policies[0]
        if policy.get('policy_type') != 'HATE_SPEECH':
            pytest.fail(f'Expected HATE_SPEECH, got {policy.get("policy_type")}')
        if policy.get('threshold') != 0.3:
            pytest.fail(f'Expected threshold 0.3, got {policy.get("threshold")}')

    def test_build_request_mixed_metrics(self) -> None:
        """Request body handles mixed plain and config metrics."""
        request = GuardrailsClient._build_request(
            content='Mixed test',
            policies=[
                ChecksEvaluationMetricType.DANGEROUS_CONTENT,
                ChecksEvaluationMetricConfig(
                    type=ChecksEvaluationMetricType.HARASSMENT,
                    threshold=0.7,
                ),
            ],
        )
        policies = request['policies']
        if len(policies) != 2:
            pytest.fail(f'Expected 2 policies, got {len(policies)}')
        if 'threshold' in policies[0]:
            pytest.fail(f'First policy should not have threshold: {policies[0]}')
        if policies[1].get('threshold') != 0.7:
            pytest.fail(f'Expected threshold 0.7, got {policies[1].get("threshold")}')


class TestResponseModels:
    """Tests for the Pydantic response models."""

    def test_parse_api_response(self) -> None:
        """ClassifyContentResponse parses API JSON correctly."""
        api_json = {
            'policyResults': [
                {
                    'policyType': 'DANGEROUS_CONTENT',
                    'score': 0.1,
                    'violationResult': 'NON_VIOLATIVE',
                },
                {
                    'policyType': 'HATE_SPEECH',
                    'score': 0.9,
                    'violationResult': 'VIOLATIVE',
                },
            ],
        }
        response = ClassifyContentResponse.model_validate(api_json)
        if len(response.policy_results) != 2:
            pytest.fail(f'Expected 2 results, got {len(response.policy_results)}')
        if response.policy_results[0].policy_type != 'DANGEROUS_CONTENT':
            pytest.fail(f'Expected DANGEROUS_CONTENT, got {response.policy_results[0].policy_type}')
        if response.policy_results[1].violation_result != 'VIOLATIVE':
            pytest.fail(f'Expected VIOLATIVE, got {response.policy_results[1].violation_result}')


class TestGetViolatedPolicies:
    """Tests for the _get_violated_policies helper."""

    @pytest.mark.asyncio
    async def test_no_violations(self) -> None:
        """Returns empty list when no policies are violated."""
        mock_response = ClassifyContentResponse(
            policyResults=[
                PolicyResult(policyType='DANGEROUS_CONTENT', score=0.1, violationResult='NON_VIOLATIVE'),
            ],
        )
        guardrails = MagicMock(spec=GuardrailsClient)
        guardrails.classify_content = AsyncMock(return_value=mock_response)

        violated = await _get_violated_policies(
            guardrails,
            'safe text',
            [ChecksEvaluationMetricType.DANGEROUS_CONTENT],
        )
        if violated:
            pytest.fail(f'Expected no violations, got {violated}')

    @pytest.mark.asyncio
    async def test_with_violations(self) -> None:
        """Returns violated policy types."""
        mock_response = ClassifyContentResponse(
            policyResults=[
                PolicyResult(policyType='DANGEROUS_CONTENT', score=0.1, violationResult='NON_VIOLATIVE'),
                PolicyResult(policyType='HATE_SPEECH', score=0.9, violationResult='VIOLATIVE'),
            ],
        )
        guardrails = MagicMock(spec=GuardrailsClient)
        guardrails.classify_content = AsyncMock(return_value=mock_response)

        violated = await _get_violated_policies(
            guardrails,
            'hateful text',
            [
                ChecksEvaluationMetricType.DANGEROUS_CONTENT,
                ChecksEvaluationMetricType.HATE_SPEECH,
            ],
        )
        if violated != ['HATE_SPEECH']:
            pytest.fail(f'Expected ["HATE_SPEECH"], got {violated}')


class TestChecksPlugin:
    """Tests for the Checks plugin class."""

    def test_plugin_name(self) -> None:
        """Plugin has the correct name."""
        plugin = Checks(project_id='test-project')
        if plugin.name != 'checks':
            pytest.fail(f'Expected "checks", got {plugin.name}')

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_init_raises_without_project_id(self) -> None:
        """init() raises ValueError when no project_id is set."""
        plugin = Checks()
        with pytest.raises(ValueError, match='project_id'):
            await plugin.init()

    @pytest.mark.asyncio
    async def test_init_with_project_id(self) -> None:
        """init() succeeds with a project_id."""
        plugin = Checks(project_id='test-project')
        result = await plugin.init()
        if result != []:
            pytest.fail(f'Expected empty list, got {result}')

    @pytest.mark.asyncio
    async def test_resolve_returns_none(self) -> None:
        """resolve() always returns None for Checks plugin."""
        plugin = Checks(project_id='test-project')
        result = await plugin.resolve(ActionKind.MODEL, 'checks/test')
        if result is not None:
            pytest.fail(f'Expected None, got {result}')

    @pytest.mark.asyncio
    async def test_list_actions_returns_empty(self) -> None:
        """list_actions() returns an empty list."""
        plugin = Checks(project_id='test-project')
        result = await plugin.list_actions()
        if result != []:
            pytest.fail(f'Expected empty list, got {result}')


class TestDefineChecksEvaluators:
    """Tests for the define_checks_evaluators function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_raises_without_project_id(self) -> None:
        """Raises ValueError when no project_id is available."""
        ai = MagicMock()
        with pytest.raises(ValueError, match='project_id'):
            define_checks_evaluators(ai, metrics=[ChecksEvaluationMetricType.DANGEROUS_CONTENT])

    def test_no_op_with_empty_metrics(self) -> None:
        """Does nothing when metrics list is empty."""
        ai = MagicMock()
        define_checks_evaluators(ai, project_id='test-project', metrics=[])
        ai.define_evaluator.assert_not_called()

    @patch('genkit.plugins.checks.plugin.GuardrailsClient')
    @patch('genkit.plugins.checks.plugin.create_checks_evaluators')
    def test_calls_create_checks_evaluators(self, mock_create: MagicMock, _mock_client: MagicMock) -> None:
        """Calls create_checks_evaluators with the right arguments."""
        ai = MagicMock()
        metrics = [ChecksEvaluationMetricType.DANGEROUS_CONTENT]
        define_checks_evaluators(ai, project_id='test-project', metrics=metrics)
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        if call_kwargs.kwargs['metrics'] != metrics:
            pytest.fail(f'Expected metrics {metrics}, got {call_kwargs.kwargs["metrics"]}')


class TestPackageExports:
    """Tests that all public symbols are exported."""

    def test_all_exports(self) -> None:
        """All expected symbols are in __all__."""
        expected_exports = [
            'Checks',
            'ChecksEvaluationConfig',
            'ChecksEvaluationMetric',
            'ChecksEvaluationMetricConfig',
            'ChecksEvaluationMetricType',
            'GuardrailsClient',
            'checks_middleware',
            'define_checks_evaluators',
            'package_name',
        ]
        for name in expected_exports:
            if name not in checks_module.__all__:
                pytest.fail(f'{name} not in __all__')
            if not hasattr(checks_module, name):
                pytest.fail(f'{name} not importable from genkit.plugins.checks')

    def test_package_name(self) -> None:
        """package_name() returns the correct value."""
        if package_name() != 'genkit.plugins.checks':
            pytest.fail(f'Expected "genkit.plugins.checks", got {package_name()}')
