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

"""Genkit plugin entry point for Google Checks AI Safety.

This module provides both:

1. A ``Checks`` plugin class for use with ``Genkit(plugins=[Checks(...)])``.
2. A ``define_checks_evaluators()`` standalone function for explicit
   registration (matching the evaluators plugin pattern).

The middleware is provided separately as ``checks_middleware()`` which can
be passed to ``ai.generate(use=[...])``.

See Also:
    - JS reference: ``js/plugins/checks/src/index.ts``
"""

from __future__ import annotations

import os

from google.auth.credentials import Credentials
from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.core.action import Action, ActionMetadata
from genkit.core.logging import get_logger
from genkit.core.plugin import Plugin
from genkit.core.registry import ActionKind
from genkit.plugins.checks.evaluation import create_checks_evaluators
from genkit.plugins.checks.guardrails import GuardrailsClient
from genkit.plugins.checks.metrics import ChecksEvaluationMetric

logger = get_logger(__name__)

CHECKS_PLUGIN_NAME = 'checks'


class ChecksEvaluationConfig(BaseModel):
    """Configuration for Checks evaluation metrics.

    Attributes:
        metrics: List of safety policies to create evaluators for.
    """

    metrics: list[ChecksEvaluationMetric] = Field(default_factory=list)


def define_checks_evaluators(
    ai: Genkit,
    project_id: str | None = None,
    metrics: list[ChecksEvaluationMetric] | None = None,
    credentials: Credentials | None = None,
) -> None:
    """Register Checks AI Safety evaluators with a Genkit instance.

    This is the standalone function API, matching the pattern used by
    ``define_genkit_evaluators()`` in the evaluators plugin. It resolves
    the project ID and creates evaluator actions for each configured metric.

    Example::

        from genkit.plugins.checks import (
            define_checks_evaluators,
            ChecksEvaluationMetricType,
        )

        ai = Genkit(...)

        define_checks_evaluators(
            ai,
            project_id='your-gcp-project-id',
            metrics=[
                ChecksEvaluationMetricType.DANGEROUS_CONTENT,
                ChecksEvaluationMetricType.HARASSMENT,
            ],
        )

    Args:
        ai: The Genkit instance to register evaluators with.
        project_id: GCP project ID. Falls back to the ``GCLOUD_PROJECT``
            environment variable.
        metrics: Safety policies to create evaluators for. If empty or
            None, no evaluators are registered.
        credentials: Optional pre-configured Google credentials. If not
            provided, credentials are resolved via ``GCLOUD_SERVICE_ACCOUNT_CREDS``
            env var or Application Default Credentials.

    Raises:
        ValueError: If no project ID can be resolved.
    """
    resolved_project_id = project_id or os.environ.get('GCLOUD_PROJECT')
    if not resolved_project_id:
        raise ValueError(
            "Checks plugin requires a 'project_id'. "
            "Set the 'GCLOUD_PROJECT' environment variable or pass "
            "'project_id' explicitly."
        )

    if not metrics:
        return

    guardrails = GuardrailsClient(
        project_id=resolved_project_id,
        credentials=credentials,
    )
    create_checks_evaluators(
        registry=ai,
        guardrails=guardrails,
        metrics=metrics,
    )


class Checks(Plugin):
    """Google Checks AI Safety plugin for Genkit.

    Provides safety evaluators that classify content against Google Checks
    AI Safety policies. The plugin authenticates using Application Default
    Credentials and registers evaluator actions for each configured metric.

    Example::

        ai = Genkit(
            plugins=[
                Checks(project_id='my-gcp-project'),
            ],
        )

        # Register evaluators separately (plugin validates project_id):
        define_checks_evaluators(
            ai,
            project_id='your-gcp-project-id',
            metrics=[
                ChecksEvaluationMetricType.DANGEROUS_CONTENT,
                ChecksEvaluationMetricType.HARASSMENT,
            ],
        )

    Args:
        project_id: GCP project ID. Falls back to ``GCLOUD_PROJECT`` env var.
        credentials: Optional pre-configured Google credentials.
    """

    name = CHECKS_PLUGIN_NAME

    def __init__(
        self,
        project_id: str | None = None,
        credentials: Credentials | None = None,
    ) -> None:
        """Initialize the Checks plugin.

        Note: Evaluator registration requires registry access, which the
        Plugin base class does not provide. Use
        ``define_checks_evaluators(ai, ...)`` after creating the Genkit
        instance to register evaluators.

        Args:
            project_id: GCP project ID. Falls back to ``GCLOUD_PROJECT``.
            credentials: Optional pre-configured Google credentials.
        """
        self._project_id = project_id or os.environ.get('GCLOUD_PROJECT')
        self._credentials = credentials

    async def init(self) -> list[Action]:
        """Initialize the plugin and validate configuration.

        Validates that a project ID is configured. Evaluator registration
        must be done separately via ``define_checks_evaluators()``.

        Returns:
            An empty list.

        Raises:
            ValueError: If no project ID is configured.
        """
        if not self._project_id:
            raise ValueError(
                "Checks plugin requires a 'project_id'. "
                "Set the 'GCLOUD_PROJECT' environment variable or pass "
                "'project_id' to the Checks plugin."
            )
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by name.

        The Checks plugin does not support dynamic action resolution.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced name of the action.

        Returns:
            None — the plugin does not support lazy resolution.
        """
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        """List available actions.

        Returns an empty list. Actions are registered via
        ``define_checks_evaluators()``.

        Returns:
            An empty list.
        """
        return []
