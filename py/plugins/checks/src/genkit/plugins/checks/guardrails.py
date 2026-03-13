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

"""Client for the Checks AI Safety ``classifyContent`` API.

This module provides ``GuardrailsClient``, which communicates with the
Checks ``v1alpha`` endpoint to classify text against configurable safety
policies.

API endpoint:
    ``POST https://checks.googleapis.com/v1alpha/aisafety:classifyContent``

Authentication (resolved in priority order):
    1. Explicit ``credentials`` passed to the constructor.
    2. ``GCLOUD_SERVICE_ACCOUNT_CREDS`` env var containing a JSON service
       account key (matches js/plugins/checks/src/index.ts).
    3. Google Application Default Credentials (ADC).

All credential paths request the ``cloud-platform`` and ``checks`` OAuth
scopes, matching the JS canonical implementation.

See Also:
    - JS reference implementation: ``js/plugins/checks/src/guardrails.ts``
    - JS auth initialization: ``js/plugins/checks/src/index.ts``
    - Checks AI Safety: https://checks.google.com/ai-safety
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from google.auth import default as google_auth_default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from pydantic import BaseModel, Field

from genkit.core.error import GenkitError
from genkit.core.http_client import get_cached_client
from genkit.core.logging import get_logger
from genkit.plugins.checks.metrics import (
    ChecksEvaluationMetric,
    ChecksEvaluationMetricConfig,
)

logger = get_logger(__name__)

GUARDRAILS_URL = 'https://checks.googleapis.com/v1alpha/aisafety:classifyContent'

# OAuth scopes required by the Checks API.
# Matches the JS plugin: js/plugins/checks/src/index.ts
_CHECKS_OAUTH_SCOPES: list[str] = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/checks',
]


class PolicyResult(BaseModel):
    """Result of a single policy classification.

    Attributes:
        policy_type: The policy that was evaluated (e.g. ``DANGEROUS_CONTENT``).
        score: Optional confidence score from the API.
        violation_result: ``"VIOLATIVE"`` or ``"NON_VIOLATIVE"``.
    """

    policy_type: str = Field(alias='policyType')
    score: float | None = None
    violation_result: str = Field(alias='violationResult')

    model_config = {'populate_by_name': True}


class ClassifyContentResponse(BaseModel):
    """Response from the Checks ``classifyContent`` endpoint.

    Attributes:
        policy_results: Per-policy classification results.
    """

    policy_results: list[PolicyResult] = Field(alias='policyResults')

    model_config = {'populate_by_name': True}


def _resolve_credentials(
    credentials: Credentials | None = None,
) -> Credentials:
    """Resolve Google credentials in priority order.

    Matches the JS plugin's ``initializeAuth()`` in ``index.ts``:
    1. Explicit credentials passed by the caller.
    2. ``GCLOUD_SERVICE_ACCOUNT_CREDS`` env var (JSON service account key).
    3. Application Default Credentials (ADC).

    Args:
        credentials: Optional pre-configured credentials.

    Returns:
        Resolved credentials with the required OAuth scopes.
    """
    if credentials is not None:
        return credentials

    # Match JS: process.env.GCLOUD_SERVICE_ACCOUNT_CREDS
    sa_creds_json = os.environ.get('GCLOUD_SERVICE_ACCOUNT_CREDS')
    if sa_creds_json:
        try:
            info = json.loads(sa_creds_json)
            return service_account.Credentials.from_service_account_info(
                info,
                scopes=_CHECKS_OAUTH_SCOPES,
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                'Failed to parse GCLOUD_SERVICE_ACCOUNT_CREDS. Falling back to Application Default Credentials.',
                error=str(e),
            )

    resolved, _ = google_auth_default(scopes=_CHECKS_OAUTH_SCOPES)
    return resolved


class GuardrailsClient:
    """Client for the Checks AI Safety classifyContent API.

    Handles authentication and sends classification requests to the Checks
    API. Tokens are refreshed per-request because they may expire between
    calls.

    Supports the same credential resolution as the JS plugin:
    explicit credentials, ``GCLOUD_SERVICE_ACCOUNT_CREDS`` env var, or ADC.

    Args:
        project_id: The GCP project ID used for billing via
            ``x-goog-user-project``.
        credentials: Optional pre-configured Google credentials. If not
            provided, credentials are resolved via env vars or ADC.
    """

    def __init__(
        self,
        project_id: str,
        credentials: Credentials | None = None,
    ) -> None:
        """Initialize the guardrails client.

        Args:
            project_id: GCP project ID for billing.
            credentials: Optional pre-configured credentials.
        """
        self._project_id = project_id
        self._credentials = _resolve_credentials(credentials)

        # Match JS: warn when credential's quota project differs from
        # configured project_id. This mirrors the GoogleAuth.getClient()
        # check in js/plugins/checks/src/index.ts.
        quota_project = getattr(self._credentials, 'quota_project_id', None)
        if quota_project and quota_project != project_id:
            logger.warning(
                'Checks: Your credentials have a default quota project which will override the configured project_id.',
                quota_project=quota_project,
                configured_project_id=project_id,
            )

    async def classify_content(
        self,
        content: str,
        policies: list[ChecksEvaluationMetric],
    ) -> ClassifyContentResponse:
        """Classify text against the specified safety policies.

        Sends a POST request to the Checks ``classifyContent`` endpoint.

        Args:
            content: The text to classify.
            policies: Safety policies to evaluate against. Each policy can be
                a plain ``ChecksEvaluationMetricType`` enum value or a
                ``ChecksEvaluationMetricConfig`` with an explicit threshold.

        Returns:
            The classification response with per-policy results.

        Raises:
            GenkitError: If authentication fails or the API returns an error.
        """
        request_body = self._build_request(content, policies)

        # Refresh credentials on each call. Use asyncio.to_thread to avoid
        # blocking the event loop during the synchronous token refresh.
        await asyncio.to_thread(self._credentials.refresh, Request())
        token = self._credentials.token

        if not token:
            raise GenkitError(
                message=(
                    'Unable to authenticate with Google Cloud. '
                    'Ensure you have valid credentials configured '
                    '(e.g. via Application Default Credentials).'
                ),
                status='UNAUTHENTICATED',
            )

        headers: dict[str, str] = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'x-goog-user-project': self._project_id,
        }

        client = get_cached_client(
            cache_key='checks-guardrails',
            timeout=60.0,
        )

        try:
            response = await client.post(
                GUARDRAILS_URL,
                headers=headers,
                json=request_body,
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
                    message=f'Checks classifyContent API error: [{response.status_code}] {error_message}',
                    status='INTERNAL',
                )

            return ClassifyContentResponse.model_validate(response.json())

        except Exception as e:
            if isinstance(e, GenkitError):
                raise
            raise GenkitError(
                message=f'Failed to call Checks classifyContent API: {e}',
                status='UNAVAILABLE',
            ) from e

    @staticmethod
    def _build_request(
        content: str,
        policies: list[ChecksEvaluationMetric],
    ) -> dict[str, Any]:
        """Build the JSON request body for the classifyContent endpoint.

        Args:
            content: The text to classify.
            policies: Safety policies with optional thresholds.

        Returns:
            A dict ready for JSON serialization.
        """
        policy_list: list[dict[str, Any]] = []
        for policy in policies:
            if isinstance(policy, ChecksEvaluationMetricConfig):
                entry: dict[str, Any] = {'policy_type': str(policy.type)}
                if policy.threshold is not None:
                    entry['threshold'] = policy.threshold
            else:
                entry = {'policy_type': str(policy)}
            policy_list.append(entry)

        return {
            'input': {
                'text_input': {
                    'content': content,
                },
            },
            'policies': policy_list,
        }
