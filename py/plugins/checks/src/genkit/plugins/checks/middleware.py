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

"""Model middleware for Checks AI Safety content classification.

Intercepts model requests and responses, classifying text content against
configured safety policies via the Checks ``classifyContent`` API. If any
input or output violates a policy, the middleware returns a ``blocked``
response instead of forwarding the request / returning the generated content.

Data Flow::

    User prompt
         │
         ▼
    ┌─ Input Guard ──────────────────────────────────────────────┐
    │  For each message in the request:                          │
    │    • Extract text content                                  │
    │    • Call classifyContent API                               │
    │    • If VIOLATIVE → return blocked response immediately    │
    └────────────────────┬───────────────────────────────────────┘
                         │ (all inputs safe)
                         ▼
                    ┌─ Model ──┐
                    │ Generate │
                    └────┬─────┘
                         │
                         ▼
    ┌─ Output Guard ─────────────────────────────────────────────┐
    │  For each candidate in the response:                       │
    │    • Extract text content                                  │
    │    • Call classifyContent API                               │
    │    • If VIOLATIVE → return blocked response immediately    │
    └────────────────────┬───────────────────────────────────────┘
                         │ (all outputs safe)
                         ▼
                    Return to user

See Also:
    - JS reference: ``js/plugins/checks/src/middleware.ts``
"""

from __future__ import annotations

from google.auth.credentials import Credentials

from genkit.blocks.model import ModelMiddleware, ModelMiddlewareNext
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.core.typing import FinishReason, GenerateRequest, GenerateResponse
from genkit.plugins.checks.guardrails import GuardrailsClient
from genkit.plugins.checks.metrics import ChecksEvaluationMetric

logger = get_logger(__name__)


def checks_middleware(
    project_id: str,
    metrics: list[ChecksEvaluationMetric],
    credentials: Credentials | None = None,
) -> ModelMiddleware:
    """Create a model middleware that enforces Checks AI Safety policies.

    The middleware classifies both input messages and generated output against
    the configured safety policies. If any text violates a policy, the
    middleware returns a ``blocked`` response with details about which policies
    were violated.

    Example::

        response = await ai.generate(
            model='googleai/gemini-2.0-flash',
            prompt='Tell me a story',
            use=[
                checks_middleware(
                    project_id='your-gcp-project-id',
                    metrics=[
                        ChecksEvaluationMetricType.DANGEROUS_CONTENT,
                        ChecksEvaluationMetricType.HARASSMENT,
                    ],
                ),
            ],
        )

    Args:
        project_id: The GCP project ID for Checks API billing.
        metrics: Safety policies to enforce on input and output.
        credentials: Optional pre-configured Google credentials.

    Returns:
        A ``ModelMiddleware`` function that can be passed to ``ai.generate(use=[...])``.
    """
    guardrails = GuardrailsClient(project_id=project_id, credentials=credentials)

    async def middleware(
        req: GenerateRequest,
        ctx: ActionRunContext,
        next_fn: ModelMiddlewareNext,
    ) -> GenerateResponse:
        """Classify input and output text, blocking on policy violations.

        Args:
            req: The incoming generate request.
            ctx: The action run context.
            next_fn: The next middleware or model function in the chain.

        Returns:
            The model response if safe, or a blocked response if violated.
        """
        # Classify each input message against safety policies.
        for message in req.messages:
            for part in message.content:
                text = part.root.text
                if text is not None:
                    text_str = str(text.root) if hasattr(text, 'root') else str(text)
                    if text_str:
                        violated = await _get_violated_policies(guardrails, text_str, metrics)
                        if violated:
                            policy_names = ' '.join(violated)
                            return GenerateResponse(
                                finish_reason=FinishReason.BLOCKED,
                                finish_message=(
                                    f'Model input violated Checks policies: '
                                    f'[{policy_names}], further processing blocked.'
                                ),
                            )

        generated = await next_fn(req, ctx)

        # Classify each output candidate against safety policies.
        if generated.candidates:
            for candidate in generated.candidates:
                if candidate.message and candidate.message.content:
                    for part in candidate.message.content:
                        text = part.root.text
                        if text is not None:
                            text_str = str(text.root) if hasattr(text, 'root') else str(text)
                            if text_str:
                                violated = await _get_violated_policies(guardrails, text_str, metrics)
                                if violated:
                                    policy_names = ' '.join(violated)
                                    return GenerateResponse(
                                        finish_reason=FinishReason.BLOCKED,
                                        finish_message=(
                                            f'Model output violated Checks policies: [{policy_names}], output blocked.'
                                        ),
                                    )

        # Check top-level message only when candidates is not set, to avoid
        # double-checking the same content that was already classified above.
        if not generated.candidates and generated.message and generated.message.content:
            for part in generated.message.content:
                text = part.root.text
                if text is not None:
                    text_str = str(text.root) if hasattr(text, 'root') else str(text)
                    if text_str:
                        violated = await _get_violated_policies(guardrails, text_str, metrics)
                        if violated:
                            policy_names = ' '.join(violated)
                            return GenerateResponse(
                                finish_reason=FinishReason.BLOCKED,
                                finish_message=(
                                    f'Model output violated Checks policies: [{policy_names}], output blocked.'
                                ),
                            )

        return generated

    return middleware


async def _get_violated_policies(
    guardrails: GuardrailsClient,
    content: str,
    metrics: list[ChecksEvaluationMetric],
) -> list[str]:
    """Classify content and return a list of violated policy type names.

    Args:
        guardrails: The guardrails API client.
        content: The text to classify.
        metrics: Safety policies to check.

    Returns:
        A list of violated policy type strings. Empty if all policies pass.
    """
    response = await guardrails.classify_content(content, metrics)
    return [result.policy_type for result in response.policy_results if result.violation_result == 'VIOLATIVE']
