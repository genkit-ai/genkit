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

"""Google Checks AI Safety sample for Genkit.

This sample demonstrates two ways to use the Checks AI Safety plugin:

1. **Evaluators** — Register safety evaluators that score model outputs
   against Checks policies. These appear in the Genkit Dev UI and can
   be used in automated evaluation pipelines.

2. **Middleware** — Wrap model calls with ``checks_middleware`` to
   automatically block unsafe input and output in real-time.

How it works::

    ┌─────────────────────────────────────────────────────────────────────┐
    │                      Sample Architecture                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  Evaluators (registered at startup):                                │
    │    checks/dangerous_content                                         │
    │    checks/harassment                                                │
    │    checks/hate_speech                                               │
    │                                                                     │
    │  Flows:                                                             │
    │    safe_generate ─── uses checks_middleware ──┐                     │
    │         │                                     │                     │
    │         ▼                                     ▼                     │
    │    ┌──────────┐                     ┌───────────────────┐           │
    │    │  Gemini  │                     │  Checks API       │           │
    │    │  Model   │                     │  classifyContent  │           │
    │    └──────────┘                     └───────────────────┘           │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Prerequisites:
    - ``GCLOUD_PROJECT`` env var set to a GCP project with Checks API enabled
    - ``GEMINI_API_KEY`` env var set to a valid Gemini API key
    - Application Default Credentials configured (``gcloud auth application-default login``)

Testing:
    1. Run ``./run.sh`` to start the sample
    2. Open the Genkit Dev UI URL printed in the terminal
    3. Try the ``safe_generate`` flow with safe and unsafe prompts
    4. Check the evaluators tab for registered Checks evaluators

See Also:
    - Google Checks AI Safety: https://checks.google.com/ai-safety
    - Genkit Documentation: https://genkit.dev/
"""

import asyncio
import os

from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.core.logging import get_logger
from genkit.core.typing import FinishReason
from genkit.plugins.checks import (
    ChecksEvaluationMetricType,
    checks_middleware,
    define_checks_evaluators,
)
from genkit.plugins.google_genai import GoogleAI

logger = get_logger(__name__)

PROJECT_ID = os.environ.get('GCLOUD_PROJECT', '')

ai = Genkit(
    plugins=[GoogleAI()],
)

define_checks_evaluators(
    ai,
    project_id=PROJECT_ID,
    metrics=[
        ChecksEvaluationMetricType.DANGEROUS_CONTENT,
        ChecksEvaluationMetricType.HARASSMENT,
        ChecksEvaluationMetricType.HATE_SPEECH,
    ],
)

safety_middleware = checks_middleware(
    project_id=PROJECT_ID,
    metrics=[
        ChecksEvaluationMetricType.DANGEROUS_CONTENT,
        ChecksEvaluationMetricType.HARASSMENT,
        ChecksEvaluationMetricType.HATE_SPEECH,
    ],
)


class SafeGenerateInput(BaseModel):
    """Input for safe_generate flow."""

    prompt: str = Field(
        default='Tell me a fun fact about dolphins.',
        description='The text prompt to send to the model.',
    )


@ai.flow()
async def safe_generate(input: SafeGenerateInput) -> str:
    """Generate text with Checks AI Safety middleware.

    The middleware checks both the input prompt and the model's output
    against the configured safety policies. If either violates a policy,
    the response will have ``finish_reason=FinishReason.BLOCKED``.

    Args:
        input: The input containing the text prompt.

    Returns:
        The model's response text, or a blocked message if safety
        policies were violated.
    """
    response = await ai.generate(
        model='googleai/gemini-2.0-flash',
        prompt=input.prompt,
        use=[safety_middleware],
    )
    if response.finish_reason == FinishReason.BLOCKED:
        return f'[BLOCKED] {response.finish_message}'
    return response.text


async def main() -> None:
    """Keep alive for Dev UI."""
    while True:
        await asyncio.sleep(3600)


if __name__ == '__main__':
    ai.run_main(main())
