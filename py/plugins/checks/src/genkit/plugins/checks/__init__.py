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

"""Google Checks AI Safety plugin for Genkit.

This plugin integrates Google Checks AI Safety guardrails into Genkit,
providing both evaluators and model middleware for content safety
classification.

Key Concepts::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Guardrails          │ Safety fences around your AI. Like guardrails     │
    │                     │ on a bowling lane — they keep the ball on track.  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Policy              │ A specific safety rule (e.g., "no hate speech").  │
    │                     │ Each policy checks for one type of violation.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Threshold           │ How strict a policy is. Lower = stricter.         │
    │                     │ Like a sensitivity dial on a smoke detector.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Violation           │ When content breaks a policy rule.                │
    │                     │ "This text contains hate speech" → VIOLATIVE.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Middleware           │ A safety layer that wraps model calls. Checks    │
    │                     │ input before sending and output after receiving.  │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow (Evaluator)::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   HOW CHECKS EVALUATION WORKS                          │
    │                                                                         │
    │    Your AI Output                                                       │
    │    "Here's how to..."                                                   │
    │         │                                                               │
    │         │  (1) Send text to Checks classifyContent API                 │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Checks API     │   Classifies content against safety policies    │
    │    │  (Google Cloud) │                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Returns per-policy scores and violations             │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Evaluation     │   DANGEROUS_CONTENT: 0.1 (safe)                 │
    │    │  Results        │   HARASSMENT: 0.05 (safe)                       │
    │    │                 │   HATE_SPEECH: 0.9 (VIOLATIVE!)                 │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Data Flow (Middleware)::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  HOW CHECKS MIDDLEWARE WORKS                            │
    │                                                                         │
    │    User Request                                                         │
    │         │                                                               │
    │         │  (1) Classify input text                                     │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Input Guard    │──── Violation? → Return "blocked" response       │
    │    └────────┬────────┘                                                  │
    │             │  (2) Pass to model                                       │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  LLM Model      │   Generates response                            │
    │    └────────┬────────┘                                                  │
    │             │  (3) Classify output text                                 │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Output Guard   │──── Violation? → Return "blocked" response       │
    │    └────────┬────────┘                                                  │
    │             │  (4) Safe output                                          │
    │             ▼                                                           │
    │    Return to user                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

Example (Evaluator)::

    from genkit.ai import Genkit
    from genkit.plugins.checks import (
        ChecksEvaluationMetricType,
        define_checks_evaluators,
    )

    ai = Genkit(...)

    define_checks_evaluators(
        ai,
        project_id='my-gcp-project',
        metrics=[
            ChecksEvaluationMetricType.DANGEROUS_CONTENT,
            ChecksEvaluationMetricType.HARASSMENT,
        ],
    )

Example (Middleware)::

    from genkit.plugins.checks import checks_middleware, ChecksEvaluationMetricType

    response = await ai.generate(
        model='googleai/gemini-1.5-flash-latest',
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

Architecture::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         genkit-plugin-checks                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  __init__.py - Plugin entry point and public API exports                │
    │  ├── package_name() - Plugin identifier                                 │
    │  ├── Checks (Plugin class for Genkit(plugins=[...]))                    │
    │  ├── define_checks_evaluators() - Standalone registration function      │
    │  └── checks_middleware() - Model middleware factory                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  metrics.py - Safety policy type definitions                            │
    │  ├── ChecksEvaluationMetricType (StrEnum: DANGEROUS_CONTENT, etc.)      │
    │  ├── ChecksEvaluationMetricConfig (Pydantic: type + threshold)          │
    │  └── ChecksEvaluationMetric (union type alias)                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  guardrails.py - Checks API client                                      │
    │  ├── GuardrailsClient (ADC auth + classifyContent POST)                 │
    │  ├── PolicyResult (per-policy API response)                             │
    │  └── ClassifyContentResponse (full API response)                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  evaluation.py - Evaluator action factory                               │
    │  └── create_checks_evaluators() - Registers evaluators with registry    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  middleware.py - Model middleware                                        │
    │  ├── checks_middleware() - Creates input/output safety middleware        │
    │  └── _get_violated_policies() - Classify and filter violations          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  plugin.py - Plugin class and standalone API                             │
    │  ├── Checks (Plugin subclass with init/resolve/list_actions)             │
    │  ├── ChecksEvaluationConfig (config model for Plugin class)             │
    │  └── define_checks_evaluators() (standalone registration function)       │
    └─────────────────────────────────────────────────────────────────────────┘

See Also:
    - Google Checks: https://developers.google.com/checks
    - Checks API: https://checks.googleapis.com
    - Genkit documentation: https://genkit.dev/
"""

from genkit.plugins.checks.guardrails import GuardrailsClient
from genkit.plugins.checks.metrics import (
    ChecksEvaluationMetric,
    ChecksEvaluationMetricConfig,
    ChecksEvaluationMetricType,
)
from genkit.plugins.checks.middleware import checks_middleware
from genkit.plugins.checks.plugin import (
    Checks,
    ChecksEvaluationConfig,
    define_checks_evaluators,
)


def package_name() -> str:
    """Get the package name for the Checks plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit.plugins.checks'


__all__ = [
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
