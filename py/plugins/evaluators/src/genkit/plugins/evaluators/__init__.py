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


"""Evaluators Plugin for Genkit.

This plugin provides evaluation metrics for RAG (Retrieval-Augmented Generation)
applications, built as wrappers on the RAGAS framework.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Evaluation          │ Grading how well your AI answers questions.       │
    │                     │ Like a teacher checking your homework.            │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Judge Model         │ Another AI that grades the first AI's answers.    │
    │                     │ "Is this answer good?" → Score: 0.85              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Relevancy           │ Does the answer actually address the question?    │
    │                     │ Q: "What's 2+2?" A: "Blue" → Low relevancy!       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Faithfulness        │ Does the answer stick to the provided context?    │
    │                     │ Or is the AI making stuff up (hallucinating)?     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Accuracy            │ Is the answer actually correct?                   │
    │                     │ Compared against ground truth if available.       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Maliciousness       │ Is the response harmful or inappropriate?         │
    │                     │ Safety check for AI outputs.                      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ RAGAS               │ A popular framework for RAG evaluation.           │
    │                     │ We build on top of their proven metrics.          │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   HOW AI EVALUATION WORKS                               │
    │                                                                         │
    │    Your RAG System                                                      │
    │    Question: "What causes rain?"                                        │
    │    Context: [retrieved documents about weather]                         │
    │    Answer: "Rain is caused by..."                                       │
    │         │                                                               │
    │         │  (1) Send Q, A, Context to evaluator                          │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Evaluator      │   Packages data for the judge                    │
    │    │  (Genkit)       │                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Ask judge model to grade                             │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Judge Model    │   "Is this answer relevant?"                     │
    │    │  (Gemini, etc.) │   "Is it faithful to the context?"              │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) Scores returned                                      │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Evaluation     │   relevancy: 0.92                                │
    │    │  Results        │   faithfulness: 0.88                             │
    │    │                 │   accuracy: 0.95                                 │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (4) Use scores to improve your system                    │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Your App       │   Track metrics, fix low scores,                 │
    │    │                 │   improve prompts and retrieval                  │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       Evaluators Plugin                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Plugin Entry Point (__init__.py)                                       │
    │  ├── define_genkit_evaluators() - Register evaluators with Genkit       │
    │  ├── evaluators_name() - Helper for namespaced evaluator names          │
    │  ├── GenkitMetricType - Enum of available metrics                       │
    │  ├── MetricConfig - Configuration for individual metrics                │
    │  └── PluginOptions - Plugin-wide configuration                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  constant.py - Constants and Configuration Types                        │
    │  ├── GenkitMetricType (enum of metric types)                            │
    │  ├── MetricConfig (per-metric configuration)                            │
    │  └── PluginOptions (plugin configuration)                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  helpers.py - Evaluator Registration                                    │
    │  ├── define_genkit_evaluators() - Main registration function            │
    │  └── Metric implementation wrappers                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  prompts/ - Evaluation Prompts                                          │
    │  ├── answer_accuracy.prompt                                             │
    │  ├── answer_relevancy.prompt                                            │
    │  ├── faithfulness_*.prompt                                              │
    │  └── maliciousness.prompt                                               │
    └─────────────────────────────────────────────────────────────────────────┘

Supported Metrics:
    - ANSWER_RELEVANCY: How relevant is the answer to the question?
    - FAITHFULNESS: Is the answer faithful to the retrieved context?
    - ANSWER_ACCURACY: How accurate is the answer?
    - MALICIOUSNESS: Does the response contain harmful content?

Example:
    ```python
    from genkit import Genkit
    from genkit.plugins.evaluators import (
        define_genkit_evaluators,
        GenkitMetricType,
        PluginOptions,
    )

    ai = Genkit(...)

    # Register evaluators
    define_genkit_evaluators(
        ai,
        PluginOptions(
            metrics=[GenkitMetricType.ANSWER_RELEVANCY, GenkitMetricType.FAITHFULNESS],
            judge_model='googleai/gemini-2.0-flash',
        ),
    )

    # Run evaluation
    result = await ai.evaluate(
        evaluator='genkit/answer_relevancy',
        input={'question': '...', 'answer': '...', 'context': '...'},
    )
    ```

Caveats:
    - Requires a judge model to be configured
    - Some metrics require context (retrieved documents)
    - Evaluation quality depends on the judge model

See Also:
    - RAGAS: https://docs.ragas.io/
    - Genkit documentation: https://genkit.dev/
"""

from genkit.plugins.evaluators.constant import (
    GenkitMetricType,
    MetricConfig,
    PluginOptions,
)
from genkit.plugins.evaluators.helpers import define_genkit_evaluators, evaluators_name


def package_name() -> str:
    """Get the package name for the Evaluators plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit.plugins.evaluators'


__all__ = [
    'GenkitMetricType',
    'MetricConfig',
    'PluginOptions',
    'define_genkit_evaluators',
    'evaluators_name',
    'package_name',
]
