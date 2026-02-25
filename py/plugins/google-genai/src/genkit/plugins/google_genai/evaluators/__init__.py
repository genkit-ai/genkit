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

"""Vertex AI Evaluators for the Genkit framework.

This module provides evaluation metrics using the Vertex AI Evaluation API.
These evaluators assess model outputs for quality metrics like BLEU, ROUGE,
fluency, safety, groundedness, and summarization quality.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Evaluator           │ A "grader" that scores your AI's answers.         │
    │                     │ Like a teacher checking homework.                  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ BLEU Score          │ Compares AI output to a "correct" answer.         │
    │                     │ Higher = closer to the reference text.            │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ROUGE Score         │ Measures how much key info is captured.           │
    │                     │ Good for checking if summaries hit key points.    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Fluency             │ How natural and readable the text is.             │
    │                     │ Does it sound like a human wrote it?              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Safety              │ Is the content appropriate and safe?              │
    │                     │ No harmful, biased, or inappropriate content.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Groundedness        │ Does the answer stick to the facts given?         │
    │                     │ No making things up (hallucinations).             │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      EVALUATION PIPELINE                                │
    │                                                                         │
    │   Test Dataset                                                          │
    │   [input, output, reference, context]                                  │
    │        │                                                                │
    │        ▼                                                                │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │                    Vertex AI Evaluators                          │  │
    │   │                                                                   │  │
    │   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │  │
    │   │  │  BLEU   │  │  ROUGE  │  │ Fluency │  │    Groundedness     │ │  │
    │   │  │ (0.72)  │  │ (0.68)  │  │ (4/5)   │  │       (5/5)         │ │  │
    │   │  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘ │  │
    │   │                                                                   │  │
    │   │  ┌─────────┐  ┌──────────────┐  ┌───────────────────────────────┐│  │
    │   │  │ Safety  │  │ Summarization│  │   Summarization Helpfulness   ││  │
    │   │  │ (5/5)   │  │ Quality (4/5)│  │            (4/5)              ││  │
    │   │  └─────────┘  └──────────────┘  └───────────────────────────────┘│  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │        │                                                                │
    │        ▼                                                                │
    │   Evaluation Report                                                     │
    │   {"score": 0.85, "details": {"reasoning": "..."}}                     │
    └─────────────────────────────────────────────────────────────────────────┘

Overview:
    Vertex AI offers built-in evaluation metrics that use machine learning
    to score model outputs. These evaluators are useful for:

    - **Automated testing**: CI/CD quality gates for LLM outputs
    - **Model comparison**: Compare different models or prompts
    - **Quality assurance**: Catch regressions in output quality
    - **Safety checks**: Ensure outputs meet safety standards

Available Metrics:
    +-----------------------------+-------------------------------------------+
    | Metric                      | Description                               |
    +-----------------------------+-------------------------------------------+
    | BLEU                        | Compare output to reference (translation) |
    | ROUGE                       | Compare output to reference (summarization)|
    | FLUENCY                     | Assess language mastery and readability   |
    | SAFETY                      | Check for harmful/inappropriate content   |
    | GROUNDEDNESS                | Verify output is grounded in context      |
    | SUMMARIZATION_QUALITY       | Overall summarization ability             |
    | SUMMARIZATION_HELPFULNESS   | Usefulness as a summary substitute        |
    | SUMMARIZATION_VERBOSITY     | Conciseness of the summary                |
    +-----------------------------+-------------------------------------------+

Example:
    Running evaluations:

        >>> from genkit import Genkit
        >>> from genkit.plugins.google_genai import VertexAI
        >>> from genkit.plugins.google_genai.evaluators import VertexAIEvaluationMetricType
        >>>
        >>> ai = Genkit(plugins=[VertexAI(project='my-project')])
        >>>
        >>> # Prepare test dataset
        >>> dataset = [
        ...     {
        ...         'input': 'Summarize this article about AI...',
        ...         'output': 'AI is transforming industries...',
        ...         'reference': 'The article discusses how AI impacts...',
        ...         'context': ['Article content here...'],
        ...     }
        ... ]
        >>>
        >>> # Run fluency evaluation
        >>> results = await ai.evaluate(
        ...     evaluator='vertexai/fluency',
        ...     dataset=dataset,
        ... )
        >>>
        >>> for result in results:
        ...     print(f'Score: {result.evaluation.score}')
        ...     print(f'Reasoning: {result.evaluation.details.get("reasoning")}')

Caveats:
    - Requires Google Cloud project with Vertex AI API enabled
    - Evaluators are billed per API call
    - Some metrics require specific fields (e.g., GROUNDEDNESS needs context)
    - Scores are subjective assessments, not ground truth

See Also:
    - Vertex AI Evaluation API: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/evaluation
    - Genkit evaluation docs: https://genkit.dev/docs/evaluation
"""

from genkit.plugins.google_genai.evaluators.evaluation import (
    VertexAIEvaluationMetricType,
    create_vertex_evaluators,
)

__all__ = [
    'VertexAIEvaluationMetricType',
    'create_vertex_evaluators',
]
