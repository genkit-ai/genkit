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

Example:
    ```python
    from genkit import Genkit
    from genkit.evaluator import BaseDataPoint
    from genkit_google_genai import VertexAI

    ai = Genkit(plugins=[VertexAI(project='my-project')])

    dataset = [
        BaseDataPoint(
            input='Write about AI.',
            output='AI is transforming industries through intelligent automation.',
        ),
    ]

    results = await ai.evaluate(
        evaluator='vertexai/fluency',
        dataset=dataset,
    )

    for result in results.root:
        print(f'Score: {result.evaluation.score}')
    ```
"""

from genkit_google_genai.evaluators.evaluation import (
    VertexAIEvaluationMetricType,
    create_vertex_evaluators,
)

__all__ = [
    'VertexAIEvaluationMetricType',
    'create_vertex_evaluators',
]
