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
    >>> from genkit import Genkit
    >>> from genkit_googleai import VertexAI
    >>>
    >>> # 1. Initialize Genkit with VertexAI plugin
    >>> ai = Genkit(plugins=[VertexAI(project='my-project', location='us-central1')])
    >>>
    >>> # 2. Prepare dataset with input and model output
    >>> dataset = [
    ...     {
    ...         'input': 'What is the capital of France?',
    ...         'output': 'Paris is the capital of France.',
    ...     }
    ... ]
    >>>
    >>> # 3. Evaluate output fluency using Vertex AI Evaluators
    >>> results = await ai.evaluate(
    ...     evaluator='vertexai/fluency',
    ...     dataset=dataset,
    ... )
    >>>
    >>> # 4. Inspect evaluation score directly
    >>> print(results[0].score)
    # => 5.0
"""

from genkit_googleai.evaluators.evaluation import (
    VertexAIEvaluationMetricType,
    create_vertex_evaluators,
)

__all__ = [
    'VertexAIEvaluationMetricType',
    'create_vertex_evaluators',
]
