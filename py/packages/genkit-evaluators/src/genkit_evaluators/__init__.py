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

r"""Genkit built-in evaluators: regex, deep_equal, jsonata.

Example:
    ```python
    from genkit import Genkit
    from genkit_evaluators import register_genkit_evaluators

    # 1. Initialize Genkit and register built-in evaluators
    ai = Genkit()
    register_genkit_evaluators(ai)

    # 2. Evaluate regex pattern matching on model outputs
    dataset = [{'output': 'Order #12345 confirmed.', 'testCaseId': 'tc1'}]
    results = await ai.evaluate(
        evaluator='genkit/regex',
        dataset=dataset,
        options={'pattern': r'Order #\d+'},
    )

    # 3. Inspect evaluation score
    print(results[0].score)
    # => 1.0
    ```
"""

from genkit_evaluators.plugin import genkit_eval_name, register_genkit_evaluators

__all__ = ['genkit_eval_name', 'register_genkit_evaluators']
