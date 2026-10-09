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

"""Evaluator namespace module for Genkit.

Everything needed to call or write an evaluator lives here.

Example:
    ```python
    from genkit import Genkit
    from genkit.evaluator import BaseDataPoint, EvalFnResponse, Score

    ai = Genkit()


    # 1. Score one datapoint
    async def allergen_check(datapoint: BaseDataPoint, _options: object | None = None) -> EvalFnResponse:
        ok = 'peanut' not in str(datapoint.output).lower()
        return EvalFnResponse(test_case_id=datapoint.test_case_id or '', evaluation=[Score(score=ok)])


    ai.define_evaluator(
        name='allergen_check', display_name='Allergen check', definition='Flags peanuts.', fn=allergen_check
    )

    # 2. Run it over a dataset
    results = await ai.evaluate(
        evaluator='allergen_check',
        dataset=[BaseDataPoint(input='Suggest a dish.', output='Peanut noodles')],
    )
    print(results[0].evaluation[0].score)
    # => False
    ```
"""

from genkit._ai._evaluator import (
    EvaluatorRef,
    evaluator_action_metadata,
    evaluator_ref,
)
from genkit._core._typing import (
    BaseDataPoint,
    EvalFnResponse,
    EvalRequest,
    Score,
    ScoreDetails,
    ScoreStatus,
)

__all__ = [
    # Dataset rows
    'BaseDataPoint',
    # Request/Response types
    'EvalRequest',
    'EvalFnResponse',
    # Score types
    'Score',
    'ScoreDetails',
    # Status
    'ScoreStatus',
    # Factory functions
    'evaluator_ref',
    # Reference types
    'EvaluatorRef',
    # Plugin list_actions
    'evaluator_action_metadata',
]
