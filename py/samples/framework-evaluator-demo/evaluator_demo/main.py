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


"""Evaluator demo - Custom evaluation metrics for AI outputs.

This sample demonstrates how to define custom evaluators in Genkit for
assessing AI-generated outputs against quality metrics.

See README.md for testing instructions.

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Custom Evaluator Definition             | `ai.define_evaluator()`             |
| Evaluation Logic                        | `random_eval`                       |
| Evaluation Response Structure           | `EvalFnResponse`, `Score`           |
| PDF RAG evaluation                      | `pdf_rag` module                    |
"""

import argparse
import asyncio
import random
from collections.abc import Coroutine
from typing import Any, cast

from samples.shared.logging import setup_sample

setup_sample()

# Import flows so they get registered
from evaluator_demo import (
    eval_in_code,  # noqa: F401
    pdf_rag,  # noqa: F401
    setup as setup_module,  # noqa: F401
)
from evaluator_demo.genkit_demo import ai
from genkit.core.typing import BaseDataPoint, Details, EvalFnResponse, EvalStatusEnum, Score


# Test evaluator that generates random scores and randomly fails
async def random_eval(datapoint: BaseDataPoint, _options: dict[str, object] | None = None) -> EvalFnResponse:
    """Evaluate a datapoint with random results.

    Args:
        datapoint: The datapoint to evaluate.
        options: Optional configuration.

    Returns:
        The evaluation response.
    """
    score = random.random()
    # Throw if score is 0.5x (10% prob.)
    if 0.5 <= score < 0.6:
        raise ValueError('Simulated error')

    # PASS if score > 0.5, else FAIL
    status = EvalStatusEnum.FAIL if score < 0.5 else EvalStatusEnum.PASS_
    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=Score(
            score=score,
            status=status,
            details=Details(reasoning='Randomly failed' if status == EvalStatusEnum.FAIL else 'Randomly passed'),
        ),
    )


ai.define_evaluator(
    name='custom/test_evaluator',
    display_name='TEST - Random Eval',
    definition='Randomly generates scores, for testing Evals UI only',
    fn=random_eval,
)


async def main() -> None:
    """Keep alive for Dev UI."""
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluator Demo')
    parser.add_argument('--setup', action='store_true', help='Perform initial setup (indexing docs)')
    args = parser.parse_args()

    if args.setup:
        from evaluator_demo.setup import setup as run_setup

        ai.run_main(cast(Coroutine[Any, Any, object], run_setup()))
    else:
        ai.run_main(main())
