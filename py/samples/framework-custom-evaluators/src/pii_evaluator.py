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

"""PII detection evaluator using LLM-as-a-judge."""

from functools import partial

from pydantic import BaseModel

from genkit.ai import Genkit
from genkit.core.typing import BaseDataPoint, Details, EvalFnResponse, Score


class PiiDetectionResponse(BaseModel):
    """Response schema for PII detection evaluator."""

    reason: str
    verdict: bool


async def pii_detection_score(
    ai: Genkit,
    judge: str,
    datapoint: BaseDataPoint,
    _options: dict[str, object] | None = None,
    judge_config: dict[str, object] | None = None,
) -> EvalFnResponse:
    """Score a datapoint for PII presence using an LLM judge.

    Args:
        ai: Genkit instance with loaded prompts.
        judge: Model name to use as judge (e.g., 'googleai/gemini-2.0-flash').
        datapoint: The evaluation datapoint containing output to check.
        _options: (Unused) Evaluation options passed by Genkit.
        judge_config: Optional configuration for the judge model.

    Returns:
        Score with boolean verdict and reasoning.

    Raises:
        ValueError: If output is missing.
    """
    if not datapoint.output:
        raise ValueError('Output is required for PII detection')

    pii_prompt = ai.prompt('pii_detection')
    rendered = await pii_prompt.render(input={'output': str(datapoint.output)})

    response = await ai.generate(
        model=judge,
        messages=rendered.messages,
        config=judge_config,
        output={'schema': PiiDetectionResponse},
    )

    if not response.output:
        raise ValueError(f'Unable to parse evaluator response: {response.text}')

    parsed = PiiDetectionResponse.model_validate(response.output)
    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=Score(
            score=parsed.verdict,
            details=Details(reasoning=parsed.reason),
        ),
    )


def register_pii_evaluator(
    ai: Genkit,
    judge: str,
    judge_config: dict[str, object] | None = None,
) -> None:
    """Register the PII detection evaluator.

    Args:
        ai: Genkit instance to register evaluator with.
        judge: Model name to use as judge.
        judge_config: Optional configuration for the judge model.
    """
    ai.define_evaluator(
        name='byo/pii_detection',
        display_name='PII Detection',
        definition='Detects whether PII is present in the output.',
        fn=partial(pii_detection_score, ai, judge, judge_config=judge_config),
    )
