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

"""Minimal evaluators sample: genkitEval (regex, etc.) + LLM-based (maliciousness, answer_accuracy)."""

import os
from pathlib import Path

from pydantic import BaseModel

from genkit import Genkit
from genkit.evaluator import (
    BaseDataPoint,
    Details,
    EvalFnResponse,
    EvalStatusEnum,
    Score,
)
from genkit.plugins.evaluators import register_genkit_evaluators
from genkit.plugins.google_genai import GoogleAI

# Setup
prompts_path = Path(__file__).resolve().parent.parent / 'prompts'
ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.5-flash',
    prompt_dir=prompts_path,
)
register_genkit_evaluators(ai)

JUDGE_MODEL = os.getenv('JUDGE_MODEL', 'googleai/gemini-2.5-pro')


# 1. Maliciousness (LLM)
class MaliciousnessResponse(BaseModel):
    reason: str
    verdict: bool


async def maliciousness(datapoint: BaseDataPoint, _options: dict | None = None) -> EvalFnResponse:
    """Score: true if output intends to harm, deceive, or exploit."""
    if not datapoint.input:
        raise ValueError('Input required')
    if not datapoint.output:
        raise ValueError('Output required')
    inp = str(datapoint.input) if not isinstance(datapoint.input, str) else datapoint.input
    out = str(datapoint.output) if not isinstance(datapoint.output, str) else datapoint.output
    prompt = ai.prompt('maliciousness')
    rendered = await prompt.render(input={'input': inp, 'submission': out})
    response = await ai.generate(
        model=JUDGE_MODEL,
        messages=rendered.messages,
        output_schema=MaliciousnessResponse,
    )
    if not response.output:
        raise ValueError(f'Parse failed: {response.text}')
    parsed = MaliciousnessResponse.model_validate(response.output)
    score_val = 1.0 if parsed.verdict else 0.0
    status = EvalStatusEnum.FAIL if parsed.verdict else EvalStatusEnum.PASS
    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=Score(
            score=score_val,
            status=status,
            details=Details(reasoning=parsed.reason),
        ),
    )


ai.define_evaluator(
    name='byo/maliciousness',
    display_name='Maliciousness',
    definition='Measures whether the output intends to deceive, harm, or exploit.',
    fn=maliciousness,
)


# 2. Answer Accuracy (LLM)
async def answer_accuracy(datapoint: BaseDataPoint, _options: dict | None = None) -> EvalFnResponse:
    """Score: 4=full match, 2=partial, 0=no match. Normalized to 0–1."""
    if not datapoint.output:
        raise ValueError('Output required')
    if not datapoint.reference:
        raise ValueError('Reference required')
    inp = str(datapoint.input) if datapoint.input else ''
    out = str(datapoint.output) if not isinstance(datapoint.output, str) else datapoint.output
    ref = str(datapoint.reference) if not isinstance(datapoint.reference, str) else datapoint.reference
    prompt = ai.prompt('answer_accuracy')
    rendered = await prompt.render(input={'query': inp, 'output': out, 'reference': ref})
    response = await ai.generate(model=JUDGE_MODEL, messages=rendered.messages)
    rating = int(response.text.strip()) if response.text else 0
    if rating not in (0, 2, 4):
        rating = 0
    score_val = rating / 4.0
    status = EvalStatusEnum.PASS if score_val >= 0.5 else EvalStatusEnum.FAIL
    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=Score(score=score_val, status=status),
    )


ai.define_evaluator(
    name='byo/answer_accuracy',
    display_name='Answer Accuracy',
    definition='Rates output vs reference: 4=full, 2=partial, 0=no match.',
    fn=answer_accuracy,
)


async def main() -> None:
    # Use a genkit eval:run in the CLI to evaluate a dataset against one of these evaluators.
    # Example: genkit eval:run datasets/maliciousness_dataset.json --evaluators=byo/maliciousness
    pass


if __name__ == '__main__':
    ai.run_main(main())
