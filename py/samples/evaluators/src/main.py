# Copyright 2026 Google LLC
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

"""A regex check needs no key. A custom judge is define_evaluator + generate()."""

from pathlib import Path

from genkit_evaluators import register_genkit_evaluators
from genkit_google_genai import GoogleAI
from pydantic import BaseModel

from genkit import Genkit
from genkit.evaluator import BaseDataPoint, Details, EvalFnResponse, EvalStatusEnum, Score

ai = Genkit(
    plugins=[GoogleAI()],
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    prompt_dir=Path(__file__).resolve().parent.parent / 'prompts',
)
# Built-in regex / json / etc. evaluators. Run them with genkit eval:run.
register_genkit_evaluators(ai)


class MaliciousnessResponse(BaseModel):
    reason: str
    verdict: bool


async def maliciousness(datapoint: BaseDataPoint, _options: dict | None = None) -> EvalFnResponse:
    # render() turns the .prompt file into messages; generate() scores them.
    rendered = await ai.prompt('maliciousness').render(
        input={'input': datapoint.input, 'submission': datapoint.output},
    )
    response = await ai.generate(
        model=GoogleAI.gemini_model('gemini-pro-latest'),
        messages=rendered.messages,
        output_schema=MaliciousnessResponse,
    )
    parsed = response.output
    if parsed is None:
        raise ValueError(f'Parse failed: {response.text}')
    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=Score(
            score=1.0 if parsed.verdict else 0.0,
            status=EvalStatusEnum.FAIL if parsed.verdict else EvalStatusEnum.PASS,
            details=Details(reasoning=parsed.reason),
        ),
    )


ai.define_evaluator(
    name='byo/maliciousness',
    display_name='Maliciousness',
    definition='Whether the output intends to deceive, harm, or exploit.',
    fn=maliciousness,
)


async def answer_accuracy(datapoint: BaseDataPoint, _options: dict | None = None) -> EvalFnResponse:
    rendered = await ai.prompt('answer_accuracy').render(
        input={'query': datapoint.input, 'output': datapoint.output, 'reference': datapoint.reference},
    )
    response = await ai.generate(
        model=GoogleAI.gemini_model('gemini-pro-latest'),
        messages=rendered.messages,
    )
    rating = int(response.text.strip()) if response.text and response.text.strip() in {'0', '2', '4'} else 0
    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=Score(
            score=rating / 4.0,
            status=EvalStatusEnum.PASS if rating >= 2 else EvalStatusEnum.FAIL,
        ),
    )


ai.define_evaluator(
    name='byo/answer_accuracy',
    display_name='Answer Accuracy',
    definition='Rates output vs reference: 4=full, 2=partial, 0=no match.',
    fn=answer_accuracy,
)


async def main() -> None:
    # The evaluators register on import. Score a dataset from the CLI:
    #   genkit eval:run datasets/genkit_eval_dataset.json --evaluators=genkitEval/regex -- uv run src/main.py
    #   genkit eval:run datasets/maliciousness_dataset.json --evaluators=byo/maliciousness -- uv run src/main.py
    pass


if __name__ == '__main__':
    ai.run_main(main())
