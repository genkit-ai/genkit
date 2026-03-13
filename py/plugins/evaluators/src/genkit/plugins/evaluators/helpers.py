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


"""Helper functions for Genkit evaluators."""

import json
import os
import pathlib
import re
from collections.abc import Callable
from typing import Any, cast

import jsonata

from genkit.ai import Genkit, Output
from genkit.core.typing import BaseDataPoint, EvalFnResponse, EvalStatusEnum, Score
from genkit.plugins.evaluators.constant import (
    AnswerRelevancyResponseSchema,
    GenkitMetricType,
    LongFormResponseSchema,
    MaliciousnessResponseSchema,
    MetricConfig,
    NliResponse,
    PluginOptions,
)
from genkit.plugins.metrics.helper import load_prompt_file, render_text


def _get_prompt_path(filename: str) -> str:
    """Get absolute path to a prompt file in the prompts directory."""
    plugin_dir = pathlib.Path(pathlib.Path(__file__).resolve()).parent
    return os.path.join(plugin_dir, '..', '..', '..', '..', 'prompts', filename)


def evaluators_name(name: str) -> str:
    """Create an evaluators plugin name.

    Args:
        name: Name for the evaluator.

    Returns:
        The fully qualified genkitEval action name.
    """
    return f'genkitEval/{name}'


def fill_scores(
    datapoint: BaseDataPoint,
    score: Score,
    status_override_fn: Callable[[Score], EvalStatusEnum] | None = None,
) -> EvalFnResponse:
    """Adds status overrides if provided."""
    status = score.status
    if status_override_fn is not None:
        status = status_override_fn(score)
    score.status = status
    return EvalFnResponse(test_case_id=datapoint.test_case_id or '', evaluation=score)


def define_genkit_evaluators(ai: Genkit, params: PluginOptions | list[MetricConfig]) -> None:
    """Register Genkit evaluators on the provided Genkit instance."""
    if isinstance(params, list):
        params = PluginOptions(root=params)
    for param in params.root:
        _configure_evaluator(ai=ai, param=param)


def _configure_evaluator(ai: Genkit, param: MetricConfig) -> None:
    """Validates and configures supported evaluators."""
    metric_type = param.metric_type
    match metric_type:
        case GenkitMetricType.ANSWER_RELEVANCY:

            async def _relevancy_eval(datapoint: BaseDataPoint, options: object | None) -> EvalFnResponse:
                assert param.judge is not None, 'judge is required for ANSWER_RELEVANCY metric'
                assert datapoint.output is not None, 'output is required'
                output_string = datapoint.output if isinstance(datapoint.output, str) else json.dumps(datapoint.output)
                input_string = datapoint.input if isinstance(datapoint.input, str) else json.dumps(datapoint.input)
                prompt_function = await load_prompt_file(_get_prompt_path('answer_relevancy.prompt'))
                context = ' '.join(json.dumps(e) for e in (datapoint.context or []))
                prompt = await render_text(
                    prompt_function, {'question': input_string, 'answer': output_string, 'context': context}
                )

                response = await ai.generate(
                    model=param.judge.name,
                    prompt=prompt,
                    config=param.judge_config,
                    output=Output(schema=AnswerRelevancyResponseSchema),
                )
                # TODO(#4358): embedding comparison between the input and the result of the llm
                answered = False
                if response.output and hasattr(response.output, 'answered'):
                    answered = bool(response.output.answered)
                status = EvalStatusEnum.PASS_ if answered else EvalStatusEnum.FAIL
                return fill_scores(
                    datapoint,
                    Score(score=answered, status=status),
                    param.status_override_fn,
                )

            ai.define_evaluator(
                name=evaluators_name(str(GenkitMetricType.ANSWER_RELEVANCY).lower()),
                display_name='Answer Relevancy',
                definition='Assesses how pertinent the generated answer is to the given prompt',
                fn=_relevancy_eval,
            )
        case GenkitMetricType.FAITHFULNESS:
            faithfulness_prompts = {}

            async def _faithfulness_eval(datapoint: BaseDataPoint, options: object | None) -> EvalFnResponse:
                assert param.judge is not None, 'judge is required for FAITHFULNESS metric'
                assert datapoint.output is not None, 'output is required'
                output_string = datapoint.output if isinstance(datapoint.output, str) else json.dumps(datapoint.output)
                input_string = datapoint.input if isinstance(datapoint.input, str) else json.dumps(datapoint.input)
                context_list = [(json.dumps(e) if not isinstance(e, str) else e) for e in (datapoint.context or [])]

                if 'longform' not in faithfulness_prompts:
                    faithfulness_prompts['longform'] = await load_prompt_file(
                        _get_prompt_path('faithfulness_long_form.prompt')
                    )
                if 'nli' not in faithfulness_prompts:
                    faithfulness_prompts['nli'] = await load_prompt_file(_get_prompt_path('faithfulness_nli.prompt'))

                prompt = await render_text(
                    faithfulness_prompts['longform'], {'question': input_string, 'answer': output_string}
                )
                longform_response = await ai.generate(
                    model=param.judge.name,
                    prompt=prompt,
                    config=param.judge_config,
                    output=Output(schema=LongFormResponseSchema),
                )
                statements: list[str] = []
                if isinstance(longform_response.output, dict):
                    statements = longform_response.output.get('statements', [])
                elif longform_response.output and hasattr(longform_response.output, 'statements'):
                    statements = longform_response.output.statements
                if not statements:
                    raise ValueError('No statements returned')

                all_statements = '\n'.join([f'statement: {s}' for s in statements])
                all_context = '\n'.join(context_list)
                prompt = await render_text(
                    faithfulness_prompts['nli'], {'context': all_context, 'statements': all_statements}
                )

                nli_response = await ai.generate(
                    model=param.judge.name,
                    prompt=prompt,
                    config=param.judge_config,
                    output=Output(schema=NliResponse),
                )

                nli_output = nli_response.output
                responses: list[object] = []
                if isinstance(nli_output, dict):
                    nli_dict = cast(dict[str, Any], nli_output)
                    raw_resp = nli_dict.get('responses')
                    responses = raw_resp if isinstance(raw_resp, list) else []
                elif nli_output and hasattr(nli_output, 'responses'):
                    responses = nli_output.responses

                if not responses:
                    raise ValueError('Evaluator response empty')

                def _get_verdict(r: object) -> bool:
                    if isinstance(r, dict):
                        r_dict = cast(dict[str, Any], r)
                        return bool(r_dict.get('verdict'))
                    return bool(getattr(r, 'verdict', False))

                def _get_reason(r: object) -> str:
                    if isinstance(r, dict):
                        r_dict = cast(dict[str, Any], r)
                        return str(r_dict.get('reason', ''))
                    return str(getattr(r, 'reason', ''))

                faithful_count = sum(1 for r in responses if _get_verdict(r))
                score_val = faithful_count / len(responses)
                reasoning = '; '.join([_get_reason(r) for r in responses])
                status = EvalStatusEnum.PASS_ if score_val > 0.5 else EvalStatusEnum.FAIL

                return fill_scores(
                    datapoint,
                    Score(score=score_val, status=status, details={'reasoning': reasoning}),  # type: ignore[arg-type]
                    param.status_override_fn,
                )

            ai.define_evaluator(
                name=evaluators_name(str(GenkitMetricType.FAITHFULNESS).lower()),
                display_name='Faithfulness',
                definition='Measures the factual consistency of the generated answer against the given context',
                fn=_faithfulness_eval,
            )

        case GenkitMetricType.MALICIOUSNESS:

            async def _maliciousness_eval(datapoint: BaseDataPoint, options: object | None) -> EvalFnResponse:
                assert param.judge is not None, 'judge is required for MALICIOUSNESS metric'
                assert datapoint.output is not None, 'output is required'
                output_string = datapoint.output if isinstance(datapoint.output, str) else json.dumps(datapoint.output)
                input_string = datapoint.input if isinstance(datapoint.input, str) else json.dumps(datapoint.input)
                prompt_function = await load_prompt_file(_get_prompt_path('maliciousness.prompt'))
                context = ' '.join(json.dumps(e) for e in (datapoint.context or []))
                prompt = await render_text(
                    prompt_function, {'input': input_string, 'output': output_string, 'context': context}
                )

                response = await ai.generate(
                    model=param.judge.name,
                    prompt=prompt,
                    config=param.judge_config,
                    output=Output(schema=MaliciousnessResponseSchema),
                )
                is_malicious = bool(
                    response.output.malicious if response.output and hasattr(response.output, 'malicious') else False
                )
                status = EvalStatusEnum.FAIL if is_malicious else EvalStatusEnum.PASS_
                return fill_scores(datapoint, Score(score=is_malicious, status=status), param.status_override_fn)

            ai.define_evaluator(
                name=evaluators_name(str(GenkitMetricType.MALICIOUSNESS).lower()),
                display_name='Maliciousness',
                definition='Measures whether the generated output intends to deceive, harm, or exploit',
                fn=_maliciousness_eval,
            )
        case GenkitMetricType.REGEX:

            async def _regex_eval(datapoint: BaseDataPoint, options: object | None) -> EvalFnResponse:
                assert datapoint.output is not None, 'output is required'
                assert datapoint.reference is not None, 'reference is required'
                assert isinstance(datapoint.reference, str), 'reference must be of string (regex)'
                output_string = datapoint.output if isinstance(datapoint.output, str) else json.dumps(datapoint.output)
                pattern = re.compile(datapoint.reference)
                score = pattern.search(output_string) is not None
                status = EvalStatusEnum.PASS_ if score else EvalStatusEnum.FAIL
                return fill_scores(datapoint, Score(score=score, status=status), param.status_override_fn)

            ai.define_evaluator(
                name=evaluators_name(str(GenkitMetricType.REGEX).lower()),
                display_name='RegExp',
                definition='Tests output against the regexp provided as reference',
                fn=_regex_eval,
            )

        case GenkitMetricType.DEEP_EQUAL:

            async def _deep_equal_eval(datapoint: BaseDataPoint, options: object | None) -> EvalFnResponse:
                assert datapoint.reference is not None, 'reference is required'
                assert datapoint.output is not None, 'output is required'
                score = False
                if type(datapoint.output) is type(datapoint.reference):
                    if datapoint.output == datapoint.reference:
                        score = True
                status = EvalStatusEnum.PASS_ if score else EvalStatusEnum.FAIL
                return fill_scores(datapoint, Score(score=score, status=status), param.status_override_fn)

            ai.define_evaluator(
                name=evaluators_name(str(GenkitMetricType.DEEP_EQUAL).lower()),
                display_name='Deep Equals',
                definition="""Tests equality of output against the provided reference""",
                fn=_deep_equal_eval,
            )

        case GenkitMetricType.JSONATA:

            async def _jsonata_eval(datapoint: BaseDataPoint, options: object | None) -> EvalFnResponse:
                assert datapoint.output is not None, 'output is required'
                assert datapoint.reference is not None, 'reference is required'
                assert isinstance(datapoint.reference, str), 'reference must be of string (jsonata)'
                expr = jsonata.Jsonata(datapoint.reference)
                score = expr.evaluate(datapoint.output)
                status = EvalStatusEnum.PASS_ if bool(score) else EvalStatusEnum.FAIL
                return fill_scores(datapoint, Score(score=score, status=status), param.status_override_fn)

            ai.define_evaluator(
                name=evaluators_name(str(GenkitMetricType.JSONATA).lower()),
                display_name='JSONata',
                definition="""Tests JSONata expression (provided in reference) against output""",
                fn=_jsonata_eval,
            )
