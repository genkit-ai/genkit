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

"""Tests for genkitEval evaluators (matching Go evaluators_test.go)."""

import pytest
from genkit_evaluators import register_genkit_evaluators

from genkit import BaseDataPoint, Genkit
from genkit.evaluator import EvalRequest, Score


def _as_score(evaluation: Score | list[Score]) -> Score:
    """These evaluators always return one Score; reject a list so .score is safe."""
    assert isinstance(evaluation, Score)
    return evaluation


@pytest.fixture
def ai() -> Genkit:
    ai = Genkit()
    register_genkit_evaluators(ai)
    return ai


@pytest.mark.asyncio
async def test_deep_equal(ai: Genkit) -> None:
    """Deep equal evaluator: output must equal reference."""
    dataset = [
        {'input': 'sample', 'reference': 'hello world', 'output': 'hello world'},
        {'input': 'sample', 'output': 'Foo bar', 'reference': 'gablorken'},
        {'input': 'sample', 'output': 'Foo bar'},
    ]
    eval_action = await ai._registry.resolve_evaluator('genkitEval/deep_equal')
    assert eval_action is not None
    req = EvalRequest(
        dataset=[BaseDataPoint.model_validate(d) for d in dataset],
        eval_run_id='testrun',
    )
    resp = await eval_action.run(input=req)
    results = resp.response.root
    assert len(results) == 3
    assert _as_score(results[0].evaluation).score is True
    assert _as_score(results[1].evaluation).score is False
    assert _as_score(results[2].evaluation).error is not None


@pytest.mark.asyncio
async def test_regex(ai: Genkit) -> None:
    """Regex evaluator: reference is regex pattern, output must match."""
    dataset = [
        {'input': 'sample', 'reference': 'ba?a?a', 'output': 'banana'},
        {'input': 'sample', 'reference': 'ba?a?a', 'output': 'apple'},
        {'input': 'sample', 'reference': 12345, 'output': 'apple'},
    ]
    eval_action = await ai._registry.resolve_evaluator('genkitEval/regex')
    assert eval_action is not None
    req = EvalRequest(
        dataset=[BaseDataPoint.model_validate(d) for d in dataset],
        eval_run_id='testrun',
    )
    resp = await eval_action.run(input=req)
    results = resp.response.root
    assert len(results) == 3
    assert _as_score(results[0].evaluation).score is True
    assert _as_score(results[1].evaluation).score is False
    assert _as_score(results[2].evaluation).error is not None


@pytest.mark.asyncio
async def test_jsonata(ai: Genkit) -> None:
    """JSONata evaluator: reference is expression, evaluated against output."""
    dataset = [
        {'input': 'sample', 'reference': 'age=33', 'output': {'name': 'Bob', 'age': 33}},
        {'input': 'sample', 'reference': 'age=31', 'output': {'name': 'Bob', 'age': 33}},
        {'input': 'sample', 'reference': 123456, 'output': {'name': 'Bob', 'age': 33}},
    ]
    eval_action = await ai._registry.resolve_evaluator('genkitEval/jsonata')
    assert eval_action is not None
    req = EvalRequest(
        dataset=[BaseDataPoint.model_validate(d) for d in dataset],
        eval_run_id='testrun',
    )
    resp = await eval_action.run(input=req)
    results = resp.response.root
    assert len(results) == 3
    first = _as_score(results[0].evaluation)
    assert first.score is not False and first.score != ''
    # age=31 with age 33 -> false or empty result -> FAIL
    second = _as_score(results[1].evaluation)
    assert second.score is False or second.status == 'FAIL'
    assert _as_score(results[2].evaluation).error is not None
