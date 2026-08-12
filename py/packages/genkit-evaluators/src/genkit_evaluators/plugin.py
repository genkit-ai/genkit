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

"""Genkit built-in evaluators: regex, deep_equal, jsonata."""

import json
import re
from typing import Any

from genkit import Genkit
from genkit._core._typing import (
    BaseDataPoint,
    EvalFnResponse,
    EvalStatusEnum,
    Score,
)

try:
    from jsonata import Jsonata
except ImportError:
    Jsonata = None  # type: ignore[misc, assignment]

PROVIDER = 'genkitEval'


def genkit_eval_name(local: str) -> str:
    """Return namespaced evaluator name."""
    return f'{PROVIDER}/{local}'


async def _regex_impl(datapoint: BaseDataPoint, _options: object | None = None) -> EvalFnResponse:
    """Regex evaluator: reference must be a regex string; output tested against it."""
    if datapoint.output is None:
        raise ValueError('output was not provided')
    if datapoint.reference is None:
        raise ValueError('reference was not provided')
    if not isinstance(datapoint.reference, str):
        raise ValueError('reference must be a string (regex)')
    output_str = datapoint.output if isinstance(datapoint.output, str) else json.dumps(datapoint.output)
    match = bool(re.search(datapoint.reference, output_str))
    status = EvalStatusEnum.PASS if match else EvalStatusEnum.FAIL
    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=Score(score=match, status=status),
    )


async def _deep_equal_impl(datapoint: BaseDataPoint, _options: object | None = None) -> EvalFnResponse:
    """Deep equal evaluator: output must equal reference."""
    if datapoint.output is None:
        raise ValueError('output was not provided')
    if datapoint.reference is None:
        raise ValueError('reference was not provided')
    equal = datapoint.output == datapoint.reference
    status = EvalStatusEnum.PASS if equal else EvalStatusEnum.FAIL
    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=Score(score=equal, status=status),
    )


async def _jsonata_impl(datapoint: BaseDataPoint, _options: object | None = None) -> EvalFnResponse:
    """JSONata evaluator: reference is a JSONata expression; evaluated against output."""
    if datapoint.output is None:
        raise ValueError('output was not provided')
    if datapoint.reference is None:
        raise ValueError('reference was not provided')
    if not isinstance(datapoint.reference, str):
        raise ValueError('reference must be a string (jsonata)')
    if Jsonata is None:
        raise RuntimeError('jsonata-python is required for jsonata evaluator')
    expr = Jsonata(datapoint.reference)
    result = expr.evaluate(datapoint.output)
    passed = result not in (False, '', None)
    status = EvalStatusEnum.PASS if passed else EvalStatusEnum.FAIL
    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=Score(score=result, status=status),
    )


def register_genkit_evaluators(ai: Genkit, metrics: list[str] | None = None) -> None:
    """Register built-in Genkit evaluators (regex, deep_equal, jsonata) on an ai instance.

        ai = Genkit(...)
        register_genkit_evaluators(ai)

    Args:
        ai: The Genkit instance to register evaluators on.
        metrics: Optional list of metric names to register. Defaults to all three
            ('regex', 'deep_equal', 'jsonata').
    """
    _all: dict[str, Any] = {
        'regex': {
            'display_name': 'RegExp',
            'definition': 'Tests output against the regexp provided as reference',
            'fn': _regex_impl,
        },
        'deep_equal': {
            'display_name': 'Deep Equals',
            'definition': 'Tests equality of output against the provided reference',
            'fn': _deep_equal_impl,
        },
        'jsonata': {
            'display_name': 'JSONata',
            'definition': 'Tests JSONata expression (provided in reference) against output',
            'fn': _jsonata_impl,
        },
    }
    selected = metrics if metrics is not None else list(_all.keys())
    for key in selected:
        cfg = _all[key]
        ai.define_evaluator(
            name=genkit_eval_name(key),
            display_name=cfg['display_name'],
            definition=cfg['definition'],
            is_billed=False,
            fn=cfg['fn'],
        )
