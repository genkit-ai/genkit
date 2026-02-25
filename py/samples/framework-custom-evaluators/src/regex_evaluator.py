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

"""Regex-based evaluator factory.

This module provides a factory pattern for creating regex-based evaluators
that match patterns in output text without using an LLM.
"""

import re
from collections.abc import Callable, Coroutine
from re import Pattern
from typing import Any, cast

from genkit.ai import Genkit
from genkit.core.typing import BaseDataPoint, Details, EvalFnResponse, Score


def regex_matcher(suffix: str, pattern: Pattern[str]) -> dict[str, object]:
    """Create a regex matcher configuration.

    Args:
        suffix: Suffix for the evaluator name (e.g., 'url', 'us_phone').
        pattern: Compiled regex pattern to match against.

    Returns:
        Configuration dict with name and regex pattern.
    """
    return {
        'name': f'regex_match_{suffix}',
        'regex': pattern,
    }


async def regex_match_score(datapoint: BaseDataPoint, regex: Pattern[str]) -> EvalFnResponse:
    """Score a datapoint using regex matching.

    Args:
        datapoint: The evaluation datapoint containing output to check.
        regex: The regex pattern to match against.

    Returns:
        Score with boolean match result and reasoning.

    Raises:
        ValueError: If output is missing or not a string.
    """
    if not datapoint.output or not isinstance(datapoint.output, str):
        raise ValueError('String output is required for regex matching')

    matches = bool(regex.search(datapoint.output))
    reasoning = f'Output matched regex {regex.pattern}' if matches else f'Output did not match regex {regex.pattern}'

    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=Score(
            score=matches,
            details=Details(reasoning=reasoning),
        ),
    )


def _regex_eval_fn_factory(
    regex_pattern: re.Pattern[str],
) -> Callable[[BaseDataPoint, dict[str, Any] | None], Coroutine[Any, Any, EvalFnResponse]]:
    """Factory to create a callable for regex evaluators."""

    async def _eval_fn(datapoint: BaseDataPoint, options: dict[str, Any] | None = None) -> EvalFnResponse:
        return await regex_match_score(datapoint, regex_pattern)

    return _eval_fn


def register_regex_evaluators(ai: Genkit, patterns: list[dict[str, Any]]) -> None:
    """Register regex-based evaluators with Genkit.

    Args:
        ai: Genkit instance to register evaluators with.
        patterns: List of pattern configurations from regex_matcher().
    """
    for pattern_config in patterns:
        name = str(pattern_config['name'])
        regex = cast(re.Pattern[str], pattern_config['regex'])
        if not isinstance(regex, re.Pattern):
            continue

        ai.define_evaluator(
            name=f'byo/{name}',
            display_name=f'Regex Match ({name.split("_")[-1]})',
            definition='Runs the output against a regex and responds with 1 if a match is found and 0 otherwise.',
            is_billed=False,
            fn=_regex_eval_fn_factory(regex),
        )
