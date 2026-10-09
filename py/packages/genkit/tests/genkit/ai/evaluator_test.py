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

"""Plugin authors build evaluators with genkit.evaluator.evaluator and list them with evaluator_action_metadata."""

from typing import Any, cast

import pytest
from pydantic import BaseModel

from genkit import Genkit, GenkitError
from genkit.evaluator import BaseDataPoint, EvalFnResponse, EvalRequest, Score, evaluator, evaluator_action_metadata
from genkit.plugin_api import Action, ActionKind, ActionMetadata, Plugin


class ThresholdOptions(BaseModel):
    threshold: float = 0.5


async def _exact_match(datapoint: BaseDataPoint, options: object | None) -> EvalFnResponse:
    return EvalFnResponse(
        test_case_id=datapoint.test_case_id or '',
        evaluation=[Score(score=datapoint.output == datapoint.reference)],
    )


def _two_rows() -> list[BaseDataPoint]:
    return [
        BaseDataPoint(input='2+2', output='4', reference='4', test_case_id='case1'),
        BaseDataPoint(input='3+3', output='7', reference='6', test_case_id='case2'),
    ]


class ExactMatchPlugin(Plugin):
    """Plugin that hands Genkit an evaluator built with evaluator(...) on resolve."""

    name = 'p'

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        if action_type != ActionKind.EVALUATOR or name != 'e':
            return None
        return evaluator('p/e', _exact_match, display_name='Exact match', definition='output equals reference')

    async def list_actions(self) -> list[ActionMetadata]:
        return [evaluator_action_metadata('p/e', display_name='Exact match', definition='output equals reference')]


@pytest.mark.asyncio
async def test_evaluator_builds_action_without_registering() -> None:
    """evaluator('t/e', fn, ...) is an EVALUATOR action; an app with no plugin returning it can't evaluate with it."""
    ai = Genkit()

    action = evaluator('t/e', _exact_match, display_name='Exact match', definition='output equals reference')

    assert action.kind == ActionKind.EVALUATOR
    assert action.name == 't/e'
    with pytest.raises(GenkitError) as raised:
        await ai.evaluate(evaluator='t/e', dataset=_two_rows())
    assert raised.value.status == 'NOT_FOUND'


@pytest.mark.asyncio
async def test_evaluator_action_runs_fn_per_datapoint() -> None:
    """Running the built action over two rows returns two scored rows, in dataset order."""
    action = evaluator('t/e', _exact_match, display_name='Exact match', definition='output equals reference')

    result = await action.run(EvalRequest(dataset=_two_rows(), eval_run_id='run-1'))

    rows = result.response.root
    assert [row.test_case_id for row in rows] == ['case1', 'case2']
    assert [score.score for score in rows[0].evaluation] == [True]
    assert [score.score for score in rows[1].evaluation] == [False]


def test_evaluator_sync_fn_raises_type_error() -> None:
    """evaluator(...) rejects a sync fn when built, instead of failing every row at run time."""

    def sync_match(datapoint: BaseDataPoint, options: object | None) -> EvalFnResponse:
        return EvalFnResponse(test_case_id=datapoint.test_case_id or '', evaluation=[Score(score=True)])

    with pytest.raises(TypeError, match="Got sync function for 't/e'"):
        evaluator('t/e', cast(Any, sync_match), display_name='Exact match', definition='output equals reference')


def test_evaluator_action_metadata_carries_display_name_definition_billed() -> None:
    """The listing for an evaluator carries the display name, definition, billed flag and options the Dev UI shows."""
    listing = evaluator_action_metadata(
        'p/fluency',
        display_name='Fluency',
        definition='Assesses the language mastery of an output',
        is_billed=True,
        config_schema=ThresholdOptions,
    )

    assert listing.action_type == ActionKind.EVALUATOR
    assert listing.name == 'p/fluency'
    assert listing.metadata == {
        'evaluator': {
            'evaluatorDisplayName': 'Fluency',
            'evaluatorDefinition': 'Assesses the language mastery of an output',
            'evaluatorIsBilled': True,
            'label': 'p/fluency',
            'customOptions': {
                'properties': {'threshold': {'default': 0.5, 'title': 'Threshold', 'type': 'number'}},
                'title': 'ThresholdOptions',
                'type': 'object',
            },
        }
    }


@pytest.mark.parametrize('config_schema', [None, ThresholdOptions], ids=['no_options', 'with_options'])
def test_evaluator_and_metadata_agree_for_same_args(config_schema: type[BaseModel] | None) -> None:
    """The built evaluator and its listing show the Dev UI the same card for the same args."""
    action = evaluator(
        'p/e',
        _exact_match,
        display_name='Exact match',
        definition='output equals reference',
        is_billed=True,
        config_schema=config_schema,
    )
    listing = evaluator_action_metadata(
        'p/e',
        display_name='Exact match',
        definition='output equals reference',
        is_billed=True,
        config_schema=config_schema,
    )

    assert listing.metadata is not None
    assert action.metadata['evaluator'] == listing.metadata['evaluator']


@pytest.mark.asyncio
async def test_plugin_returning_evaluator_registers_it_on_app() -> None:
    """A plugin whose resolve() returns evaluator(...) makes ai.evaluate(evaluator='p/e', ...) score every row."""
    ai = Genkit(plugins=[ExactMatchPlugin()])

    results = await ai.evaluate(evaluator='p/e', dataset=_two_rows())

    assert [row.test_case_id for row in results] == ['case1', 'case2']
    assert [score.score for score in results[0].evaluation] == [True]
    assert [score.score for score in results[1].evaluation] == [False]
