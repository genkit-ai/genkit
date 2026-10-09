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

"""One config pipeline for models, embedders and evaluators.

The definition sets the config class (annotation and/or config_schema) and
its defaults. Callers layer on the values they set. The fn sees the same
input from ai.*, a ref, or a Dev UI-shaped Action.run.
"""

from typing import Any, cast

import pytest
from pydantic import BaseModel

from genkit import Genkit, GenkitError
from genkit._core._action import ActionRunContext
from genkit._core._model import EmbedRequest, EvalRequest, Message, ModelRequest, ModelResponse, Part
from genkit._core._typing import (
    BaseDataPoint,
    Embedding,
    EmbedResponse,
    EvalFnResponse,
    EvalResponse,
    Operation,
    Role,
    Score,
)


class TableConfig(BaseModel):
    """A restaurant model's settings, with real defaults."""

    temperature: float = 0.7
    allergens: list[str] = ['peanut']


class CrmEmbedConfig(BaseModel):
    """A CRM search embedder's settings."""

    dimensions: int = 768
    task_type: str | None = None


class AllergyJudgeConfig(BaseModel):
    """An allergy-check evaluator's settings."""

    strict: bool = True
    allergens: list[str] = ['peanut']


class MenuOnlyConfig(BaseModel):
    """A class no action here declares."""

    course: str | None = None


ROWS = [BaseDataPoint(input='satay', output='contains peanut', test_case_id='row-1')]


def _ok() -> ModelResponse:
    return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))


def _score() -> EvalFnResponse:
    return EvalFnResponse(test_case_id='row-1', evaluation=[Score(score=1)])


def _embedding() -> EmbedResponse:
    return EmbedResponse(embeddings=[Embedding(embedding=[0.1])])


# -----------------------------------------------------------------------------
# Definition: one config class per action
# -----------------------------------------------------------------------------


def test_model_annotation_and_config_schema_must_agree() -> None:
    ai = Genkit()

    async def bistro(request: ModelRequest[TableConfig], ctx: ActionRunContext) -> ModelResponse:
        return _ok()

    with pytest.raises(GenkitError, match=r"model 'bistro': config_schema is .*MenuOnlyConfig, but the function"):
        ai.define_model(name='bistro', fn=bistro, config_schema=MenuOnlyConfig)


def test_background_model_annotation_and_config_schema_must_agree() -> None:
    ai = Genkit()

    async def start(request: ModelRequest[TableConfig], ctx: ActionRunContext) -> Operation:
        raise NotImplementedError

    async def check(operation: Operation, ctx: ActionRunContext) -> Operation:
        raise NotImplementedError

    with pytest.raises(GenkitError, match='config_schema is .*MenuOnlyConfig'):
        ai.define_background_model(name='video', start=start, check=check, config_schema=MenuOnlyConfig)


def test_embedder_annotation_and_config_schema_must_agree() -> None:
    ai = Genkit()

    async def crm_search(request: EmbedRequest[CrmEmbedConfig]) -> EmbedResponse:
        return _embedding()

    with pytest.raises(GenkitError, match=r"embedder 'crm': config_schema is .*MenuOnlyConfig, but the function"):
        ai.define_embedder('crm', crm_search, config_schema=MenuOnlyConfig)


def test_per_row_evaluator_annotation_and_config_schema_must_agree() -> None:
    ai = Genkit()

    async def allergy_check(datapoint: BaseDataPoint, options: AllergyJudgeConfig) -> EvalFnResponse:
        return _score()

    with pytest.raises(GenkitError, match=r"evaluator 'allergy': config_schema is .*MenuOnlyConfig"):
        ai.define_evaluator(
            name='allergy',
            display_name='Allergy',
            definition='Flags allergens.',
            fn=allergy_check,
            config_schema=MenuOnlyConfig,
        )


def test_batch_evaluator_annotation_and_config_schema_must_agree() -> None:
    ai = Genkit()

    async def allergy_batch(request: EvalRequest[AllergyJudgeConfig]) -> list[EvalFnResponse]:
        return [_score()]

    with pytest.raises(GenkitError, match=r"evaluator 'allergy': config_schema is .*MenuOnlyConfig"):
        ai.define_batch_evaluator(
            name='allergy',
            display_name='Allergy',
            definition='Flags allergens.',
            fn=allergy_batch,
            config_schema=MenuOnlyConfig,
        )


def test_same_class_in_annotation_and_config_schema_is_fine() -> None:
    ai = Genkit()

    async def bistro(request: ModelRequest[TableConfig], ctx: ActionRunContext) -> ModelResponse:
        return _ok()

    action = ai.define_model(name='bistro', fn=bistro, config_schema=TableConfig)

    assert action.config_schema is TableConfig


def test_annotation_alone_sets_config_schema_and_dev_ui_form() -> None:
    ai = Genkit()

    async def bistro(request: ModelRequest[TableConfig], ctx: ActionRunContext) -> ModelResponse:
        return _ok()

    async def crm_search(request: EmbedRequest[CrmEmbedConfig]) -> EmbedResponse:
        return _embedding()

    async def allergy_check(datapoint: BaseDataPoint, options: AllergyJudgeConfig) -> EvalFnResponse:
        return _score()

    async def allergy_batch(request: EvalRequest[AllergyJudgeConfig]) -> list[EvalFnResponse]:
        return [_score()]

    model = ai.define_model(name='bistro', fn=bistro)
    embedder = ai.define_embedder('crm', crm_search)
    per_row = ai.define_evaluator(name='row', display_name='Row', definition='d', fn=allergy_check)
    batch = ai.define_batch_evaluator(name='batch', display_name='Batch', definition='d', fn=allergy_batch)

    assert model.config_schema is TableConfig
    assert embedder.config_schema is CrmEmbedConfig
    assert per_row.config_schema is AllergyJudgeConfig
    assert batch.config_schema is AllergyJudgeConfig
    model_meta = cast(dict[str, Any], model.metadata['model'])
    embedder_meta = cast(dict[str, Any], embedder.metadata['embedder'])
    per_row_meta = cast(dict[str, Any], per_row.metadata['evaluator'])
    batch_meta = cast(dict[str, Any], batch.metadata['evaluator'])
    assert model_meta['customOptions']['properties']['temperature']['default'] == 0.7
    assert embedder_meta['customOptions']['properties']['dimensions']['default'] == 768
    assert per_row_meta['customOptions']['properties']['strict']['default'] is True
    assert batch_meta['customOptions']['properties']['strict']['default'] is True


def test_embedder_config_schema_sets_the_class_and_form() -> None:
    ai = Genkit()

    async def crm_search(request: EmbedRequest) -> EmbedResponse:
        return _embedding()

    action = ai.define_embedder('crm', crm_search, config_schema=CrmEmbedConfig)

    assert action.config_schema is CrmEmbedConfig
    embedder_meta = cast(dict[str, Any], action.metadata['embedder'])
    assert embedder_meta['customOptions']['properties']['dimensions']['default'] == 768


@pytest.mark.asyncio
async def test_unannotated_embedder_gets_a_request_from_a_raw_run() -> None:
    ai = Genkit()
    seen: dict[str, Any] = {}

    async def crm_search(request):  # noqa: ANN001, ANN202
        seen['request'] = request
        return _embedding()

    action = ai.define_embedder('crm', crm_search)
    await action.run({'input': [{'content': [{'text': 'acme corp'}]}], 'options': None})

    assert isinstance(seen['request'], EmbedRequest)
    assert seen['request'].options == {}
    assert seen['request'].input[0].text == 'acme corp'


@pytest.mark.asyncio
async def test_unannotated_batch_evaluator_gets_a_request_from_a_raw_run() -> None:
    ai = Genkit()
    seen: dict[str, Any] = {}

    async def allergy_batch(request):  # noqa: ANN001, ANN202
        seen['request'] = request
        return [_score()]

    action = ai.define_batch_evaluator(name='allergy', display_name='Allergy', definition='d', fn=allergy_batch)
    response = await action.run({'dataset': [{'input': 'satay'}], 'evalRunId': 'run-1', 'options': None})

    assert isinstance(seen['request'], EvalRequest)
    assert seen['request'].options == {}
    assert isinstance(response.response, EvalResponse)


@pytest.mark.asyncio
async def test_typed_per_row_evaluator_gets_its_options_class() -> None:
    ai = Genkit()
    seen: dict[str, Any] = {}

    async def allergy_check(datapoint: BaseDataPoint, options: AllergyJudgeConfig) -> EvalFnResponse:
        seen['options'] = options
        return _score()

    action = ai.define_evaluator(name='allergy', display_name='Allergy', definition='d', fn=allergy_check)
    await action.run({'dataset': [{'input': 'satay'}], 'evalRunId': 'run-1', 'options': {'strict': False}})

    assert seen['options'] == AllergyJudgeConfig(strict=False, allergens=['peanut'])
