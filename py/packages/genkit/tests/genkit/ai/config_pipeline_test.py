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

import dataclasses
from typing import Any, cast

import pytest
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit import Genkit, GenkitError
from genkit._ai._embedding import EmbedderRef, create_embedder_ref
from genkit._ai._evaluator import EvaluatorRef, evaluator_ref
from genkit._ai._model import model_ref
from genkit._core._action import ActionRunContext
from genkit._core._model import EmbedRequest, EvalRequest, Message, ModelConfig, ModelRequest, ModelResponse, Part
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


# -----------------------------------------------------------------------------
# Action boundary: the definition's non-None defaults
# -----------------------------------------------------------------------------


class NotSetConfig(BaseModel):
    """Every field defaults to None, like every built-in plugin config."""

    temperature: float | None = None
    voice: str | None = None


def _pipeline_app(config_schema: type[BaseModel] = TableConfig) -> tuple[Genkit, dict[str, Any]]:
    """Untyped model, embedder, per-row and batch evaluator, all with config_schema."""
    ai = Genkit()
    seen: dict[str, Any] = {}

    async def bistro(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        seen['model'] = request.config
        return _ok()

    async def crm_search(request: EmbedRequest) -> EmbedResponse:
        seen['embedder'] = request.options
        return _embedding()

    async def allergy_check(datapoint: BaseDataPoint, options: dict[str, Any]) -> EvalFnResponse:
        seen['per_row'] = options
        return _score()

    async def allergy_batch(request: EvalRequest) -> list[EvalFnResponse]:
        seen['batch'] = request.options
        return [_score()]

    ai.define_model(name='bistro', fn=bistro, config_schema=config_schema)
    ai.define_embedder('crm', crm_search, config_schema=config_schema)
    ai.define_evaluator(name='row', display_name='Row', definition='d', fn=allergy_check, config_schema=config_schema)
    ai.define_batch_evaluator(
        name='batch', display_name='Batch', definition='d', fn=allergy_batch, config_schema=config_schema
    )
    return ai, seen


TABLE_DEFAULTS = {'temperature': 0.7, 'allergens': ['peanut']}


@pytest.mark.asyncio
async def test_untyped_fns_see_schema_defaults_from_ai_calls() -> None:
    ai, seen = _pipeline_app()

    await ai.generate(model='bistro', prompt='a table for two')
    await ai.embed(embedder='crm', content='acme corp')
    await ai.evaluate(evaluator='row', dataset=ROWS)
    await ai.evaluate(evaluator='batch', dataset=ROWS)

    assert seen == {
        'model': TABLE_DEFAULTS,
        'embedder': TABLE_DEFAULTS,
        'per_row': TABLE_DEFAULTS,
        'batch': TABLE_DEFAULTS,
    }


@pytest.mark.asyncio
async def test_untyped_fns_see_schema_defaults_from_a_dev_ui_run() -> None:
    ai, seen = _pipeline_app()
    eval_payload = {'dataset': [{'input': 'satay', 'testCaseId': 'row-1'}], 'evalRunId': 'run-1', 'options': None}

    for kind, name, payload in [
        ('model', 'bistro', {'messages': [], 'config': None}),
        ('embedder', 'crm', {'input': [], 'options': None}),
        ('evaluator', 'row', eval_payload),
        ('evaluator', 'batch', eval_payload),
    ]:
        action = await ai.registry.resolve_action_by_key(f'/{kind}/{name}')
        assert action is not None
        await action.run(payload)

    assert seen == {
        'model': TABLE_DEFAULTS,
        'embedder': TABLE_DEFAULTS,
        'per_row': TABLE_DEFAULTS,
        'batch': TABLE_DEFAULTS,
    }


@pytest.mark.asyncio
async def test_dev_ui_run_overlays_set_fields_and_drops_none() -> None:
    """An explicit None falls back to the definition's default, the same as through ai.*."""
    ai, seen = _pipeline_app()
    action = await ai.registry.resolve_action_by_key('/model/bistro')
    assert action is not None

    await action.run({'messages': [], 'config': {'allergens': ['shellfish'], 'temperature': None, 'seat': 'patio'}})

    assert seen['model'] == {'temperature': 0.7, 'allergens': ['shellfish'], 'seat': 'patio'}


@pytest.mark.asyncio
async def test_none_defaults_are_not_injected() -> None:
    ai, seen = _pipeline_app(config_schema=NotSetConfig)
    action = await ai.registry.resolve_action_by_key('/model/bistro')
    assert action is not None

    await action.run({'messages': [], 'config': None})
    await ai.embed(embedder='crm', content='acme corp')

    assert seen['model'] is None
    assert seen['embedder'] == {}


@pytest.mark.asyncio
async def test_typed_fns_see_the_class_with_defaults() -> None:
    ai = Genkit()
    seen: dict[str, Any] = {}

    async def bistro(request: ModelRequest[TableConfig], ctx: ActionRunContext) -> ModelResponse:
        seen['model'] = request.config
        return _ok()

    async def allergy_check(datapoint: BaseDataPoint, options: AllergyJudgeConfig) -> EvalFnResponse:
        seen['per_row'] = options
        return _score()

    ai.define_model(name='bistro', fn=bistro)
    ai.define_evaluator(name='row', display_name='Row', definition='d', fn=allergy_check)
    model = await ai.registry.resolve_action_by_key('/model/bistro')
    assert model is not None

    await model.run({'messages': [], 'config': None})
    assert seen['model'] == TableConfig()
    await model.run({'messages': [], 'config': {'temperature': None, 'allergens': ['shellfish']}})
    assert seen['model'] == TableConfig(allergens=['shellfish'])
    assert seen['model'].model_fields_set == {'allergens'}
    await ai.generate(model='bistro', prompt='a table for two')
    assert seen['model'] == TableConfig()
    await ai.evaluate(evaluator='row', dataset=ROWS)
    assert seen['per_row'] == AllergyJudgeConfig()


# -----------------------------------------------------------------------------
# Refs: EmbedderRef and EvaluatorRef have ModelRef's shape
# -----------------------------------------------------------------------------


def test_typed_embedder_ref_checks_and_copies_config() -> None:
    config = CrmEmbedConfig(task_type='query')
    ref = create_embedder_ref('crm', config_schema=CrmEmbedConfig, config=config, version='v2')
    config.task_type = 'document'

    assert ref.config == CrmEmbedConfig(task_type='query')
    assert ref.version == 'v2'
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.name = 'other'  # type: ignore[misc]
    with pytest.raises(GenkitError, match=r'crm: config must be an instance of .*CrmEmbedConfig, got .*MenuOnlyConfig'):
        EmbedderRef(name='crm', config_schema=CrmEmbedConfig, config=cast(Any, MenuOnlyConfig()))


def test_untyped_refs_take_a_mapping_and_copy_it() -> None:
    options: dict[str, Any] = {'allergens': ['peanut']}
    embedder = EmbedderRef(name='crm', config=options)
    evaluator = evaluator_ref('allergy', config=options)
    options['allergens'].append('shellfish')

    assert embedder.config == {'allergens': ['peanut']}
    assert evaluator.config == {'allergens': ['peanut']}
    with pytest.raises(GenkitError, match='config must be a mapping when config_schema is not set'):
        EvaluatorRef(name='allergy', config=AllergyJudgeConfig())


def test_typed_evaluator_ref_checks_config() -> None:
    ref = evaluator_ref('allergy', config_schema=AllergyJudgeConfig, config=AllergyJudgeConfig(strict=False))

    assert ref.config == AllergyJudgeConfig(strict=False)
    assert ref.config_schema is AllergyJudgeConfig
    with pytest.raises(GenkitError, match='config must be an instance of'):
        evaluator_ref('allergy', config_schema=AllergyJudgeConfig, config=cast(Any, {'strict': False}))


# -----------------------------------------------------------------------------
# ai.generate / ai.embed / ai.evaluate: one merge, ref < call
# -----------------------------------------------------------------------------


def _typed_app() -> tuple[Genkit, dict[str, Any]]:
    """Typed model, embedder, per-row and batch evaluator."""
    ai = Genkit()
    seen: dict[str, Any] = {}

    async def bistro(request: ModelRequest[TableConfig], ctx: ActionRunContext) -> ModelResponse:
        seen['model'] = request.config
        return _ok()

    async def crm_search(request: EmbedRequest[TableConfig]) -> EmbedResponse:
        seen['embedder'] = request.options
        return _embedding()

    async def allergy_check(datapoint: BaseDataPoint, options: TableConfig) -> EvalFnResponse:
        seen['per_row'] = options
        return _score()

    async def allergy_batch(request: EvalRequest[TableConfig]) -> list[EvalFnResponse]:
        seen['batch'] = request.options
        return [_score()]

    ai.define_model(name='bistro', fn=bistro)
    ai.define_embedder('crm', crm_search)
    ai.define_evaluator(name='row', display_name='Row', definition='d', fn=allergy_check)
    ai.define_batch_evaluator(name='batch', display_name='Batch', definition='d', fn=allergy_batch)
    return ai, seen


@pytest.mark.asyncio
async def test_ref_config_survives_unset_call_fields_and_call_wins_per_field() -> None:
    ai, seen = _typed_app()
    ref_config = TableConfig(temperature=0.1)
    call_config = TableConfig(allergens=['shellfish'])

    await ai.generate(
        model=model_ref('bistro', config_schema=TableConfig, config=ref_config), prompt='hi', config=call_config
    )
    await ai.embed(
        embedder=create_embedder_ref('crm', config_schema=TableConfig, config=ref_config),
        content='acme corp',
        config=call_config,
    )
    for name in ('row', 'batch'):
        await ai.evaluate(
            evaluator=evaluator_ref(name, config_schema=TableConfig, config=ref_config),
            dataset=ROWS,
            config=call_config,
        )

    expected = TableConfig(temperature=0.1, allergens=['shellfish'])
    assert seen == {'model': expected, 'embedder': expected, 'per_row': expected, 'batch': expected}


@pytest.mark.asyncio
async def test_untyped_fns_get_ref_and_call_layers_over_defaults() -> None:
    ai, seen = _pipeline_app()

    await ai.embed(
        embedder=create_embedder_ref('crm', config={'temperature': 0.1, 'region': 'emea'}),
        content='acme corp',
        config={'allergens': ['shellfish']},
    )
    await ai.evaluate(evaluator=evaluator_ref('row', config={'temperature': 0.1}), dataset=ROWS)

    assert seen['embedder'] == {'temperature': 0.1, 'allergens': ['shellfish'], 'region': 'emea'}
    assert seen['per_row'] == {'temperature': 0.1, 'allergens': ['peanut']}


@pytest.mark.asyncio
async def test_explicit_none_clears_down_to_the_definition_default() -> None:
    """None never reaches the fn. It clears the ref's value, and the schema default fills in."""
    ai, seen = _pipeline_app()
    ref = create_embedder_ref('crm', config={'temperature': 0.1, 'region': 'emea'})

    await ai.embed(embedder=ref, content='acme corp', config={'temperature': None, 'region': None})
    await ai.generate(
        model=model_ref('bistro', config_schema=TableConfig, config=TableConfig(temperature=0.1)),
        prompt='hi',
        config={'temperature': None},
    )

    assert seen['embedder'] == TABLE_DEFAULTS
    assert seen['model'] == TABLE_DEFAULTS


@pytest.mark.asyncio
async def test_ref_with_a_foreign_config_schema_raises_at_call_time() -> None:
    ai, seen = _typed_app()

    with pytest.raises(GenkitError, match=r"model 'bistro' takes config .*TableConfig, but the ref's config_schema"):
        await ai.generate(model=model_ref('bistro', config_schema=MenuOnlyConfig), prompt='hi')
    with pytest.raises(GenkitError, match=r"embedder 'crm' takes config .*TableConfig, but the ref's config_schema"):
        await ai.embed(embedder=create_embedder_ref('crm', config_schema=MenuOnlyConfig), content='acme corp')
    with pytest.raises(GenkitError, match=r"evaluator 'row' takes config .*TableConfig, but the ref's config_schema"):
        await ai.evaluate(evaluator=evaluator_ref('row', config_schema=MenuOnlyConfig), dataset=ROWS)
    assert seen == {}


@pytest.mark.asyncio
async def test_generic_model_config_on_a_ref_defers_to_the_model_class() -> None:
    ai, seen = _typed_app()

    await ai.generate(model=model_ref('bistro', config_schema=ModelConfig), prompt='hi', config={'temperature': 0.2})

    assert seen['model'] == TableConfig(temperature=0.2)


@pytest.mark.asyncio
async def test_call_config_of_another_class_raises_before_the_fn_runs() -> None:
    ai, seen = _typed_app()

    with pytest.raises(GenkitError, match=r'crm: config must be .*TableConfig or a mapping, got .*MenuOnlyConfig'):
        await ai.embed(embedder='crm', content='acme corp', config=MenuOnlyConfig())
    with pytest.raises(GenkitError, match=r'batch: config must be .*TableConfig or a mapping, got .*MenuOnlyConfig'):
        await ai.evaluate(evaluator='batch', dataset=ROWS, config=MenuOnlyConfig())
    with pytest.raises(GenkitError, match=r"row: config 'temperature'"):
        await ai.evaluate(evaluator='row', dataset=ROWS, config={'temperature': 'warm'})
    assert seen == {}


class AliasedEmbedConfig(BaseModel):
    """camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra='forbid')

    task_type: str | None = None
    output_dimensionality: int | None = None


@pytest.mark.asyncio
async def test_embed_folds_aliases_to_the_embedder_class() -> None:
    ai = Genkit()
    seen: dict[str, Any] = {}

    async def crm_search(request: EmbedRequest) -> EmbedResponse:
        seen['options'] = request.options
        return _embedding()

    ai.define_embedder('crm', crm_search, config_schema=AliasedEmbedConfig)

    await ai.embed(
        embedder=create_embedder_ref('crm', config={'taskType': 'RETRIEVAL_DOCUMENT', 'outputDimensionality': 256}),
        content='acme corp',
        config={'task_type': 'RETRIEVAL_QUERY'},
    )

    assert seen['options'] == {'task_type': 'RETRIEVAL_QUERY', 'output_dimensionality': 256}


@pytest.mark.asyncio
async def test_version_is_the_lowest_caller_layer_for_models_and_embedders() -> None:
    """ref.version < ref.config['version'] < call config['version'], for both kinds."""
    ai, seen = _pipeline_app(config_schema=NotSetConfig)

    await ai.embed(embedder=create_embedder_ref('crm', version='v1'), content='acme corp')
    assert seen['embedder'] == {'version': 'v1'}
    await ai.embed(embedder=create_embedder_ref('crm', version='v1', config={'version': 'v2'}), content='acme corp')
    assert seen['embedder'] == {'version': 'v2'}
    await ai.embed(
        embedder=create_embedder_ref('crm', version='v1', config={'version': 'v2'}),
        content='acme corp',
        config={'version': 'v3'},
    )
    assert seen['embedder'] == {'version': 'v3'}

    class VersionedConfig(NotSetConfig):
        version: str | None = None

    await ai.generate(model=model_ref('bistro', config_schema=ModelConfig, version='v1'), prompt='hi')
    assert seen['model'] == {'version': 'v1'}
    await ai.generate(
        model=model_ref('bistro', config_schema=NotSetConfig, version='v1', config=VersionedConfig(version='v2')),
        prompt='hi',
    )
    assert seen['model'] == {'version': 'v2'}
    await ai.generate(
        model=model_ref('bistro', config_schema=NotSetConfig, version='v1', config=VersionedConfig(version='v2')),
        prompt='hi',
        config={'version': 'v3'},
    )
    assert seen['model'] == {'version': 'v3'}
