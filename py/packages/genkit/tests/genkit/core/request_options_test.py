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

"""EmbedRequest[Cfg] and EvalRequest[Cfg] follow ModelRequest[Cfg]'s config rule."""

from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from genkit._core._error import GenkitError
from genkit._core._model import EmbedRequest, EvalRequest, ModelRequest, declared_config_type
from genkit._core._typing import BaseDataPoint


class CrmEmbedConfig(BaseModel):
    """Options a CRM search embedder reads."""

    dimensions: int = 768
    task_type: str | None = None


class AllergyJudgeConfig(BaseModel):
    """Options an allergy-check evaluator reads."""

    strict: bool = True
    allergens: list[str] = ['peanut']


class MenuModelConfig(BaseModel):
    """Unrelated class, used to check the wrong-class error."""

    temperature: float = 0.7


def test_bare_embed_request_options_are_a_dict() -> None:
    assert EmbedRequest(input=[]).options == {}
    assert EmbedRequest(input=[], options=None).options == {}
    assert EmbedRequest(input=[], options=CrmEmbedConfig(task_type='query')).options == {'task_type': 'query'}


def test_typed_embed_request_validates_options_into_the_class() -> None:
    request = EmbedRequest[CrmEmbedConfig].model_validate({'input': [], 'options': {'task_type': 'query'}})

    assert request.options == CrmEmbedConfig(dimensions=768, task_type='query')
    assert request.options.model_fields_set == {'task_type'}


@pytest.mark.parametrize('options', [None, {}], ids=['none', 'empty'])
def test_typed_embed_request_without_options_gets_class_defaults(options: object) -> None:
    payload: dict[str, Any] = {'input': []} if options == {} else {'input': [], 'options': options}
    request = EmbedRequest[CrmEmbedConfig].model_validate(payload)

    assert request.options == CrmEmbedConfig()
    assert request.options.model_fields_set == set()


def test_typed_eval_request_validates_options_into_the_class() -> None:
    request = EvalRequest[AllergyJudgeConfig].model_validate({
        'dataset': [BaseDataPoint(input='satay')],
        'eval_run_id': 'run-1',
        'options': {'allergens': ['shellfish']},
    })

    assert request.options == AllergyJudgeConfig(strict=True, allergens=['shellfish'])


def test_typed_request_rejects_an_instance_of_another_class() -> None:
    with pytest.raises(
        ValidationError, match='options must be .*AllergyJudgeConfig or a mapping, got .*MenuModelConfig'
    ):
        EvalRequest[AllergyJudgeConfig].model_validate({
            'dataset': [],
            'eval_run_id': 'run-1',
            'options': MenuModelConfig(),
        })
    with pytest.raises(ValidationError, match='options must be a BaseModel or mapping, got str'):
        EmbedRequest[CrmEmbedConfig].model_validate({'input': [], 'options': 'query'})


def test_bare_request_is_revalidated_into_the_typed_class() -> None:
    """The action boundary validates a bare request against the fn's annotation."""
    bare = EmbedRequest(input=[], options={'task_type': 'query'})

    typed = TypeAdapter(EmbedRequest[CrmEmbedConfig]).validate_python(bare)

    assert typed.options == CrmEmbedConfig(task_type='query')


@pytest.mark.parametrize(
    ('request_type', 'field'),
    [(ModelRequest, 'config'), (EmbedRequest, 'options'), (EvalRequest, 'options')],
)
def test_non_model_type_argument_raises_when_annotated(request_type: Any, field: str) -> None:  # noqa: ANN401
    with pytest.raises(GenkitError, match=f'the {field} type must be a pydantic BaseModel subclass'):
        request_type[dict[str, Any]]


def test_declared_config_type_reads_every_request_type() -> None:
    assert declared_config_type(EmbedRequest[CrmEmbedConfig]) is CrmEmbedConfig
    assert declared_config_type(EvalRequest[AllergyJudgeConfig]) is AllergyJudgeConfig
    assert declared_config_type(EmbedRequest) is None
    assert declared_config_type(EvalRequest[Any]) is None


def test_typed_options_json_schema_references_the_class() -> None:
    schema = TypeAdapter(EmbedRequest[CrmEmbedConfig]).json_schema()

    assert schema['properties']['options'] == {'$ref': '#/$defs/CrmEmbedConfig'}
