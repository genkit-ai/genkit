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

"""EmbedRequest[Cfg] follows ModelRequest[Cfg]'s config rule."""

from typing import Any, Generic, Literal, TypeVar

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from genkit._core._action import Action, ActionKind
from genkit._core._error import GenkitError
from genkit._core._model import (
    EmbedRequest,
    ModelRequest,
    declared_config_type,
)


class CrmEmbedConfig(BaseModel):
    """Options a CRM search embedder reads."""

    dimensions: int = 768
    task_type: str | None = None


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


def test_typed_request_rejects_an_instance_of_another_class() -> None:
    with pytest.raises(ValidationError, match='options must be a BaseModel or mapping, got str'):
        EmbedRequest[CrmEmbedConfig].model_validate({'input': [], 'options': 'query'})


def test_bare_request_is_revalidated_into_the_typed_class() -> None:
    """The action boundary validates a bare request against the fn's annotation."""
    bare = EmbedRequest(input=[], options={'task_type': 'query'})

    typed = TypeAdapter(EmbedRequest[CrmEmbedConfig]).validate_python(bare)

    assert typed.options == CrmEmbedConfig(task_type='query')


@pytest.mark.parametrize(
    ('request_type', 'field'),
    [
        (ModelRequest, 'config'),
        (EmbedRequest, 'options'),
    ],
)
def test_non_model_type_argument_raises_when_annotated(request_type: Any, field: str) -> None:  # noqa: ANN401
    with pytest.raises(GenkitError, match=f'the {field} type must be a pydantic BaseModel subclass'):
        request_type[dict[str, Any]]


def test_declared_config_type_reads_every_request_type() -> None:
    assert declared_config_type(EmbedRequest[CrmEmbedConfig]) is CrmEmbedConfig
    assert declared_config_type(EmbedRequest[Any]) is None
    assert declared_config_type(EmbedRequest) is None


def test_typed_options_json_schema_references_the_class() -> None:
    schema = TypeAdapter(EmbedRequest[CrmEmbedConfig]).json_schema()

    assert schema['properties']['options'] == {'$ref': '#/$defs/CrmEmbedConfig'}


@pytest.mark.parametrize('request_type', [EmbedRequest, EmbedRequest[Any]], ids=['bare', 'any'])
@pytest.mark.parametrize('mode', ['validation', 'serialization'])
def test_untyped_options_json_schema_is_an_object(
    request_type: Any,  # noqa: ANN401
    mode: Literal['validation', 'serialization'],
) -> None:
    schema = TypeAdapter(request_type).json_schema(mode=mode)

    assert schema['properties']['options'] == {
        'additionalProperties': True,
        'title': 'Options',
        'type': 'object',
    }


class CrmEmbedRequest(EmbedRequest[CrmEmbedConfig]):
    """A plugin's own request type: a subclass of the parameterized EmbedRequest."""


class MenuModelRequest(ModelRequest[MenuModelConfig]):
    """A plugin's own request type: a subclass of the parameterized ModelRequest."""


class CrmEmbedRequestChild(CrmEmbedRequest):
    """A subclass of a subclass keeps the config class too."""


_ExtraT = TypeVar('_ExtraT')
_OptionsT = TypeVar('_OptionsT')


class TaggedEmbedRequest(EmbedRequest[_OptionsT], Generic[_ExtraT, _OptionsT]):
    """A generic subclass whose first type parameter is not the options class."""


@pytest.mark.parametrize(
    ('request_type', 'expected'),
    [
        (CrmEmbedRequest, CrmEmbedConfig),
        (CrmEmbedRequestChild, CrmEmbedConfig),
        (MenuModelRequest, MenuModelConfig),
        (TaggedEmbedRequest[MenuModelConfig, CrmEmbedConfig], CrmEmbedConfig),
        (ModelRequest[MenuModelConfig], MenuModelConfig),
        (ModelRequest, None),
        (ModelRequest[Any], None),
        (TaggedEmbedRequest, None),
        (CrmEmbedConfig, None),
    ],
    ids=[
        'embed-subclass',
        'embed-grandchild',
        'model-subclass',
        'generic-subclass-config-not-first',
        'model-typed',
        'model-bare',
        'model-any',
        'generic-subclass-unparameterized',
        'not-a-request',
    ],
)
def test_declared_config_type_reads_subclasses_of_typed_requests(
    request_type: type,
    expected: type[BaseModel] | None,
) -> None:
    assert declared_config_type(request_type) is expected


def test_typed_embed_request_subclass_json_schema_references_the_class() -> None:
    schema = TypeAdapter(CrmEmbedRequest).json_schema()

    assert schema['properties']['options'] == {'$ref': '#/$defs/CrmEmbedConfig'}
    assert 'CrmEmbedConfig' in schema['$defs']


def test_typed_embed_request_subclass_validates_options_into_the_class() -> None:
    request = CrmEmbedRequest.model_validate({'input': [], 'options': {'task_type': 'query'}})

    assert request.options == CrmEmbedConfig(task_type='query')


@pytest.mark.parametrize(
    ('request_type', 'payload', 'field'),
    [
        (CrmEmbedRequest, {'input': [], 'options': MenuModelConfig(temperature=0.2)}, 'options'),
        (MenuModelRequest, {'messages': [], 'config': CrmEmbedConfig()}, 'config'),
    ],
    ids=['embed', 'model'],
)
def test_typed_request_subclass_rejects_an_instance_of_another_class(
    request_type: Any,  # noqa: ANN401
    payload: dict[str, Any],
    field: str,
) -> None:
    with pytest.raises(ValidationError, match=f'{field} must be .* or a mapping, got '):
        request_type.model_validate(payload)


def test_action_config_schema_comes_from_a_typed_request_subclass_annotation() -> None:
    async def run_model(req: MenuModelRequest) -> str:
        return ''

    action = Action(kind=ActionKind.MODEL, name='menu-model', fn=run_model)

    assert action.config_schema is MenuModelConfig
