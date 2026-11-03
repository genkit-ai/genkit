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

"""Config classes expose Genkit types only, and still send what the SDK types sent.

Enum fields are Literals of the google-genai enum values and nested messages
are flat fields. These tests check that no google-genai class leaks into a
config annotation or schema, and that values needing conversion (datetimes,
enum strings) reach the HTTP body.
"""

import json
import types
from collections.abc import Iterator
from typing import Any, Literal, get_args, get_origin
from unittest.mock import patch

import pytest
from genkit_google_genai import GeminiConfig, GeminiImageConfig, GeminiTtsConfig, GemmaConfig, VeoConfig, VertexAI
from google.auth.credentials import AnonymousCredentials
from google.genai import _api_client, types as genai_types
from pydantic import BaseModel

from genkit import Genkit

_PUBLIC_CONFIGS: list[type[BaseModel]] = [GeminiConfig, GeminiTtsConfig, GeminiImageConfig, GemmaConfig, VeoConfig]


# googlegenai_gemini_test.py pins each Literal to its google-genai enum.


def _reachable(annotation: object, seen: set[type]) -> None:
    """Collect every class a field annotation can hold, through unions, containers and nested models."""
    origin = get_origin(annotation)
    if origin is Literal:
        return
    if origin is not None or isinstance(annotation, types.UnionType):
        for arg in get_args(annotation):
            _reachable(arg, seen)
        return
    if not isinstance(annotation, type) or annotation in seen:
        return
    seen.add(annotation)
    if issubclass(annotation, BaseModel):
        for field in annotation.model_fields.values():
            _reachable(field.annotation, seen)


@pytest.mark.parametrize('config_class', _PUBLIC_CONFIGS, ids=lambda c: c.__name__)
def test_public_config_fields_hold_no_google_genai_types(config_class: type[BaseModel]) -> None:
    """No field, at any depth, is annotated with a google.genai class."""
    seen: set[type] = set()
    _reachable(config_class, seen)

    assert sorted(c.__qualname__ for c in seen if c.__module__.startswith('google.')) == []


@pytest.mark.parametrize('config_class', _PUBLIC_CONFIGS, ids=lambda c: c.__name__)
def test_public_config_schema_defs_are_genkit_classes(config_class: type[BaseModel]) -> None:
    """Every $defs entry in the Dev UI schema is a Genkit class reachable from the config."""
    seen: set[type] = set()
    _reachable(config_class, seen)
    genkit_names = {c.__name__ for c in seen if c.__module__.startswith('genkit_google_genai.')}

    defs = set(config_class.model_json_schema(by_alias=True).get('$defs', {}))

    assert defs <= genkit_names


def test_sdk_enum_member_still_validates_as_its_literal() -> None:
    """A google-genai enum member passed at runtime is stored as its plain string value."""
    config = GeminiImageConfig.model_validate({'prominent_people': genai_types.ProminentPeople.BLOCK_PROMINENT_PEOPLE})

    assert config.prominent_people == 'BLOCK_PROMINENT_PEOPLE'
    assert type(config.prominent_people) is str


# -- on the wire ---------------------------------------------------------------


@pytest.fixture
def sent_bodies() -> Iterator[list[dict[str, Any]]]:
    """Capture every request body the SDK sends, and answer with a canned success."""
    bodies: list[dict[str, Any]] = []

    async def fake_request(
        _self: object, http_request: _api_client.HttpRequest, http_options: object = None, stream: bool = False
    ) -> _api_client.HttpResponse:
        assert isinstance(http_request.data, dict)
        bodies.append(http_request.data)
        if ':predictLongRunning' in http_request.url:
            body: dict[str, Any] = {'name': 'operations/1', 'done': False}
        else:
            body = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'ok'}]}, 'finishReason': 'STOP'}]}
        return _api_client.HttpResponse(headers={}, response_stream=[json.dumps(body)])

    with patch.object(_api_client.BaseApiClient, '_async_request', fake_request):
        yield bodies


def _vertexai() -> Genkit:
    return Genkit(plugins=[VertexAI(project='p', location='us-central1', credentials=AnonymousCredentials())])


@pytest.mark.asyncio
async def test_generate_google_search_time_range_reaches_request(sent_bodies: list[dict[str, Any]]) -> None:
    """`google_search_start_time` / `_end_time` as ISO strings are sent as the SDK sends an Interval."""
    await _vertexai().generate(
        model='vertexai/gemini-2.5-flash',
        prompt='hi',
        config={
            'google_search': True,
            'google_search_exclude_domains': ['spam.example'],
            'google_search_start_time': '2024-01-01T00:00:00Z',
            'google_search_end_time': '2024-02-01T00:00:00Z',
        },
    )

    # Key spelling inside googleSearch is the SDK's business (it varies by
    # backend); pin the values.
    [tool] = sent_bodies[-1]['tools']
    exclude_domains, time_range = tool['googleSearch'].values()
    assert exclude_domains == ['spam.example']
    assert sorted(time_range.values()) == [
        '2024-01-01T00:00:00+00:00',
        '2024-02-01T00:00:00+00:00',
    ]


@pytest.mark.asyncio
async def test_generate_operation_veo_literal_fields_reach_request(sent_bodies: list[dict[str, Any]]) -> None:
    """`compressionQuality` and `resizeMode` strings are sent as the SDK enum values."""
    await _vertexai().generate_operation(
        model='vertexai/veo-3.0-generate-001',
        prompt='a cat',
        config={'compressionQuality': 'LOSSLESS', 'resizeMode': 'PAD'},
    )

    parameters = sent_bodies[-1]['parameters']
    assert parameters['compressionQuality'] == 'LOSSLESS'
    assert parameters['resizeMode'] == 'PAD'
