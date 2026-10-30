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

"""A Django-served flow sends Genkit types nested in its result as wire JSON."""

import json
import sys
import types
from collections.abc import Iterator

import pytest
from django.test import AsyncClient
from django.test.utils import override_settings
from django.urls import path
from genkit_django import genkit_django_handler
from pydantic import BaseModel

from genkit import Genkit, Message, Part


class Turn(BaseModel):
    reply: Message
    turns: int


@pytest.fixture
def turn_urlconf(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Mount a flow that returns a user model holding a Message."""
    ai = Genkit()

    @genkit_django_handler(ai)
    @ai.flow()
    async def turn(_: str) -> Turn:
        return Turn(reply=Message(role='model', content=[Part.from_text('hi')]), turns=1)

    module = types.ModuleType('genkit_django_nested_types_urls')
    module.urlpatterns = [path('turn', turn)]  # type: ignore[attr-defined]  # pyrefly: ignore[missing-attribute]
    monkeypatch.setitem(sys.modules, 'genkit_django_nested_types_urls', module)
    with override_settings(ROOT_URLCONF='genkit_django_nested_types_urls'):
        yield


@pytest.mark.asyncio
async def test_django_flow_returning_model_with_nested_message_sends_wire_json(turn_urlconf: None) -> None:  # noqa: ARG001
    """A flow returning Turn(reply=Message(...)) responds with camelCase, null-free JSON for the message."""
    response = await AsyncClient().post('/turn', data=json.dumps({'data': 'go'}), content_type='application/json')

    assert response.status_code == 200
    assert json.loads(response.content) == {
        'result': {'reply': {'role': 'model', 'content': [{'text': 'hi'}]}, 'turns': 1}
    }
