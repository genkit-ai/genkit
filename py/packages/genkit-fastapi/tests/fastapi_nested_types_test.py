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

"""FastAPI responses send Genkit types nested in a user model as wire JSON."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from genkit_fastapi import genkit_fastapi_handler
from pydantic import BaseModel

from genkit import Genkit, Message, Part


class Turn(BaseModel):
    reply: Message
    turns: int


def test_fastapi_flow_returning_model_with_nested_message_unchanged() -> None:
    """A served flow returning Turn(reply=Message(...)) responds with camelCase, null-free JSON, as before."""
    ai = Genkit()
    app = FastAPI()

    @app.post('/turn', response_model=None)
    @genkit_fastapi_handler(ai)
    @ai.flow()
    async def turn(_: str) -> Turn:
        return Turn(reply=Message(role='model', content=[Part.from_text('hi')]), turns=1)

    response = TestClient(app).post('/turn', json={'data': 'go'})

    assert response.status_code == 200
    assert response.json() == {'result': {'reply': {'role': 'model', 'content': [{'text': 'hi'}]}, 'turns': 1}}


def test_fastapi_response_model_with_message_returns_camel_case_json() -> None:
    """A plain FastAPI route whose response_model holds a Message returns camelCase JSON with no nulls."""
    app = FastAPI()

    @app.get('/turn', response_model=Turn)
    async def turn() -> Turn:
        return Turn(reply=Message(role='model', content=[Part.from_tool_request('lookup', {'q': 1})]), turns=1)

    response = TestClient(app).get('/turn')

    assert response.status_code == 200
    assert response.json() == {
        'reply': {'role': 'model', 'content': [{'toolRequest': {'name': 'lookup', 'input': {'q': 1}}}]},
        'turns': 1,
    }
