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

"""A Flask-served flow sends Genkit types nested in its result as wire JSON."""

from flask import Flask
from genkit_flask import genkit_flask_handler
from pydantic import BaseModel

from genkit import Genkit, Message, Part


class Turn(BaseModel):
    reply: Message
    turns: int


def test_flask_flow_returning_model_with_nested_message_sends_wire_json() -> None:
    """A flow returning Turn(reply=Message(...)) responds with camelCase, null-free JSON for the message."""
    ai = Genkit()
    app = Flask(__name__)

    @app.post('/turn')
    @genkit_flask_handler(ai)
    @ai.flow()
    async def turn(_: str) -> Turn:
        return Turn(reply=Message(role='model', content=[Part.from_text('hi')]), turns=1)

    response = app.test_client().post('/turn', json={'data': 'go'})

    assert response.status_code == 200
    assert response.json == {'result': {'reply': {'role': 'model', 'content': [{'text': 'hi'}]}, 'turns': 1}}
