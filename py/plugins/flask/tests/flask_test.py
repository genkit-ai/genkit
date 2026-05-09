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


"""Tests for the Flask plugin."""

from typing import Any

from flask import Flask, Request

from genkit import ActionRunContext, Genkit
from genkit.plugin_api import RequestData
from genkit.plugins.flask import genkit_flask_handler


def create_app() -> Flask:
    """Create a Flask application for testing."""
    ai = Genkit()

    app = Flask(__name__)
    app.config.update({
        'TESTING': True,
    })

    async def my_context_provider(request_data: RequestData[Request]) -> dict[str, Any]:
        """Provide a context for the flow."""
        return {'username': request_data.request.headers.get('authorization')}

    @app.post('/chat')
    @genkit_flask_handler(ai, context_provider=my_context_provider)
    @ai.flow()
    async def say_hi(name: str, ctx: ActionRunContext) -> dict[str, str]:
        ctx.send_chunk(1)
        ctx.send_chunk({'username': ctx.context.get('username')})
        ctx.send_chunk({'foo': 'bar'})
        return {'bar': 'baz'}

    return app


def test_simple_post() -> None:
    """Test a simple POST request to the chat endpoint."""
    client = create_app().test_client()
    response = client.post(
        '/chat', json={'data': 'banana'}, headers={'Authorization': 'Pavel', 'content-Type': 'application/json'}
    )

    assert response.json == {
        'result': {
            'bar': 'baz',
        },
    }


def test_streaming() -> None:
    """Test a streaming POST request to the chat endpoint."""
    client = create_app().test_client()
    response = client.post(
        '/chat',
        json={'data': 'banana'},
        headers={'Authorization': 'Pavel', 'content-Type': 'application/json', 'accept': 'text/event-stream'},
    )

    assert response.is_streamed

    chunks = []
    for chunk in response.response:
        chunks.append(chunk)

    assert chunks == [
        b'data: {"message":1}\n\n',
        b'data: {"message":{"username":"Pavel"}}\n\n',
        b'data: {"message":{"foo":"bar"}}\n\n',
        b'data: {"result":{"bar":"baz"}}\n\n',
    ]
