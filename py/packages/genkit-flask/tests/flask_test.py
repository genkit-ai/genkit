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

import json
from typing import Any

from flask import Flask, abort
from genkit_flask import genkit_flask_handler

from genkit import ActionRunContext, Genkit, GenkitError, PublicError, RequestData
from genkit.plugin_api import wrap_http_error


def sse_error_event(chunks: list[bytes]) -> dict:
    """Return the first SSE ``error`` payload, or fail."""
    text = b''.join(chunks).decode()
    for line in text.splitlines():
        if line.startswith('data: '):
            payload = json.loads(line[6:])
            if 'error' in payload:
                return payload['error']
    raise AssertionError(f'no SSE error event in {text!r}')


def create_app() -> Flask:
    """Create a Flask application for testing."""
    ai = Genkit()

    app = Flask(__name__)
    app.config.update({
        'TESTING': True,
    })

    async def my_context_provider(request_data: RequestData) -> dict[str, Any]:
        """Provide a context for the flow."""
        return {'username': request_data.headers.get('authorization')}

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


def test_flask_flow_raising_unauthenticated_returns_500_internal_error() -> None:
    """Flask POST to a flow that raises GenkitError UNAUTHENTICATED is 500, not 401."""
    ai = Genkit()
    app = Flask(__name__)
    app.config.update({'TESTING': True})

    @app.post('/login')
    @genkit_flask_handler(ai)
    @ai.flow()
    async def login(_: str) -> None:
        raise GenkitError(status='UNAUTHENTICATED', message='token for alice@example.com expired')

    response = app.test_client().post('/login', json={'data': 'x'})

    assert response.status_code == 500
    assert json.loads(response.data) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert b'alice@example.com' not in response.data


def test_flask_flow_raising_public_error_returns_its_status_and_message() -> None:
    """Flask POST to a flow that raises PublicError NOT_FOUND returns 404 with that message."""
    ai = Genkit()
    app = Flask(__name__)
    app.config.update({'TESTING': True})

    @app.post('/lookup')
    @genkit_flask_handler(ai)
    @ai.flow()
    async def lookup(_: str) -> None:
        raise PublicError('NOT_FOUND', 'no order 99')

    response = app.test_client().post('/lookup', json={'data': '99'})

    assert response.status_code == 404
    assert json.loads(response.data) == {'message': 'no order 99', 'status': 'NOT_FOUND'}


def test_flask_flow_raising_value_error_returns_500_internal_error_without_stack() -> None:
    """Flask POST to a flow that raises ValueError returns a generic 500."""
    ai = Genkit()
    app = Flask(__name__)
    app.config.update({'TESTING': True})

    @app.post('/boom')
    @genkit_flask_handler(ai)
    @ai.flow()
    async def boom(_: str) -> None:
        raise ValueError('secret')

    response = app.test_client().post('/boom', json={'data': 'x'})

    assert response.status_code == 500
    body = json.loads(response.data)
    assert body == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert b'secret' not in response.data
    assert 'stack' not in body


def test_flask_stream_flow_raising_value_error_sends_sse_internal_error_without_stack() -> None:
    """Flask SSE to a flow that raises ValueError sends a generic Internal Error."""
    ai = Genkit()
    app = Flask(__name__)
    app.config.update({'TESTING': True})

    @app.post('/boom')
    @genkit_flask_handler(ai)
    @ai.flow()
    async def boom(_: str) -> None:
        raise ValueError('secret')

    response = app.test_client().post(
        '/boom',
        json={'data': 'x'},
        headers={'accept': 'text/event-stream'},
    )

    chunks = list(response.response)
    error = sse_error_event(chunks)
    assert error == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert b'secret' not in b''.join(chunks)
    assert 'stack' not in error


def test_flask_context_provider_sees_method_lowercase_headers_and_input() -> None:
    """Flask context_provider sees method, lowercase headers, and input."""
    ai = Genkit()
    app = Flask(__name__)
    app.config.update({'TESTING': True})

    async def provider(request_data: RequestData) -> dict[str, Any]:
        return {
            'method': request_data.method,
            'authorization': request_data.headers['authorization'],
            'input': request_data.input,
        }

    @app.post('/echo')
    @genkit_flask_handler(ai, context_provider=provider)
    @ai.flow()
    async def echo(_: str, ctx: ActionRunContext) -> dict[str, Any]:
        return {
            'method': ctx.context['method'],
            'authorization': ctx.context['authorization'],
            'input': ctx.context['input'],
        }

    response = app.test_client().post(
        '/echo',
        json={'data': 'hello'},
        headers={'Authorization': 'Bearer tok'},
    )

    assert response.status_code == 200
    assert response.json == {
        'result': {
            'method': 'POST',
            'authorization': 'Bearer tok',
            'input': 'hello',
        }
    }


def test_flask_flow_rejects_non_dict_json_payload_with_400() -> None:
    """A JSON payload that is not an object (e.g. ['data']) returns 400."""
    client = create_app().test_client()
    response = client.post(
        '/chat',
        json=['data'],
        headers={'Authorization': 'Pavel', 'Content-Type': 'application/json'},
    )

    assert response.status_code == 400
    assert json.loads(response.data) == {
        'message': 'flow request must be wrapped in {"data": data} object',
        'status': 'INVALID_ARGUMENT',
    }


def test_flask_missing_data_wrapper_returns_the_wrap_message() -> None:
    """A POST without ``{"data": ...}`` tells the caller to wrap the body."""
    response = create_app().test_client().post('/chat', json={'foo': 'bar'})

    assert response.status_code == 400
    assert json.loads(response.data) == {
        'message': 'flow request must be wrapped in {"data": data} object',
        'status': 'INVALID_ARGUMENT',
    }


def test_flask_malformed_json_body_returns_400_valid_json_message() -> None:
    """A Flask POST that is not JSON is 400 request body must be valid JSON."""
    response = (
        create_app()
        .test_client()
        .post(
            '/chat',
            data='{bad',
            content_type='application/json',
        )
    )

    assert response.status_code == 400
    assert json.loads(response.data) == {
        'message': 'request body must be valid JSON',
        'status': 'INVALID_ARGUMENT',
    }


def test_flask_provider_401_returns_500_internal_error() -> None:
    """A Flask flow whose model call fails with a provider 401 is 500 Internal Error."""
    ai = Genkit()
    app = Flask(__name__)
    app.config.update({'TESTING': True})

    @app.post('/ask')
    @genkit_flask_handler(ai)
    @ai.flow()
    async def ask(_: str) -> str:
        raise wrap_http_error(RuntimeError('API key not valid'), status_code=401)

    response = app.test_client().post('/ask', json={'data': 'hi'})

    assert response.status_code == 500
    assert json.loads(response.data) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert b'API key not valid' not in response.data


def test_flask_stream_provider_401_sends_sse_internal_error() -> None:
    """A streamed Flask flow whose model call fails with a provider 401 ends with Internal Error."""
    ai = Genkit()
    app = Flask(__name__)
    app.config.update({'TESTING': True})

    @app.post('/ask')
    @genkit_flask_handler(ai)
    @ai.flow()
    async def ask(_: str) -> str:
        raise wrap_http_error(RuntimeError('API key not valid'), status_code=401)

    response = app.test_client().post(
        '/ask',
        json={'data': 'hi'},
        headers={'accept': 'text/event-stream'},
    )

    chunks = list(response.response)
    assert sse_error_event(chunks) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert b'API key not valid' not in b''.join(chunks)


def test_flask_context_provider_public_error_returns_its_status_and_message() -> None:
    """A PublicError from Flask's context_provider is mapped like a flow failure."""
    ai = Genkit()
    app = Flask(__name__)
    app.config.update({'TESTING': True})

    def deny(_request: RequestData) -> dict[str, Any]:
        raise PublicError('UNAUTHENTICATED', 'not signed in')

    @app.post('/chat')
    @genkit_flask_handler(ai, context_provider=deny)
    @ai.flow()
    async def chat(_: str) -> str:
        return 'ok'

    response = app.test_client().post('/chat', json={'data': 'x'})

    assert response.status_code == 401
    assert json.loads(response.data) == {'message': 'not signed in', 'status': 'UNAUTHENTICATED'}


def test_flask_context_provider_abort_keeps_its_status() -> None:
    """abort(401) from context_provider is the app's own response, not a 500."""
    ai = Genkit()
    app = Flask(__name__)
    app.config.update({'TESTING': True})

    def require_token(_request: RequestData) -> dict[str, Any]:
        abort(401)

    @app.post('/chat')
    @genkit_flask_handler(ai, context_provider=require_token)
    @ai.flow()
    async def chat(_: str) -> str:
        return 'ok'

    response = app.test_client().post('/chat', json={'data': 'x'})

    assert response.status_code == 401
