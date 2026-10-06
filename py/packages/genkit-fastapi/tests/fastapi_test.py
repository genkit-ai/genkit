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


"""Tests for the FastAPI plugin."""

import json
import logging

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from genkit_fastapi import genkit_fastapi_handler, serve_flow
from pydantic import BaseModel

from genkit import ActionRunContext, Genkit, GenkitError, PublicError, RequestData
from genkit.plugin_api import wrap_http_error


def assert_is_error_response(parsed: dict) -> None:
    """Assert parsed dict has a callable error body (message + status)."""
    assert isinstance(parsed, dict)
    assert all(k in parsed for k in ('message', 'status'))
    assert 'stack' not in parsed.get('details', {})


def sse_error_event(text: str) -> dict:
    """Return the first SSE ``error`` payload, or fail."""
    for line in text.splitlines():
        if line.startswith('data: '):
            payload = json.loads(line[6:])
            if 'error' in payload:
                return payload['error']
    raise AssertionError(f'no SSE error event in {text!r}')


def create_app() -> FastAPI:
    """Create a FastAPI application for testing."""
    ai = Genkit()
    app = FastAPI()

    @app.post('/chat', response_model=None)
    @genkit_fastapi_handler(ai)
    @ai.flow()
    async def say_hi(name: str, ctx: ActionRunContext) -> dict[str, str]:
        return {'greeting': f'Hi {name}'}

    @ai.flow()
    async def void_flow() -> dict[str, str]:
        return {'ok': 'true'}

    @ai.flow()
    async def raise_error(_: str) -> None:
        raise ValueError('Intentional test error')

    app.include_router(serve_flow(void_flow, base_path='/void_flow'))
    app.include_router(serve_flow(raise_error, base_path='/error_flow'))

    return app


def test_void_flow_accepts_empty_body() -> None:
    """runFlow() with no input sends {}; void flows should still run."""
    client = TestClient(create_app())
    response = client.post('/void_flow', json={})
    assert response.status_code == 200
    assert response.json()['result'] == {'ok': 'true'}


def test_void_flow_accepts_explicit_null_data() -> None:
    """Explicit ``{"data": null}`` is equivalent to a missing input."""
    client = TestClient(create_app())
    response = client.post('/void_flow', json={'data': None})
    assert response.status_code == 200
    assert response.json()['result'] == {'ok': 'true'}


def test_required_input_empty_body_fails_at_action_not_wire() -> None:
    """Missing input on a required-parameter flow is an action INVALID_ARGUMENT."""
    client = TestClient(create_app())
    response = client.post('/chat', json={})
    assert response.status_code == 400
    parsed = json.loads(response.text)
    assert_is_error_response(parsed)
    assert parsed['status'] == 'INVALID_ARGUMENT'
    assert parsed['message'] == 'Invalid argument'


def test_fastapi_flow_posted_wrong_input_type_returns_400_without_validation_text() -> None:
    """Posting a dict to a ``str`` flow is a 400 INVALID_ARGUMENT that doesn't echo the input."""
    client = TestClient(create_app())
    response = client.post('/chat', json={'data': {'ssn': '123-45-6789'}})

    assert response.status_code == 400
    assert response.json() == {
        'message': 'Invalid argument',
        'status': 'INVALID_ARGUMENT',
    }
    assert '123-45-6789' not in response.text


def test_unknown_body_shape_still_returns_400() -> None:
    """Bodies with unrecognized keys still require a data wrapper."""
    client = TestClient(create_app())
    response = client.post('/chat', json={'foo': 'bar'})
    assert response.status_code == 400
    assert response.json() == {
        'message': 'Action request must be wrapped in {"data": ...} object',
        'status': 'INVALID_ARGUMENT',
    }


def test_500_flow_exception_returns_valid_json() -> None:
    """500 (flow exception) must return valid JSON (not TypeError).

    get_callable_json now returns a dict, so json.dumps works directly.

    Uses real code snippet (SQL injection pattern) to exercise error path realistically.
    """
    client = TestClient(create_app())
    code_snippet = 'query = f"SELECT * FROM users WHERE id={user_input}"'
    response = client.post('/error_flow', json={'data': code_snippet})
    assert response.status_code == 500
    parsed = json.loads(response.text)
    assert_is_error_response(parsed)


def test_context_dependency_value_reaches_action() -> None:
    """A value resolved through FastAPI's DI graph lands in the action context."""
    ai = Genkit()

    @ai.flow()
    async def whoami(_: str, ctx: ActionRunContext) -> str:
        return str(ctx.context.get('uid'))

    async def current_uid() -> str:
        return 'user-123'

    # A dependency with its own sub-dependency, proving the whole graph resolves.
    async def user_context(uid: str = Depends(current_uid)) -> dict[str, object]:
        return {'uid': uid}

    app = FastAPI()
    app.include_router(serve_flow(whoami, base_path='/whoami', context_dependency=user_context))
    client = TestClient(app)

    response = client.post('/whoami', json={'data': 'x'})

    assert response.status_code == 200
    assert response.json()['result'] == 'user-123'


def test_fastapi_flow_raising_not_found_returns_404_with_generic_message() -> None:
    """FastAPI POST to a flow that raises GenkitError NOT_FOUND returns 404 'Not found', not its text."""
    ai = Genkit()

    @ai.flow()
    async def missing(_: str) -> None:
        raise GenkitError(status='NOT_FOUND', message='no recipe for alice@example.com')

    app = FastAPI()
    app.include_router(serve_flow(missing, base_path='/missing'))
    response = TestClient(app).post('/missing', json={'data': 'x'})

    assert response.status_code == 404
    assert response.json() == {'message': 'Not found', 'status': 'NOT_FOUND'}
    assert 'alice@example.com' not in response.text


def test_fastapi_flow_raising_public_error_returns_its_status_and_message() -> None:
    """FastAPI POST to a flow that raises PublicError NOT_FOUND returns 404 with that message."""
    ai = Genkit()

    @ai.flow()
    async def lookup(_: str) -> None:
        raise PublicError('NOT_FOUND', 'no order 99')

    app = FastAPI()
    app.include_router(serve_flow(lookup, base_path='/lookup'))
    response = TestClient(app).post('/lookup', json={'data': '99'})

    assert response.status_code == 404
    assert response.json() == {'message': 'no order 99', 'status': 'NOT_FOUND'}


def test_fastapi_flow_raising_value_error_returns_500_internal_error_without_stack() -> None:
    """FastAPI POST to a flow that raises ValueError returns a generic 500."""
    ai = Genkit()

    @ai.flow()
    async def boom(_: str) -> None:
        raise ValueError('secret')

    app = FastAPI()
    app.include_router(serve_flow(boom, base_path='/boom'))
    response = TestClient(app).post('/boom', json={'data': 'x'})

    assert response.status_code == 500
    body = json.loads(response.text)
    assert body == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert 'secret' not in response.text
    assert 'stack' not in body


def test_fastapi_stream_flow_raising_not_found_sends_sse_error_with_generic_message() -> None:
    """FastAPI SSE to a flow that raises GenkitError NOT_FOUND ends with a 'Not found' error event."""
    ai = Genkit()

    @ai.flow()
    async def missing(_: str) -> None:
        raise GenkitError(status='NOT_FOUND', message='no recipe for alice@example.com')

    app = FastAPI()
    app.include_router(serve_flow(missing, base_path='/missing'))
    response = TestClient(app).post(
        '/missing',
        json={'data': 'x'},
        headers={'Accept': 'text/event-stream'},
    )

    assert sse_error_event(response.text) == {'message': 'Not found', 'status': 'NOT_FOUND'}
    assert 'alice@example.com' not in response.text


def test_fastapi_stream_flow_raising_public_error_sends_its_status_and_message() -> None:
    """FastAPI SSE to a flow that raises PublicError ends with an error event carrying its message."""
    ai = Genkit()

    @ai.flow()
    async def lookup(_: str) -> None:
        raise PublicError('NOT_FOUND', 'no order 99')

    app = FastAPI()
    app.include_router(serve_flow(lookup, base_path='/lookup'))
    response = TestClient(app).post(
        '/lookup',
        json={'data': '99'},
        headers={'Accept': 'text/event-stream'},
    )

    assert sse_error_event(response.text) == {'message': 'no order 99', 'status': 'NOT_FOUND'}


def test_fastapi_stream_flow_raising_value_error_sends_sse_internal_error_without_stack() -> None:
    """FastAPI SSE to a flow that raises ValueError sends a generic Internal Error."""
    ai = Genkit()

    @ai.flow()
    async def boom(_: str) -> None:
        raise ValueError('secret')

    app = FastAPI()
    app.include_router(serve_flow(boom, base_path='/boom'))
    response = TestClient(app).post(
        '/boom',
        json={'data': 'x'},
        headers={'Accept': 'text/event-stream'},
    )

    error = sse_error_event(response.text)
    assert error == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert 'secret' not in response.text
    assert 'stack' not in error


def test_fastapi_context_provider_public_error_returns_its_status_and_message() -> None:
    """A PublicError from FastAPI's context_provider is mapped like a flow failure."""
    ai = Genkit()

    def deny(_request: RequestData) -> dict[str, object]:
        raise PublicError('UNAUTHENTICATED', 'not signed in')

    @ai.flow()
    async def chat(_: str) -> str:
        return 'ok'

    app = FastAPI()

    @app.post('/chat', response_model=None)
    @genkit_fastapi_handler(ai, context_provider=deny)
    async def chat_route():
        return chat

    response = TestClient(app).post('/chat', json={'data': 'x'})

    assert response.status_code == 401
    assert response.json() == {'message': 'not signed in', 'status': 'UNAUTHENTICATED'}


def test_served_flow_wrong_server_key_is_500_internal_error() -> None:
    """A served flow whose model call fails with a provider 401 is 500 Internal Error."""
    ai = Genkit()

    @ai.flow()
    async def ask(_: str) -> str:
        raise wrap_http_error(RuntimeError('API key not valid'), status_code=401)

    app = FastAPI()
    app.include_router(serve_flow(ask, base_path='/ask'))
    response = TestClient(app).post('/ask', json={'data': 'hi'})

    assert response.status_code == 500
    assert response.json() == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert 'API key not valid' not in response.text


def test_served_flow_public_error_keeps_status_and_message() -> None:
    """A PublicError the app raises keeps its status and message on the wire."""
    ai = Genkit()

    @ai.flow()
    async def lookup(_: str) -> None:
        raise PublicError('NOT_FOUND', 'no order 99')

    app = FastAPI()
    app.include_router(serve_flow(lookup, base_path='/lookup'))
    response = TestClient(app).post('/lookup', json={'data': '99'})

    assert response.status_code == 404
    assert response.json() == {'message': 'no order 99', 'status': 'NOT_FOUND'}


def test_served_flow_provider_failure_logs_traceback(caplog: pytest.LogCaptureFixture) -> None:
    """A provider failure on a served flow is logged at error with a traceback."""
    ai = Genkit()

    @ai.flow()
    async def ask(_: str) -> str:
        raise wrap_http_error(RuntimeError('API key not valid'), status_code=401)

    app = FastAPI()
    app.include_router(serve_flow(ask, base_path='/ask'))
    with caplog.at_level(logging.ERROR, logger='genkit_fastapi.handler'):
        response = TestClient(app).post('/ask', json={'data': 'hi'})

    assert response.status_code == 500
    assert any(record.exc_info for record in caplog.records)


def test_fastapi_stream_provider_401_sends_sse_internal_error() -> None:
    """A streamed flow whose model call fails with a provider 401 ends with Internal Error."""
    ai = Genkit()

    @ai.flow()
    async def ask(_: str) -> str:
        raise wrap_http_error(RuntimeError('API key not valid'), status_code=401)

    app = FastAPI()
    app.include_router(serve_flow(ask, base_path='/ask'))
    response = TestClient(app).post(
        '/ask',
        json={'data': 'hi'},
        headers={'Accept': 'text/event-stream'},
    )

    assert sse_error_event(response.text) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert 'API key not valid' not in response.text


def test_fastapi_stream_public_error_with_nested_model_details_sends_error_event() -> None:
    """A streamed PublicError with models in details still ends with a JSON error event."""

    class FieldViolation(BaseModel):
        field: str

    ai = Genkit()

    @ai.flow()
    async def lookup(_: str) -> None:
        raise PublicError('INVALID_ARGUMENT', 'bad', details={'violations': [FieldViolation(field='a')]})

    app = FastAPI()
    app.include_router(serve_flow(lookup, base_path='/lookup'))
    response = TestClient(app).post(
        '/lookup',
        json={'data': 'x'},
        headers={'Accept': 'text/event-stream'},
    )

    assert sse_error_event(response.text) == {
        'message': 'bad',
        'status': 'INVALID_ARGUMENT',
        'details': {'violations': [{'field': 'a'}]},
    }


def test_fastapi_malformed_json_body_returns_400_valid_json_message() -> None:
    """A FastAPI POST that is not JSON is 400 request body must be valid JSON."""
    client = TestClient(create_app())
    response = client.post(
        '/chat',
        content=b'{bad',
        headers={'content-type': 'application/json'},
    )

    assert response.status_code == 400
    assert response.json() == {
        'message': 'request body must be valid JSON',
        'status': 'INVALID_ARGUMENT',
    }


def test_fastapi_context_provider_route_malformed_json_returns_400() -> None:
    """A malformed body on a context_provider route is the same 400 and never calls the provider."""
    ai = Genkit()
    called = False

    def deny(_request: RequestData) -> dict[str, object]:
        nonlocal called
        called = True
        raise PublicError('UNAUTHENTICATED', 'not signed in')

    @ai.flow()
    async def chat(_: str) -> str:
        return 'ok'

    app = FastAPI()

    @app.post('/chat', response_model=None)
    @genkit_fastapi_handler(ai, context_provider=deny)
    async def chat_route():
        return chat

    response = TestClient(app).post(
        '/chat',
        content=b'{bad',
        headers={'content-type': 'application/json'},
    )

    assert called is False
    assert response.status_code == 400
    assert response.json() == {
        'message': 'request body must be valid JSON',
        'status': 'INVALID_ARGUMENT',
    }


def test_fastapi_handler_sync_wrapper_returns_json_internal_error() -> None:
    """A sync wrapper under genkit_fastapi_handler is JSON 500 Internal Error."""
    ai = Genkit()

    @ai.flow()
    async def chat(_: str) -> str:
        return 'ok'

    app = FastAPI()

    @app.post('/chat', response_model=None)
    @genkit_fastapi_handler(ai)
    def chat_route():
        return chat

    response = TestClient(app).post('/chat', json={'data': 'x'})

    assert response.status_code == 500
    assert response.json() == {'message': 'Internal Error', 'status': 'INTERNAL'}


def test_fastapi_handler_wrapper_returning_non_action_returns_json_internal_error() -> None:
    """A wrapper that does not return an Action is JSON 500 Internal Error."""
    ai = Genkit()
    app = FastAPI()

    @app.post('/chat', response_model=None)
    @genkit_fastapi_handler(ai)
    async def chat_route():
        return 'not an action'

    response = TestClient(app).post('/chat', json={'data': 'x'})

    assert response.status_code == 500
    assert response.json() == {'message': 'Internal Error', 'status': 'INTERNAL'}
