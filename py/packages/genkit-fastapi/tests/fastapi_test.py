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

import asyncio
import json
import logging
from typing import Any

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from genkit_fastapi import genkit_fastapi_handler, serve_flow
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

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


def test_serve_flow_message_body_is_rejected_as_bad_request() -> None:
    """POST {"message": "hi"} to a flow route is 400, not a 500 from inside the flow."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str) -> str:
        return f'Hi {name}'

    app = FastAPI()
    app.include_router(serve_flow(greet, base_path='/greet'))
    client = TestClient(app)

    response = client.post('/greet', json={'message': 'hi'})

    assert response.status_code == 400
    parsed = json.loads(response.text)
    assert_is_error_response(parsed)
    assert 'must be wrapped in {"data": ...}' in parsed['message']


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


def test_fastapi_flow_raising_not_found_returns_500_internal_error() -> None:
    """FastAPI POST to a flow that raises GenkitError NOT_FOUND is 500, not a missing route."""
    ai = Genkit()

    @ai.flow()
    async def missing(_: str) -> None:
        raise GenkitError(status='NOT_FOUND', message='no recipe for alice@example.com')

    app = FastAPI()
    app.include_router(serve_flow(missing, base_path='/missing'))
    response = TestClient(app).post('/missing', json={'data': 'x'})

    assert response.status_code == 500
    assert response.json() == {'message': 'Internal Error', 'status': 'INTERNAL'}
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


def test_fastapi_stream_flow_raising_not_found_sends_sse_internal_error() -> None:
    """FastAPI SSE to a flow that raises GenkitError NOT_FOUND ends with Internal Error."""
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

    assert sse_error_event(response.text) == {'message': 'Internal Error', 'status': 'INTERNAL'}
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


def test_fastapi_context_provider_sees_method_lowercase_headers_and_input() -> None:
    """FastAPI context_provider sees method, lowercase headers, and input."""
    ai = Genkit()
    app = FastAPI()

    async def provider(request_data: RequestData) -> dict[str, object]:
        return {
            'method': request_data.method,
            'authorization': request_data.headers['authorization'],
            'input': request_data.input,
        }

    @app.post('/echo', response_model=None)
    @genkit_fastapi_handler(ai, context_provider=provider)
    @ai.flow()
    async def echo(_: str, ctx: ActionRunContext) -> dict[str, object]:
        return {
            'method': ctx.context['method'],
            'authorization': ctx.context['authorization'],
            'input': ctx.context['input'],
        }

    response = TestClient(app).post(
        '/echo',
        json={'data': 'hello'},
        headers={'Authorization': 'Bearer tok'},
    )

    assert response.status_code == 200
    assert response.json()['result'] == {
        'method': 'POST',
        'authorization': 'Bearer tok',
        'input': 'hello',
    }


def test_request_data_duplicate_authorization_is_comma_joined() -> None:
    """Two Authorization values become one comma-joined string."""
    ai = Genkit()
    app = FastAPI()

    async def provider(request_data: RequestData) -> dict[str, object]:
        return {'authorization': request_data.headers.get('authorization')}

    @app.post('/echo', response_model=None)
    @genkit_fastapi_handler(ai, context_provider=provider)
    @ai.flow()
    async def echo(_: str, ctx: ActionRunContext) -> dict[str, object]:
        return {'authorization': ctx.context['authorization']}

    response = TestClient(app).post(
        '/echo',
        json={'data': 'hello'},
        headers=[('Authorization', 'Bearer a'), ('Authorization', 'Bearer b')],
    )

    assert response.status_code == 200
    assert response.json()['result'] == {'authorization': 'Bearer a, Bearer b'}


def test_request_data_duplicate_x_forwarded_for_is_comma_joined() -> None:
    """Two X-Forwarded-For values keep the full proxy chain."""
    ai = Genkit()
    app = FastAPI()

    async def provider(request_data: RequestData) -> dict[str, object]:
        return {'xff': request_data.headers.get('x-forwarded-for')}

    @app.post('/echo', response_model=None)
    @genkit_fastapi_handler(ai, context_provider=provider)
    @ai.flow()
    async def echo(_: str, ctx: ActionRunContext) -> dict[str, object]:
        return {'xff': ctx.context['xff']}

    response = TestClient(app).post(
        '/echo',
        json={'data': 'hello'},
        headers=[('X-Forwarded-For', '203.0.113.1'), ('X-Forwarded-For', '198.51.100.2')],
    )

    assert response.status_code == 200
    assert response.json()['result'] == {'xff': '203.0.113.1, 198.51.100.2'}


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


def _defaulted_and_bad_return_app() -> FastAPI:
    ai = Genkit()
    app = FastAPI()

    @ai.flow()
    async def greet(name: str = 'world') -> str:
        return f'hello {name}'

    @ai.flow()
    async def charge(name: str) -> dict[str, int]:
        return {'account': name}  # type: ignore[dict-item]

    app.include_router(serve_flow(greet, base_path='/greet'))
    app.include_router(serve_flow(charge, base_path='/charge'))
    return app


def test_fastapi_flow_with_default_and_empty_body_uses_python_default() -> None:
    """POST `{}` to a served `greet(name: str = 'world')` returns the default's result."""
    client = TestClient(_defaulted_and_bad_return_app())
    response = client.post('/greet', json={})
    assert response.status_code == 200
    assert response.json() == {'result': 'hello world'}


def test_fastapi_flow_with_default_and_null_data_uses_python_default() -> None:
    """POST `{"data": null}` behaves like an omitted input."""
    client = TestClient(_defaulted_and_bad_return_app())
    response = client.post('/greet', json={'data': None})
    assert response.status_code == 200
    assert response.json() == {'result': 'hello world'}


def test_fastapi_stream_flow_with_default_and_null_data_uses_python_default() -> None:
    """Streaming POST `{"data": null}` ends with the default's result."""
    client = TestClient(_defaulted_and_bad_return_app())
    response = client.post('/greet', json={'data': None}, headers={'accept': 'text/event-stream'})
    assert response.status_code == 200
    assert 'data: {"result":"hello world"}' in response.text


def test_fastapi_flow_returning_wrong_shape_returns_500_internal_error() -> None:
    """A served flow whose return doesn't match `-> dict[str, int]` gets 500 Internal Error, not the bad value."""
    client = TestClient(_defaulted_and_bad_return_app())
    response = client.post('/charge', json={'data': 'acme'})
    assert response.status_code == 500
    assert response.json() == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert 'acme' not in response.text


class Receipt(BaseModel):
    table: int
    note: str | None = None


def test_fastapi_flow_returning_dict_for_model_sends_model_dump() -> None:
    """POST to a `-> Receipt` flow that returns `{'table': 4, 'note': None}` gives `{"result": {"table": 4}}`."""
    ai = Genkit()
    app = FastAPI()

    @ai.flow()
    async def close_tab(table: int) -> Receipt:
        return {'table': table, 'note': None}  # type: ignore[return-value]

    @app.post('/close_tab', response_model=None)
    @genkit_fastapi_handler(ai)
    async def close_tab_route():
        return close_tab

    response = TestClient(app).post('/close_tab', json={'data': 4})

    assert response.status_code == 200
    assert response.json() == {'result': {'table': 4}}


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


def test_served_flow_client_error_logs_one_warning_without_traceback(caplog: pytest.LogCaptureFixture) -> None:
    """A 4xx on a served flow is the caller's mistake: one warning, no traceback."""
    ai = Genkit()

    @ai.flow()
    async def lookup_order(_: str) -> str:
        raise PublicError('NOT_FOUND', 'order not found')

    app = FastAPI()
    app.include_router(serve_flow(lookup_order, base_path='/orders'))
    with caplog.at_level(logging.DEBUG, logger='genkit_fastapi.handler'):
        response = TestClient(app).post('/orders', json={'data': 'A-1001'})

    records = [record for record in caplog.records if record.name == 'genkit_fastapi.handler']
    assert response.status_code == 404
    assert [record.levelno for record in records] == [logging.WARNING]
    assert records[0].exc_info is None


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


def test_fastapi_context_provider_http_exception_keeps_its_status() -> None:
    """An HTTPException from context_provider is the app's own response, not a 500."""
    ai = Genkit()

    def require_token(_request: RequestData) -> dict[str, object]:
        raise HTTPException(status_code=401, detail='no token', headers={'WWW-Authenticate': 'Bearer'})

    @ai.flow()
    async def chat(_: str) -> str:
        return 'ok'

    app = FastAPI()

    @app.post('/chat', response_model=None)
    @genkit_fastapi_handler(ai, context_provider=require_token)
    async def chat_route():
        return chat

    response = TestClient(app).post('/chat', json={'data': 'x'})

    assert response.status_code == 401
    assert response.json() == {'detail': 'no token'}
    assert response.headers['www-authenticate'] == 'Bearer'


def test_served_flow_missing_model_returns_500() -> None:
    """POST a flow whose generate names a model that isn't registered is 500, not 404."""
    ai = Genkit()

    @ai.flow()
    async def poem(_: None) -> str:
        res = await ai.generate(model='googleai/typo-flash', prompt='hi')
        return res.text

    app = FastAPI()
    app.include_router(serve_flow(poem, base_path='/poem'))
    response = TestClient(app).post('/poem', json={'data': None})

    assert response.status_code == 500
    assert response.json() == {'message': 'Internal Error', 'status': 'INTERNAL'}


def test_fastapi_context_provider_starlette_http_exception_keeps_401() -> None:
    """A Starlette HTTPException from context_provider stays 401, not 500."""
    ai = Genkit()

    def require_token(_request: RequestData) -> dict[str, object]:
        raise StarletteHTTPException(status_code=401, detail='no token')

    @ai.flow()
    async def chat(_: str) -> str:
        return 'ok'

    app = FastAPI()

    @app.post('/chat', response_model=None)
    @genkit_fastapi_handler(ai, context_provider=require_token)
    async def chat_route():
        return chat

    response = TestClient(app).post('/chat', json={'data': 'x'})

    assert response.status_code == 401
    assert response.json() == {'detail': 'no token'}


def test_fastapi_wrapper_http_exception_keeps_503() -> None:
    """An HTTPException from the async wrapper is the app's own response, not a 500."""
    ai = Genkit()

    app = FastAPI()

    @app.post('/chat', response_model=None)
    @genkit_fastapi_handler(ai)
    async def chat_route():
        raise HTTPException(status_code=503, detail='not ready')

    response = TestClient(app).post('/chat', json={'data': 'x'})

    assert response.status_code == 503
    assert response.json() == {'detail': 'not ready'}


async def post_sse_then_hang_up(app: FastAPI, path: str) -> list[bytes]:
    """POST `{"data": "x"}` asking for SSE, then disconnect right after the first body chunk arrives."""
    first_chunk = asyncio.Event()
    request_read = False
    bodies: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_read
        if not request_read:
            request_read = True
            return {'type': 'http.request', 'body': json.dumps({'data': 'x'}).encode(), 'more_body': False}
        await first_chunk.wait()
        return {'type': 'http.disconnect'}

    async def send(message: dict[str, Any]) -> None:
        if message['type'] == 'http.response.body' and message.get('body'):
            bodies.append(message['body'])
            first_chunk.set()

    scope = {
        'type': 'http',
        'asgi': {'version': '3.0', 'spec_version': '2.3'},
        'http_version': '1.1',
        'method': 'POST',
        'scheme': 'http',
        'path': path,
        'raw_path': path.encode(),
        'root_path': '',
        'query_string': b'',
        'headers': [(b'content-type', b'application/json'), (b'accept', b'text/event-stream')],
        'client': ('testclient', 50000),
        'server': ('testserver', 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    return bodies


@pytest.mark.asyncio
async def test_fastapi_streaming_client_disconnect_cancels_flow() -> None:
    """A client that hangs up after the first SSE chunk of a streamed flow route cancels the flow."""
    ai = Genkit()
    flow_cancelled = asyncio.Event()

    @ai.flow()
    async def slow(_: str, ctx: ActionRunContext) -> str:
        ctx.send_chunk(1)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            flow_cancelled.set()
            raise
        return 'never'

    app = FastAPI()
    app.include_router(serve_flow(slow, base_path='/slow'))

    bodies = await post_sse_then_hang_up(app, '/slow')

    assert bodies == [b'data: {"message":1}\n\n']
    assert flow_cancelled.is_set()
