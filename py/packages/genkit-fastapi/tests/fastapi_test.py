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
from typing import Any, cast

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from genkit_fastapi import genkit_fastapi_handler, serve_flow
from pydantic import BaseModel, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from genkit import (
    FlowContext,
    Genkit,
    GenkitError,
    Interrupt,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    PublicError,
)
from genkit._core._typing import FinishReason, Role
from genkit.plugin_api import provider_error
from genkit.testing import define_scripted_model
from genkit.web import RequestData, to_sse


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
    async def say_hi(name: str, ctx: FlowContext) -> dict[str, str]:
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


def test_empty_body_runs_no_input_flow_on_every_adapter() -> None:
    """POST {} to a served flow that takes no input returns its result."""
    client = TestClient(create_app())
    response = client.post('/void_flow', json={})
    assert response.status_code == 200
    assert response.json() == {'result': {'ok': 'true'}}


def test_input_key_body_is_400_on_every_adapter() -> None:
    """POST {"input": 1} to a served flow is a 400 telling the caller to wrap in data."""
    client = TestClient(create_app())
    response = client.post('/void_flow', json={'input': 1})
    assert response.status_code == 400
    assert (
        response.content
        == b'{"message":"Flow request must be wrapped in {\\"data\\": ...}","status":"INVALID_ARGUMENT"}'
    )


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
    assert parsed['message'] == "Flow 'say_hi' requires input but none was provided."


def test_fastapi_flow_posted_wrong_input_type_returns_400_naming_the_problem_not_the_value() -> None:
    """Posting a dict to a ``str`` flow is a 400 that says what was expected and doesn't echo the input."""
    client = TestClient(create_app())
    response = client.post('/chat', json={'data': {'ssn': '123-45-6789'}})

    assert response.status_code == 400
    assert response.json() == {
        'message': "Invalid input for flow 'say_hi': Input should be a valid string",
        'status': 'INVALID_ARGUMENT',
    }
    assert '123-45-6789' not in response.text


def test_fastapi_nested_flow_called_with_bad_input_is_500_not_400() -> None:
    """A flow that calls another flow with the wrong shape is a server bug: 500, nothing echoed."""
    ai = Genkit()

    class Order(BaseModel):
        order_id: str

    @ai.flow()
    async def place(order: Order) -> str:
        return order.order_id

    @ai.flow()
    async def checkout(cart_id: str) -> str:
        return await place(cast(Any, {'qty': 'internal-row-42'}))

    app = FastAPI()
    app.include_router(serve_flow(checkout, base_path='/checkout'))
    response = TestClient(app).post('/checkout', json={'data': 'c1'})

    assert response.status_code == 500
    assert response.json() == {'message': 'Internal Error', 'status': 'INTERNAL'}


def test_fastapi_custom_validator_text_never_reaches_the_400() -> None:
    """A field_validator's ValueError naming the value and a tenant is 'Invalid value' on the wire."""
    ai = Genkit()

    class Card(BaseModel):
        number: str

        @field_validator('number')
        @classmethod
        def known_card(cls, v: str) -> str:
            raise ValueError(f'card {v} rejected for tenant acme-internal-db-7')

    @ai.flow()
    async def pay(card: Card) -> str:
        return card.number

    app = FastAPI()
    app.include_router(serve_flow(pay, base_path='/pay'))
    response = TestClient(app).post('/pay', json={'data': {'number': '4111111111111111'}})

    assert response.status_code == 400
    assert response.json() == {
        'message': "Invalid input for flow 'pay': number: Invalid value",
        'status': 'INVALID_ARGUMENT',
    }
    assert '4111' not in response.text
    assert 'acme' not in response.text


def test_unknown_body_shape_still_returns_400() -> None:
    """Bodies with unrecognized keys still require a data wrapper."""
    client = TestClient(create_app())
    response = client.post('/chat', json={'foo': 'bar'})
    assert response.status_code == 400
    assert response.json() == {
        'message': 'Flow request must be wrapped in {"data": ...}',
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
    async def whoami(_: str, ctx: FlowContext) -> str:
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


def test_fastapi_flow_error_response_is_redacted_body_and_status() -> None:
    """A flow raising GenkitError('secret') is a compact 500 Internal Error; a bad input stays a compact 400."""
    ai = Genkit()

    @ai.flow()
    async def save(_: str) -> None:
        raise GenkitError(status='INVALID_ARGUMENT', message='secret')

    app = FastAPI()
    app.include_router(serve_flow(save, base_path='/save'))
    client = TestClient(app)

    crashed = client.post('/save', json={'data': 'x'})
    assert crashed.status_code == 500
    assert crashed.content == b'{"message":"Internal Error","status":"INTERNAL"}'
    assert crashed.headers['content-type'] == 'application/json'

    bad_input = client.post('/save', json={'data': {'ssn': '123-45-6789'}})
    assert bad_input.status_code == 400
    assert bad_input.json() == {
        'message': "Invalid input for flow 'save': Input should be a valid string",
        'status': 'INVALID_ARGUMENT',
    }
    assert b'123-45-6789' not in bad_input.content


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


def test_streamed_flow_failure_ends_with_same_error_frame() -> None:
    """A streamed flow raising PublicError('NOT_FOUND', 'no order 99') ends with that exact error frame."""
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

    assert response.content.endswith(b'data: {"error":{"message":"no order 99","status":"NOT_FOUND"}}\n\n')


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


def test_fastapi_flow_raising_interrupt_returns_500_and_sse_internal_error() -> None:
    """A flow raising Interrupt (a BaseException) still gets the Genkit 500 body and SSE error frame.

    The action boundary turns it into GenkitError INTERNAL, so the adapter's
    `except Exception` sees it; without that the run is a bare 500 and the
    stream a 200 with no error event.
    """
    ai = Genkit()

    @ai.flow()
    async def refund(order_id: str) -> str:
        raise Interrupt({'reason': 'needs approval', 'order_id': order_id})

    app = FastAPI()
    app.include_router(serve_flow(refund, base_path='/refund'))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post('/refund', json={'data': 'A-17'})
    assert response.status_code == 500
    assert response.json() == {'message': 'Internal Error', 'status': 'INTERNAL'}

    response = client.post('/refund', json={'data': 'A-17'}, headers={'Accept': 'text/event-stream'})
    assert sse_error_event(response.text) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert 'A-17' not in response.text


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
    async def echo(_: str, ctx: FlowContext) -> dict[str, object]:
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
    async def echo(_: str, ctx: FlowContext) -> dict[str, object]:
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
    async def echo(_: str, ctx: FlowContext) -> dict[str, object]:
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
        raise provider_error(RuntimeError('API key not valid'), http_status=401)

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
        raise provider_error(RuntimeError('API key not valid'), http_status=401)

    app = FastAPI()
    app.include_router(serve_flow(ask, base_path='/ask'))
    with caplog.at_level(logging.ERROR, logger='genkit_fastapi._handler'):
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
    with caplog.at_level(logging.DEBUG, logger='genkit_fastapi._handler'):
        response = TestClient(app).post('/orders', json={'data': 'A-1001'})

    records = [record for record in caplog.records if record.name == 'genkit_fastapi._handler']
    assert response.status_code == 404
    assert [record.levelno for record in records] == [logging.WARNING]
    assert records[0].exc_info is None


def test_fastapi_stream_provider_401_sends_sse_internal_error() -> None:
    """A streamed flow whose model call fails with a provider 401 ends with Internal Error."""
    ai = Genkit()

    @ai.flow()
    async def ask(_: str) -> str:
        raise provider_error(RuntimeError('API key not valid'), http_status=401)

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
    @genkit_fastapi_handler(ai)  # pyright: ignore[reportArgumentType]
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
    @genkit_fastapi_handler(ai)  # pyright: ignore[reportArgumentType]
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


class _Turn:
    """A generate_stream-shaped handle for to_sse tests."""

    def __init__(
        self,
        chunks: list[object],
        result: object = None,
        error: Exception | None = None,
        *,
        fail_after_chunks: bool = False,
    ) -> None:
        self._chunks = chunks
        self._result = result
        self._error = error
        self._fail_after_chunks = fail_after_chunks

    @property
    def stream(self):
        return self._iter()

    async def _iter(self):
        if self._error is not None and not self._chunks and not self._fail_after_chunks:
            raise self._error
        for chunk in self._chunks:
            yield chunk
        if self._error is not None and self._fail_after_chunks:
            raise self._error

    @property
    def response(self):
        return self._done()

    async def _done(self):
        if self._error is not None and not self._fail_after_chunks:
            raise self._error
        return self._result


def _sse_lines(turn: _Turn) -> list[str]:
    async def collect() -> list[str]:
        return [line async for line in to_sse(turn)]

    return asyncio.run(collect())


def test_to_sse_text_chunk_then_result_are_data_frames() -> None:
    """A chunk then a result become data: {message} and data: {result}."""
    lines = _sse_lines(_Turn(chunks=[{'text': 'Hi'}], result={'ok': True}))

    assert lines == [
        'data: {"message":{"text":"Hi"}}\n\n',
        'data: {"result":{"ok":true}}\n\n',
    ]


def test_to_sse_no_chunks_is_only_the_result_frame() -> None:
    """A settled turn with no chunks is a single result frame."""
    lines = _sse_lines(_Turn(chunks=[], result={'ok': True}))

    assert lines == ['data: {"result":{"ok":true}}\n\n']


def test_to_sse_error_after_a_chunk_is_chunk_then_error_no_result() -> None:
    """A failure after a chunk keeps the chunk and sends error, not result."""
    lines = _sse_lines(
        _Turn(
            chunks=[{'text': 'Hi'}],
            error=GenkitError(status='INTERNAL', message='boom'),
            fail_after_chunks=True,
        )
    )

    assert len(lines) == 2
    assert lines[0] == 'data: {"message":{"text":"Hi"}}\n\n'
    assert lines[1].startswith('data: {')
    payload = json.loads(lines[1].removeprefix('data: ').strip())
    assert 'error' in payload
    assert 'result' not in payload


def test_to_sse_error_before_chunks_is_only_the_error_frame() -> None:
    """A failure before any chunk is a single error frame."""
    lines = _sse_lines(_Turn(chunks=[], error=GenkitError(status='INTERNAL', message='boom')))

    assert len(lines) == 1
    payload = json.loads(lines[0].removeprefix('data: ').strip())
    assert 'error' in payload
    assert 'result' not in payload
    assert 'message' not in payload


def test_to_sse_lives_on_genkit_web_only() -> None:
    """to_sse is genkit.web's; genkit_fastapi and genkit_fastapi.exp don't re-export it."""
    import genkit_fastapi
    import genkit_fastapi.exp

    import genkit.web

    assert 'to_sse' in genkit.web.__all__
    assert 'to_sse' not in genkit_fastapi.__all__
    assert 'to_sse' not in genkit_fastapi.exp.__all__


class SourceMeta(BaseModel):
    """Where a CRM record came from."""

    source: str


class CrmDoc(BaseModel):
    """A user model that happens to have a `raw` field."""

    title: str
    raw: SourceMeta


def test_serve_flow_keeps_user_model_raw_field_over_json_and_sse() -> None:
    """A flow returning a model with a `raw` field serves it whole; only agent wrappers unwrap `raw`."""
    ai = Genkit()

    @ai.flow()
    async def fetch_doc(title: str, ctx: FlowContext[CrmDoc]) -> CrmDoc:
        ctx.send_chunk(CrmDoc(title='draft', raw=SourceMeta(source='crm')))
        return CrmDoc(title=title, raw=SourceMeta(source='crm'))

    app = FastAPI()
    app.include_router(serve_flow(fetch_doc, base_path='/doc'))
    client = TestClient(app)

    as_json = client.post('/doc', json={'data': 'T'})
    as_sse = client.post('/doc?stream=true', json={'data': 'T'})

    assert as_json.json() == {'result': {'title': 'T', 'raw': {'source': 'crm'}}}
    assert as_sse.text == (
        'data: {"message":{"title":"draft","raw":{"source":"crm"}}}\n\n'
        'data: {"result":{"title":"T","raw":{"source":"crm"}}}\n\n'
    )


def test_to_sse_keeps_user_model_raw_field() -> None:
    """to_sse on a handle of user models doesn't flatten a `raw` field."""
    doc = CrmDoc(title='T', raw=SourceMeta(source='crm'))

    assert _sse_lines(_Turn(chunks=[doc], result=doc)) == [
        'data: {"message":{"title":"T","raw":{"source":"crm"}}}\n\n',
        'data: {"result":{"title":"T","raw":{"source":"crm"}}}\n\n',
    ]


def test_to_sse_logs_the_served_failure(caplog: pytest.LogCaptureFixture) -> None:
    """The error frame is logged on the genkit.web._sse stdlib logger, 5xx with the traceback."""
    with caplog.at_level(logging.ERROR, logger='genkit.web._sse'):
        _sse_lines(_Turn(chunks=[], error=GenkitError(status='INTERNAL', message='boom')))

    assert [(r.name, r.getMessage()) for r in caplog.records] == [
        ('genkit.web._sse', 'served flow stream failed'),
    ]
    assert caplog.records[0].exc_info is not None


def test_readme_chat_route_sends_text_chunks_and_text_result() -> None:
    """The README's chat route: chunks and result are plain text, and no request rides along."""
    ai = Genkit()
    reply = ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text('Pad thai, no peanuts.')]),
    )
    define_scripted_model(
        ai,
        responses=[reply],
        chunks=[
            [
                ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Pad thai')]),
                ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(', no peanuts.')]),
            ]
        ],
    )

    @ai.flow()
    async def ask_chef(prompt: str, ctx: FlowContext[str]) -> str:
        stream = ai.generate_stream(model='scriptedModel', system='Comp VIPs up to $50.', prompt=prompt)
        async for chunk in stream:
            ctx.send_chunk(chunk.text)
        return (await stream.response).text

    app = FastAPI()

    @app.post('/chat/stream')
    async def chat_stream(prompt: str) -> StreamingResponse:
        stream = ask_chef.stream(prompt, context={'user_id': 'u_42'})
        return StreamingResponse(to_sse(stream), media_type='text/event-stream')

    response = TestClient(app).post('/chat/stream', params={'prompt': 'Dinner?'})

    assert response.text == (
        'data: {"message":"Pad thai"}\n\n'
        'data: {"message":", no peanuts."}\n\n'
        'data: {"result":"Pad thai, no peanuts."}\n\n'
    )
