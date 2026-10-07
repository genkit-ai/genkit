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

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from genkit_fastapi import genkit_fastapi_handler, serve_flow
from genkit_fastapi.exp import to_sse

from genkit import ActionRunContext, Genkit, GenkitError


def assert_is_error_response(parsed: dict) -> None:
    """Assert parsed dict has HttpErrorWireFormat shape (message, status, details)."""
    assert isinstance(parsed, dict)
    assert all(k in parsed for k in ('message', 'status', 'details'))


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
    """Missing input on a required-parameter flow is an action error, not 400."""
    client = TestClient(create_app())
    response = client.post('/chat', json={})
    assert response.status_code == 500
    parsed = json.loads(response.text)
    assert_is_error_response(parsed)


def test_unknown_body_shape_still_returns_400() -> None:
    """Bodies with unrecognized keys still require a data wrapper."""
    client = TestClient(create_app())
    response = client.post('/chat', json={'foo': 'bar'})
    assert response.status_code == 400
    parsed = json.loads(response.text)
    assert_is_error_response(parsed)


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


def test_to_sse_is_experimental_only() -> None:
    """to_sse comes from genkit_fastapi.exp, not the stable import."""
    import genkit_fastapi
    import genkit_fastapi.exp

    assert 'to_sse' not in genkit_fastapi.__all__
    assert not hasattr(genkit_fastapi, 'to_sse')
    assert 'to_sse' in genkit_fastapi.exp.__all__
