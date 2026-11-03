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

"""Django, Flask and FastAPI serve a failing flow with the same bytes and logs.

Flask and Django keep their own copy of the failure logging and the JSON/SSE
error framing, so they can be read and changed with the framework's idioms;
FastAPI streams through ``genkit.web.to_sse``. This test
is what keeps those copies in step: every case runs the same flow through all
three adapters and checks the status, the exact response bytes and the log
records against one expected result. A fix to the error frame or a log level
that lands in one adapter only fails here.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
from django.http import HttpResponse, StreamingHttpResponse
from django.test import AsyncRequestFactory
from fastapi import FastAPI
from fastapi.testclient import TestClient
from flask import Flask
from genkit_django import genkit_django_handler
from genkit_fastapi import genkit_fastapi_handler
from genkit_flask import genkit_flask_handler

from genkit import FlowContext, Genkit, PublicError
from genkit.web import ContextProvider, RequestData

FlowBody = Callable[[str, FlowContext], Awaitable[Any]]

SSE = {'accept': 'text/event-stream'}


@dataclass(frozen=True)
class Served:
    """What a client and the server log see for one request."""

    status: int
    body: bytes
    # (level, message, has_traceback) for each record the adapter logged.
    logs: tuple[tuple[str, str, bool], ...]


async def _ok(_: str, _ctx: FlowContext) -> str:
    return 'served'


async def _missing_order(_: str, ctx: FlowContext) -> None:
    raise PublicError('NOT_FOUND', 'no order 99')


async def _card_declined(_: str, ctx: FlowContext) -> None:
    raise ValueError('card 4111 declined for alice@example.com')


async def _starter_then_missing(_: str, ctx: FlowContext) -> None:
    ctx.send_chunk({'course': 'starter'})
    raise PublicError('NOT_FOUND', 'no order 99')


async def _starter_then_declined(_: str, ctx: FlowContext) -> None:
    ctx.send_chunk({'course': 'starter'})
    raise ValueError('card 4111 declined for alice@example.com')


def _deny_unauthenticated(_request: RequestData) -> dict[str, object]:
    raise PublicError('UNAUTHENTICATED', 'not signed in')


def _crash_auth_lookup(_request: RequestData) -> dict[str, object]:
    raise ValueError('auth db down for alice@example.com')


def _adapter_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        # genkit.web._sse: FastAPI streams through genkit.web.to_sse.
        if r.name.startswith(('genkit.web', 'genkit_django', 'genkit_flask', 'genkit_fastapi'))
    ]


def _adapter_logs(caplog: pytest.LogCaptureFixture) -> tuple[tuple[str, str, bool], ...]:
    return tuple((r.levelname, r.getMessage(), r.exc_info is not None) for r in _adapter_records(caplog))


def _serve_flask(
    body: FlowBody,
    headers: dict[str, str],
    context_provider: ContextProvider | None = None,
) -> tuple[int, bytes]:
    ai = Genkit()
    app = Flask(__name__)

    @app.post('/order')
    @genkit_flask_handler(ai, context_provider=context_provider)
    @ai.flow(name='order')
    async def order(data: str, ctx: FlowContext) -> Any:  # noqa: ANN401
        return await body(data, ctx)

    response = app.test_client().post('/order', json={'data': 'table 4'}, headers=headers)
    return response.status_code, response.get_data()


def _serve_fastapi(
    body: FlowBody,
    headers: dict[str, str],
    context_provider: ContextProvider | None = None,
) -> tuple[int, bytes]:
    ai = Genkit()
    app = FastAPI()

    @app.post('/order', response_model=None)
    @genkit_fastapi_handler(ai, context_provider=context_provider)
    @ai.flow(name='order')
    async def order(data: str, ctx: FlowContext) -> Any:  # noqa: ANN401
        return await body(data, ctx)

    response = TestClient(app).post('/order', json={'data': 'table 4'}, headers=headers)
    return response.status_code, response.content


def _serve_django(
    body: FlowBody,
    headers: dict[str, str],
    context_provider: ContextProvider | None = None,
) -> tuple[int, bytes]:
    ai = Genkit()

    @genkit_django_handler(ai, context_provider=context_provider)
    @ai.flow(name='order')
    async def order(data: str, ctx: FlowContext) -> Any:  # noqa: ANN401
        return await body(data, ctx)

    async def call() -> tuple[int, bytes]:
        request = AsyncRequestFactory().post(
            '/order', data='{"data":"table 4"}', content_type='application/json', headers=headers
        )
        response = await order(request)
        if isinstance(response, StreamingHttpResponse):
            parts = [part async for part in response.streaming_content]
            content = b''.join(p if isinstance(p, bytes) else bytes(p) for p in parts)
        else:
            assert isinstance(response, HttpResponse)
            content = response.content
        return response.status_code, content

    return asyncio.run(call())


ADAPTERS = {'django': _serve_django, 'flask': _serve_flask, 'fastapi': _serve_fastapi}

INTERNAL = b'{"message":"Internal Error","status":"INTERNAL"}'
NOT_FOUND = b'{"message":"no order 99","status":"NOT_FOUND"}'
UNAUTHENTICATED = b'{"message":"not signed in","status":"UNAUTHENTICATED"}'

CASES = [
    pytest.param(
        _missing_order,
        {},
        None,
        Served(404, NOT_FOUND, (('WARNING', "served action 'order' run failed: NOT_FOUND: no order 99", False),)),
        id='public-error',
    ),
    pytest.param(
        _card_declined,
        {},
        None,
        Served(500, INTERNAL, (('ERROR', "served action 'order' run failed", True),)),
        id='internal-error',
    ),
    pytest.param(
        _starter_then_missing,
        SSE,
        None,
        Served(
            200,
            b'data: {"message":{"course":"starter"}}\n\n' + b'data: {"error":' + NOT_FOUND + b'}\n\n',
            (('WARNING', "served action 'order' stream failed: NOT_FOUND: no order 99", False),),
        ),
        id='stream-public-error',
    ),
    pytest.param(
        _starter_then_declined,
        SSE,
        None,
        Served(
            200,
            b'data: {"message":{"course":"starter"}}\n\n' + b'data: {"error":' + INTERNAL + b'}\n\n',
            (('ERROR', "served action 'order' stream failed", True),),
        ),
        id='stream-internal-error',
    ),
    pytest.param(
        _ok,
        {},
        _deny_unauthenticated,
        Served(
            401,
            UNAUTHENTICATED,
            (('WARNING', "served action 'order' context provider failed: UNAUTHENTICATED: not signed in", False),),
        ),
        id='context-provider-public-error',
    ),
    pytest.param(
        _ok,
        {},
        _crash_auth_lookup,
        Served(500, INTERNAL, (('ERROR', "served action 'order' context provider failed", True),)),
        id='context-provider-internal-error',
    ),
]


@pytest.mark.parametrize('adapter', list(ADAPTERS))
@pytest.mark.parametrize(('body', 'headers', 'context_provider', 'expected'), CASES)
def test_failed_flow_is_served_the_same_by_every_adapter(
    adapter: str,
    body: FlowBody,
    headers: dict[str, str],
    context_provider: ContextProvider | None,
    expected: Served,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing flow gets the same status, bytes and log records from Django, Flask and FastAPI."""
    # 1. Serve the flow through one adapter and capture its log records
    with caplog.at_level(logging.DEBUG):
        status, content = ADAPTERS[adapter](body, headers, context_provider)

    # 2. The client and the server log see exactly the shared contract
    assert Served(status, content, _adapter_logs(caplog)) == expected
    assert [getattr(r, 'genkit_action', None) for r in _adapter_records(caplog)] == ['order'] * len(expected.logs)

    # 3. The internal message never reaches the client
    assert b'alice@example.com' not in content
