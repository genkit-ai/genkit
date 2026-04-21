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

from fastapi import FastAPI
from fastapi.testclient import TestClient

from genkit import ActionRunContext, Genkit
from genkit.plugins.fastapi import genkit_fastapi_handler


def _assert_is_error_response(parsed: dict) -> None:
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

    @app.post('/error_flow', response_model=None)
    @genkit_fastapi_handler(ai)
    @ai.flow()
    async def raise_error(_: str) -> None:
        raise ValueError('Intentional test error')

    return app


def test_400_missing_data_returns_valid_json() -> None:
    """400 (missing data) must return valid JSON."""
    client = TestClient(create_app())
    response = client.post('/chat', json={})  # no 'data' key
    assert response.status_code == 400
    parsed = json.loads(response.text)
    _assert_is_error_response(parsed)


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
    _assert_is_error_response(parsed)
