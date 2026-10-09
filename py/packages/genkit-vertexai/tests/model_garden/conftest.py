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

"""A fake Vertex Claude endpoint shared by the Model Garden tests."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from anthropic import AsyncAnthropicVertex


class FakeVertexClaude:
    """Records every rawPredict call and answers like the Messages API."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200,
            json={
                'id': 'msg_1',
                'type': 'message',
                'role': 'assistant',
                'model': 'claude-sonnet-4-6',
                'content': [{'type': 'text', 'text': 'ok'}],
                'stop_reason': 'end_turn',
                'stop_sequence': None,
                'usage': {'input_tokens': 1, 'output_tokens': 1},
            },
        )


@pytest.fixture
def vertex_requests() -> Iterator[list[httpx.Request]]:
    """Route Model Garden Claude through FakeVertexClaude and yield the requests it received."""
    fake = FakeVertexClaude()

    def client(**kwargs: Any) -> AsyncAnthropicVertex:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
        return AsyncAnthropicVertex(access_token='google-token', http_client=http_client, **kwargs)

    with patch('genkit_vertexai._model_garden._anthropic.AsyncAnthropicVertex', client):
        yield fake.requests
