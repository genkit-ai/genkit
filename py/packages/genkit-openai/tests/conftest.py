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


"""Test configuration for the OpenAI compatible plugin."""

from collections.abc import Callable
from typing import Any

import pytest
from genkit_openai.typing import OpenAIConfig
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from genkit import (
    Message,
    ModelRequest,
    Part,
    Role,
    TextPart,
)


@pytest.fixture
def sample_request() -> ModelRequest:
    """Fixture to create a sample ModelRequest object."""
    return ModelRequest(
        messages=[
            Message(
                role=Role.SYSTEM,
                content=[Part(root=TextPart(text='You are an assistant'))],
            ),
            Message(role=Role.USER, content=[Part(root=TextPart(text='Hello, world!'))]),
        ],
        config=OpenAIConfig(
            model='gpt-4',
            top_p=0.9,
            temperature=0.7,
            stop=['stop'],
            max_tokens=100,
        ),
    )


@pytest.fixture
def make_completion() -> Callable[..., ChatCompletion]:
    """Build a chat completion the way the SDK builds one off the wire."""

    def factory(*, content: str = 'Hello, user!', choice: dict[str, Any] | None = None, **extra: Any) -> ChatCompletion:
        payload: dict[str, Any] = {
            'id': 'chatcmpl-abc',
            'created': 1700000000,
            'model': 'gpt-4o-2024-08-06',
            'object': 'chat.completion',
            'choices': [
                {
                    'index': 0,
                    'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': content},
                    **(choice or {}),
                }
            ],
        }
        payload.update(extra)
        return ChatCompletion.construct(**payload)

    return factory


@pytest.fixture
def make_chunk() -> Callable[..., ChatCompletionChunk]:
    """Build a streamed chunk the way the SDK builds one off the wire."""

    def factory(
        *, content: str | None = None, choice: dict[str, Any] | None = None, **extra: Any
    ) -> ChatCompletionChunk:
        payload: dict[str, Any] = {
            'id': 'chatcmpl-stream',
            'created': 1700000000,
            'model': 'grok-4',
            'object': 'chat.completion.chunk',
            'choices': [{'index': 0, 'delta': {'content': content}, **(choice or {})}],
        }
        payload.update(extra)
        return ChatCompletionChunk.construct(**payload)

    return factory
