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

"""Flask integration sample - Serve Genkit flows via Flask.

This sample demonstrates how to integrate Genkit flows with a Flask web server,
enabling HTTP API endpoints that leverage AI capabilities.

See README.md for testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Flask               │ A simple Python web framework. Like a waiter       │
    │                     │ that takes HTTP requests and serves responses.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ HTTP Endpoint       │ A URL that accepts requests. Like a phone number   │
    │                     │ your app answers when called.                      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ genkit_flask_handler│ Connects Flask routes to Genkit flows. Does the    │
    │                     │ plumbing so you can focus on AI logic.             │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Context Provider    │ A function that adds extra info to each request.   │
    │                     │ Like adding the username from headers.             │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Request Headers     │ Metadata sent with HTTP requests. Like the         │
    │                     │ "From:" on an envelope.                            │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Flask Integration                       | `genkit_flask_handler`              |
| Context Provider                        | `my_context_provider`               |
| Request Header Access                   | `request.request.headers`           |
| Flow Context Usage                      | `ctx.context.get("username")`       |
"""

import os
from typing import cast

from flask import Flask
from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.blocks.model import GenerateResponseWrapper
from genkit.core.action import ActionRunContext
from genkit.core.context import RequestData
from genkit.plugins.flask import genkit_flask_handler
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.google_genai.models.gemini import GoogleAIGeminiVersion
from samples.shared.logging import setup_sample

setup_sample()

if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = input('Please enter your GEMINI_API_KEY: ')

ai = Genkit(
    plugins=[GoogleAI()],
    model=f'googleai/{GoogleAIGeminiVersion.GEMINI_3_FLASH_PREVIEW}',
)

app = Flask(__name__)


class SayHiInput(BaseModel):
    """Input for say_hi flow."""

    name: str = Field(default='Mittens', description='Name to greet')


async def my_context_provider(request: RequestData[dict[str, object]]) -> dict[str, object]:
    """Provide a context for the flow."""
    headers_raw = request.request.get('headers') if isinstance(request.request, dict) else None
    headers = cast(dict[str, str], headers_raw) if isinstance(headers_raw, dict) else {}
    auth_header = headers.get('authorization')
    return {'username': auth_header}


@app.post('/chat')
@genkit_flask_handler(ai, context_provider=my_context_provider)
@ai.flow()
async def say_hi(
    input: SayHiInput,
    ctx: ActionRunContext | None = None,
) -> GenerateResponseWrapper:
    """Say hi to the user."""
    username = ctx.context.get('username') if ctx is not None else 'unknown'
    return await ai.generate(
        on_chunk=ctx.send_chunk if ctx is not None else None,
        prompt=f'tell a medium sized joke about {input.name} for user {username}',
    )
