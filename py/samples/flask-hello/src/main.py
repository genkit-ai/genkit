#!/usr/bin/env python3
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

"""Flask + Genkit - Serve flows as HTTP endpoints. Requires GEMINI_API_KEY.

Run directly:
    uv run src/main.py
Or inspect live execution and traces in Dev UI:
    genkit start -- uv run src/main.py
"""

from __future__ import annotations

from typing import cast

from flask import Flask
from pydantic import BaseModel, Field

from genkit import ActionRunContext, Genkit, ModelResponse
from genkit.plugin_api import RequestData
from genkit.plugins.flask import genkit_flask_handler
from genkit.plugins.google_genai import GoogleAI

# 1. Initialize Flask app and Genkit AI
app = Flask(__name__)
ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')


class SayHiInput(BaseModel):
    name: str = Field(default='Mittens', description='Name to greet')


# 2. Extract request headers or auth tokens via a custom context provider
async def my_context_provider(request: RequestData[dict[str, object]]) -> dict[str, object]:
    headers_raw = request.request.get('headers') if isinstance(request.request, dict) else None
    headers = cast(dict[str, str], headers_raw) if isinstance(headers_raw, dict) else {}
    return {'username': headers.get('authorization', 'guest')}


# 3. Expose a streaming Genkit flow directly as a Flask POST route
@app.post('/chat')
@genkit_flask_handler(ai, context_provider=my_context_provider)
@ai.flow()
async def say_hi(
    input: SayHiInput,
    ctx: ActionRunContext | None = None,
) -> ModelResponse:
    """Say hi to the user while streaming chunks over HTTP."""
    username = ctx.context.get('username') if ctx is not None else 'unknown'
    stream_response = ai.generate_stream(
        prompt=f'Tell a short, punchy joke about {input.name} for user {username}',
    )
    async for chunk in stream_response.stream:
        if ctx is not None and chunk.text:
            ctx.send_chunk(chunk.text)
    # => Streams live chunks to HTTP client: "Why did Mittens sit on the computer? To keep an eye on the mouse!"
    return await stream_response.response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # noqa: S104
