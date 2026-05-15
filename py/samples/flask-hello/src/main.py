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

"""Flask + Genkit - Serve flows as HTTP endpoints. See README.md."""

from typing import cast

from flask import Flask
from pydantic import BaseModel, Field

from genkit import Genkit, ModelResponse
from genkit._core._action import ActionRunContext
from genkit._core._context import RequestData
from genkit.plugins.flask import genkit_flask_handler
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.google_genai.models.gemini import GoogleAIGeminiVersion

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
) -> ModelResponse:
    """Say hi to the user."""
    username = ctx.context.get('username') if ctx is not None else 'unknown'
    stream_response = ai.generate_stream(
        prompt=f'tell a medium sized joke about {input.name} for user {username}',
    )
    async for chunk in stream_response.stream:
        if ctx is not None and chunk.text:
            ctx.send_chunk(chunk.text)
    return await stream_response.response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # noqa: S104
