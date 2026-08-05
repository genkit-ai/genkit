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

"""Django + Genkit - Serve flows as HTTP endpoints. See README.md."""

from collections.abc import Mapping
from typing import Any, cast

from django.http import HttpRequest
from genkit_django import genkit_django_handler
from genkit_google_genai import GoogleAI
from pydantic import BaseModel, Field

from genkit import Genkit, ModelResponse
from genkit._core._action import ActionRunContext
from genkit.plugin_api import RequestData

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-flash-latest',
)


class SayHiInput(BaseModel):
    """Input for say_hi flow."""

    name: str = Field(default='Mittens', description='Name to greet')


async def my_context_provider(request: RequestData[HttpRequest]) -> dict[str, Any]:
    """Provide a context for the flow."""
    # Django types `HttpRequest.headers` as a cached_property which trips static
    # checkers; cast to a Mapping so .get() resolves.
    headers = cast(Mapping[str, str], request.request.headers)
    return {'username': headers.get('authorization')}


@genkit_django_handler(ai, context_provider=my_context_provider)
@ai.flow()
async def say_hi(
    input: SayHiInput,
    ctx: ActionRunContext | None = None,
) -> ModelResponse:
    """Say hi to the user, streaming the model output."""
    username = ctx.context.get('username') if ctx is not None else 'unknown'
    stream_response = ai.generate_stream(
        prompt=f'tell a medium sized joke about {input.name} for user {username}',
    )
    async for chunk in stream_response.stream:
        if ctx is not None and chunk.text:
            ctx.send_chunk(chunk.text)
    return await stream_response.response
