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

"""A flow as a Django view. Streaming and request context included."""

from django.http import HttpRequest
from genkit_django import genkit_django_handler
from genkit_google_genai import GoogleAI
from pydantic import BaseModel

from genkit import ActionRunContext, Genkit, ModelResponse
from genkit.plugin_api import RequestData

ai = Genkit(
    plugins=[GoogleAI()],
    model=GoogleAI.gemini_model('gemini-flash-latest'),
)


class SayHiInput(BaseModel):
    name: str = 'Mittens'


async def auth_context(request: RequestData[HttpRequest]) -> dict[str, object]:
    # The caller is identified from the request, not the JSON body.
    return {'username': request.request.META.get('HTTP_AUTHORIZATION') or 'guest'}


@genkit_django_handler(ai, context_provider=auth_context)
@ai.flow()
async def say_hi(input: SayHiInput, ctx: ActionRunContext) -> ModelResponse:
    username = ctx.context.get('username', 'guest')
    stream = ai.generate_stream(
        prompt=f'tell a medium sized joke about {input.name} for user {username}',
        context=ctx.context,
    )
    async for chunk in stream.stream:
        if chunk.text:
            ctx.send_chunk(chunk.text)
    return await stream.response
