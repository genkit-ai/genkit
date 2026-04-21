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

"""Middleware - inspect or modify requests before they reach the model."""

from collections.abc import Awaitable, Callable

import structlog
from pydantic import BaseModel, Field

from genkit import Genkit, Message, ModelRequest, ModelResponse, Part, Role, TextPart
from genkit._core._action import ActionRunContext
from genkit.plugins.google_genai import GoogleAI

logger = structlog.get_logger(__name__)
ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-2.5-flash')


class PromptInput(BaseModel):
    """Input shared by middleware flows."""

    prompt: str = Field(default='Explain recursion simply.', description='Prompt to send to the model')


async def logging_middleware(
    req: ModelRequest,
    ctx: ActionRunContext,
    next_handler: Callable[[ModelRequest, ActionRunContext], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Log request/response details without changing behavior."""

    await logger.ainfo('middleware saw request', message_count=len(req.messages))
    response = await next_handler(req, ctx)
    await logger.ainfo('middleware saw response', finish_reason=response.finish_reason)
    return response


async def concise_reply_middleware(
    req: ModelRequest,
    ctx: ActionRunContext,
    next_handler: Callable[[ModelRequest, ActionRunContext], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Add a short system instruction before the model call."""

    system_message = Message(
        role=Role.SYSTEM,
        content=[Part(root=TextPart(text='Answer in one short paragraph.'))],
    )
    return await next_handler(req.model_copy(update={'messages': [system_message, *req.messages]}), ctx)


@ai.flow()
async def logging_demo(input: PromptInput) -> str:
    """Run a prompt through a read-only middleware."""

    response = await ai.generate(prompt=input.prompt, use=[logging_middleware])
    return response.text


@ai.flow()
async def request_modifier_demo(input: PromptInput) -> str:
    """Run a prompt through a request-modifying middleware."""

    response = await ai.generate(prompt=input.prompt, use=[concise_reply_middleware])
    return response.text


async def main() -> None:
    """Run both middleware demos once."""
    try:
        print(await logging_demo(PromptInput()))  # noqa: T201
        print(await request_modifier_demo(PromptInput(prompt='Write a haiku about recursion.')))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
