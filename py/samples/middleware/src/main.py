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

from pathlib import Path

import structlog
from genkit_google_genai import GoogleAI
from genkit_middleware import Middleware
from pydantic import BaseModel, Field

from genkit import Genkit, Message, Part, Role, TextPart
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext

logger = structlog.get_logger(__name__)


class PromptInput(BaseModel):
    """Input shared by middleware flows."""

    prompt: str = Field(
        default='Explain recursion simply.',
        description='Prompt to send to the model',
    )


ai = Genkit(
    plugins=[GoogleAI(), Middleware()],
    model='googleai/gemini-flash-latest',
    prompt_dir=Path(__file__).resolve().parent.parent / 'prompts',
)


class LoggingMiddleware(BaseMiddleware):
    """Log request/response details without changing behavior."""

    async def wrap_model(self, params, ctx: GenerateMiddlewareContext, next_fn):
        await logger.ainfo('middleware saw request', message_count=len(params.request.messages))
        response = await next_fn(params, ctx)
        await logger.ainfo('middleware saw response', finish_reason=response.finish_reason)
        return response


class ConciseReplyConfig(BaseModel):
    """Per-call system instruction for ConciseReplyMiddleware."""

    instruction: str = 'Answer in one short paragraph.'


@ai.middleware(name='concise_reply_mw')
class ConciseReplyMiddleware(BaseMiddleware[ConciseReplyConfig]):
    """Prepend a short system instruction before the model call.

    Each call can supply its own value by constructing a fresh instance:
    ``ConciseReplyMiddleware(instruction=...)``.
    """

    async def wrap_model(self, params, ctx: GenerateMiddlewareContext, next_fn):
        system_message = Message(
            role=Role.SYSTEM,
            content=[Part(root=TextPart(text=self.config.instruction))],
        )
        params.request = params.request.model_copy()
        params.request.messages = [system_message, *params.request.messages]
        return await next_fn(params, ctx)


@ai.flow()
async def logging_demo(input: PromptInput) -> str:
    """Pass a ``BaseMiddleware`` instance directly: no registration needed in-process."""

    response = await ai.generate(prompt=input.prompt, use=[LoggingMiddleware()])
    return response.text


@ai.flow()
async def request_modifier_demo(input: PromptInput) -> str:
    """Pass a configured middleware instance with a per-call override of ``instruction``."""

    response = await ai.generate(
        prompt=input.prompt,
        use=[ConciseReplyMiddleware(instruction='Answer in a single haiku.')],
    )
    return response.text


@ai.flow()
async def middleware_prompt_demo(input: PromptInput) -> str:
    """Run ``middleware_demo.prompt`` with plugin retry and ``concise_reply_mw``."""

    response = await ai.prompt('middleware_demo')(input={'prompt': input.prompt})
    return response.text


async def main() -> None:
    """Run both middleware demos once."""
    print(await logging_demo(PromptInput()))  # noqa: T201
    print(await request_modifier_demo(PromptInput(prompt='Write a haiku about recursion.')))  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
