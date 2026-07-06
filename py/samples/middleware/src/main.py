#!/usr/bin/env python3
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

"""Middleware - inspect or modify requests before they reach the model. Requires GEMINI_API_KEY."""

from __future__ import annotations

from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from genkit import Genkit, Message, Part, Role, TextPart
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.middleware import Middleware

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


# 1. Define custom middleware by subclassing `BaseMiddleware`
class LoggingMiddleware(BaseMiddleware):
    """Log request and response metrics before/after the model executes."""

    async def wrap_model(self, params, ctx: GenerateMiddlewareContext, next_fn):
        await logger.ainfo('middleware saw request', message_count=len(params.request.messages))
        response = await next_fn(params, ctx)
        await logger.ainfo('middleware saw response', finish_reason=response.finish_reason)
        # => Logs: [info] middleware saw request message_count=1 ... middleware saw response finish_reason=STOP
        return response


class ConciseReplyConfig(BaseModel):
    instruction: str = 'Answer in one short paragraph.'


# 2. Register typed middleware that inspects and mutates `params.request`
@ai.middleware(name='concise_reply_mw')
class ConciseReplyMiddleware(BaseMiddleware[ConciseReplyConfig]):
    """Prepend a short system instruction right before calling the model."""

    async def wrap_model(self, params, ctx: GenerateMiddlewareContext, next_fn):
        system_msg = Message(role=Role.SYSTEM, content=[Part(TextPart(text=self.config.instruction))])
        params.request = params.request.model_copy()
        params.request.messages = [system_msg, *params.request.messages]
        return await next_fn(params, ctx)


# 3. Pass middleware instances directly to `ai.generate(use=[...])`
@ai.flow()
async def logging_demo(input: PromptInput) -> str:
    """Pass a `BaseMiddleware` instance directly without global registration."""
    response = await ai.generate(prompt=input.prompt, use=[LoggingMiddleware()])
    # => "Recursion is a technique where a function calls itself to solve smaller instances of the same problem."
    return response.text


@ai.flow()
async def request_modifier_demo(input: PromptInput) -> str:
    """Pass a configured middleware instance overriding the default instruction."""
    response = await ai.generate(
        prompt=input.prompt,
        use=[ConciseReplyMiddleware(instruction='Answer in a single haiku.')],
    )
    # => "Function calls itself,\nSmaller steps toward the end,\nLoop without a loop."
    return response.text


@ai.flow()
async def middleware_prompt_demo(input: PromptInput) -> str:
    """Run `middleware_demo.prompt` configured with plugin retry and `concise_reply_mw`."""
    response = await ai.prompt('middleware_demo')(input={'prompt': input.prompt})
    # => "To understand recursion, you must first understand recursion."
    return response.text


async def main() -> None:
    """Run both middleware demos once."""
    print(await logging_demo(PromptInput()))
    print(await request_modifier_demo(PromptInput(prompt='Write a haiku about recursion.')))


if __name__ == '__main__':
    ai.run_main(main())
