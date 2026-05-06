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
from typing import ClassVar

import structlog
from pydantic import BaseModel, Field

from genkit import Genkit, Message, MiddlewareRef, Part, Role, TextPart
from genkit.middleware import BaseMiddleware, ModelHookParams
from genkit.plugins.google_genai import GoogleAI

logger = structlog.get_logger(__name__)


class PromptInput(BaseModel):
    """Input shared by middleware flows."""

    prompt: str = Field(
        default='Explain recursion simply.',
        description='Prompt to send to the model',
    )


class LoggingMiddleware(BaseMiddleware):
    """Log request/response details without changing behavior."""

    name: ClassVar[str] = 'logging_mw'

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn: Callable[[ModelHookParams], Awaitable],
    ):
        await logger.ainfo('middleware saw request', message_count=len(params.request.messages))
        response = await next_fn(params)
        await logger.ainfo(
            'middleware saw response',
            finish_reason=response.finish_reason,
        )
        return response


class ConciseReplyMiddleware(BaseMiddleware):
    """Add a short system instruction before the model call.

    ``instruction`` is a pydantic field: override it per call via ``ConciseReplyMiddleware(instruction=...)``
    or via ``MiddlewareRef(name='concise_reply_mw', config={'instruction': ...})``.
    """

    name: ClassVar[str] = 'concise_reply_mw'
    instruction: str = 'Answer in one short paragraph.'

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn: Callable[[ModelHookParams], Awaitable],
    ):
        system_message = Message(
            role=Role.SYSTEM,
            content=[Part(root=TextPart(text=self.instruction))],
        )
        new_req = params.request.model_copy(update={'messages': [system_message, *params.request.messages]})
        return await next_fn(
            ModelHookParams(
                request=new_req,
                on_chunk=params.on_chunk,
                context=params.context,
            )
        )


ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.5-flash',
)

# Register ``ConciseReplyMiddleware`` by name so it can be reached via ``MiddlewareRef``
# (from the Dev UI or cross-process callers); ``LoggingMiddleware`` is used inline below.
ai.define_middleware(ConciseReplyMiddleware)


@ai.flow()
async def logging_demo(input: PromptInput) -> str:
    """Pass a ``BaseMiddleware`` instance directly: no registration needed in-process."""

    response = await ai.generate(prompt=input.prompt, use=[LoggingMiddleware()])
    return response.text


@ai.flow()
async def request_modifier_demo(input: PromptInput) -> str:
    """Resolve a registered middleware by name, with a per-call config override."""

    response = await ai.generate(
        prompt=input.prompt,
        use=[MiddlewareRef(name='concise_reply_mw', config={'instruction': 'Answer in a single haiku.'})],
    )
    return response.text


async def main() -> None:
    """Run both middleware demos once."""
    print(await logging_demo(PromptInput()))  # noqa: T201
    print(await request_modifier_demo(PromptInput(prompt='Write a haiku about recursion.')))  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
