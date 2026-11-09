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

"""One class on generate(..., use=[...]). This one redacts emails first."""

import re
from collections.abc import Awaitable, Callable

from genkit_google_genai import GoogleAI
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, MultipartToolResponse, Part
from genkit.middleware import GenerateHookParams, Middleware, MiddlewareContext, ToolHookParams

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))

_EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')


def _redact(text: str) -> str:
    return _EMAIL.sub('[REDACTED_EMAIL]', text)


def _redact_parts(parts: list[Part]) -> list[Part]:
    return [Part.from_text(_redact(p.text), metadata=p.metadata) if p.text is not None else p for p in parts]


def _redact_message(message: Message) -> Message:
    return message.model_copy(update={'content': _redact_parts(message.content)})


class PiiRedactConfig(BaseModel):
    pass


@ai.middleware
class PiiRedact(Middleware[PiiRedactConfig]):
    name = 'pii_redact'

    # wrap_generate sees the history before Genkit builds the model request
    # and response.messages from it, so scrubbing here keeps the email out of
    # the provider call and out of the history you store.
    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[GenerateHookParams, MiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        messages = [_redact_message(m) for m in params.options.messages]
        params.options = params.options.model_copy(update={'messages': messages})
        return await next_fn(params, ctx)

    # Tool results go back to the model on the next turn; scrub them too.
    async def wrap_tool(
        self,
        params: ToolHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[ToolHookParams, MiddlewareContext], Awaitable[MultipartToolResponse]],
    ) -> MultipartToolResponse:
        result = await next_fn(params, ctx)
        update: dict[str, object] = {}
        if isinstance(result.output, str):
            update['output'] = _redact(result.output)
        if result.content:
            update['content'] = _redact_parts(result.content)
        return result.model_copy(update=update) if update else result


@ai.tool()
async def lookup_customer(order_id: str) -> str:
    """Look up the customer on an order."""
    return f'Order {order_id} belongs to Ada Lovelace, ada@example.com.'


async def main() -> None:
    ticket = 'Charged twice on order 1042. Email me at ada@example.com when the refund lands.'
    response = await ai.generate(
        prompt=ticket,
        system='Draft a short support reply. Do not ask the customer to repeat contact details.',
        tools=[lookup_customer],
        use=[PiiRedact()],
    )
    print(response.text)
    # => Thanks for flagging the double charge on order 1042. We've issued the refund...


if __name__ == '__main__':
    ai.run_main(main())
