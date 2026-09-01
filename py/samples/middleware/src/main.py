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

from genkit import Genkit, ModelResponse, Part, TextPart
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ModelHookParams

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))

_EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')


class PiiRedactConfig(BaseModel):
    pass


@ai.middleware(name='pii_redact')
class PiiRedact(BaseMiddleware[PiiRedactConfig]):
    # The provider sees whatever we put on params.request, including on
    # retries. This is the last place a ticket's email can be removed.
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        new_messages = []
        for message in params.request.messages:
            new_parts = []
            for part in message.content:
                root = part.root
                if isinstance(root, TextPart):
                    redacted = _EMAIL.sub('[REDACTED_EMAIL]', root.text)
                    new_parts.append(Part(root=root.model_copy(update={'text': redacted})))
                else:
                    new_parts.append(part)
            new_messages.append(message.model_copy(update={'content': new_parts}))
        params.request = params.request.model_copy(update={'messages': new_messages})
        return await next_fn(params, ctx)


async def main() -> None:
    ticket = 'Charged twice. Email me at ada@example.com when the refund lands.'
    response = await ai.generate(
        prompt=ticket,
        system='Draft a short support reply. Do not ask the customer to repeat contact details.',
        use=[PiiRedact()],
    )
    print(response.text)


if __name__ == '__main__':
    ai.run_main(main())
