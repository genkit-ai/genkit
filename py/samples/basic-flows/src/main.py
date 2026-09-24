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

"""A flow is a typed, traced, streamable function. No model, no API key."""

import asyncio

from fastapi import FastAPI, Header
from genkit_fastapi import serve_flow
from pydantic import BaseModel

from genkit import ActionRunContext, Genkit

ai = Genkit()


class StatusChunk(BaseModel):
    step: int
    label: str


@ai.flow()
async def welcome(name: str) -> str:
    # ai.run names a step so it shows up as its own span when something
    # looks wrong. Nested flows nest the same way.
    async def lookup() -> str:
        return f'customer: {name}'

    found = await ai.run(name='lookup', fn=lookup)

    async def draft() -> str:
        return f'welcome, {found}'

    return await ai.run(name='draft', fn=draft)


@ai.flow(chunk_type=StatusChunk)
async def status(count: int, ctx: ActionRunContext) -> str:
    # send_chunk is what Dev UI and the HTTP stream paint as progress.
    for i in range(count):
        await asyncio.sleep(0.1)
        ctx.send_chunk(StatusChunk(step=i + 1, label='working'))
    return f'done after {count} updates'


@ai.flow()
async def whoami(name: str, ctx: ActionRunContext) -> str:
    # Request context (auth, tenant) lands here, not as extra args.
    return f'{name} as {ctx.context}'


async def caller(authorization: str | None = Header(default=None)) -> dict[str, str]:
    return {'user': authorization or 'anonymous'}


app = FastAPI()
app.include_router(serve_flow(welcome), prefix='/api')
app.include_router(serve_flow(status), prefix='/api')
app.include_router(serve_flow(whoami, context_dependency=caller), prefix='/api')


async def main() -> None:
    print(await welcome('Ada'))

    streamed = status.stream(3)
    async for chunk in streamed.stream:
        print(chunk)
    print(await streamed.response)


if __name__ == '__main__':
    ai.run_main(main())
