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

"""Claude through the same generate() as Gemini."""

from genkit_anthropic import Anthropic
from pydantic import BaseModel

from genkit import Genkit

ai = Genkit(
    plugins=[Anthropic()],
    model=Anthropic.claude_model('claude-sonnet-4-6'),
)


class Cat(BaseModel):
    name: str
    breed: str
    age: int
    personality: str


async def main() -> None:
    haiku = await ai.generate(prompt='Write a haiku about coding.')
    print(haiku.text)

    # output_schema is the Pydantic model you get back on response.output.
    cat = await ai.generate(
        prompt='Invent a cat named Mittens.',
        output_schema=Cat,
    )
    print(cat.output)

    # thinking= is how you see the model's reasoning stream.
    stream = ai.generate_stream(
        prompt='What is 17 * 23? Think it through, then state the answer.',
        config={'thinking': {'enabled': True, 'budgetTokens': 1024}},
    )
    async for chunk in stream.stream:
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    await stream.response


if __name__ == '__main__':
    ai.run_main(main())
