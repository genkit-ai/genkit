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

"""Cache a handbook once. Follow-up questions skip those tokens."""

from pathlib import Path

from genkit_google_genai import GoogleAI

from genkit import Genkit, Message, Role, TextPart

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-3-flash-preview'))

handbook = (Path(__file__).resolve().parent.parent / 'handbook.md').read_text()
# Gemini only caches prefixes past ~180k characters. A real handbook is
# already that long; this stand-in is padded so the second call shows
# cached_content_tokens.
handbook = handbook + ('\n# appendix\n' * 20_000)

# The cache lives on the model turn of this prefix. Later generate()
# calls pass the same messages and only pay for the new question.
cached = [
    Message(role=Role.USER, content=[TextPart(text=handbook)]),
    Message(
        role=Role.MODEL,
        content=[TextPart(text='Handbook cached. Ask a question.')],
        metadata={'cache': {'ttl_seconds': 300}},
    ),
]


async def main() -> None:
    for question in (
        'How many PTO days does a full-time employee get?',
        'What is the parental leave policy?',
    ):
        response = await ai.generate(messages=cached, prompt=question)
        usage = response.usage
        print(f'Q: {question}')
        print(f'A: {response.text}')
        print(
            f'   cached_content_tokens={usage.cached_content_tokens if usage else None} '
            f'input_tokens={usage.input_tokens if usage else None}'
        )


if __name__ == '__main__':
    ai.run_main(main())
