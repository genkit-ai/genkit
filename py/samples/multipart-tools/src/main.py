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

"""Multipart tools — bare return vs ``response()`` with a PNG."""

from genkit_google_genai import GoogleAI

from genkit import Genkit, Media, MediaPart, MultipartToolResponse, response

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))

# 1x1 PNG so the tool message has real media without a camera.
_PNG = MediaPart(
    media=Media(
        content_type='image/png',
        url='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
    )
)


@ai.tool()
async def weather(city: str) -> str:
    """Look up the weather. A bare return is wrapped as output-only."""
    return f'Sunny in {city}'


@ai.tool()
async def screenshot(label: str) -> MultipartToolResponse:
    """Take a screenshot. ``response()`` is the action result, PNG included."""
    return response({'ok': True, 'label': label}, parts=[_PNG], metadata={'src': 'lab-cam'})


async def main() -> None:
    res = await ai.generate(
        prompt='What is the weather in Austin, and take a screenshot of the lab camera.',
        tools=[weather, screenshot],
    )
    print(res.text)


if __name__ == '__main__':
    ai.run_main(main())
