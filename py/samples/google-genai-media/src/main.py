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

"""Google GenAI media - one simple example each for speech, image, and video. Requires GEMINI_API_KEY."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from genkit import Genkit
from genkit.model import Operation
from genkit.plugins.google_genai import GoogleAI

# 1. Initialize Genkit with Google GenAI plugin
ai = Genkit(plugins=[GoogleAI()])


def _first_media_url(response: Any) -> str | None:
    """Return the first media URL in a model response."""
    message = getattr(response, 'message', None)
    if not message:
        return None
    for part in message.content:
        media = getattr(part.root, 'media', None)
        if media and getattr(media, 'url', None):
            return media.url
    return None


async def _poll_video(operation: Operation) -> Operation:
    """Wait for a background video operation to finish."""
    started_at = time.monotonic()
    while not operation.done:
        if time.monotonic() - started_at > 180:
            raise TimeoutError('Timed out waiting for Veo output')
        await asyncio.sleep(3)
        operation = await ai.check_operation(operation)
    return operation


async def main() -> None:
    """Run fast media generation directly without intermediate flow wrappers."""
    try:
        # --- 1. Text-to-Speech (TTS) Generation ---
        speech_res = await ai.generate(
            model='googleai/gemini-2.5-flash-preview-tts',
            prompt='Welcome to Genkit media generation.',
            config={'speech_config': {'voice_config': {'prebuilt_voice_config': {'voice_name': 'Kore'}}}},
        )
        print(f'Audio URL: {_first_media_url(speech_res)}')
        # => Audio URL: data:audio/wav;base64,UklGR...

        # --- 2. Image Generation (Imagen) ---
        image_res = await ai.generate(
            model='googleai/imagen-3.0-generate-002',
            prompt='A watercolor postcard of San Francisco at sunrise',
            config={'number_of_images': 1},
        )
        print(f'Image URL: {_first_media_url(image_res)}')
        # => Image URL: data:image/png;base64,iVBOR...

        # --- 3. Video Generation (Veo Background Operations) ---
        # Note: Video generation takes several minutes; uncomment below to run and poll via ai.generate_operation:
        # operation = await ai.generate_operation(
        #     model='googleai/veo-3.1-generate-preview',
        #     prompt='A paper airplane gliding through a classroom in slow motion',
        # )
        # operation = await _poll_video(operation)
        # => operation.output['message']['content'][0]['media']['url'] = "https://..."
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
