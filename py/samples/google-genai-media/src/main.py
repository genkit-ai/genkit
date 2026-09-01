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
# See the License for the specific language governing
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Speech and image are generate(). Video is generate_operation + poll."""

import asyncio

from genkit_google_genai import GoogleAI, VertexAI

from genkit import Genkit

ai = Genkit(plugins=[GoogleAI(), VertexAI()])


async def main() -> None:
    # Speech: pick a TTS model, read the audio off response.media.
    voice = await ai.generate(
        model=GoogleAI.gemini_tts_model('gemini-2.5-flash-preview-tts'),
        prompt='Welcome to the Genkit media sample.',
        config={'speech_config': {'voice_config': {'prebuilt_voice_config': {'voice_name': 'Kore'}}}},
    )
    print(voice.media[0].url if voice.media else 'no audio')

    # Image: same generate(), different model.
    poster = await ai.generate(
        model=GoogleAI.gemini_image_model('gemini-2.5-flash-image'),
        prompt='A watercolor postcard of San Francisco at sunrise',
    )
    print(poster.media[0].url if poster.media else 'no image')

    # Video is a job, not a round-trip. generate_operation hands back a
    # ticket; check_operation is how you find out when the video is ready.
    # Vertex often returns the mp4 inline; operation.output still has a url.
    operation = await ai.generate_operation(
        model=VertexAI.veo_model('veo-3.1-generate-001'),
        prompt='A paper airplane gliding through a bright classroom',
    )
    while not operation.done:
        await asyncio.sleep(3)
        operation = await ai.check_operation(operation)
    print(operation.output)


if __name__ == '__main__':
    ai.run_main(main())
