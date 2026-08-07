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

"""Google GenAI media - simple examples for speech and image generation."""

from genkit_google_genai import GoogleAI
from pydantic import BaseModel, Field

from genkit import Genkit

ai = Genkit(plugins=[GoogleAI()])


class SpeechInput(BaseModel):
    """Input for TTS."""

    text: str = Field(default='Welcome to the Genkit media sample.', description='Text to speak')
    voice: str = Field(default='Kore', description='Prebuilt voice name')


class ImageInput(BaseModel):
    """Input for image generation."""

    prompt: str = Field(default='A watercolor postcard of San Francisco at sunrise', description='Image prompt')


def _first_media_url(response: object) -> str | None:
    """Extract media URL from first candidate message part if present."""
    message = getattr(response, 'message', None)
    if message is None:
        return None
    content = getattr(message, 'content', [])
    for part in content:
        media = getattr(part.root, 'media', None)
        if media is not None and getattr(media, 'url', None):
            return media.url
    return None


@ai.flow(name='generate_speech')
async def tts_speech_generator(input: SpeechInput) -> dict[str, str | None]:
    """Generate audio bytes with Gemini TTS."""

    response = await ai.generate(
        model='googleai/gemini-2.5-flash-preview-tts',
        prompt=input.text,
        config={'speech_config': {'voice_config': {'prebuilt_voice_config': {'voice_name': input.voice}}}},
    )
    return {'model': 'googleai/gemini-2.5-flash-preview-tts', 'audio_url': _first_media_url(response)}


@ai.flow(name='generate_image')
async def imagen_image_generator(input: ImageInput) -> dict[str, str | None]:
    """Generate one image with Imagen."""

    response = await ai.generate(
        model='googleai/imagen-3.0-generate-002',
        prompt=input.prompt,
        config={'number_of_images': 1},
    )
    return {'model': 'googleai/imagen-3.0-generate-002', 'image_url': _first_media_url(response)}


async def main() -> None:
    """Run the fast media demos once."""
    try:
        print(await tts_speech_generator(SpeechInput()))  # noqa: T201
        print(await imagen_image_generator(ImageInput()))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
