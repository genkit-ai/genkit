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

"""Google GenAI media - one simple example each for speech, image, and video."""

import asyncio

from genkit_google_genai import GoogleAI, VeoConfigSchema
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


class VideoInput(BaseModel):
    """Input for Veo."""

    model: str = Field(default='googleai/veo-3.1-generate-preview', description='Veo model for generation')
    prompt: str = Field(
        default='A paper airplane gliding through a bright classroom, cinematic slow motion',
        description='Video prompt',
    )
    config: VeoConfigSchema = Field(
        default_factory=lambda: VeoConfigSchema(aspect_ratio='16:9', duration_seconds=5),
        description='Veo model configuration',
    )


@ai.flow(name='generate_speech')
async def tts_speech_generator(input: SpeechInput) -> str | None:
    """Turn text into speech with one TTS call."""
    response = await ai.generate(
        model='googleai/gemini-2.5-flash-preview-tts',
        prompt=input.text,
        config={'speech_config': {'voice_config': {'prebuilt_voice_config': {'voice_name': input.voice}}}},
    )
    return response.media[0].url if response.media else None


@ai.flow(name='generate_image')
async def imagen_image_generator(input: ImageInput) -> str | None:
    """Generate one image with Imagen."""
    response = await ai.generate(
        model='googleai/imagen-3.0-generate-002',
        prompt=input.prompt,
        config={'number_of_images': 1},
    )
    return response.media[0].url if response.media else None


@ai.flow(name='generate_video')
async def veo_video_generator(input: VideoInput) -> str | None:
    """Generate one Veo video with generate_operation() and poll to completion."""
    operation = await ai.generate_operation(
        model=input.model,
        prompt=input.prompt,
        config=input.config,
    )
    while not operation.done:
        await asyncio.sleep(3)
        operation = await ai.check_operation(operation)

    return operation.output.media[0].url if operation.output and operation.output.media else None


async def main() -> None:
    """Run the fast media demos once."""
    try:
        print(await tts_speech_generator(SpeechInput()))  # noqa: T201
        print(await imagen_image_generator(ImageInput()))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
