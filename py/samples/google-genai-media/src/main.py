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
import time
from typing import Any

from pydantic import BaseModel, Field

from genkit import Genkit, ModelConfig
from genkit._core._background import lookup_background_action
from genkit._core._typing import Operation, Part, Role, TextPart
from genkit.model import Message, ModelRequest
from genkit.plugins.google_genai import GoogleAI

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

    prompt: str = Field(
        default='A paper airplane gliding through a bright classroom, cinematic slow motion',
        description='Video prompt',
    )
    aspect_ratio: str = Field(default='16:9', description='Video aspect ratio')
    duration_seconds: int = Field(default=5, description='Video duration in seconds')


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


@ai.flow(name='generate_speech')
async def tts_speech_generator(input: SpeechInput) -> dict[str, str | None]:
    """Turn text into speech with one TTS call."""

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


async def _poll_video(operation: Operation) -> Operation:
    """Wait for a background video operation to finish."""

    action = await lookup_background_action(ai.registry, '/background-model/googleai/veo-2.0-generate-001')
    if action is None:
        raise ValueError('Veo background model not found')

    started_at = time.monotonic()
    while not operation.done:
        if time.monotonic() - started_at > 180:
            raise TimeoutError('Timed out waiting for Veo output')
        await asyncio.sleep(3)
        operation = await action.check(operation)
    return operation


@ai.flow(name='generate_video')
async def veo_video_generator(input: VideoInput) -> dict[str, str | int | None]:
    """Generate one video by starting and polling a background model."""

    action = await lookup_background_action(ai.registry, '/background-model/googleai/veo-2.0-generate-001')
    if action is None:
        raise ValueError('Veo background model not found')

    operation = await action.start(
        ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text=input.prompt))])],
            config=ModelConfig.model_validate({
                'aspect_ratio': input.aspect_ratio,
                'duration_seconds': input.duration_seconds,
            }),
        )
    )
    operation = await _poll_video(operation)

    video_url = None
    if isinstance(operation.output, dict):
        message = operation.output.get('message', {})
        content = message.get('content', [])
        if content:
            media = content[0].get('media', {})
            video_url = media.get('url')

    return {
        'model': 'googleai/veo-2.0-generate-001',
        'operation_id': operation.id,
        'video_url': video_url,
        'duration_seconds': input.duration_seconds,
    }


async def main() -> None:
    """Run the fast media demos once."""
    try:
        print(await tts_speech_generator(SpeechInput()))  # noqa: T201
        print(await imagen_image_generator(ImageInput()))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
