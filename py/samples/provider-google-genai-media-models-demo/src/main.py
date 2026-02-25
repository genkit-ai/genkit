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

"""Media Generation Models Demo - Veo, TTS, Lyria, Imagen, and Gemini Image.

This demo showcases all media generation capabilities in the Google GenAI plugin.
See README.md for detailed testing instructions and configuration options.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Veo                 │ Google's video generation AI. Describe a scene,    │
    │                     │ get a video clip. Takes 30s to 5min.               │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ TTS                 │ Text-To-Speech. AI reads text aloud with           │
    │                     │ realistic human-like voices.                       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Lyria               │ AI music generation. Describe a song and           │
    │                     │ get an audio file back.                            │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Gemini Image        │ Image generation from Gemini. Describe what        │
    │                     │ you want to see, AI draws it.                      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Background Model    │ Long-running generation (like Veo). Start it,      │
    │                     │ poll for status, get result when ready.            │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Standard Model      │ Quick generation (TTS, Image). Get results         │
    │                     │ directly in the response.                          │
    └─────────────────────┴────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────────────────┐
    │                    Media Generation Models Overview                     │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   Model Type      │ Output         │ API              │ Latency        │
    │   ────────────────┼────────────────┼──────────────────┼────────────────│
    │   Veo (Video)     │ MP4 video URL  │ Background Model │ 30s - 5min     │
    │   TTS (Speech)    │ Audio (WAV)    │ Standard Model   │ ~1-5 seconds   │
    │   Lyria (Audio)   │ Audio (WAV)    │ Standard Model   │ ~5-30 seconds  │
    │   Imagen          │ Image (PNG)    │ Standard Model   │ ~5-15 seconds  │
    │   Gemini Image    │ Image          │ Standard Model   │ ~5-15 seconds  │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

Why Different Model Patterns?
=============================
- **Standard Models** (TTS, Lyria, Gemini Image): Return results directly.
  Response time is typically under 30 seconds.

- **Background Models** (Veo): Long-running operations that can take minutes.
  Uses a start/poll/complete pattern with Operation objects.

    ┌──────────────────────────────────────────────────────────────────────┐
    │                    Response Time Comparison                           │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │   Text:   |█| ~100ms                                                 │
    │   TTS:    |████| ~1-5s                                               │
    │   Image:  |████████| ~5-15s                                          │
    │   Audio:  |████████████| ~5-30s                                      │
    │   Video:  |████████████████████████████████████| 30s - 5min          │
    │                                                                       │
    └──────────────────────────────────────────────────────────────────────┘

Available Flows
===============
- `tts_speech_generator` - Text-to-speech with voice selection
- `imagen_image_generator` - Image generation with Imagen (predict API)
- `gemini_image_generator` - Image generation with Gemini (native)
- `lyria_audio_generator` - Music/audio generation (Vertex AI)
- `veo_video_generator` - Video generation (background model)
- `describe_image_with_gemini` - Image-to-text description
- `generate_images` - Multimodal image generation with photos
- `multipart_tool_calling` - Tool calling with image input/output
- `gemini_image_editing` - Image editing (inpainting/outpainting)
- `nano_banana_pro` - 4K image config with aspect ratio
- `gemini_media_resolution` - Media resolution control
- `multimodal_input` - Multimodal prompting
"""

import asyncio
import base64
import os
import pathlib
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.blocks.background_model import lookup_background_action
from genkit.blocks.model import GenerateResponseWrapper
from genkit.core.action import ActionRunContext
from genkit.core.typing import (
    Error,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Message,
    ModelInfo,
    Operation,
    Part,
    Role,
    Supports,
    TextPart,
)
from genkit.types import (
    GenerationCommonConfig,
    Media,
    MediaPart,
    Metadata,
)
from samples.shared.logging import setup_sample

setup_sample()

HAS_GEMINI_API_KEY = bool(os.getenv('GEMINI_API_KEY'))
HAS_GCP_PROJECT = bool(os.getenv('GOOGLE_CLOUD_PROJECT'))


def _exception_chain_message(exc: BaseException) -> str:
    """Collect error messages from the full exception ``__cause__`` chain.

    The Genkit framework wraps provider exceptions in ``GenkitError``,
    so the top-level ``str(e)`` only contains a generic wrapper message
    like ``'INTERNAL: Error while running action ...'``.  The original
    API error—e.g. ``'404 NOT_FOUND'``—is buried in ``e.__cause__``.
    This helper concatenates messages from every exception in the chain
    so that keyword checks (``'NOT_FOUND'``, ``'quota'``, etc.) work
    regardless of wrapping depth.
    """
    parts: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        parts.append(str(current))
        current = current.__cause__
    return ' '.join(parts)


# Initialize Genkit
if HAS_GEMINI_API_KEY:
    from genkit.plugins.google_genai import (
        GeminiConfigSchema,
        GeminiImageConfigSchema,
        GoogleAI,
    )

    ai = Genkit(plugins=[GoogleAI()])
else:
    ai = Genkit()

if HAS_GCP_PROJECT:
    pass


class TtsInput(BaseModel):
    """Input for TTS speech generation flow."""

    text: str = Field(
        default='Meow! The magnificent cat leaps gracefully across the rooftops at sunset!',
        description='The text to convert to speech.',
    )
    voice: str = Field(
        default='Kore',
        description='The voice to use for speech generation. Options: Zephyr, Puck, Charon, Kore, etc.',
    )


class ImageInput(BaseModel):
    """Input for image generation flow."""

    prompt: str = Field(
        default='A fluffy orange cat wearing a tiny crown, sitting on a velvet cushion',
        description='Description of the image to generate.',
    )
    aspect_ratio: str = Field(
        default='16:9',
        description='Image aspect ratio (e.g., 16:9, 1:1, 9:16).',
    )


class ImagenInput(BaseModel):
    """Input for Imagen image generation flow."""

    prompt: str = Field(
        default='A photorealistic image of a cat astronaut floating in space with Earth in the background',
        description='Description of the image to generate with Imagen.',
    )
    number_of_images: int = Field(
        default=1,
        description='Number of images to generate (1-4).',
    )


class AudioInput(BaseModel):
    """Input for audio/music generation flow."""

    prompt: str = Field(
        default='Playful jazz music with soft piano and gentle purring cat sounds',
        description='Description of the audio to generate.',
    )
    negative_prompt: str | None = Field(
        default=None,
        description='A description of what to avoid in the generated audio.',
    )


class VideoInput(BaseModel):
    """Input for video generation flow."""

    prompt: str = Field(
        default='A cat chasing a butterfly through a sunlit garden, cinematic slow motion',
        description='Description of the video to generate.',
    )
    aspect_ratio: str = Field(
        default='16:9',
        description='Video aspect ratio (e.g., 16:9, 9:16).',
    )
    duration_seconds: int = Field(
        default=5,
        description='Duration of the video in seconds.',
    )


class GenerateImagesInput(BaseModel):
    """Input for multimodal image generation flow."""

    name: str = Field(default='a fluffy cat', description='Subject to generate images about')


class DescribeImageInput(BaseModel):
    """Input for image description flow."""

    data: str = Field(
        default='auto',
        description=(
            "Image data as a data URI (e.g., 'data:image/jpeg;base64,...'). "
            "Set to 'auto' to use the bundled default image."
        ),
    )


class ImageEditingInput(BaseModel):
    """Input for image editing flow."""

    prompt: str = Field(
        default='add the plant to my room',
        description='Editing instruction for how to combine/modify the images.',
    )


class ToolCallingInput(BaseModel):
    """Input for multipart tool calling flow."""

    prompt: str = Field(
        default="Tell me what I'm seeing on the screen.",
        description='Prompt asking the model to use the screenshot tool.',
    )


class NanoBananaProInput(BaseModel):
    """Input for 4K image generation flow."""

    prompt: str = Field(
        default='Generate a picture of a sunset in the mountains by a lake',
        description='Description of the image to generate in 4K.',
    )


class MediaResolutionInput(BaseModel):
    """Input for media resolution query flow."""

    prompt: str = Field(
        default='What is in this picture?',
        description='Question to ask about the bundled image.',
    )


class MultimodalInput(BaseModel):
    """Input for multimodal prompting flow."""

    prompt: str = Field(
        default='describe this photo',
        description='Instruction for describing the bundled photo.',
    )


_operations: dict[str, dict[str, Any]] = {}


class SimulatedTtsConfig(BaseModel):
    """Configuration for simulated TTS."""

    voice_name: str = Field(default='Kore', description='Voice to use')


class SimulatedImageConfig(BaseModel):
    """Configuration for simulated image generation."""

    aspect_ratio: str = Field(default='16:9', description='Image aspect ratio')


class SimulatedVeoConfig(BaseModel):
    """Configuration for simulated Veo."""

    duration_seconds: int = Field(default=5, description='Video duration')
    aspect_ratio: str = Field(default='16:9', description='Video aspect ratio')


class SimulatedLyriaConfig(BaseModel):
    """Configuration for simulated Lyria."""

    sample_count: int = Field(default=1, description='Number of audio samples')


def _extract_prompt(request: GenerateRequest) -> str:
    """Extract text prompt from request."""
    if request.messages:
        for msg in request.messages:
            for part in msg.content:
                if hasattr(part.root, 'text') and part.root.text:
                    return str(part.root.text)
    return ''


# --- Simulated TTS ---
async def simulated_tts_generate(
    request: GenerateRequest,
    ctx: ActionRunContext,
) -> GenerateResponse:
    """Simulate TTS audio generation."""
    _extract_prompt(request)

    await asyncio.sleep(1)  # Simulate processing

    # Create a simple WAV header + silence (for demo)
    # Real TTS would return actual audio
    fake_audio = base64.b64encode(b'RIFF' + b'\x00' * 100).decode()

    return GenerateResponse(
        message=Message(
            role=Role.MODEL,
            content=[
                Part.model_validate({
                    'media': {
                        'url': f'data:audio/wav;base64,{fake_audio}',
                        'contentType': 'audio/wav',
                    }
                })
            ],
        ),
        finish_reason=FinishReason.STOP,
    )


# --- Simulated Image ---
async def simulated_image_generate(
    request: GenerateRequest,
    ctx: ActionRunContext,
) -> GenerateResponse:
    """Simulate image generation."""
    _extract_prompt(request)

    await asyncio.sleep(2)  # Simulate processing

    # Create a tiny PNG (1x1 pixel)
    fake_png = base64.b64encode(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
    ).decode()

    return GenerateResponse(
        message=Message(
            role=Role.MODEL,
            content=[
                Part.model_validate({
                    'media': {
                        'url': f'data:image/png;base64,{fake_png}',
                        'contentType': 'image/png',
                    }
                })
            ],
        ),
        finish_reason=FinishReason.STOP,
    )


# --- Simulated Lyria ---
async def simulated_lyria_generate(
    request: GenerateRequest,
    ctx: ActionRunContext,
) -> GenerateResponse:
    """Simulate audio generation."""
    _extract_prompt(request)

    await asyncio.sleep(3)  # Simulate processing

    fake_audio = base64.b64encode(b'RIFF' + b'\x00' * 200).decode()

    return GenerateResponse(
        message=Message(
            role=Role.MODEL,
            content=[
                Part.model_validate({
                    'media': {
                        'url': f'data:audio/wav;base64,{fake_audio}',
                        'contentType': 'audio/wav',
                    }
                })
            ],
        ),
        finish_reason=FinishReason.STOP,
    )


# --- Simulated Veo (Background Model) ---
async def simulated_veo_start(
    request: GenerateRequest,
    ctx: ActionRunContext,
) -> Operation:
    """Start simulated video generation."""
    prompt = _extract_prompt(request)
    op_id = f'operations/veo-{uuid.uuid4().hex[:12]}'

    _operations[op_id] = {
        'prompt': prompt,
        'start_time': time.time(),
        'progress': 0,
    }

    return Operation(id=op_id, done=False, metadata={'progress': 0})


async def simulated_veo_check(operation: Operation) -> Operation:
    """Check simulated video generation status."""
    op_data = _operations.get(operation.id)
    if not op_data:
        return Operation(id=operation.id, done=True, error=Error(message='Not found'))

    elapsed = time.time() - op_data['start_time']
    progress = min(100, int(elapsed * 10))  # 10 second generation

    if progress >= 100:
        video_url = f'https://storage.example.com/{operation.id.split("/")[-1]}.mp4'
        return Operation(
            id=operation.id,
            done=True,
            metadata={'progress': 100},
            output={
                'finishReason': 'stop',
                'message': {
                    'role': 'model',
                    'content': [{'media': {'url': video_url, 'contentType': 'video/mp4'}}],
                },
            },
        )

    return Operation(
        id=operation.id,
        done=False,
        metadata={'progress': progress, 'estimatedSeconds': max(0, 10 - elapsed)},
    )


# Register simulated models if no API key
if not HAS_GEMINI_API_KEY:
    ai.define_model(
        name='simulated-tts',
        fn=simulated_tts_generate,  # type: ignore[arg-type]
        config_schema=SimulatedTtsConfig,
        info=ModelInfo(
            label='Simulated TTS',
            supports=Supports(multiturn=False, media=False, tools=False),
        ),
    )

    ai.define_model(
        name='simulated-image',
        fn=simulated_image_generate,  # type: ignore[arg-type]
        config_schema=SimulatedImageConfig,
        info=ModelInfo(
            label='Simulated Image',
            supports=Supports(multiturn=False, media=True, tools=False),
        ),
    )

    ai.define_model(
        name='simulated-lyria',
        fn=simulated_lyria_generate,  # type: ignore[arg-type]
        config_schema=SimulatedLyriaConfig,
        info=ModelInfo(
            label='Simulated Lyria',
            supports=Supports(multiturn=False, media=True, tools=False),
        ),
    )

    ai.define_background_model(
        name='simulated-veo',
        start=simulated_veo_start,
        check=simulated_veo_check,
        config_schema=SimulatedVeoConfig,
        info=ModelInfo(
            label='Simulated Veo',
            supports=Supports(multiturn=False, media=True, tools=False, output=['media']),
        ),
    )


def get_tts_model() -> str:
    """Get the TTS model name based on environment.

    TTS models use the '-preview-' suffix:
    - gemini-2.5-flash-preview-tts (optimized for latency)
    - gemini-2.5-pro-preview-tts (optimized for quality)
    """
    if HAS_GEMINI_API_KEY:
        return 'googleai/gemini-2.5-flash-preview-tts'
    return 'simulated-tts'


def get_image_model() -> str:
    """Get the image model name based on environment.

    Available image generation models:
    - gemini-2.5-flash-image: Fast, efficient image generation (Nano Banana)
    - gemini-3-pro-image-preview: Professional quality (Nano Banana Pro)
    """
    if HAS_GEMINI_API_KEY:
        return 'googleai/gemini-2.5-flash-image'
    return 'simulated-image'


def get_imagen_model() -> str:
    """Get the Imagen model name based on environment.

    Imagen models use the predict API (not generateContent) and are
    discovered dynamically by the GoogleAI plugin. They produce
    high-quality images with excellent prompt following.

    Available models (discovered dynamically):
    - imagen-3.0-generate-002: Production quality
    - imagen-3.0-fast-generate-001: Fast generation
    - imagen-4.0-generate-001: Latest Imagen model
    """
    if HAS_GEMINI_API_KEY:
        return 'googleai/imagen-3.0-generate-002'
    return 'simulated-image'


def get_lyria_model() -> str:
    """Get the Lyria model name based on environment."""
    if HAS_GCP_PROJECT:
        return 'vertexai/lyria-002'
    return 'simulated-lyria'


def get_veo_model() -> str:
    """Get the Veo model name based on environment."""
    if HAS_GEMINI_API_KEY:
        return 'googleai/veo-2.0-generate-001'
    return 'simulated-veo'


@ai.flow(name='tts_speech_generator', description='Generate speech from text using TTS')
async def tts_speech_generator_flow(input: TtsInput | None = None) -> dict[str, Any]:
    """Generate speech audio from text.

    Text-to-Speech (TTS) converts written text into natural-sounding speech.
    Gemini TTS supports controllable generation via natural language prompts
    for style, accent, pace, and tone.

    Available Voices (30 prebuilt options)
    ======================================
    ┌─────────────────┬─────────────┬─────────────────┬─────────────┐
    │ Voice           │ Style       │ Voice           │ Style       │
    ├─────────────────┼─────────────┼─────────────────┼─────────────┤
    │ Zephyr          │ Bright      │ Puck            │ Upbeat      │
    │ Charon          │ Informative │ Kore            │ Firm        │
    │ Fenrir          │ Excitable   │ Leda            │ Youthful    │
    │ Orus            │ Firm        │ Aoede           │ Breezy      │
    │ Callirrhoe      │ Easy-going  │ Autonoe         │ Bright      │
    │ Enceladus       │ Breathy     │ Iapetus         │ Clear       │
    │ Umbriel         │ Easy-going  │ Algieba         │ Smooth      │
    │ Despina         │ Smooth      │ Erinome         │ Clear       │
    │ Algenib         │ Gravelly    │ Rasalgethi      │ Informative │
    │ Laomedeia       │ Upbeat      │ Achernar        │ Soft        │
    │ Alnilam         │ Firm        │ Schedar         │ Even        │
    │ Gacrux          │ Mature      │ Pulcherrima     │ Forward     │
    │ Achird          │ Friendly    │ Zubenelgenubi   │ Casual      │
    │ Vindemiatrix    │ Gentle      │ Sadachbia       │ Lively      │
    │ Sadaltager      │ Knowledgeable│ Sulafat        │ Warm        │
    └─────────────────┴─────────────┴─────────────────┴─────────────┘

    Try voices at: https://aistudio.google.com/generate-speech

    Args:
        input: TtsInput containing the text to convert to speech.

    Returns:
        Dictionary with audio data (base64) or URL.

    Example:
        >>> result = await tts_speech_generator_flow(TtsInput(text='Good morning!'))
        >>> print(result['audio_url'][:50])
        data:audio/wav;base64,...
    """
    if input is None:
        input = TtsInput()
    text = input.text
    voice = input.voice
    model = get_tts_model()

    config: dict[str, Any] = {}
    if HAS_GEMINI_API_KEY:
        config = {'speech_config': {'voice_config': {'prebuilt_voice_config': {'voice_name': voice}}}}

    try:
        response = await ai.generate(model=model, prompt=text, config=config)
    except Exception as e:
        error_msg = _exception_chain_message(e)
        if 'RESOURCE_EXHAUSTED' in error_msg or 'quota' in error_msg.lower():
            return {
                'error': 'QUOTA_EXCEEDED',
                'message': 'TTS requires GCP billing or quota reset.',
                'details': 'Visit https://ai.google.dev/gemini-api/docs/rate-limits for info.',
                'model': model,
            }
        if 'FAILED_PRECONDITION' in error_msg and 'billing' in error_msg.lower():
            return {
                'error': 'GCP_BILLING_REQUIRED',
                'message': 'TTS requires GCP billing.',
                'details': 'Visit https://console.cloud.google.com/billing to enable billing.',
                'model': model,
            }
        raise

    # Extract audio from response
    audio_url = None
    if response.message and response.message.content:
        for part in response.message.content:
            if hasattr(part.root, 'media') and part.root.media:
                audio_url = getattr(part.root.media, 'url', None)
                break

    return {
        'model': model,
        'voice': voice,
        'text': text,
        'audio_url': audio_url,
        # pyrefly: ignore[unbound-name] - HAS_GEMINI_API_KEY defined at module level
        'using_real_model': HAS_GEMINI_API_KEY,
    }


@ai.flow(name='imagen_image_generator', description='Generate images using Imagen (predict API)')
async def imagen_image_generator_flow(input: ImagenInput | None = None) -> dict[str, Any]:
    """Generate images using Imagen.

    Imagen models use the predict API (not generateContent) and produce
    photorealistic, high-quality images with excellent prompt following.
    They are discovered dynamically by the GoogleAI plugin.

    Imagen vs Gemini Image:
    - Imagen: Dedicated image gen model, predict API, photorealistic output
    - Gemini Image: Multimodal model, generateContent API, good for editing

    Args:
        input: ImagenInput containing the prompt description.

    Returns:
        Dictionary with image data (base64 PNG).

    Example:
        >>> result = await imagen_image_generator_flow(ImagenInput(prompt='A cat in a space suit'))
    """
    if input is None:
        input = ImagenInput()
    prompt = input.prompt
    model = get_imagen_model()

    config: dict[str, Any] = {}
    if HAS_GEMINI_API_KEY:
        config = {'number_of_images': input.number_of_images}

    try:
        response = await ai.generate(model=model, prompt=prompt, config=config)
    except Exception as e:
        error_msg = _exception_chain_message(e)
        if 'NOT_FOUND' in error_msg or '404' in error_msg:
            return {
                'error': 'MODEL_NOT_FOUND',
                'message': f'{model} is not available on this API endpoint.',
                'details': 'Imagen models may not be available on v1beta with an API key. Try Vertex AI.',
                'model': model,
            }
        if 'RESOURCE_EXHAUSTED' in error_msg or 'quota' in error_msg.lower():
            return {
                'error': 'QUOTA_EXCEEDED',
                'message': 'Imagen requires quota or billing.',
                'model': model,
            }
        raise

    # Extract image from response
    image_url = None
    if response.message and response.message.content:
        for part in response.message.content:
            if hasattr(part.root, 'media') and part.root.media:
                image_url = getattr(part.root.media, 'url', None)
                break

    return {
        'model': model,
        'prompt': prompt,
        'number_of_images': input.number_of_images,
        'image_url': image_url,
        # pyrefly: ignore[unbound-name] - HAS_GEMINI_API_KEY defined at module level
        'using_real_model': HAS_GEMINI_API_KEY,
    }


@ai.flow(name='gemini_image_generator', description='Generate images using Imagen')
async def gemini_image_generator_flow(input: ImageInput | None = None) -> dict[str, Any]:
    """Generate images using Imagen image generation.

    Imagen models can generate high-quality images from text descriptions
    with excellent prompt following and artifact-free output.

    Args:
        input: ImageInput containing the prompt description.

    Returns:
        Dictionary with image data or URL.

    Example:
        >>> result = await gemini_image_generator_flow(ImageInput(prompt='A cat astronaut'))
    """
    if input is None:
        input = ImageInput()
    prompt = input.prompt
    aspect_ratio = input.aspect_ratio
    model = get_image_model()

    config: dict[str, Any] = {}
    if HAS_GEMINI_API_KEY:
        config = {'image_config': {'aspect_ratio': aspect_ratio}}

    try:
        response = await ai.generate(model=model, prompt=prompt, config=config)
    except Exception as e:
        error_msg = _exception_chain_message(e)
        if 'RESOURCE_EXHAUSTED' in error_msg or 'quota' in error_msg.lower():
            return {
                'error': 'QUOTA_EXCEEDED',
                'message': 'Image generation requires GCP billing or quota reset.',
                'details': 'Experimental models (gemini-2.0-flash-exp-*) require billing.',
                'hint': 'Visit https://console.cloud.google.com/billing to enable billing.',
                'model': model,
            }
        if 'FAILED_PRECONDITION' in error_msg and 'billing' in error_msg.lower():
            return {
                'error': 'GCP_BILLING_REQUIRED',
                'message': 'Image generation requires GCP billing.',
                'details': 'Visit https://console.cloud.google.com/billing to enable billing.',
                'model': model,
            }
        raise

    # Extract image from response
    image_url = None
    if response.message and response.message.content:
        for part in response.message.content:
            if hasattr(part.root, 'media') and part.root.media:
                image_url = getattr(part.root.media, 'url', None)
                break

    return {
        'model': model,
        'prompt': prompt,
        'aspect_ratio': aspect_ratio,
        'image_url': image_url,
        # pyrefly: ignore[unbound-name] - HAS_GEMINI_API_KEY defined at module level
        'using_real_model': HAS_GEMINI_API_KEY,
    }


@ai.flow(name='lyria_audio_generator', description='Generate music/audio using Lyria')
async def lyria_audio_generator_flow(input: AudioInput | None = None) -> dict[str, Any]:
    """Generate audio/music using Lyria.

    Lyria is Google's audio generation model available through Vertex AI.
    It can generate music, ambient sounds, and other audio from text descriptions.

    Note: Requires GOOGLE_CLOUD_PROJECT environment variable for real Lyria.

    Args:
        input: AudioInput containing the prompt description.

    Returns:
        Dictionary with audio data (base64 WAV).

    Example:
        >>> result = await lyria_audio_generator_flow(AudioInput(prompt='Dance music'))
    """
    if input is None:
        input = AudioInput()
    prompt = input.prompt
    negative_prompt = input.negative_prompt
    model = get_lyria_model()
    if negative_prompt:
        pass

    config: dict[str, Any] = {}
    if negative_prompt:
        config['negative_prompt'] = negative_prompt

    try:
        response = await ai.generate(model=model, prompt=prompt, config=config)
    except Exception as e:
        error_msg = _exception_chain_message(e)
        if 'resolve model' in error_msg.lower() or 'not found' in error_msg.lower():
            return {
                'error': 'MODEL_NOT_AVAILABLE',
                'message': f'{model} could not be resolved.',
                'details': 'Lyria requires the Vertex AI plugin with GOOGLE_CLOUD_PROJECT set.',
                'model': model,
            }
        if 'RESOURCE_EXHAUSTED' in error_msg or 'quota' in error_msg.lower():
            return {
                'error': 'QUOTA_EXCEEDED',
                'message': 'Lyria requires GCP billing or quota reset.',
                'model': model,
            }
        if 'FAILED_PRECONDITION' in error_msg and 'billing' in error_msg.lower():
            return {
                'error': 'GCP_BILLING_REQUIRED',
                'message': 'Lyria requires Vertex AI with GCP billing.',
                'details': 'Visit https://console.cloud.google.com/billing to enable billing.',
                'model': model,
            }
        raise

    # Extract audio from response
    audio_url = None
    if response.message and response.message.content:
        for part in response.message.content:
            if hasattr(part.root, 'media') and part.root.media:
                audio_url = getattr(part.root.media, 'url', None)
                break

    return {
        'model': model,
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'audio_url': audio_url,
        'using_real_model': HAS_GCP_PROJECT,
    }


@ai.flow(name='veo_video_generator', description='Generate video using Veo (background model)')
async def veo_video_generator_flow(input: VideoInput | None = None) -> dict[str, Any]:
    """Generate video using Veo.

    Veo uses the **background model** pattern because video generation
    is a long-running operation (30 seconds to several minutes).

    The flow:
    1. Start the operation (returns immediately with job ID)
    2. Poll for completion every few seconds
    3. Return the video URL when done

    Args:
        input: VideoInput containing the prompt description.

    Returns:
        Dictionary with video URL and operation details.

    Example:
        >>> result = await veo_video_generator_flow(VideoInput(prompt='A sunset over the ocean'))
        >>> print(result['video_url'])
    """
    if input is None:
        input = VideoInput()
    prompt = input.prompt
    aspect_ratio = input.aspect_ratio
    duration_seconds = input.duration_seconds

    model = get_veo_model()

    # Get the background model using its action key
    action_key = f'/background-model/{model}'
    video_model = await lookup_background_action(ai.registry, action_key)
    if video_model is None:
        return {'error': f'Model {model} not found'}

    # Build config
    config: dict[str, Any] = {
        'aspect_ratio': aspect_ratio,
        'duration_seconds': duration_seconds,
    }

    # Start the operation
    operation = await video_model.start(
        GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text=prompt))])],
            config=config,
        )
    )

    # Poll until complete (with timeout)
    max_wait = 300  # 5 minutes
    start_time = time.time()
    poll_count = 0

    while not operation.done:
        if time.time() - start_time > max_wait:
            return {
                'operation_id': operation.id,
                'status': 'timeout',
                'message': 'Operation timed out after 5 minutes',
            }

        await asyncio.sleep(3)
        poll_count += 1
        operation = await video_model.check(operation)

        operation.metadata.get('progress', 0) if operation.metadata else 0

    # Extract video URL
    video_url = None
    if operation.output:
        output = operation.output
        if isinstance(output, dict):
            message = output.get('message', {})
            content = message.get('content', [])
            if content:
                media = content[0].get('media', {})
                video_url = media.get('url')

    if operation.error:
        return {
            'operation_id': operation.id,
            'status': 'error',
            'error': operation.error,
        }

    return {
        'operation_id': operation.id,
        'status': 'completed',
        'video_url': video_url,
        'prompt': prompt,
        'aspect_ratio': aspect_ratio,
        'duration_seconds': duration_seconds,
        'model': model,
        'using_real_model': HAS_GEMINI_API_KEY,
    }


@ai.flow(name='media_models_overview', description='Overview of all available media models')
async def media_models_overview_flow() -> dict[str, Any]:
    """Get an overview of all available media generation models.

    Returns information about which models are available based on
    the current environment configuration.
    """
    return {
        'tts': {
            'model': get_tts_model(),
            'real_available': HAS_GEMINI_API_KEY,
            'description': 'Text-to-Speech - converts text to audio',
            'voices': ['Zephyr', 'Puck', 'Charon', 'Kore', 'Fenrir', 'Leda', 'Orus'],
        },
        'imagen': {
            'model': get_imagen_model(),
            'real_available': HAS_GEMINI_API_KEY,
            'description': 'Imagen - photorealistic image generation (predict API)',
            'note': 'Discovered dynamically by GoogleAI plugin under googleai/ prefix',
        },
        'gemini_image': {
            'model': get_image_model(),
            'real_available': HAS_GEMINI_API_KEY,
            'description': 'Gemini Image - generates images from text (generateContent API)',
            'aspect_ratios': ['1:1', '16:9', '9:16', '4:3', '3:4'],
        },
        'lyria': {
            'model': get_lyria_model(),
            'real_available': HAS_GCP_PROJECT,
            'description': 'Lyria - generates music/audio (Vertex AI)',
            'note': 'Requires GOOGLE_CLOUD_PROJECT for real model',
        },
        'veo': {
            'model': get_veo_model(),
            'real_available': HAS_GEMINI_API_KEY,
            'description': 'Veo - generates videos (background model)',
            'versions': ['veo-2.0', 'veo-3.0', 'veo-3.0-fast', 'veo-3.1'],
        },
        'environment': {
            'GEMINI_API_KEY': HAS_GEMINI_API_KEY,
            'GOOGLE_CLOUD_PROJECT': HAS_GCP_PROJECT,
        },
    }


@ai.tool(name='screenshot')
def screenshot() -> dict:
    """Takes a screenshot of a room."""
    room_path = pathlib.Path(__file__).parent.parent / 'my_room.png'
    with pathlib.Path(room_path).open('rb') as f:
        room_b64 = base64.b64encode(f.read()).decode('utf-8')

    return {
        'output': 'success',
        'content': [{'media': {'url': f'data:image/png;base64,{room_b64}', 'contentType': 'image/png'}}],
    }


@ai.flow()
async def describe_image_with_gemini(input: DescribeImageInput | None = None) -> str:
    """Describe an image using Gemini.

    Args:
        input: DescribeImageInput with image data URI or 'auto' for bundled default.

    Returns:
        The description of the image.
    """
    if input is None:
        input = DescribeImageInput()
    data = input.data
    if not data or data == 'auto':
        try:
            current_dir = pathlib.Path(pathlib.Path(__file__).resolve()).parent
            image_path = os.path.join(current_dir, '..', 'image.jpg')
            with pathlib.Path(image_path).open('rb') as image_file:
                buffer = image_file.read()
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                data = f'data:image/jpeg;base64,{img_base64}'
        except FileNotFoundError as e:
            raise ValueError("Default image 'image.jpg' not found. Please provide image data.") from e

    if not (data.startswith('data:') and ',' in data):
        raise ValueError(f'Expected a data URI (e.g., "data:image/jpeg;base64,..."), but got: {data[:50]}...')

    result = await ai.generate(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='What is shown in this image?')),
                    Part(root=MediaPart(media=Media(content_type='image/jpeg', url=data))),
                ],
            ),
        ],
        model='googleai/gemini-3-flash-preview',
    )
    return result.text


@ai.flow()
async def generate_images(
    input: GenerateImagesInput | None = None,
    ctx: ActionRunContext | None = None,
) -> GenerateResponseWrapper:
    """Generate images for the given subject using multimodal prompting.

    Args:
        input: Input with subject to generate images about.
        ctx: The action run context.

    Returns:
        The generated response with text and images.
    """
    if input is None:
        input = GenerateImagesInput()
    return await ai.generate(
        model='googleai/gemini-3-pro-image-preview',
        prompt=f'tell me about {input.name} with photos',
        config=GeminiConfigSchema.model_validate({
            'response_modalities': ['text', 'image'],
            'api_version': 'v1alpha',
        }).model_dump(exclude_none=True),
    )


@ai.flow()
async def multipart_tool_calling(input: ToolCallingInput | None = None) -> str:
    """Tool calling with image input and output.

    Demonstrates a tool that returns image content (screenshot) and
    the model reasoning about the image.

    Args:
        input: ToolCallingInput with the prompt for tool use.

    Returns:
        The model's description of the screenshot.
    """
    if input is None:
        input = ToolCallingInput()
    response = await ai.generate(
        model='googleai/gemini-3-pro-preview',
        tools=['screenshot'],
        config=GenerationCommonConfig(temperature=1),
        prompt=input.prompt,
    )
    return response.text


@ai.flow()
async def gemini_image_editing(input: ImageEditingInput | None = None) -> Media | None:
    """Image editing with Gemini (inpainting/outpainting).

    Combines two images (a plant and a room) and asks Gemini to
    composite them together, demonstrating image editing capabilities.

    Args:
        input: ImageEditingInput with the editing instruction prompt.

    Returns:
        The edited image media, or None if no image was generated.
    """
    if input is None:
        input = ImageEditingInput()
    plant_path = pathlib.Path(__file__).parent.parent / 'palm_tree.png'
    room_path = pathlib.Path(__file__).parent.parent / 'my_room.png'

    with pathlib.Path(plant_path).open('rb') as f:
        plant_b64 = base64.b64encode(f.read()).decode('utf-8')
    with pathlib.Path(room_path).open('rb') as f:
        room_b64 = base64.b64encode(f.read()).decode('utf-8')

    response = await ai.generate(
        model='googleai/gemini-3-pro-image-preview',
        prompt=[
            Part(root=TextPart(text=input.prompt)),
            Part(root=MediaPart(media=Media(url=f'data:image/png;base64,{plant_b64}'))),
            Part(root=MediaPart(media=Media(url=f'data:image/png;base64,{room_b64}'))),
        ],
        config=GeminiImageConfigSchema.model_validate({
            'response_modalities': ['TEXT', 'IMAGE'],
            'image_config': {'aspect_ratio': '1:1'},
            'api_version': 'v1alpha',
        }).model_dump(exclude_none=True),
    )
    for part in response.message.content if response.message else []:
        if isinstance(part.root, MediaPart):
            return part.root.media

    return None


@ai.flow()
async def nano_banana_pro(input: NanoBananaProInput | None = None) -> Media | None:
    """Generate a 4K image with custom aspect ratio.

    Demonstrates advanced image configuration options including
    aspect ratio and image size settings.

    Args:
        input: NanoBananaProInput with the image description prompt.

    Returns:
        The generated image media, or None if no image was generated.
    """
    if input is None:
        input = NanoBananaProInput()
    response = await ai.generate(
        model='googleai/gemini-3-pro-image-preview',
        prompt=input.prompt,
        config={
            'response_modalities': ['TEXT', 'IMAGE'],
            'image_config': {
                'aspect_ratio': '21:9',
                'image_size': '4K',
            },
            'api_version': 'v1alpha',
        },
    )
    for part in response.message.content if response.message else []:
        if isinstance(part.root, MediaPart):
            return part.root.media
    return None


@ai.flow()
async def gemini_media_resolution(input: MediaResolutionInput | None = None) -> str:
    """Query an image with high media resolution.

    Demonstrates the mediaResolution metadata option for higher-fidelity
    image analysis.

    Args:
        input: MediaResolutionInput with the question about the image.

    Returns:
        The model's description of the image.
    """
    if input is None:
        input = MediaResolutionInput()
    plant_path = pathlib.Path(__file__).parent.parent / 'palm_tree.png'
    with pathlib.Path(plant_path).open('rb') as f:
        plant_b64 = base64.b64encode(f.read()).decode('utf-8')
    response = await ai.generate(
        model='googleai/gemini-3-pro-image-preview',
        prompt=[
            Part(root=TextPart(text=input.prompt)),
            Part(
                root=MediaPart(
                    media=Media(url=f'data:image/png;base64,{plant_b64}'),
                    metadata=Metadata({'mediaResolution': {'level': 'MEDIA_RESOLUTION_HIGH'}}),
                )
            ),
        ],
        config={'api_version': 'v1alpha'},
    )
    return response.text


@ai.flow()
async def multimodal_input(input: MultimodalInput | None = None) -> str:
    """Describe a photo using multimodal prompting.

    Demonstrates sending both text and image content to the model
    in a single prompt.

    Args:
        input: MultimodalInput with the instruction for describing the photo.

    Returns:
        The model's description of the photo.
    """
    if input is None:
        input = MultimodalInput()
    photo_path = pathlib.Path(__file__).parent.parent / 'photo.jpg'
    with pathlib.Path(photo_path).open('rb') as f:
        photo_b64 = base64.b64encode(f.read()).decode('utf-8')

    response = await ai.generate(
        model='googleai/gemini-3-pro-image-preview',
        prompt=[
            Part(root=TextPart(text=input.prompt)),
            Part(root=MediaPart(media=Media(url=f'data:image/jpeg;base64,{photo_b64}', content_type='image/jpeg'))),
        ],
    )
    return response.text


async def main() -> None:
    """Keep the server alive for the Dev UI."""
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
