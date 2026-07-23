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

"""Typed ModelRef helpers for Google GenAI model families."""

from typing import Literal

from genkit.model import ModelRef, model_ref
from genkit_google_genai.models.gemini import (
    GeminiConfig,
    GeminiImageConfig,
    GeminiTtsConfig,
    GemmaConfig,
)
from genkit_google_genai.models.imagen import ImagenConfig
from genkit_google_genai.models.lyria import LyriaConfig
from genkit_google_genai.models.veo import VeoConfig

# Short *-latest aliases exist only for Gemini text (same as the JS plugin).
# Other families have no latest shortcuts today — list their known IDs instead.
KnownGemini = Literal[
    'gemini-pro-latest',
    'gemini-flash-latest',
    'gemini-flash-lite-latest',
    'gemini-3.5-flash',
    'gemini-3.1-flash-lite',
    'gemini-3.1-flash-lite-preview',
    'gemini-3.1-pro-preview',
    'gemini-3.1-pro-preview-customtools',
    'gemini-3-flash-preview',
    'gemini-3-pro-preview',
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
]

KnownGeminiTts = Literal[
    'gemini-2.5-flash-preview-tts',
    'gemini-2.5-pro-preview-tts',
    'gemini-3.1-flash-tts-preview',
]

KnownGeminiImage = Literal[
    'gemini-2.5-flash-image',
    'gemini-2.5-flash-image-preview',
    'gemini-3.1-flash-image',
    'gemini-3.1-flash-image-preview',
    'gemini-3-pro-image',
    'gemini-3-pro-image-preview',
]

KnownGemma = Literal[
    'gemma-4-26b-a4b-it',
    'gemma-4-31b-it',
    'gemma-3-1b-it',
    'gemma-3-4b-it',
    'gemma-3-12b-it',
    'gemma-3-27b-it',
    'gemma-3n-e4b-it',
]

KnownImagen = Literal[
    'imagen-4.0-generate-001',
    'imagen-4.0-fast-generate-001',
    'imagen-4.0-ultra-generate-001',
    'imagen-3.0-generate-002',
    'imagen-3.0-fast-generate-001',
    'imagegeneration@006',
]

KnownVeo = Literal[
    'veo-3.1-generate-preview',
    'veo-3.1-fast-generate-preview',
    'veo-3.1-lite-generate-preview',
    'veo-3.1-generate-001',
    'veo-3.1-fast-generate-001',
    'veo-3.0-generate-001',
    'veo-3.0-fast-generate-001',
    'veo-2.0-generate-001',
    'veo-2.0-generate-exp',
]

KnownLyria = Literal[
    'lyria-002',
    'lyria-3-clip-preview',
    'lyria-3-pro-preview',
]


def gemini_model(
    name: KnownGemini | str,
    *,
    config: GeminiConfig | None = None,
) -> ModelRef[GeminiConfig]:
    """Return a typed ref for standard Gemini text models."""
    return model_ref(
        name,
        config_schema=GeminiConfig,
        namespace='googleai',
        config=config,
    )


def gemini_tts_model(
    name: KnownGeminiTts | str,
    *,
    config: GeminiTtsConfig | None = None,
) -> ModelRef[GeminiTtsConfig]:
    """Return a typed ref for Gemini text-to-speech models."""
    return model_ref(
        name,
        config_schema=GeminiTtsConfig,
        namespace='googleai',
        config=config,
    )


def gemini_image_model(
    name: KnownGeminiImage | str,
    *,
    config: GeminiImageConfig | None = None,
) -> ModelRef[GeminiImageConfig]:
    """Return a typed ref for Gemini native image generation models."""
    return model_ref(
        name,
        config_schema=GeminiImageConfig,
        namespace='googleai',
        config=config,
    )


def gemma_model(
    name: KnownGemma | str,
    *,
    config: GemmaConfig | None = None,
) -> ModelRef[GemmaConfig]:
    """Return a typed ref for Gemma open-weight models."""
    return model_ref(
        name,
        config_schema=GemmaConfig,
        namespace='googleai',
        config=config,
    )


def imagen_model(
    name: KnownImagen | str,
    *,
    config: ImagenConfig | None = None,
) -> ModelRef[ImagenConfig]:
    """Return a typed ref for Imagen text-to-image models."""
    return model_ref(
        name,
        config_schema=ImagenConfig,
        namespace='googleai',
        config=config,
    )


def veo_model(
    name: KnownVeo | str,
    *,
    config: VeoConfig | None = None,
) -> ModelRef[VeoConfig]:
    """Return a typed ref for Veo video generation models."""
    return model_ref(
        name,
        config_schema=VeoConfig,
        namespace='googleai',
        config=config,
    )


def lyria_model(
    name: KnownLyria | str,
    *,
    config: LyriaConfig | None = None,
) -> ModelRef[LyriaConfig]:
    """Return a typed ref for Lyria audio generation models."""
    return model_ref(
        name,
        config_schema=LyriaConfig,
        namespace='googleai',
        config=config,
    )
