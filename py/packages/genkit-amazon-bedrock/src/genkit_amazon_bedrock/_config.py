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

"""Configuration types for the Amazon Bedrock plugin."""

from typing import Any, Literal

from pydantic import Field

from genkit.model import ModelConfig
from genkit.plugin_api import GenkitConfig

DEFAULT_MAX_RETRIES = 3
# Socket read timeout: the gap botocore tolerates between two reads, which
# resets on every byte received. Generous because Bedrock generations can
# legitimately run for many minutes (Nova allows 60-minute inference).
DEFAULT_READ_TIMEOUT = 3600.0
DEFAULT_CONNECT_TIMEOUT = 60.0
# Whole-call deadline. Unlike the read timeout this does not reset, so a
# connection that dribbles a byte at a time still ends.
DEFAULT_TOTAL_TIMEOUT = 3600.0
# The botocore default of 10 pooled connections throttles LLM concurrency.
DEFAULT_MAX_POOL_CONNECTIONS = 50


class BedrockConfig(ModelConfig):
    """Per-call configuration for Bedrock text models (the Converse API).

    Stability image models take ``StabilityImageConfig`` instead, which
    ``Bedrock.stability()`` binds.

    Unknown keys are rejected: only the declared fields reach the Converse
    API, so a tolerated typo (``maxTokens`` for ``maxOutputTokens``) would run
    the call with the knob silently unset. Model-specific options go in
    ``extra``, which is sent as-is as Converse ``additionalModelRequestFields``
    (e.g. Claude extended thinking).
    """

    tool_choice: str | None = None
    """Tool choice mode: ``auto``, ``required``/``any``, ``none``, or a tool name."""


_EXTRA_DESCRIPTION = (
    'Bedrock body fields this class does not declare, in wire names, deep-merged into the InvokeModel body after '
    "everything else (for example {'mode': 'text-to-image'}). Nested objects merge key by key. Fields built from "
    'the prompt are rejected. Not checked otherwise; do not put secrets here.'
)


class AmazonImageConfig(GenkitConfig):
    """Config for Titan Image and Nova Canvas, which Bedrock retired; not exported.

    Bedrock retired Titan Image (v2 on 2026-06-30) and Nova Canvas (on
    2026-09-30), so these models get no typed fields. The raw model ID still
    routes to InvokeModel, and ``extra`` carries every body field the caller
    wants beyond the prompt, e.g. ``{'imageGenerationConfig': {'seed': 7}}``.
    Any other key raises as unknown, and Genkit's message names ``extra``.
    """

    extra: dict[str, Any] | None = Field(default=None, description=_EXTRA_DESCRIPTION)


StabilityAspectRatio = Literal['16:9', '1:1', '21:9', '2:3', '3:2', '4:5', '5:4', '9:16', '9:21']


class StabilityImageConfig(GenkitConfig):
    """Per-call configuration for Stable Diffusion 3.5, Stable Image Core and Stable Image Ultra.

    Field names are the Stability wire names and are sent at the top level of
    the request body. Extends ``GenkitConfig``, so Genkit's generic knobs
    (``temperature``, ...) raise as unknown keys.

    Example:
        ```python
        from genkit_amazon_bedrock import StabilityImageConfig

        # 1. Widescreen jpeg, pinned seed
        response = await ai.generate(
            model='bedrock/stability.sd3-5-large-v1:0',
            prompt='A food truck at dusk, string lights.',
            config=StabilityImageConfig(aspect_ratio='16:9', seed=42, negative_prompt='blurry', output_format='jpeg'),
        )

        # 2. The MIME type follows output_format
        print(response.media[0].content_type)
        # => image/jpeg
        ```
    """

    aspect_ratio: StabilityAspectRatio | None = Field(default=None, description='Output aspect ratio. Default 1:1.')
    seed: int | None = Field(
        default=None,
        ge=0,
        le=4294967295,
        description=(
            'Randomness seed; 0 or unset picks a random one. '
            'Stable Image Core and Ultra accept up to 4294967295; Stable Diffusion 3.5 caps at 4294967294.'
        ),
    )
    negative_prompt: str | None = Field(default=None, max_length=10000, description='What to keep out of the image.')
    output_format: Literal['png', 'jpeg', 'webp'] | None = Field(
        default=None, description='Output format. Default png. webp is Stable Diffusion 3.5 only.'
    )
    extra: dict[str, Any] | None = Field(default=None, description=_EXTRA_DESCRIPTION)


ImageConfig = StabilityImageConfig | AmazonImageConfig
"""Any of the per-family image config classes."""
