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

from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from genkit import GenkitError
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

    Image models take their own family's class instead: ``NovaCanvasConfig``,
    ``TitanImageConfig`` or ``StabilityImageConfig``. ``Bedrock.text()`` and
    the image ref constructors (``Bedrock.nova_canvas()``, ...) bind the
    right one.

    Unknown keys are rejected: only the declared fields reach the Converse
    API, so a tolerated typo (``maxTokens`` for ``maxOutputTokens``) would run
    the call with the knob silently unset. Model-specific options go in
    ``extra``, which is sent as-is as Converse ``additionalModelRequestFields``
    (e.g. Claude extended thinking).
    """

    tool_choice: str | None = None
    """Tool choice mode: ``auto``, ``required``/``any``, ``none``, or a tool name."""


# The nested object Titan Image and Nova Canvas used to take from config,
# before its fields moved to the top level. Both spellings were read.
_NESTED_IMAGE_KEYS = ('imageGenerationConfig', 'image_generation_config')


_EXTRA_DESCRIPTION = (
    'Bedrock body fields this class does not declare, in wire names, deep-merged into the InvokeModel body after '
    "everything else (for example {'textToImageParams': {'negativeText': 'blurry'}}). Nested objects merge key by "
    'key. Fields built from the prompt are rejected. Not checked otherwise; do not put secrets here.'
)


class _AmazonImageConfig(GenkitConfig):
    """Fields Titan Image and Nova Canvas share; the plugin nests them under ``imageGenerationConfig``.

    The camelCase aliases are the Bedrock wire names, so ``number_of_images``
    is sent as ``imageGenerationConfig.numberOfImages``.

    Extends ``GenkitConfig``, like ``VeoConfig``, not ``ModelConfig``: the
    image APIs take none of Genkit's generic knobs, so ``temperature`` or
    ``top_p`` raise as unknown keys instead of being dropped.
    """

    number_of_images: int | None = Field(default=None, ge=1, le=5, description='Images to generate. Default 1.')
    cfg_scale: float | None = Field(
        default=None,
        ge=1.1,
        le=10.0,
        description='How strictly the image follows the prompt; lower is more random. Default 8.0.',
    )
    seed: int | None = Field(default=None, ge=0, le=2147483646, description='Initial noise seed. Default 0.')
    quality: Literal['standard', 'premium'] | None = Field(default=None, description='Image quality tier.')
    extra: dict[str, Any] | None = Field(default=None, description=_EXTRA_DESCRIPTION)

    @model_validator(mode='before')
    @classmethod
    def _reject_nested_image_generation_config(cls, data: Any) -> Any:  # noqa: ANN401
        """Name the flat fields when a caller passes the old nested object."""
        if not isinstance(data, Mapping):
            return data
        for key in _NESTED_IMAGE_KEYS:
            if key in cast(Mapping[str, Any], data):
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=(
                        f'bedrock image: {key} was flattened; set number_of_images=, height=, width=, '
                        "cfg_scale=, seed= and quality= on the config, or send raw body fields in config['extra']"
                    ),
                )
        return data


class TitanImageConfig(_AmazonImageConfig):
    """Per-call configuration for Amazon Titan Image Generator (v1, v2).

    Example:
        ```python
        from genkit_amazon_bedrock import TitanImageConfig

        # 1. Two 512x512 images, pinned seed
        response = await ai.generate(
            model='bedrock/amazon.titan-image-generator-v2:0',
            prompt='A bowl of ramen on a cedar counter.',
            config=TitanImageConfig(number_of_images=2, width=512, height=512, seed=7),
        )

        # 2. One media part per image
        print(len(response.media))
        # => 2
        ```

    ``quality`` is not sent unless set. Titan accepts only fixed width and
    height pairs (1024x1024, 768x1152, 1280x768, ...); Bedrock rejects any
    other pair with a ``ValidationException``.
    """

    height: int | None = Field(default=None, gt=0, description='Image height in pixels. Default 1024.')
    width: int | None = Field(default=None, gt=0, description='Image width in pixels. Default 1024.')


class NovaCanvasConfig(_AmazonImageConfig):
    """Per-call configuration for Amazon Nova Canvas.

    Example:
        ```python
        from genkit_amazon_bedrock import NovaCanvasConfig

        # 1. Premium quality, negativeText through extra
        response = await ai.generate(
            model='bedrock/amazon.nova-canvas-v1:0',
            prompt='A dim sum spread on a marble table.',
            config=NovaCanvasConfig(
                quality='premium',
                cfg_scale=6.5,
                extra={'textToImageParams': {'negativeText': 'blurry'}},
            ),
        )

        # 2. Nova Canvas always returns PNG
        print(response.media[0].content_type)
        # => image/png
        ```
    """

    height: int | None = Field(
        default=None,
        ge=320,
        le=4096,
        multiple_of=16,
        description='Image height in pixels, 320-4096 and divisible by 16. Default 1024.',
    )
    width: int | None = Field(
        default=None,
        ge=320,
        le=4096,
        multiple_of=16,
        description='Image width in pixels, 320-4096 and divisible by 16. Default 1024.',
    )


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
        default=None, ge=0, le=4294967294, description='Randomness seed; 0 or unset picks a random one.'
    )
    negative_prompt: str | None = Field(default=None, max_length=10000, description='What to keep out of the image.')
    output_format: Literal['png', 'jpeg', 'webp'] | None = Field(
        default=None, description='Output format. Default png. webp is Stable Diffusion 3.5 only.'
    )
    extra: dict[str, Any] | None = Field(default=None, description=_EXTRA_DESCRIPTION)


ImageConfig = TitanImageConfig | NovaCanvasConfig | StabilityImageConfig
"""Any of the per-family image config classes."""
