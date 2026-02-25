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

"""Type definitions for Cloudflare Workers AI plugin.

This module contains Pydantic configuration schemas for the Cloudflare Workers AI
plugin, including model-specific configurations and common parameters.

See: https://developers.cloudflare.com/workers-ai/models/
"""

from pydantic import Field

from genkit.types import GenerationCommonConfig


class CloudflareConfig(GenerationCommonConfig):
    """Configuration schema for Cloudflare Workers AI models.

    This configuration applies to all Cloudflare text generation models.
    Parameters match the Cloudflare Workers AI API specification.

    See: https://developers.cloudflare.com/workers-ai/models/llama-3.1-8b-instruct/

    Attributes:
        temperature: Controls randomness (0-5). Higher = more random. Default: 0.6.
        max_output_tokens: Maximum tokens to generate. Default: 256.
        top_p: Nucleus sampling threshold (0-2). Default: None.
        top_k: Top-k sampling (1-50). Limits token selection. Default: None.
        seed: Random seed for reproducibility (1-9999999999). Default: None.
        repetition_penalty: Penalty for repeated tokens (0-2). Default: None.
        frequency_penalty: Penalty for frequent tokens (0-2). Default: None.
        presence_penalty: Bonus for new topics (0-2). Default: None.
        lora: Name of LoRA adapter to use for fine-tuning. Default: None.
        raw: If True, skip chat template formatting. Default: None.
    """

    top_k: float | None = Field(
        default=None,
        ge=1,
        le=50,
        description='Limits the AI to choose from the top k most probable words.',
    )
    seed: int | None = Field(
        default=None,
        ge=1,
        le=9999999999,
        description='Random seed for reproducibility.',
    )
    repetition_penalty: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description='Penalty for repeated tokens; higher values discourage repetition.',
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description='Decreases the likelihood of repeating the same lines.',
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description='Increases the likelihood of introducing new topics.',
    )
    lora: str | None = Field(
        default=None,
        description='Name of the LoRA model to fine-tune the base model.',
    )
    raw: bool | None = Field(
        default=None,
        description='If true, chat template is not applied.',
    )


class CloudflareEmbedConfig(GenerationCommonConfig):
    """Configuration schema for Cloudflare embedding models.

    See: https://developers.cloudflare.com/workers-ai/models/bge-base-en-v1.5/

    Attributes:
        pooling: The pooling method used in the embedding process.
            - 'mean': Default pooling method. Works for most use cases.
            - 'cls': Generates more accurate embeddings on larger inputs,
              but embeddings created with 'cls' pooling are NOT compatible
              with embeddings generated with 'mean' pooling.
    """

    pooling: str | None = Field(
        default=None,
        pattern='^(mean|cls)$',
        description=(
            "Pooling method: 'mean' (default) or 'cls'. "
            "'cls' gives better accuracy on larger inputs but is incompatible with 'mean'."
        ),
    )


__all__ = [
    'CloudflareConfig',
    'CloudflareEmbedConfig',
]
