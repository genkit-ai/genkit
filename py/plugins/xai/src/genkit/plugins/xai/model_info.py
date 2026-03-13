# Copyright 2025 Google LLC
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

"""xAI model information."""

import sys

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from genkit.types import ModelInfo, Supports

__all__ = ['SUPPORTED_XAI_MODELS', 'get_model_info']


# Source: https://docs.x.ai/docs/models
LANGUAGE_MODEL_SUPPORTS = Supports(
    multiturn=True,
    tools=True,
    media=False,
    system_role=True,
    output=['text', 'json'],
)

REASONING_MODEL_SUPPORTS = Supports(
    multiturn=True,
    tools=True,
    media=False,
    system_role=True,
    output=['text', 'json'],
)

VISION_MODEL_SUPPORTS = Supports(
    multiturn=False,
    tools=True,
    media=True,
    system_role=False,
    output=['text', 'json'],
)

# --- Grok 3 family (legacy) ---
GROK_3 = ModelInfo(label='xAI - Grok 3', versions=['grok-3'], supports=LANGUAGE_MODEL_SUPPORTS)
GROK_3_FAST = ModelInfo(label='xAI - Grok 3 Fast', versions=['grok-3-fast'], supports=LANGUAGE_MODEL_SUPPORTS)
GROK_3_MINI = ModelInfo(label='xAI - Grok 3 Mini', versions=['grok-3-mini'], supports=LANGUAGE_MODEL_SUPPORTS)
GROK_3_MINI_FAST = ModelInfo(
    label='xAI - Grok 3 Mini Fast', versions=['grok-3-mini-fast'], supports=LANGUAGE_MODEL_SUPPORTS
)

# --- Grok 4 family ---
GROK_4 = ModelInfo(label='xAI - Grok 4', versions=['grok-4'], supports=REASONING_MODEL_SUPPORTS)
GROK_4_FAST_REASONING = ModelInfo(
    label='xAI - Grok 4 Fast (Reasoning)',
    versions=['grok-4-fast-reasoning'],
    supports=REASONING_MODEL_SUPPORTS,
)
GROK_4_FAST_NON_REASONING = ModelInfo(
    label='xAI - Grok 4 Fast (Non-Reasoning)',
    versions=['grok-4-fast-non-reasoning'],
    supports=LANGUAGE_MODEL_SUPPORTS,
)

# --- Grok 4.1 family ---
# NOTE: "grok-4.1" is an alias available only on the OpenAI-compatible REST API.
# The native xai_sdk (gRPC) does not support this alias; use the explicit model IDs below.
GROK_4_1_FAST_REASONING = ModelInfo(
    label='xAI - Grok 4.1 Fast (Reasoning)',
    versions=['grok-4-1-fast-reasoning'],
    supports=REASONING_MODEL_SUPPORTS,
)
GROK_4_1_FAST_NON_REASONING = ModelInfo(
    label='xAI - Grok 4.1 Fast (Non-Reasoning)',
    versions=['grok-4-1-fast-non-reasoning'],
    supports=LANGUAGE_MODEL_SUPPORTS,
)

# --- Specialist models ---
GROK_CODE_FAST_1 = ModelInfo(
    label='xAI - Grok Code Fast 1',
    versions=['grok-code-fast-1'],
    supports=LANGUAGE_MODEL_SUPPORTS,
)

# --- Vision models ---
GROK_2_VISION_1212 = ModelInfo(
    label='xAI - Grok 2 Vision',
    versions=['grok-2-vision-1212'],
    supports=VISION_MODEL_SUPPORTS,
)


class XAIGrokVersion(StrEnum):
    """xAI Grok models.

    Source: https://docs.x.ai/docs/models

    | Model                         | Description                    | Status     |
    |-------------------------------|--------------------------------|------------|
    | `grok-3`                      | Grok 3                         | Supported  |
    | `grok-3-fast`                 | Grok 3 Fast                    | Supported  |
    | `grok-3-mini`                 | Grok 3 Mini                    | Supported  |
    | `grok-3-mini-fast`            | Grok 3 Mini Fast               | Supported  |
    | `grok-4`                      | Grok 4 (reasoning)             | Supported  |
    | `grok-4-fast-reasoning`       | Grok 4 Fast (reasoning)        | Supported  |
    | `grok-4-fast-non-reasoning`   | Grok 4 Fast (non-reasoning)    | Supported  |
    | `grok-4-1-fast-reasoning`     | Grok 4.1 Fast (reasoning)      | Supported  |
    | `grok-4-1-fast-non-reasoning` | Grok 4.1 Fast (non-reasoning)  | Supported  |
    | `grok-code-fast-1`            | Grok Code Fast 1 (coding)      | Supported  |
    | `grok-2-vision-1212`          | Grok 2 Vision                  | Supported  |
    """

    GROK_3 = 'grok-3'
    GROK_3_FAST = 'grok-3-fast'
    GROK_3_MINI = 'grok-3-mini'
    GROK_3_MINI_FAST = 'grok-3-mini-fast'
    GROK_4 = 'grok-4'
    GROK_4_FAST_REASONING = 'grok-4-fast-reasoning'
    GROK_4_FAST_NON_REASONING = 'grok-4-fast-non-reasoning'
    GROK_4_1_FAST_REASONING = 'grok-4-1-fast-reasoning'
    GROK_4_1_FAST_NON_REASONING = 'grok-4-1-fast-non-reasoning'
    GROK_CODE_FAST_1 = 'grok-code-fast-1'
    GROK_2_VISION_1212 = 'grok-2-vision-1212'


SUPPORTED_XAI_MODELS: dict[str, ModelInfo] = {
    XAIGrokVersion.GROK_3: GROK_3,
    XAIGrokVersion.GROK_3_FAST: GROK_3_FAST,
    XAIGrokVersion.GROK_3_MINI: GROK_3_MINI,
    XAIGrokVersion.GROK_3_MINI_FAST: GROK_3_MINI_FAST,
    XAIGrokVersion.GROK_4: GROK_4,
    XAIGrokVersion.GROK_4_FAST_REASONING: GROK_4_FAST_REASONING,
    XAIGrokVersion.GROK_4_FAST_NON_REASONING: GROK_4_FAST_NON_REASONING,
    XAIGrokVersion.GROK_4_1_FAST_REASONING: GROK_4_1_FAST_REASONING,
    XAIGrokVersion.GROK_4_1_FAST_NON_REASONING: GROK_4_1_FAST_NON_REASONING,
    XAIGrokVersion.GROK_CODE_FAST_1: GROK_CODE_FAST_1,
    XAIGrokVersion.GROK_2_VISION_1212: GROK_2_VISION_1212,
}


def get_model_info(name: str) -> ModelInfo:
    """Get information for a specific model."""
    return SUPPORTED_XAI_MODELS.get(
        name,
        ModelInfo(
            label=f'xAI - {name}',
            supports=LANGUAGE_MODEL_SUPPORTS,
        ),
    )
