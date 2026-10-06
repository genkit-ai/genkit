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

"""Catalog of OpenAI-compatible models served by Vertex AI Model Garden."""

from genkit.model import ModelInfo, Supports

LLAMA_3_1 = 'meta/llama-3.1-405b-instruct-maas'
LLAMA_3_2 = 'meta/llama-3.2-90b-vision-instruct-maas'

SUPPORTED_OPENAI_COMPAT_MODELS: dict[str, ModelInfo] = {
    LLAMA_3_1: ModelInfo(
        label='ModelGarden - Meta - llama-3.1',
        supports=Supports(
            multiturn=True,
            media=False,
            tools=True,
            system_role=True,
            long_running=False,
            output=['json', 'text'],
        ),
    ),
    LLAMA_3_2: ModelInfo(
        label='ModelGarden - Meta - llama-3.2',
        supports=Supports(
            multiturn=True,
            media=True,
            tools=True,
            system_role=True,
            output=['json', 'text'],
        ),
    ),
}

DEFAULT_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['json', 'text'],
)


def get_default_model_info(name: str) -> ModelInfo:
    """Gets the default model info given a name."""
    return ModelInfo(
        label=f'ModelGarden - {name}',
        supports=DEFAULT_SUPPORTS,
    )
