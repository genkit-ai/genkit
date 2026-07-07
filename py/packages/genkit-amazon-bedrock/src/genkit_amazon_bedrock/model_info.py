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

"""Model capability registry for the Amazon Bedrock plugin.

Ported from the Go plugin's ``models.go``. Capabilities are keyed by base
Bedrock model ID; cross-region inference-profile prefixes are stripped for
lookup only - the full original model ID is always sent to Bedrock untouched.
Unknown chat/text models fall back to modern Converse defaults (multimodal +
tools) at the unstable stage, so newer or inference-profile-only models remain
callable without a plugin release.
"""

from typing import Literal, NamedTuple

from genkit import Constrained, ModelInfo, Stage, Supports

INFERENCE_PROFILE_PREFIXES = (
    'global.',
    'us-gov.',
    'us.',
    'eu.',
    'jp.',
    'apac.',
    'au.',
)


class ModelCapability(NamedTuple):
    """Capabilities of a Bedrock chat/text model."""

    multimodal: bool
    tools: bool


MODEL_CAPABILITIES: dict[str, ModelCapability] = {
    # Anthropic Claude 3 models
    'anthropic.claude-3-haiku-20240307-v1:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-3-sonnet-20240229-v1:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-3-opus-20240229-v1:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-3-5-haiku-20241022-v1:0': ModelCapability(multimodal=False, tools=True),
    'anthropic.claude-3-5-sonnet-20240620-v1:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-3-5-sonnet-20241022-v2:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-3-7-sonnet-20250219-v1:0': ModelCapability(multimodal=True, tools=True),
    # Anthropic Claude 4/4.5/4.6 models
    'anthropic.claude-haiku-4-5-20251001-v1:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-opus-4-1-20250805-v1:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-opus-4-20250514-v1:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-sonnet-4-20250514-v1:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-sonnet-4-5-20250929-v1:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-opus-4-5-20251101-v1:0': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-sonnet-4-6': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-opus-4-6-v1': ModelCapability(multimodal=True, tools=True),
    # Provisioned-throughput variants (28k/48k/200k context)
    'anthropic.claude-3-haiku-20240307-v1:0:48k': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-3-haiku-20240307-v1:0:200k': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-3-sonnet-20240229-v1:0:28k': ModelCapability(multimodal=True, tools=True),
    'anthropic.claude-3-sonnet-20240229-v1:0:200k': ModelCapability(multimodal=True, tools=True),
    # Amazon Nova models
    'amazon.nova-micro-v1:0': ModelCapability(multimodal=False, tools=True),
    'amazon.nova-lite-v1:0': ModelCapability(multimodal=True, tools=True),
    'amazon.nova-pro-v1:0': ModelCapability(multimodal=True, tools=True),
    'amazon.nova-premier-v1:0': ModelCapability(multimodal=True, tools=True),
    # Cohere Command models
    'cohere.command-r-v1:0': ModelCapability(multimodal=False, tools=True),
    'cohere.command-r-plus-v1:0': ModelCapability(multimodal=False, tools=True),
    # Mistral models
    'mistral.mistral-large-2402-v1:0': ModelCapability(multimodal=False, tools=True),
    'mistral.mistral-large-2407-v1:0': ModelCapability(multimodal=False, tools=True),
    'mistral.mistral-small-2402-v1:0': ModelCapability(multimodal=False, tools=True),
    'mistral.pixtral-large-2502-v1:0': ModelCapability(multimodal=True, tools=True),
    # AI21 Labs Jamba models
    'ai21.jamba-1-5-large-v1:0': ModelCapability(multimodal=False, tools=True),
    'ai21.jamba-1-5-mini-v1:0': ModelCapability(multimodal=False, tools=True),
    # Meta Llama models
    'meta.llama3-8b-instruct-v1:0': ModelCapability(multimodal=False, tools=True),
    'meta.llama3-70b-instruct-v1:0': ModelCapability(multimodal=False, tools=True),
    'meta.llama3-1-8b-instruct-v1:0': ModelCapability(multimodal=False, tools=True),
    'meta.llama3-1-70b-instruct-v1:0': ModelCapability(multimodal=False, tools=True),
    'meta.llama3-1-405b-instruct-v1:0': ModelCapability(multimodal=False, tools=True),
    'meta.llama3-2-1b-instruct-v1:0': ModelCapability(multimodal=False, tools=True),
    'meta.llama3-2-3b-instruct-v1:0': ModelCapability(multimodal=False, tools=True),
    'meta.llama3-2-11b-instruct-v1:0': ModelCapability(multimodal=True, tools=True),
    'meta.llama3-2-90b-instruct-v1:0': ModelCapability(multimodal=True, tools=True),
    'meta.llama3-3-70b-instruct-v1:0': ModelCapability(multimodal=False, tools=True),
    'meta.llama4-maverick-17b-instruct-v1:0': ModelCapability(multimodal=True, tools=True),
    'meta.llama4-scout-17b-instruct-v1:0': ModelCapability(multimodal=True, tools=True),
    # DeepSeek models
    'deepseek.r1-v1:0': ModelCapability(multimodal=False, tools=True),
    # Writer models
    'writer.palmyra-x4-v1:0': ModelCapability(multimodal=False, tools=True),
    'writer.palmyra-x5-v1:0': ModelCapability(multimodal=False, tools=True),
    # TwelveLabs models
    'twelvelabs.pegasus-1-2-v1:0': ModelCapability(multimodal=False, tools=True),
}


def strip_inference_profile_prefix(model_id: str) -> str:
    """Strips a cross-region inference-profile prefix from a model ID.

    Used for capability lookup only; requests always carry the original ID.

    Args:
        model_id: Bedrock model ID, possibly prefixed (e.g. ``us.anthropic...``).

    Returns:
        The base model ID without the inference-profile prefix.
    """
    for prefix in INFERENCE_PROFILE_PREFIXES:
        if model_id.startswith(prefix):
            return model_id.removeprefix(prefix)
    return model_id


def get_model_info(
    model_name: str,
    model_type: Literal['chat', 'image', 'embedding'] = 'chat',
) -> ModelInfo:
    """Infers Genkit model info for a Bedrock model.

    Args:
        model_name: Bedrock model ID or inference-profile ID.
        model_type: Routing type for the model.

    Returns:
        ModelInfo with capabilities from the registry, or modern Converse
        defaults at the unstable stage for unknown chat/text models.
    """
    if model_type == 'image':
        return ModelInfo(
            label=model_name,
            stage=Stage.STABLE,
            supports=Supports(
                multiturn=False,
                tools=False,
                tool_choice=False,
                system_role=False,
                media=True,
                constrained=Constrained.NONE,
            ),
        )

    if model_type == 'embedding':
        return ModelInfo(
            label=model_name,
            stage=Stage.STABLE,
            supports=Supports(
                multiturn=False,
                tools=False,
                tool_choice=False,
                system_role=False,
                media=False,
                constrained=Constrained.NONE,
            ),
        )

    capability = MODEL_CAPABILITIES.get(strip_inference_profile_prefix(model_name))
    stage = Stage.STABLE if capability is not None else Stage.UNSTABLE
    if capability is None:
        capability = ModelCapability(multimodal=True, tools=True)

    return ModelInfo(
        label=model_name,
        stage=stage,
        supports=Supports(
            multiturn=True,
            tools=capability.tools,
            tool_choice=capability.tools,
            system_role=True,
            media=capability.multimodal,
            constrained=Constrained.NONE,
        ),
    )
