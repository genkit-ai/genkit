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

"""AWS Bedrock plugin for Genkit.

This plugin provides access to AWS Bedrock models through the Genkit framework.
AWS Bedrock is a fully managed service that provides access to foundation models
from multiple providers through a unified API.

Documentation Links:
    - AWS Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html
    - Supported Models: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
    - Converse API: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
    - Boto3 Reference: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime.html

The plugin supports:
    - Chat completion models via Converse API
    - Text embedding models (Titan, Cohere, Nova)
    - Tool/function calling
    - Streaming responses
    - Multimodal inputs (images, video for supported models)

Authentication:
    The plugin uses the standard AWS credential chain:
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    2. AWS credentials file (~/.aws/credentials)
    3. IAM role (for EC2, Lambda, ECS, etc.)
    4. AWS profile

Example::

    from genkit import Genkit
    from genkit.plugins.amazon_bedrock import AmazonBedrock, bedrock_model

    ai = Genkit(
        plugins=[AmazonBedrock(region='us-east-1')],
        model=bedrock_model('anthropic.claude-sonnet-4-5-20250929-v1:0'),
    )

    response = await ai.generate(prompt='Tell me a joke.')
    print(response.text)

Trademark Notice:
    This is a community plugin and is not officially supported by Amazon Web Services.
    "Amazon", "AWS", "Amazon Bedrock", and related marks are trademarks of
    Amazon.com, Inc. or its affiliates.
"""

import json
import os
from typing import Any

import aioboto3
from botocore.config import Config

from genkit.ai import Plugin
from genkit.blocks.embedding import EmbedderOptions, EmbedderSupports, embedder_action_metadata
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.logging import get_logger
from genkit.core.registry import ActionKind
from genkit.plugins.amazon_bedrock.models.model import BedrockModel
from genkit.plugins.amazon_bedrock.models.model_info import (
    SUPPORTED_BEDROCK_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    get_model_info,
)
from genkit.plugins.amazon_bedrock.typing import (
    AI21JambaConfig,
    AmazonNovaConfig,
    AnthropicConfig,
    BedrockConfig,
    CohereConfig,
    DeepSeekConfig,
    GoogleGemmaConfig,
    MetaLlamaConfig,
    MiniMaxConfig,
    MistralConfig,
    MoonshotConfig,
    NvidiaConfig,
    OpenAIConfig,
    QwenConfig,
    StabilityConfig,
    TitanConfig,
    WriterConfig,
)
from genkit.types import Embedding, EmbedRequest, EmbedResponse

# Regional prefixes for inference profiles (must match model.py)
_INFERENCE_PROFILE_PREFIXES = ('us.', 'eu.', 'apac.')


def _strip_inference_profile_prefix(model_id: str) -> str:
    """Strip the regional inference profile prefix from a model ID.

    Converts inference profile IDs (e.g., ``us.amazon.titan-embed-text-v2:0``)
    back to base model IDs (``amazon.titan-embed-text-v2:0``).

    The InvokeModel API (used for embeddings) does NOT support inference profile
    IDs — only Converse/ConverseStream do. This helper ensures we always pass
    base model IDs to InvokeModel.

    Args:
        model_id: A model ID, possibly with a regional prefix.

    Returns:
        The base model ID with any regional prefix stripped.
    """
    for prefix in _INFERENCE_PROFILE_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return model_id


_MODEL_CONFIG_PREFIX_MAP: dict[str, type] = {
    # Amazon models
    'amazon.nova': AmazonNovaConfig,
    'amazon.titan': TitanConfig,
    # Anthropic Claude models
    'anthropic.claude': AnthropicConfig,
    # AI21 Labs Jamba models
    'ai21.jamba': AI21JambaConfig,
    # Cohere models
    'cohere.command': CohereConfig,
    'cohere.embed': CohereConfig,
    # DeepSeek models
    'deepseek': DeepSeekConfig,
    # Google Gemma models
    'google.gemma': GoogleGemmaConfig,
    # Meta Llama models
    'meta.llama': MetaLlamaConfig,
    # MiniMax models
    'minimax': MiniMaxConfig,
    # Mistral AI models
    'mistral': MistralConfig,
    # Moonshot AI models
    'moonshot': MoonshotConfig,
    # NVIDIA models
    'nvidia': NvidiaConfig,
    # OpenAI models (GPT-OSS on Bedrock)
    'openai': OpenAIConfig,
    # Qwen models
    'qwen': QwenConfig,
    # Writer models
    'writer': WriterConfig,
    # Stability AI models
    'stability': StabilityConfig,
}
"""Mapping from model ID prefixes to their configuration classes."""


def get_config_schema_for_model(model_id: str) -> type:
    """Get the appropriate config schema for a model based on its ID.

    This function maps model IDs to their model-specific configuration classes,
    enabling the DevUI to show relevant parameters for each model family.

    Handles both direct model IDs and cross-region inference profile IDs:
    - Direct: 'anthropic.claude-sonnet-4-5-20250929-v1:0'
    - Inference profile: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

    Args:
        model_id: The model ID or inference profile ID.

    Returns:
        The appropriate config class for the model. Returns BedrockConfig as default.
    """
    model_lower = model_id.lower()

    # Strip regional prefix for inference profiles (e.g., 'us.', 'eu.', 'ap.')
    # Inference profile format: {region}.{provider}.{model}
    if '.' in model_lower:
        parts = model_lower.split('.', 1)
        # Check if first part is a region code (2-3 chars like 'us', 'eu', 'ap')
        if len(parts[0]) <= 3 and parts[0].isalpha():
            model_lower = parts[1]

    for prefix, config_class in _MODEL_CONFIG_PREFIX_MAP.items():
        if model_lower.startswith(prefix):
            return config_class

    # Default: standard Bedrock config
    return BedrockConfig


# Plugin name
AMAZON_BEDROCK_PLUGIN_NAME = 'amazon-bedrock'

# Logger for this module
logger = get_logger(__name__)


def bedrock_name(model_id: str) -> str:
    """Get fully qualified AWS Bedrock model name.

    Args:
        model_id: The Bedrock model ID (e.g., 'anthropic.claude-sonnet-4-5-20250929-v1:0').

    Returns:
        Fully qualified model name (e.g., 'amazon-bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0').
    """
    return f'{AMAZON_BEDROCK_PLUGIN_NAME}/{model_id}'


class AmazonBedrock(Plugin):
    """AWS Bedrock plugin for Genkit.

    This plugin provides access to AWS Bedrock models including:
    - Amazon: Nova Pro, Nova Lite, Nova Micro, Titan
    - Anthropic: Claude Opus, Sonnet, Haiku
    - AI21 Labs: Jamba 1.5
    - Cohere: Command R, Command R+, Embed
    - DeepSeek: R1, V3
    - Google: Gemma 3
    - Meta: Llama 3.x, Llama 4
    - MiniMax: M2
    - Mistral AI: Large 3, Pixtral, Ministral
    - Moonshot AI: Kimi K2
    - NVIDIA: Nemotron
    - OpenAI: GPT-OSS
    - Qwen: Qwen3
    - Writer: Palmyra

    See: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

    Attributes:
        name: Plugin name ('amazon-bedrock').
    """

    name = AMAZON_BEDROCK_PLUGIN_NAME

    def __init__(
        self,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        profile_name: str | None = None,
        connect_timeout: int = 60,
        read_timeout: int = 3600,
        **boto_config: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the AWS Bedrock plugin.

        Args:
            region: AWS region (e.g., 'us-east-1'). Falls back to AWS_REGION env var.
            access_key_id: AWS access key ID. Falls back to AWS_ACCESS_KEY_ID env var.
            secret_access_key: AWS secret access key. Falls back to AWS_SECRET_ACCESS_KEY env var.
            session_token: AWS session token for temporary credentials.
            profile_name: AWS profile name to use from credentials file.
            connect_timeout: Connection timeout in seconds. Default: 60.
            read_timeout: Read timeout in seconds. Default: 3600 (1 hour for Nova models).
            **boto_config: Additional parameters passed to boto3 Config.

        Example:
            # Using environment variables (recommended):
            plugin = AmazonBedrock(region='us-east-1')

            # Using explicit credentials:
            plugin = AmazonBedrock(
                region='us-east-1',
                access_key_id='your-access-key',
                secret_access_key='your-secret-key',
            )

            # Using AWS profile:
            plugin = AmazonBedrock(
                region='us-east-1',
                profile_name='my-profile',
            )
        """
        # Resolve region from environment
        resolved_region = region or os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')

        if not resolved_region:
            raise ValueError('AWS region is required. Set AWS_REGION environment variable or pass region parameter.')

        # Build botocore config with timeouts
        # Nova models have 60-minute inference timeout, so we set read_timeout high
        self._botocore_config = Config(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            **boto_config,
        )

        # Build session kwargs
        session_kwargs: dict[str, Any] = {
            'region_name': resolved_region,
        }

        if profile_name:
            session_kwargs['profile_name'] = profile_name

        # Create aioboto3 session (sync — session creation is lightweight)
        self._session = aioboto3.Session(**session_kwargs)

        # Client kwargs for bedrock-runtime
        self._client_kwargs: dict[str, Any] = {
            'config': self._botocore_config,
        }

        # Add explicit credentials if provided
        if access_key_id and secret_access_key:
            self._client_kwargs['aws_access_key_id'] = access_key_id
            self._client_kwargs['aws_secret_access_key'] = secret_access_key
            if session_token:
                self._client_kwargs['aws_session_token'] = session_token

        # Async client — created in init(), closed in close()
        self._client: Any = None  # noqa: ANN401
        self._client_ctx: Any = None  # noqa: ANN401
        self._region = resolved_region

        logger.debug(
            'Initialized AWS Bedrock plugin',
            region=resolved_region,
        )

    async def init(self) -> list[Action]:
        """Initialize plugin and register supported models.

        Creates the async bedrock-runtime client and registers all
        supported models and embedders.

        Returns:
            List of Action objects for supported models and embedders.
        """
        # Create the async bedrock-runtime client.
        # aioboto3 requires async context managers for client creation.
        # We enter the context here and exit in close().
        self._client_ctx = self._session.client('bedrock-runtime', **self._client_kwargs)
        self._client = await self._client_ctx.__aenter__()

        actions: list[Action] = []

        # Register all supported models from predefined list
        for model_id in SUPPORTED_BEDROCK_MODELS:
            actions.append(self._create_model_action(bedrock_name(model_id)))

        # Register all supported embedding models
        for model_id in SUPPORTED_EMBEDDING_MODELS:
            actions.append(self._create_embedder_action(bedrock_name(model_id)))

        return actions

    async def close(self) -> None:
        """Close the async bedrock-runtime client.

        Exits the aioboto3 async context manager entered in init(),
        releasing network connections and other resources.
        """
        if self._client_ctx is not None:
            await self._client_ctx.__aexit__(None, None, None)
            self._client = None
            self._client_ctx = None

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by type and name.

        This enables lazy loading of models not pre-registered during init().

        Args:
            action_type: The kind of action to resolve (MODEL or EMBEDDER).
            name: The namespaced name of the action.

        Returns:
            Action object if resolvable, None otherwise.
        """
        if action_type == ActionKind.MODEL:
            return self._create_model_action(name)
        elif action_type == ActionKind.EMBEDDER:
            return self._create_embedder_action(name)
        return None

    def _create_model_action(self, name: str) -> Action:
        """Create an Action object for a chat completion model.

        Args:
            name: The namespaced model name (e.g., 'amazon-bedrock/anthropic.claude-...').

        Returns:
            Action object for the model.
        """
        # Extract model ID (remove plugin prefix)
        prefix = f'{AMAZON_BEDROCK_PLUGIN_NAME}/'
        model_id = name[len(prefix) :] if name.startswith(prefix) else name

        model = BedrockModel(
            model_id=model_id,
            client=self._client,
        )
        model_info = get_model_info(model_id)

        # Get the appropriate config schema for this model family
        config_schema = get_config_schema_for_model(model_id)

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=model.generate,
            metadata=model_action_metadata(
                name=name,
                info=model_info.model_dump(by_alias=True, exclude_none=True),
                config_schema=config_schema,
            ).metadata,
        )

    def _create_embedder_action(self, name: str) -> Action:
        """Create an Action object for an embedding model.

        Args:
            name: The namespaced embedder name.

        Returns:
            Action object for the embedder.
        """
        prefix = f'{AMAZON_BEDROCK_PLUGIN_NAME}/'
        model_id = name[len(prefix) :] if name.startswith(prefix) else name

        # Get embedder info
        embedder_info = SUPPORTED_EMBEDDING_MODELS.get(
            model_id,
            {
                'label': f'Bedrock - {model_id}',
                'dimensions': 1024,
                'supports': {'input': ['text']},
            },
        )

        async def embed_fn(request: EmbedRequest) -> EmbedResponse:
            """Generate embeddings using AWS Bedrock.

            The InvokeModel API does NOT support inference profile IDs
            (regional prefixes like ``us.``, ``eu.``, ``apac.``). Only
            the Converse/ConverseStream APIs support them. We must strip
            any inference profile prefix before calling InvokeModel.
            """
            # Strip inference profile prefix — InvokeModel requires base model IDs
            base_model_id = _strip_inference_profile_prefix(model_id)

            # Extract text from document content
            texts: list[str] = []
            for doc in request.input:
                text_parts: list[str] = []
                for part in doc.content:
                    if hasattr(part.root, 'text') and part.root.text:
                        text_parts.append(str(part.root.text))
                doc_text = ''.join(text_parts)
                texts.append(doc_text)

            embeddings: list[Embedding] = []

            # Process each text (Bedrock embedding API typically handles one at a time)
            for text in texts:
                # Build request body based on model type
                if base_model_id.startswith('amazon.titan-embed'):
                    body = {'inputText': text}
                elif base_model_id.startswith('cohere.embed'):
                    body = {
                        'texts': [text],
                        'input_type': 'search_document',
                    }
                elif base_model_id.startswith('amazon.nova'):
                    body = {'inputText': text}
                else:
                    body = {'inputText': text}

                # Call InvokeModel for embeddings (uses base model ID, not inference profile)
                response = await self._client.invoke_model(
                    modelId=base_model_id,
                    body=json.dumps(body),
                    contentType='application/json',
                    accept='application/json',
                )

                # Parse response
                response_body = json.loads(await response['body'].read())

                # Extract embedding based on model type
                if base_model_id.startswith('amazon.titan-embed'):
                    embedding_vector = response_body.get('embedding', [])
                elif base_model_id.startswith('cohere.embed'):
                    embedding_vector = response_body.get('embeddings', [[]])[0]
                elif base_model_id.startswith('amazon.nova'):
                    embedding_vector = response_body.get('embedding', [])
                else:
                    embedding_vector = response_body.get('embedding', [])

                embeddings.append(Embedding(embedding=embedding_vector))

            return EmbedResponse(embeddings=embeddings)

        return Action(
            kind=ActionKind.EMBEDDER,
            name=name,
            fn=embed_fn,
            metadata=embedder_action_metadata(
                name=name,
                options=EmbedderOptions(
                    label=embedder_info['label'],
                    supports=EmbedderSupports(input=embedder_info['supports']['input']),
                    dimensions=embedder_info.get('dimensions'),
                ),
            ).metadata,
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """List all available models and embedders.

        Returns:
            List of ActionMetadata for all supported models and embedders.
        """
        actions: list[ActionMetadata] = []

        # Add model metadata from predefined list
        for model_id, model_info in SUPPORTED_BEDROCK_MODELS.items():
            config_schema = get_config_schema_for_model(model_id)
            actions.append(
                model_action_metadata(
                    name=bedrock_name(model_id),
                    info=model_info.model_dump(by_alias=True, exclude_none=True),
                    config_schema=config_schema,
                )
            )

        # Add embedder metadata from predefined list
        for model_id, embed_info in SUPPORTED_EMBEDDING_MODELS.items():
            actions.append(
                embedder_action_metadata(
                    name=bedrock_name(model_id),
                    options=EmbedderOptions(
                        label=embed_info['label'],
                        supports=EmbedderSupports(input=embed_info['supports']['input']),
                        dimensions=embed_info.get('dimensions'),
                    ),
                )
            )

        return actions


def bedrock_model(model_id: str) -> str:
    """Get fully qualified AWS Bedrock model name.

    Convenience function for specifying models.

    See: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

    Args:
        model_id: The Bedrock model ID (e.g., 'anthropic.claude-sonnet-4-5-20250929-v1:0').

    Returns:
        Fully qualified model name (e.g., 'amazon-bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0').

    Example:
        ai = Genkit(
            plugins=[AmazonBedrock(region='us-east-1')],
            model=bedrock_model('anthropic.claude-sonnet-4-5-20250929-v1:0'),
        )

        # Or for other models:
        response = await ai.generate(
            model=bedrock_model('meta.llama3-3-70b-instruct-v1:0'),
            prompt='Explain quantum computing.',
        )
    """
    return bedrock_name(model_id)


# =============================================================================
# Inference Profile Region Helpers
# =============================================================================
# AWS Bedrock cross-region inference profiles require a regional prefix.
# When using API keys (AWS_BEARER_TOKEN_BEDROCK), you MUST use inference profiles.
# When using IAM credentials, you can use either direct model IDs or inference profiles.
#
# See: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html


def get_inference_profile_prefix(region: str | None = None) -> str:
    """Get the inference profile prefix for a given AWS region.

    Args:
        region: AWS region code (e.g., 'us-east-1', 'eu-west-1', 'ap-northeast-1').
                If None, uses AWS_REGION or AWS_DEFAULT_REGION environment variable.

    Returns:
        The inference profile prefix ('us', 'eu', or 'apac').

    Raises:
        ValueError: If no region is specified and AWS_REGION is not set.

    Example::

        >>> get_inference_profile_prefix('us-east-1')
        'us'
        >>> get_inference_profile_prefix('eu-west-1')
        'eu'
        >>> get_inference_profile_prefix('ap-northeast-1')
        'apac'
    """
    if region is None:
        region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')

    if region is None:
        raise ValueError(
            'AWS region is required for inference profiles. '
            'Set AWS_REGION environment variable or pass region parameter.'
        )

    region_lower = region.lower()

    if region_lower.startswith('us-') or region_lower.startswith('us.'):
        return 'us'
    elif region_lower.startswith('eu-') or region_lower.startswith('eu.'):
        return 'eu'
    elif region_lower.startswith('ap-') or region_lower.startswith('ap.'):
        return 'apac'
    elif region_lower.startswith('ca-'):
        # Canada is routed through US inference profiles
        return 'us'
    elif region_lower.startswith('sa-'):
        # South America is routed through US inference profiles
        return 'us'
    elif region_lower.startswith('me-'):
        # Middle East - check AWS docs, but typically EU
        return 'eu'
    elif region_lower.startswith('af-'):
        # Africa - check AWS docs, but typically EU
        return 'eu'
    else:
        # Default to US
        return 'us'


def inference_profile(model_id: str, region: str | None = None) -> str:
    """Convert a model ID to an inference profile ID for the given region.

    Use this when you need to use API keys (AWS_BEARER_TOKEN_BEDROCK) which
    require inference profiles instead of direct model IDs.

    Args:
        model_id: Base model ID (e.g., 'anthropic.claude-sonnet-4-5-20250929-v1:0').
        region: AWS region code. If None, uses AWS_REGION environment variable.

    Returns:
        The Genkit model reference with inference profile ID.

    Example::

        # Using environment variable AWS_REGION=eu-west-1
        >>> inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0')
        'amazon-bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0'

        # Explicit region
        >>> inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0', 'ap-northeast-1')
        'amazon-bedrock/apac.anthropic.claude-sonnet-4-5-20250929-v1:0'
    """
    prefix = get_inference_profile_prefix(region)
    return bedrock_name(f'{prefix}.{model_id}')


# =============================================================================
# Pre-defined Model References
# =============================================================================
# These use DIRECT model IDs (without regional prefix) which work with:
# - IAM credentials (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
# - IAM roles (EC2, Lambda, ECS, etc.)
#
# For API keys (AWS_BEARER_TOKEN_BEDROCK), use the inference_profile() helper:
#   model = inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0')
#
# Or use the region-specific helpers defined below.

# Anthropic Claude models — grouped by family, descending version
claude_opus_4_6 = bedrock_name('anthropic.claude-opus-4-6-20260205-v1:0')
claude_opus_4_5 = bedrock_name('anthropic.claude-opus-4-5-20251101-v1:0')
claude_opus_4_1 = bedrock_name('anthropic.claude-opus-4-1-20250805-v1:0')
claude_sonnet_4_5 = bedrock_name('anthropic.claude-sonnet-4-5-20250929-v1:0')
claude_sonnet_4 = bedrock_name('anthropic.claude-sonnet-4-20250514-v1:0')
claude_haiku_4_5 = bedrock_name('anthropic.claude-haiku-4-5-20251001-v1:0')
claude_3_5_haiku = bedrock_name('anthropic.claude-3-5-haiku-20241022-v1:0')
claude_3_haiku = bedrock_name('anthropic.claude-3-haiku-20240307-v1:0')

# Amazon Nova models
nova_pro = bedrock_name('amazon.nova-pro-v1:0')
nova_lite = bedrock_name('amazon.nova-lite-v1:0')
nova_micro = bedrock_name('amazon.nova-micro-v1:0')
nova_premier = bedrock_name('amazon.nova-premier-v1:0')

# Meta Llama models
llama_3_3_70b = bedrock_name('meta.llama3-3-70b-instruct-v1:0')
llama_3_1_405b = bedrock_name('meta.llama3-1-405b-instruct-v1:0')
llama_3_1_70b = bedrock_name('meta.llama3-1-70b-instruct-v1:0')
llama_4_maverick = bedrock_name('meta.llama4-maverick-17b-instruct-v1:0')
llama_4_scout = bedrock_name('meta.llama4-scout-17b-instruct-v1:0')

# Mistral models
mistral_large_3 = bedrock_name('mistral.mistral-large-3-675b-instruct')
mistral_large = bedrock_name('mistral.mistral-large-2407-v1:0')
pixtral_large = bedrock_name('mistral.pixtral-large-2502-v1:0')

# DeepSeek models
deepseek_r1 = bedrock_name('deepseek.r1-v1:0')
deepseek_v3 = bedrock_name('deepseek.v3-v1:0')

# Cohere models
command_r_plus = bedrock_name('cohere.command-r-plus-v1:0')
command_r = bedrock_name('cohere.command-r-v1:0')

# AI21 Jamba models
jamba_large = bedrock_name('ai21.jamba-1-5-large-v1:0')
jamba_mini = bedrock_name('ai21.jamba-1-5-mini-v1:0')
