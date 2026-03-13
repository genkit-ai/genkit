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

"""Microsoft Foundry plugin for Genkit.

This plugin provides access to Microsoft Foundry models through the Genkit framework.
Microsoft Foundry (formerly Azure AI Foundry) provides access to 11,000+ AI models
from multiple providers.

Documentation:
- Microsoft Foundry Portal: https://ai.azure.com/
- Model Catalog: https://ai.azure.com/catalog/models
- SDK Overview: https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/sdk-overview
- Models: https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/models
- Switching Endpoints: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/switching-endpoints

The plugin supports:
- Chat completion models (GPT-4o, GPT-5, o-series, Claude, DeepSeek, Grok, Llama, Mistral)
- Text embedding models (text-embedding-3-small, text-embedding-3-large, embed-v-4-0)
- Tool/function calling
- Streaming responses
- Multimodal inputs (images)
- JSON output mode

Endpoint Types
==============
The plugin supports two endpoint types:

1. **Azure OpenAI endpoint** (traditional):
   Format: `https://<resource-name>.openai.azure.com/`
   Uses `AsyncAzureOpenAI` client with `api_version` parameter.

2. **Azure AI Foundry project endpoint** (new unified endpoint):
   Format: `https://<resource-name>.services.ai.azure.com/api/projects/<project-name>`
   Uses standard `AsyncOpenAI` client with `base_url` parameter.
   This endpoint eliminates the need for api-version parameters.

The plugin auto-detects the endpoint type based on the URL format.

Authentication
==============
The plugin supports two authentication methods:

1. **API Key** (simpler):
   Set `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` environment variables,
   or pass them directly to the plugin constructor.

2. **Azure AD / Managed Identity** (recommended for production):
   Use `azure_ad_token_provider` with Azure Identity credentials.

Example - Azure OpenAI Endpoint::

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, 'https://cognitiveservices.azure.com/.default')

    ai = Genkit(
        plugins=[
            MicrosoftFoundry(
                azure_ad_token_provider=token_provider,
                endpoint='https://your-resource.openai.azure.com/',
                # api_version is optional; defaults to DEFAULT_API_VERSION
            )
        ]
    )

Example - Azure AI Foundry Project Endpoint::

    ai = Genkit(
        plugins=[
            MicrosoftFoundry(
                api_key='your-api-key',
                endpoint='https://your-resource.services.ai.azure.com/api/projects/your-project',
            )
        ],
        model=microsoft_foundry_model('gpt-4o'),
    )

Example Usage
=============
```python
from genkit import Genkit
from genkit.plugins.microsoft_foundry import MicrosoftFoundry, microsoft_foundry_model

ai = Genkit(
    plugins=[
        MicrosoftFoundry(
            api_key='your-api-key',
            endpoint='https://your-resource.openai.azure.com/',
            # api_version is optional; defaults to DEFAULT_API_VERSION
        )
    ],
    model=microsoft_foundry_model('gpt-4o'),
)

response = await ai.generate(prompt='Tell me a joke.')
print(response.text)
```

See Also:
    - Microsoft Foundry Documentation: https://learn.microsoft.com/en-us/azure/ai-foundry/
    - Model Catalog: https://ai.azure.com/catalog/models
"""

import os
import re
from collections.abc import Callable, Generator
from typing import Any

import httpx
from openai import APIError, AsyncAzureOpenAI, AsyncOpenAI

from genkit.ai import Plugin
from genkit.blocks.embedding import EmbedderOptions, EmbedderSupports, embedder_action_metadata
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.logging import get_logger
from genkit.core.registry import ActionKind
from genkit.plugins.microsoft_foundry.models.model import MicrosoftFoundryModel
from genkit.plugins.microsoft_foundry.models.model_info import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_MICROSOFT_FOUNDRY_MODELS,
    get_model_info,
)
from genkit.plugins.microsoft_foundry.typing import (
    AI21JambaConfig,
    AnthropicConfig,
    ArcticConfig,
    BaichuanConfig,
    CohereConfig,
    DbrxConfig,
    DeepSeekConfig,
    FalconConfig,
    GemmaConfig,
    GlmConfig,
    GraniteConfig,
    GrokConfig,
    InflectionConfig,
    InternLMConfig,
    JaisConfig,
    LlamaConfig,
    MicrosoftFoundryConfig,
    MiniCPMConfig,
    MistralConfig,
    MptConfig,
    NvidiaConfig,
    PhiConfig,
    QwenConfig,
    RekaConfig,
    StableLMConfig,
    StarCoderConfig,
    TimeSeriesConfig,
    WriterConfig,
    XGenConfig,
    YiConfig,
)
from genkit.types import Embedding, EmbedRequest, EmbedResponse

_MODEL_CONFIG_PREFIX_MAP: dict[str, type] = {
    # Anthropic Claude models
    'claude': AnthropicConfig,
    'anthropic': AnthropicConfig,
    # Meta Llama models
    'llama': LlamaConfig,
    'meta-llama': LlamaConfig,
    # Mistral AI models (Mistral, Mixtral, Codestral)
    'mistral': MistralConfig,
    'mixtral': MistralConfig,
    'codestral': MistralConfig,
    # Cohere models (Command R, Command R+, Embed, Rerank)
    'command': CohereConfig,
    'cohere': CohereConfig,
    # DeepSeek models
    'deepseek': DeepSeekConfig,
    # Microsoft Phi models
    'phi': PhiConfig,
    # AI21 Jamba models
    'jamba': AI21JambaConfig,
    'ai21': AI21JambaConfig,
    # xAI Grok models
    'grok': GrokConfig,
    # NVIDIA NIM models (Nemotron, etc.)
    'nvidia': NvidiaConfig,
    'nemotron': NvidiaConfig,
    # Google Gemma models
    'gemma': GemmaConfig,
    # Alibaba Qwen models
    'qwen': QwenConfig,
    # Databricks DBRX
    'dbrx': DbrxConfig,
    # TII Falcon models
    'falcon': FalconConfig,
    'tiiuae': FalconConfig,
    # IBM Granite models
    'granite': GraniteConfig,
    'ibm': GraniteConfig,
    # G42 Jais (Arabic LLM)
    'jais': JaisConfig,
    # BigCode StarCoder
    'starcoder': StarCoderConfig,
    'starchat': StarCoderConfig,
    # Stability AI StableLM
    'stablelm': StableLMConfig,
    # MosaicML MPT
    'mpt': MptConfig,
    # TimesFM / Chronos (Time Series)
    'timesfm': TimeSeriesConfig,
    'chronos': TimeSeriesConfig,
    # 01.AI Yi models
    'yi': YiConfig,
    # Zhipu AI GLM models
    'glm': GlmConfig,
    'chatglm': GlmConfig,
    # Baichuan models
    'baichuan': BaichuanConfig,
    # Shanghai AI Lab InternLM
    'internlm': InternLMConfig,
    # Snowflake Arctic
    'arctic': ArcticConfig,
    'snowflake': ArcticConfig,
    # Writer Palmyra
    'palmyra': WriterConfig,
    'writer': WriterConfig,
    # Reka models
    'reka': RekaConfig,
    # OpenBMB MiniCPM
    'minicpm': MiniCPMConfig,
    # Inflection Pi
    'inflection': InflectionConfig,
    'pi': InflectionConfig,
    # Salesforce XGen / CodeGen
    'xgen': XGenConfig,
    'codegen': XGenConfig,
    'salesforce': XGenConfig,
}
"""Mapping from model name prefixes to their configuration classes."""

# Pattern matching invisible/zero-width Unicode characters commonly introduced
# by copy-paste from web UIs (e.g., Azure Portal).  These cause hard-to-debug
# ``UnicodeEncodeError: 'ascii' codec can't encode character`` failures deep
# inside HTTP transport layers when the credential string reaches a path that
# doesn't use explicit UTF-8 encoding.
#
# Categories covered:
#   \u200b  zero-width space
#   \u200c  zero-width non-joiner
#   \u200d  zero-width joiner
#   \u200e  left-to-right mark
#   \u200f  right-to-left mark
#   \ufeff  byte-order mark (BOM)
#   \u00a0  non-breaking space
#   \u2060  word joiner
#   \u2028  line separator
#   \u2029  paragraph separator
_INVISIBLE_CHARS_RE = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00a0\u2060\u2028\u2029]')


def _sanitize_credential(value: str | None) -> str | None:
    r"""Strip invisible Unicode characters from a credential string.

    Credentials (API keys, endpoint URLs, API versions) are often copied from
    web dashboards that inject invisible Unicode characters such as zero-width
    spaces (``\\u200b``).  These cause ``UnicodeEncodeError`` during HTTP
    request serialization on systems where the default encoding is ASCII.

    The function also strips leading/trailing whitespace.

    Args:
        value: The credential string to sanitize, or None.

    Returns:
        The sanitized string, or None if the input was None.
    """
    if value is None:
        return None
    return _INVISIBLE_CHARS_RE.sub('', value).strip()


def get_config_schema_for_model(model_name: str) -> type:
    """Get the appropriate config schema for a model based on its name.

    This function maps model names to their model-specific configuration classes,
    enabling the DevUI to show relevant parameters for each model family.

    Args:
        model_name: The model name (e.g., 'gpt-4o', 'claude-opus-4-5', 'llama-3.3-70b').

    Returns:
        The appropriate config class for the model. Returns MicrosoftFoundryConfig as default.
    """
    name_lower = model_name.lower()

    for prefix, config_class in _MODEL_CONFIG_PREFIX_MAP.items():
        if name_lower.startswith(prefix):
            return config_class

    # Default: OpenAI-compatible config (GPT, o-series, etc.)
    return MicrosoftFoundryConfig


class _AzureADTokenAuth(httpx.Auth):
    """Custom httpx Auth class that refreshes Azure AD tokens on each request.

    This ensures that tokens are refreshed for long-running applications,
    preventing authentication failures when tokens expire.
    """

    def __init__(self, token_provider: Callable[[], str]) -> None:
        """Initialize the auth handler.

        Args:
            token_provider: Callable that returns a fresh bearer token.
        """
        self._token_provider = token_provider

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        """Add the Authorization header with a fresh token.

        Args:
            request: The HTTP request to authenticate.

        Yields:
            The authenticated request.
        """
        # Get a fresh token on each request
        token = self._token_provider()
        request.headers['Authorization'] = f'Bearer {token}'
        yield request


# Plugin name
MICROSOFT_FOUNDRY_PLUGIN_NAME = 'microsoft-foundry'

# Default Azure OpenAI API version.
# Must be >= 2024-08-01-preview to support response_format: json_schema.
# See: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation
DEFAULT_API_VERSION = '2024-10-21'

# Logger for this module
logger = get_logger(__name__)


def microsoft_foundry_name(name: str) -> str:
    """Get fully qualified Microsoft Foundry model name.

    Args:
        name: The base model name (e.g., 'gpt-4o', 'DeepSeek-V3.2').

    Returns:
        Fully qualified model name (e.g., 'microsoft-foundry/gpt-4o').
    """
    return f'{MICROSOFT_FOUNDRY_PLUGIN_NAME}/{name}'


class MicrosoftFoundry(Plugin):
    """Microsoft Foundry plugin for Genkit.

    This plugin provides access to Microsoft Foundry models including:
    - OpenAI: GPT-4o, GPT-4, GPT-5, o-series reasoning models
    - Anthropic: Claude Opus, Sonnet, Haiku
    - DeepSeek: V3.2, R1 reasoning models
    - xAI: Grok 3, Grok 4
    - Meta: Llama 4 Maverick
    - Mistral: Mistral Large 3
    - Cohere: Command, Embed, Rerank
    - And 11,000+ more models in the catalog

    See: https://ai.azure.com/catalog/models

    Attributes:
        name: Plugin name ('microsoft-foundry').
    """

    name = MICROSOFT_FOUNDRY_PLUGIN_NAME

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
        azure_ad_token_provider: Callable[[], str] | None = None,
        discover_models: bool = False,
        **openai_params: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the Microsoft Foundry plugin.

        Args:
            api_key: Azure OpenAI API key. Falls back to AZURE_OPENAI_API_KEY env var.
            endpoint: Azure endpoint URL. Falls back to AZURE_OPENAI_ENDPOINT env var.
                Supports two formats:
                - Azure OpenAI: https://<resource>.openai.azure.com/
                - Foundry Project: https://<resource>.services.ai.azure.com/api/projects/<project>
            deployment: Default deployment name for models.
            api_version: API version (e.g., '2024-10-21'). Falls back to OPENAI_API_VERSION env var.
                Only used for Azure OpenAI endpoints. Foundry project endpoints use v1 API.
            azure_ad_token_provider: Token provider for Azure AD authentication.
                Use with `azure.identity.get_bearer_token_provider()` for managed identity.
            discover_models: If True, dynamically discover models from the Azure API.
                This queries the /openai/models endpoint to list available models.
                Default is False (uses the predefined model list).
            **openai_params: Additional parameters passed to the OpenAI client.

        Example:
            # Using API key with Azure OpenAI endpoint:
            plugin = MicrosoftFoundry(
                api_key="your-key",
                endpoint="https://your-resource.openai.azure.com/",
                # api_version is optional; defaults to DEFAULT_API_VERSION
            )

            # Using API key with Foundry project endpoint:
            plugin = MicrosoftFoundry(
                api_key="your-key",
                endpoint="https://your-resource.services.ai.azure.com/api/projects/your-project",
            )

            # Using Azure AD:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            plugin = MicrosoftFoundry(
                azure_ad_token_provider=token_provider,
                endpoint="https://your-resource.openai.azure.com/",
            )

            # With dynamic model discovery:
            plugin = MicrosoftFoundry(
                api_key="your-key",
                endpoint="https://your-resource.openai.azure.com/",
                discover_models=True,  # Fetches models from API
            )
        """
        # Resolve configuration from environment variables.
        # Sanitize all credential strings to remove invisible Unicode characters
        # (e.g., zero-width spaces) that may be present from copy-paste.
        api_key = _sanitize_credential(api_key or os.environ.get('AZURE_OPENAI_API_KEY'))
        resolved_endpoint = _sanitize_credential(endpoint or os.environ.get('AZURE_OPENAI_ENDPOINT'))
        api_version = _sanitize_credential(
            api_version
            or os.environ.get('AZURE_OPENAI_API_VERSION')
            or os.environ.get('OPENAI_API_VERSION', DEFAULT_API_VERSION)
        )

        if not resolved_endpoint:
            raise ValueError(
                'Azure OpenAI endpoint is required. '
                'Set AZURE_OPENAI_ENDPOINT environment variable or pass endpoint parameter.'
            )

        self._deployment = deployment
        self._discover_models = discover_models
        self._discovered_models: dict[str, dict[str, Any]] | None = None

        # Detect endpoint type and create appropriate client
        # Foundry project endpoints: *.services.ai.azure.com/api/projects/*
        # Azure OpenAI endpoints: *.openai.azure.com or *.cognitiveservices.azure.com
        self._is_foundry_endpoint = (
            '.services.ai.azure.com' in resolved_endpoint and '/api/projects/' in resolved_endpoint
        )

        if self._is_foundry_endpoint:
            # Azure AI Foundry project endpoint - use standard OpenAI client with base_url
            # The v1 API endpoint eliminates the need for api-version parameters
            # See: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/switching-endpoints
            base_url = resolved_endpoint.rstrip('/')
            if not base_url.endswith('/openai/v1'):
                base_url = f'{base_url}/openai/v1'

            logger.debug(
                'Using Foundry project endpoint',
                base_url=base_url,
            )

            # For Foundry endpoints with Azure AD auth, use a custom httpx client
            # that refreshes the token on each request to handle token expiration
            if azure_ad_token_provider and not api_key:
                # Create an httpx client with token-refreshing auth
                http_client = httpx.AsyncClient(auth=_AzureADTokenAuth(azure_ad_token_provider))
                self._openai_client: AsyncOpenAI | AsyncAzureOpenAI = AsyncOpenAI(
                    api_key='placeholder',  # Required but overridden by auth
                    base_url=base_url,
                    http_client=http_client,
                    **openai_params,
                )
            else:
                # Use API key directly
                self._openai_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    **openai_params,
                )
        else:
            # Standard Azure OpenAI endpoint - use AsyncAzureOpenAI
            logger.debug(
                'Using Azure OpenAI endpoint',
                endpoint=resolved_endpoint,
                api_version=api_version,
            )
            self._openai_client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=resolved_endpoint,
                api_version=api_version,
                azure_ad_token_provider=azure_ad_token_provider,
                **openai_params,
            )

    async def _discover_models_from_api(self) -> dict[str, dict[str, Any]]:
        """Discover available models from the Azure OpenAI API.

        Uses the /openai/models endpoint to list all models accessible
        by the Azure OpenAI resource, including base models and fine-tuned models.

        See: https://learn.microsoft.com/en-us/rest/api/azureopenai/models/list

        Returns:
            Dictionary mapping model IDs to their capabilities and metadata.
        """
        if self._discovered_models is not None:
            return self._discovered_models

        discovered: dict[str, dict[str, Any]] = {}
        try:
            models_response = await self._openai_client.models.list()
            for model in models_response.data:
                model_id = model.id
                # Extract capabilities from the model response
                capabilities = getattr(model, 'capabilities', None)
                # If capabilities is not provided by the API, default all to False
                # to avoid incorrectly advertising capabilities for unknown models
                discovered[model_id] = {
                    'id': model_id,
                    'capabilities': {
                        'chat_completion': getattr(capabilities, 'chat_completion', False) if capabilities else False,
                        'completion': getattr(capabilities, 'completion', False) if capabilities else False,
                        'embeddings': getattr(capabilities, 'embeddings', False) if capabilities else False,
                        'fine_tune': getattr(capabilities, 'fine_tune', False) if capabilities else False,
                        'inference': getattr(capabilities, 'inference', False) if capabilities else False,
                    },
                    'lifecycle_status': getattr(model, 'lifecycle_status', 'generally-available'),
                }
            self._discovered_models = discovered
        except APIError as e:
            # If discovery fails, log warning and fall back to predefined models
            logger.warning(
                'Failed to discover models from Azure API. Falling back to predefined models.',
                error=str(e),
            )
            self._discovered_models = {}

        return self._discovered_models

    async def init(self) -> list[Action]:
        """Initialize plugin and register supported models.

        If discover_models is True, fetches the list of available models
        from the Azure OpenAI API. Otherwise, uses the predefined model list.

        Returns:
            List of Action objects for supported models and embedders.
        """
        actions: list[Action] = []

        if self._discover_models:
            # Discover models dynamically from the API
            discovered = await self._discover_models_from_api()
            for model_id, info in discovered.items():
                caps = info.get('capabilities', {})
                if caps.get('chat_completion') or caps.get('completion'):
                    actions.append(self._create_model_action(microsoft_foundry_name(model_id)))
                if caps.get('embeddings'):
                    actions.append(self._create_embedder_action(microsoft_foundry_name(model_id)))
        else:
            # Register all supported models from predefined list
            for name in SUPPORTED_MICROSOFT_FOUNDRY_MODELS:
                actions.append(self._create_model_action(microsoft_foundry_name(name)))

            # Register all supported embedding models
            for name in SUPPORTED_EMBEDDING_MODELS:
                actions.append(self._create_embedder_action(microsoft_foundry_name(name)))

        return actions

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
            name: The namespaced model name (e.g., 'microsoft-foundry/gpt-4o').

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        prefix = f'{MICROSOFT_FOUNDRY_PLUGIN_NAME}/'
        clean_name = name[len(prefix) :] if name.startswith(prefix) else name

        model = MicrosoftFoundryModel(
            model_name=clean_name,
            client=self._openai_client,
            deployment=self._deployment,
        )
        model_info = get_model_info(clean_name)

        # Get the appropriate config schema for this model family
        config_schema = get_config_schema_for_model(clean_name)

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=model.generate,
            metadata=model_action_metadata(
                name=name,
                info=model_info.supports.model_dump() if model_info.supports else {},
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
        prefix = f'{MICROSOFT_FOUNDRY_PLUGIN_NAME}/'
        clean_name = name[len(prefix) :] if name.startswith(prefix) else name

        # Get embedder info
        embedder_info = SUPPORTED_EMBEDDING_MODELS.get(
            clean_name,
            {
                'label': f'Microsoft Foundry - {clean_name}',
                'dimensions': 1536,
                'supports': {'input': ['text']},
            },
        )

        async def embed_fn(request: EmbedRequest) -> EmbedResponse:
            """Generate embeddings using Azure OpenAI."""
            # Extract text from document content
            texts: list[str] = []
            for doc in request.input:
                text_parts: list[str] = []
                for part in doc.content:
                    if hasattr(part.root, 'text') and part.root.text:
                        text_parts.append(str(part.root.text))
                doc_text = ''.join(text_parts)
                texts.append(doc_text)

            # Get optional parameters from request
            dimensions: int | None = None
            encoding_format: str | None = None
            if request.options:
                if dim_val := request.options.get('dimensions'):
                    try:
                        dimensions = int(dim_val)
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Invalid value for 'dimensions' option: {dim_val}") from e
                if enc_val := request.options.get('encodingFormat'):
                    encoding_format = str(enc_val) if enc_val in ('float', 'base64') else None

            # Build API call kwargs - only include non-None optional params
            api_kwargs: dict[str, Any] = {
                'model': clean_name,
                'input': texts,
            }
            if dimensions is not None:
                api_kwargs['dimensions'] = dimensions
            if encoding_format is not None:
                api_kwargs['encoding_format'] = encoding_format

            # Call Azure OpenAI embeddings API
            response = await self._openai_client.embeddings.create(**api_kwargs)

            # Convert to Genkit format
            embeddings = [Embedding(embedding=item.embedding) for item in response.data]
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

        If discover_models is True, includes dynamically discovered models
        from the Azure OpenAI API alongside the predefined models.

        Returns:
            List of ActionMetadata for all supported models and embedders.
        """
        actions: list[ActionMetadata] = []

        if self._discover_models:
            # Include dynamically discovered models
            discovered = await self._discover_models_from_api()
            for model_id, info in discovered.items():
                caps = info.get('capabilities', {})
                model_info = get_model_info(model_id)

                if caps.get('chat_completion') or caps.get('completion'):
                    config_schema = get_config_schema_for_model(model_id)
                    actions.append(
                        model_action_metadata(
                            name=microsoft_foundry_name(model_id),
                            info=model_info.supports.model_dump() if model_info.supports else {},
                            config_schema=config_schema,
                        )
                    )
                if caps.get('embeddings'):
                    embed_info = SUPPORTED_EMBEDDING_MODELS.get(
                        model_id,
                        {
                            'label': f'Microsoft Foundry - {model_id}',
                            'dimensions': 1536,
                            'supports': {'input': ['text']},
                        },
                    )
                    actions.append(
                        embedder_action_metadata(
                            name=microsoft_foundry_name(model_id),
                            options=EmbedderOptions(
                                label=embed_info['label'],
                                supports=EmbedderSupports(input=embed_info['supports']['input']),
                                dimensions=embed_info.get('dimensions'),
                            ),
                        )
                    )
        else:
            # Add model metadata from predefined list
            for model_name, model_info in SUPPORTED_MICROSOFT_FOUNDRY_MODELS.items():
                config_schema = get_config_schema_for_model(model_name)
                actions.append(
                    model_action_metadata(
                        name=microsoft_foundry_name(model_name),
                        info=model_info.supports.model_dump() if model_info.supports else {},
                        config_schema=config_schema,
                    )
                )

            # Add embedder metadata from predefined list
            for embed_name, embed_info in SUPPORTED_EMBEDDING_MODELS.items():
                actions.append(
                    embedder_action_metadata(
                        name=microsoft_foundry_name(embed_name),
                        options=EmbedderOptions(
                            label=embed_info['label'],
                            supports=EmbedderSupports(input=embed_info['supports']['input']),
                            dimensions=embed_info.get('dimensions'),
                        ),
                    )
                )

        return actions


def microsoft_foundry_model(name: str) -> str:
    """Get fully qualified Microsoft Foundry model name.

    Convenience function for specifying models. Can be used with any model
    from the catalog (11,000+ models).

    See: https://ai.azure.com/catalog/models

    Args:
        name: The base model name (e.g., 'gpt-4o', 'DeepSeek-V3.2', 'claude-opus-4-5').

    Returns:
        Fully qualified model name (e.g., 'microsoft-foundry/gpt-4o').

    Example:
        ai = Genkit(
            plugins=[MicrosoftFoundry(...)],
            model=microsoft_foundry_model("gpt-4o"),
        )

        # Or with other providers:
        response = await ai.generate(
            model=microsoft_foundry_model("DeepSeek-V3.2"),
            prompt="Explain quantum computing.",
        )
    """
    return microsoft_foundry_name(name)


# Pre-defined model references for convenience
gpt4o = microsoft_foundry_name('gpt-4o')
gpt4o_mini = microsoft_foundry_name('gpt-4o-mini')
gpt4 = microsoft_foundry_name('gpt-4')
gpt35_turbo = microsoft_foundry_name('gpt-3.5-turbo')
o3_mini = microsoft_foundry_name('o3-mini')
o1 = microsoft_foundry_name('o1')
o1_mini = microsoft_foundry_name('o1-mini')
