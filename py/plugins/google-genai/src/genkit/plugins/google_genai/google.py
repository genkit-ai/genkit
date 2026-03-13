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


"""Google AI and Vertex AI plugin implementations for Genkit.

This module provides the GoogleAI and VertexAI plugins that enable Genkit to use
Google's generative AI models. Both plugins use **dynamic model discovery** to
automatically detect and register available models from the Google GenAI SDK.

Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Dynamic Model Discovery                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   Plugin Init                                                           │
    │   ┌─────────┐     ┌──────────────┐     ┌─────────────────────────────┐ │
    │   │ GoogleAI│────►│client.models │────►│ Filter & Categorize         │ │
    │   │ VertexAI│     │   .list()    │     │ ┌─────────┬───────────────┐ │ │
    │   └─────────┘     └──────────────┘     │ │ Action  │ Model Type    │ │ │
    │                                        │ ├─────────┼───────────────┤ │ │
    │                                        │ │generate │ gemini, gemma │ │ │
    │                                        │ │Content  │               │ │ │
    │                                        │ ├─────────┼───────────────┤ │ │
    │                                        │ │embed    │ text-embedding│ │ │
    │                                        │ │Content  │               │ │ │
    │                                        │ ├─────────┼───────────────┤ │ │
    │                                        │ │predict  │ imagen        │ │ │
    │                                        │ ├─────────┼───────────────┤ │ │
    │                                        │ │generate │ veo           │ │ │
    │                                        │ │Videos   │               │ │ │
    │                                        │ └─────────┴───────────────┘ │ │
    │                                        └─────────────────────────────┘ │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    ```

Key Concepts:
    +--------------------+-------------------------------------------------------+
    | Concept            | Description                                           |
    +--------------------+-------------------------------------------------------+
    | Dynamic Discovery  | Models are discovered at runtime via the API, not     |
    |                    | hardcoded. This ensures new models are automatically  |
    |                    | available without SDK updates.                        |
    +--------------------+-------------------------------------------------------+
    | Background Models  | Long-running operations (e.g., Veo video generation)  |
    |                    | use start/check pattern instead of blocking generate. |
    +--------------------+-------------------------------------------------------+
    | Action Resolution  | On-demand model instantiation when a model is first   |
    |                    | used, avoiding upfront initialization overhead.       |
    +--------------------+-------------------------------------------------------+
    | Namespacing        | Models are prefixed with plugin name (e.g.,           |
    |                    | 'googleai/gemini-2.0-flash-001').                     |
    +--------------------+-------------------------------------------------------+

Supported Model Types:
    - **Gemini/Gemma**: Text generation with generateContent action
    - **Embedders**: Text embeddings with embedContent action
    - **Imagen**: Image generation with predict action
    - **Veo**: Video generation with generateVideos action

Example:
    >>> from genkit import Genkit
    >>> from genkit.plugins.google_genai import GoogleAI
    >>>
    >>> # Models are discovered automatically
    >>> ai = Genkit(plugins=[GoogleAI()])
    >>>
    >>> # Use any available model - no pre-registration needed
    >>> response = await ai.generate(
    ...     model='googleai/gemini-2.0-flash-001',
    ...     prompt='Hello, world!',
    ... )

See Also:
    - https://ai.google.dev/gemini-api/docs
    - https://cloud.google.com/vertex-ai/generative-ai/docs
    - JS implementation: js/plugins/google-genai/src/
"""

import os
from typing import Any

from google import genai
from google.auth.credentials import Credentials
from google.genai.client import DebugConfig
from google.genai.types import HttpOptions, HttpOptionsDict

import genkit.plugins.google_genai.constants as const
from genkit.ai import GENKIT_CLIENT_HEADER, Plugin
from genkit.blocks.background_model import BackgroundAction
from genkit.blocks.document import Document
from genkit.blocks.embedding import EmbedderOptions, EmbedderSupports, embedder_action_metadata
from genkit.blocks.model import model_action_metadata
from genkit.blocks.reranker import reranker_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.error import GenkitError
from genkit.core.registry import ActionKind
from genkit.core.schema import to_json_schema
from genkit.core.typing import (
    EvalFnResponse,
    EvalRequest,
    RankedDocumentData,
    RankedDocumentMetadata,
    RerankerRequest,
    RerankerResponse,
)
from genkit.plugins.google_genai.evaluators import (
    VertexAIEvaluationMetricType,
    create_vertex_evaluators,
)
from genkit.plugins.google_genai.models.embedder import (
    Embedder,
    default_embedder_info,
)
from genkit.plugins.google_genai.models.gemini import (
    SUPPORTED_MODELS,
    GeminiModel,
    get_model_config_schema,
    google_model_info,
)
from genkit.plugins.google_genai.models.imagen import (
    SUPPORTED_MODELS as IMAGE_SUPPORTED_MODELS,
    ImagenConfigSchema,
    ImagenModel,
    vertexai_image_model_info,
)
from genkit.plugins.google_genai.models.veo import (
    VeoConfigSchema,
    VeoModel,
    is_veo_model,
    veo_model_info,
)
from genkit.plugins.google_genai.rerankers.reranker import (
    KNOWN_MODELS as RERANKER_MODELS,
    RerankRequest,
    VertexRerankerClientOptions,
    VertexRerankerConfig,
    _from_rerank_response,
    _to_reranker_doc,
    reranker_rank,
)


class GenaiModels:
    """Container for models discovered dynamically from the Google GenAI API.

    This class categorizes models by their capabilities based on the
    supported_actions field returned by the API.

    Attributes:
        gemini: List of Gemini/Gemma model names (generateContent action).
        imagen: List of Imagen model names (predict action, Vertex AI only).
        embedders: List of embedding model names (embedContent action).
        veo: List of Veo video generation model names (generateVideos action).
    """

    gemini: list[str]
    imagen: list[str]
    embedders: list[str]
    veo: list[str]

    def __init__(self) -> None:
        """Initialize empty model lists."""
        self.gemini = []
        self.imagen = []
        self.embedders = []
        self.veo = []


def _list_genai_models(client: genai.Client, is_vertex: bool) -> GenaiModels:
    """Discover and categorize available models from the Google GenAI API.

    This function queries the API for all available models and categorizes them
    based on their supported_actions field. Models marked as deprecated are
    excluded.

    The categorization logic:
        - 'embedContent' action → embedders
        - 'predict' + 'imagen' in name → imagen (Vertex AI)
        - 'generateVideos' or 'veo' in name → veo
        - 'generateContent' + 'gemini'/'gemma' in name → gemini

    Args:
        client: The Google GenAI client instance.
        is_vertex: True if using Vertex AI, False for Google AI.

    Returns:
        GenaiModels containing categorized model names.

    Note:
        Model name prefixes are stripped for consistency:
        - Vertex AI: 'publishers/google/models/' prefix removed
        - Google AI: 'models/' prefix removed
    """
    models = GenaiModels()

    for m in client.models.list():
        name = m.name
        if not name:
            continue

        # Cleanup prefix
        if is_vertex:
            if name.startswith('publishers/google/models/'):
                name = name[25:]
        elif name.startswith('models/'):
            name = name[7:]

        description = (m.description or '').lower()
        if 'deprecated' in description:
            continue

        if not m.supported_actions:
            continue

        # Embedders
        if 'embedContent' in m.supported_actions:
            models.embedders.append(name)

        # Imagen (Vertex mostly)
        if 'predict' in m.supported_actions and 'imagen' in name.lower():
            models.imagen.append(name)

        # Veo
        if 'generateVideos' in m.supported_actions or 'veo' in name.lower():
            models.veo.append(name)

        # Gemini / Gemma
        if 'generateContent' in m.supported_actions:
            lower_name = name.lower()
            if 'gemini' in lower_name or 'gemma' in lower_name:
                models.gemini.append(name)

    return models


GOOGLEAI_PLUGIN_NAME = 'googleai'
VERTEXAI_PLUGIN_NAME = 'vertexai'


def googleai_name(name: str) -> str:
    """Create a GoogleAI action name.

    Args:
        name: Base name for the action.

    Returns:
        The fully qualified Google AI action name.
    """
    return f'{GOOGLEAI_PLUGIN_NAME}/{name}'


def vertexai_name(name: str) -> str:
    """Create a VertexAI action name.

    Args:
        name: Base name for the action.

    Returns:
        The fully qualified Google AI action name.
    """
    return f'{VERTEXAI_PLUGIN_NAME}/{name}'


def _create_embedder_action(name: str, client: genai.Client, plugin_name: str) -> Action:
    """Create an Action object for an embedder.

    Args:
        name: The namespaced name of the embedder.
        client: The Google GenAI client instance.
        plugin_name: The name of the plugin (googleai or vertexai).

    Returns:
        Action object for the embedder.
    """
    clean_name = name.replace(f'{plugin_name}/', '') if name.startswith(plugin_name) else name
    embedder = Embedder(version=clean_name, client=client)
    embedder_info = default_embedder_info(clean_name)

    action_metadata = embedder_action_metadata(
        name=name,
        options=EmbedderOptions(
            label=embedder_info.get('label'),
            supports=EmbedderSupports(input=embedder_info.get('supports', {}).get('input')),
            dimensions=embedder_info.get('dimensions'),
        ),
    )

    action = Action(
        kind=ActionKind.EMBEDDER,
        name=name,
        fn=embedder.generate,
        metadata=action_metadata.metadata,
    )

    # Explicitly set schemas (no 'if' needed as they are always present in metadata)
    action.input_schema = action_metadata.input_json_schema  # type: ignore[invalid-assignment]
    action.output_schema = action_metadata.output_json_schema  # type: ignore[invalid-assignment]

    return action


class GoogleAI(Plugin):
    """GoogleAI plugin for Genkit with dynamic model discovery.

    This plugin provides access to Google AI models (Gemini, embedders, Veo)
    through the Google AI Studio API. Models are discovered dynamically at
    initialization time, ensuring new models are available without SDK updates.

    Model Types:
        +------------------+-------------------+--------------------------------+
        | Type             | Action Kind       | Example                        |
        +------------------+-------------------+--------------------------------+
        | Gemini/Gemma     | MODEL             | googleai/gemini-2.0-flash-001  |
        | Imagen           | MODEL             | googleai/imagen-3.0-generate   |
        | Embedders        | EMBEDDER          | googleai/gemini-embedding-001  |
        | Veo (video)      | BACKGROUND_MODEL  | googleai/veo-2.0-generate-001  |
        +------------------+-------------------+--------------------------------+

    Example:
        >>> from genkit import Genkit
        >>> from genkit.plugins.google_genai import GoogleAI
        >>>
        >>> ai = Genkit(plugins=[GoogleAI()])
        >>>
        >>> # Text generation
        >>> response = await ai.generate(
        ...     model='googleai/gemini-2.0-flash-001',
        ...     prompt='Explain quantum computing',
        ... )
        >>>
        >>> # Video generation (background model)
        >>> op = await ai.generate(
        ...     model='googleai/veo-2.0-generate-001',
        ...     prompt='A sunset over mountains',
        ... )
        >>> while not op.done:
        ...     await asyncio.sleep(5)
        ...     op = await ai.check_operation(op)

    Attributes:
        name: The plugin name ('googleai').
        _vertexai: Internal flag, always False for GoogleAI.

    See Also:
        - https://ai.google.dev/gemini-api/docs
        - https://aistudio.google.com/
    """

    name = GOOGLEAI_PLUGIN_NAME
    _vertexai = False

    def __init__(
        self,
        api_key: str | None = None,
        credentials: Credentials | None = None,
        debug_config: DebugConfig | None = None,
        http_options: HttpOptions | HttpOptionsDict | None = None,
        api_version: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initializes the GoogleAI plugin.

        Args:
            api_key: The API key for authenticating with the Google AI service.
                If not provided, it defaults to reading from the 'GEMINI_API_KEY'
                environment variable.
            credentials: Google Cloud credentials for authentication.
                Defaults to None, in which case the client uses default authentication
                mechanisms (e.g., application default credentials or API key).
            debug_config: Configuration for debugging the client. Defaults to None.
            http_options: HTTP options for configuring the client's network requests.
                Can be an instance of HttpOptions or a dictionary. Defaults to None.
            api_version: The API version to use (e.g., 'v1beta'). Defaults to None.
            base_url: The base URL for the API. Defaults to None.

        Raises:
            ValueError: If `api_key` is not provided and the 'GEMINI_API_KEY'
                environment variable is not set.
        """
        api_key = api_key if api_key else os.getenv('GEMINI_API_KEY')
        if not api_key and credentials is None:
            raise ValueError(
                'Gemini api key should be passed in plugin params or as a GEMINI_API_KEY environment variable'
            )

        self._client = genai.client.Client(
            vertexai=self._vertexai,
            api_key=api_key,
            credentials=credentials,
            debug_config=debug_config,
            http_options=_inject_attribution_headers(http_options, base_url, api_version),
        )

    async def init(self) -> list[Action]:
        """Initialize the plugin.

        Returns:
            List of Action objects for known/supported models.
        """
        genai_models = _list_genai_models(self._client, is_vertex=False)

        actions: list[Action] = []
        # Gemini Models
        for name in genai_models.gemini:
            actions.append(self._resolve_model(googleai_name(name)))

        # Imagen Models
        for name in genai_models.imagen:
            actions.append(self._resolve_model(googleai_name(name)))

        # Veo Models (background models)
        for name in genai_models.veo:
            bg_action = self._resolve_veo_model(googleai_name(name))
            actions.append(bg_action.start_action)
            actions.append(bg_action.check_action)

        # Embedders
        for name in genai_models.embedders:
            actions.append(self._resolve_embedder(googleai_name(name)))

        return actions

    def _list_known_models(self) -> list[Action]:
        """List known models as Action objects.

        Deprecated: Used only for internal testing if needed, but 'init' should be source of truth.
        Keeping for compatibility but redirecting to dynamic list logic if accessed directly?
        The interface defines init(), this helper was internal.
        """
        # Re-use init logic synchronously? init is async.
        # Let's implementation just mimic init logic but sync call to client.models.list is fine (it is iterator)
        genai_models = _list_genai_models(self._client, is_vertex=False)
        actions = []
        for name in genai_models.gemini:
            actions.append(self._resolve_model(googleai_name(name)))
        for name in genai_models.imagen:
            actions.append(self._resolve_model(googleai_name(name)))
        return actions

    def _list_known_veo_models(self) -> list[Action]:
        """List known Veo models as background model Action objects.

        Returns:
            List of Action objects for known Veo video generation models.
        """
        genai_models = _list_genai_models(self._client, is_vertex=False)
        actions = []
        for name in genai_models.veo:
            bg_action = self._resolve_veo_model(googleai_name(name))
            actions.append(bg_action.start_action)
            actions.append(bg_action.check_action)
        return actions

    def _list_known_embedders(self) -> list[Action]:
        """List known embedders as Action objects."""
        genai_models = _list_genai_models(self._client, is_vertex=False)
        actions = []
        for name in genai_models.embedders:
            actions.append(self._resolve_embedder(googleai_name(name)))
        return actions

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced name of the action to resolve.

        Returns:
            Action object if found, None otherwise.
        """
        if action_type == ActionKind.MODEL:
            return self._resolve_model(name)
        elif action_type == ActionKind.BACKGROUND_MODEL:
            # For Veo models, return the start action
            prefix = GOOGLEAI_PLUGIN_NAME + '/'
            clean_name = name.replace(prefix, '') if name.startswith(prefix) else name
            if is_veo_model(clean_name):
                bg_action = self._resolve_veo_model(name)
                return bg_action.start_action
            return None
        elif action_type == ActionKind.CHECK_OPERATION:
            # Check action names are in format {model_name}/check
            # Extract the model name and resolve if it's a Veo model
            if name.endswith('/check'):
                model_name = name[:-6]  # Remove '/check' suffix
                prefix = GOOGLEAI_PLUGIN_NAME + '/'
                clean_name = model_name.replace(prefix, '') if model_name.startswith(prefix) else model_name
                if is_veo_model(clean_name):
                    bg_action = self._resolve_veo_model(model_name)
                    return bg_action.check_action
            return None
        elif action_type == ActionKind.EMBEDDER:
            return self._resolve_embedder(name)
        return None

    def _resolve_veo_model(self, name: str) -> 'BackgroundAction':
        """Create a BackgroundAction for a Veo video generation model.

        Args:
            name: The namespaced name of the model.

        Returns:
            BackgroundAction for the Veo model.
        """
        clean_name = name.replace(GOOGLEAI_PLUGIN_NAME + '/', '') if name.startswith(GOOGLEAI_PLUGIN_NAME) else name

        veo = VeoModel(clean_name, self._client)

        # Create actions manually since we don't have registry access here

        # Prepare metadata matching model_action_metadata structure
        info = veo_model_info(clean_name).model_dump(by_alias=True)
        config_schema = VeoConfigSchema

        start_action = Action(
            kind=ActionKind.BACKGROUND_MODEL,
            name=name,
            fn=veo.start,
            metadata={
                'model': {**info, 'customOptions': to_json_schema(config_schema)},
                'type': 'background-model',
            },
        )

        check_action = Action(
            kind=ActionKind.CHECK_OPERATION,
            name=f'{name}/check',
            fn=lambda op, ctx: veo.check(op),
            metadata={'type': 'check-operation'},
        )

        return BackgroundAction(
            start_action=start_action,
            check_action=check_action,
            cancel_action=None,
        )

    def _resolve_model(self, name: str) -> Action:
        """Create an Action object for a Google AI model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace(GOOGLEAI_PLUGIN_NAME + '/', '') if name.startswith(GOOGLEAI_PLUGIN_NAME) else name

        # Determine model type and create appropriate model instance
        if clean_name.lower().startswith('image'):
            model_ref = vertexai_image_model_info(clean_name)
            model = ImagenModel(clean_name, self._client)
            IMAGE_SUPPORTED_MODELS[clean_name] = model_ref
            config_schema = ImagenConfigSchema
        else:
            model_ref = google_model_info(clean_name)
            model = GeminiModel(clean_name, self._client)
            SUPPORTED_MODELS[clean_name] = model_ref
            config_schema = get_model_config_schema(clean_name)

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=model.generate,
            metadata=model_action_metadata(
                name=name,
                info=model.metadata['model'],
                config_schema=config_schema,
            ).metadata,
        )

    def _resolve_embedder(self, name: str) -> Action:
        """Create an Action object for a Google AI embedder.

        Args:
            name: The namespaced name of the embedder.

        Returns:
            Action object for the embedder.
        """
        return _create_embedder_action(name, self._client, GOOGLEAI_PLUGIN_NAME)

    async def list_actions(self) -> list[ActionMetadata]:
        """Generate a list of available actions or models.

        Returns:
            list[ActionMetadata]: A list of ActionMetadata objects, each with the following attributes:
                - name (str): The name of the action or model.
                - kind (ActionKind): The type or category of the action.
                - info (dict): The metadata dictionary describing the model configuration and properties.
                - config_schema (type): The schema class used for validating the model's configuration.
        """
        genai_models = _list_genai_models(self._client, is_vertex=False)
        actions_list = []

        for name in genai_models.gemini:
            actions_list.append(
                model_action_metadata(
                    name=googleai_name(name),
                    info=google_model_info(name).model_dump(by_alias=True),
                    config_schema=get_model_config_schema(name),
                )
            )

        for name in genai_models.imagen:
            actions_list.append(
                model_action_metadata(
                    name=googleai_name(name),
                    info=vertexai_image_model_info(name).model_dump(by_alias=True),
                    config_schema=ImagenConfigSchema,
                )
            )

        for name in genai_models.veo:
            actions_list.append(
                model_action_metadata(
                    name=googleai_name(name),
                    info=veo_model_info(name).model_dump(by_alias=True),
                    config_schema=VeoConfigSchema,
                )
            )

        for name in genai_models.embedders:
            embed_info = default_embedder_info(name)
            actions_list.append(
                embedder_action_metadata(
                    name=googleai_name(name),
                    options=EmbedderOptions(
                        label=embed_info.get('label'),
                        supports=EmbedderSupports(input=embed_info.get('supports', {}).get('input')),
                        dimensions=embed_info.get('dimensions'),
                    ),
                )
            )

        return actions_list


class VertexAI(Plugin):
    """Vertex AI plugin for Genkit with dynamic model discovery.

    This plugin provides access to Google Cloud Vertex AI models including
    Gemini, Imagen, Veo, and embedders. Models are discovered dynamically,
    ensuring new models are available without SDK updates.

    Vertex AI vs Google AI:
        Vertex AI provides enterprise features including:
        - VPC Service Controls
        - Customer-managed encryption keys (CMEK)
        - Data residency controls
        - IAM-based access control
        - Imagen image generation models

    Model Types:
        +------------------+-------------------+--------------------------------+
        | Type             | Action Kind       | Example                        |
        +------------------+-------------------+--------------------------------+
        | Gemini/Gemma     | MODEL             | vertexai/gemini-2.0-flash-001  |
        | Imagen           | MODEL             | vertexai/imagen-3.0-generate   |
        | Veo (video)      | MODEL             | vertexai/veo-2.0-generate-001  |
        | Embedders        | EMBEDDER          | vertexai/text-embedding-005    |
        +------------------+-------------------+--------------------------------+

    Example:
        >>> from genkit import Genkit
        >>> from genkit.plugins.google_genai import VertexAI
        >>>
        >>> ai = Genkit(plugins=[VertexAI(project='my-project')])
        >>>
        >>> # Text generation
        >>> response = await ai.generate(
        ...     model='vertexai/gemini-2.0-flash-001',
        ...     prompt='Explain quantum computing',
        ... )
        >>>
        >>> # Image generation (Vertex AI only)
        >>> response = await ai.generate(
        ...     model='vertexai/imagen-3.0-generate-002',
        ...     prompt='A serene mountain landscape',
        ... )

    Attributes:
        name: The plugin name ('vertexai').
        _vertexai: Internal flag, always True for VertexAI.

    See Also:
        - https://cloud.google.com/vertex-ai/generative-ai/docs
    """

    _vertexai = True

    name = VERTEXAI_PLUGIN_NAME

    def __init__(
        self,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: str | None = 'us-central1',
        debug_config: DebugConfig | None = None,
        http_options: HttpOptions | HttpOptionsDict | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initializes the VertexAI plugin.

        Args:
            credentials: Google Cloud credentials for authentication.
                Defaults to None, in which case the client uses default authentication
                mechanisms (e.g., application default credentials or API key).
            project: Name of the Google Cloud project.
            location: Location of the Google Cloud project.
            debug_config: Configuration for debugging the client. Defaults to None.
            http_options: HTTP options for configuring the client's network requests.
                Can be an instance of HttpOptions or a dictionary. Defaults to None.
            api_key: The API key for authenticating with the Google AI service.
                If not provided, it defaults to reading from the 'GEMINI_API_KEY'
                environment variable.
            api_version: The API version to use. Defaults to None.
            base_url: The base URL for the API. Defaults to None.
        """
        # Store project and location on the plugin for reranker resolution.
        # This avoids reaching into client internals.
        self._project = project if project else os.getenv(const.GCLOUD_PROJECT)
        self._location = location if location else const.DEFAULT_REGION

        self._client = genai.client.Client(
            vertexai=self._vertexai,
            api_key=api_key,
            credentials=credentials,
            project=self._project,
            location=self._location,
            debug_config=debug_config,
            http_options=_inject_attribution_headers(http_options, base_url, api_version),
        )

    async def init(self) -> list[Action]:
        """Initialize the plugin.

        Returns:
            List of Action objects for known/supported models.
        """
        genai_models = _list_genai_models(self._client, is_vertex=True)
        actions: list[Action] = []

        for name in genai_models.gemini:
            actions.append(self._resolve_model(vertexai_name(name)))

        for name in genai_models.imagen:
            actions.append(self._resolve_model(vertexai_name(name)))

        for name in genai_models.veo:
            actions.append(self._resolve_model(vertexai_name(name)))

        for name in genai_models.embedders:
            actions.append(self._resolve_embedder(vertexai_name(name)))

        # Register Vertex AI rerankers
        for name in RERANKER_MODELS:
            actions.append(self._resolve_reranker(vertexai_name(name)))

        # Register Vertex AI evaluators
        # Deferred import to avoid circular dependency
        from genkit.ai._registry import GenkitRegistry

        if not self._project:
            raise ValueError(
                'VertexAI plugin requires a project ID to use evaluators. '
                'Set the project parameter or GOOGLE_CLOUD_PROJECT environment variable.'
            )
        registry = GenkitRegistry()
        actions.extend(
            create_vertex_evaluators(
                registry,
                list(VertexAIEvaluationMetricType),
                project_id=self._project,
                location=self._location,
            )
        )

        return actions

    def _list_known_models(self) -> list[Action]:
        """List known models as Action objects."""
        genai_models = _list_genai_models(self._client, is_vertex=True)
        actions = []
        for name in genai_models.gemini:
            actions.append(self._resolve_model(vertexai_name(name)))
        for name in genai_models.imagen:
            actions.append(self._resolve_model(vertexai_name(name)))
        for name in genai_models.veo:
            actions.append(self._resolve_model(vertexai_name(name)))
        return actions

    def _list_known_embedders(self) -> list[Action]:
        """List known embedders as Action objects."""
        genai_models = _list_genai_models(self._client, is_vertex=True)
        actions = []
        for name in genai_models.embedders:
            actions.append(self._resolve_embedder(vertexai_name(name)))
        return actions

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced name of the action to resolve.

        Returns:
            Action object if found, None otherwise.
        """
        if action_type == ActionKind.MODEL:
            return self._resolve_model(name)
        elif action_type == ActionKind.EMBEDDER:
            return self._resolve_embedder(name)
        elif action_type == ActionKind.RERANKER:
            return self._resolve_reranker(name)
        elif action_type == ActionKind.EVALUATOR:
            return self._resolve_evaluator(name)
        return None

    def _resolve_evaluator(self, name: str) -> Action | None:
        """Create an Action object for a Vertex AI evaluator.

        Args:
            name: The namespaced name of the evaluator.

        Returns:
            Action object for the evaluator.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace(VERTEXAI_PLUGIN_NAME + '/', '') if name.startswith(VERTEXAI_PLUGIN_NAME) else name

        try:
            metric_type = VertexAIEvaluationMetricType(clean_name.upper())
        except ValueError:
            return None

        from genkit.ai._registry import GenkitRegistry

        registry = GenkitRegistry()
        if not self._project:
            raise ValueError(
                'VertexAI plugin requires a project ID to use evaluators. '
                'Set the project parameter or GOOGLE_CLOUD_PROJECT environment variable.'
            )

        actions = create_vertex_evaluators(
            registry,
            [metric_type],
            project_id=self._project,
            location=self._location,
        )
        return actions[0] if actions else None

    def _resolve_model(self, name: str) -> Action:
        """Create an Action object for a Vertex AI model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace(VERTEXAI_PLUGIN_NAME + '/', '') if name.startswith(VERTEXAI_PLUGIN_NAME) else name

        # Determine model type and create appropriate model instance
        if clean_name.lower().startswith('image'):
            model_ref = vertexai_image_model_info(clean_name)
            model = ImagenModel(clean_name, self._client)
            IMAGE_SUPPORTED_MODELS[clean_name] = model_ref
            config_schema = ImagenConfigSchema
        elif is_veo_model(clean_name):
            model_ref = veo_model_info(clean_name)
            model = VeoModel(clean_name, self._client)
            config_schema = VeoConfigSchema
        else:
            model_ref = google_model_info(clean_name)
            model = GeminiModel(clean_name, self._client)
            SUPPORTED_MODELS[clean_name] = model_ref
            config_schema = get_model_config_schema(clean_name)

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=model.generate,
            metadata=model_action_metadata(
                name=name,
                info=model.metadata['model'],
                config_schema=config_schema,
            ).metadata,
        )

    def _resolve_embedder(self, name: str) -> Action:
        """Create an Action object for a Vertex AI embedder.

        Args:
            name: The namespaced name of the embedder.

        Returns:
            Action object for the embedder.
        """
        return _create_embedder_action(name, self._client, VERTEXAI_PLUGIN_NAME)

    def _resolve_reranker(self, name: str) -> Action:
        """Create an Action object for a Vertex AI reranker.

        Args:
            name: The namespaced name of the reranker.

        Returns:
            Action object for the reranker.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace(VERTEXAI_PLUGIN_NAME + '/', '') if name.startswith(VERTEXAI_PLUGIN_NAME) else name

        # Validate project is configured (required for reranker API)
        if not self._project:
            raise ValueError(
                'VertexAI plugin requires a project ID to use rerankers. '
                'Set the project parameter or GOOGLE_CLOUD_PROJECT environment variable.'
            )

        # Use project and location stored on the plugin instance during init.
        # This avoids accessing private attributes of the client library.
        client_options = VertexRerankerClientOptions(
            project_id=self._project,
            location=self._location,
        )

        async def wrapper(
            request: RerankerRequest,
            _ctx: Any,  # noqa: ANN401
        ) -> RerankerResponse:
            """Wrapper that takes RerankerRequest and returns RerankerResponse.

            This matches the signature expected by the Action class (max 2 args).
            """
            query_doc = Document.from_document_data(request.query)
            documents = [Document.from_document_data(d) for d in request.documents]
            options = request.options

            config = VertexRerankerConfig.model_validate(options or {})

            # Use location from config if provided, otherwise use client default
            effective_options = VertexRerankerClientOptions(
                project_id=client_options.project_id,
                location=config.location or client_options.location,
            )

            query_text = query_doc.text()
            if not query_text:
                raise GenkitError(message='Reranker query cannot be empty.')

            rerank_request = RerankRequest(
                model=clean_name,
                query=query_text,
                records=[_to_reranker_doc(doc, idx) for idx, doc in enumerate(documents)],
                top_n=config.top_n,
                ignore_record_details_in_response=config.ignore_record_details_in_response,
            )

            response = await reranker_rank(clean_name, rerank_request, effective_options)
            ranked_docs = _from_rerank_response(response, documents)

            # Convert to RerankerResponse format - ranked_docs are RankedDocument instances
            response_docs: list[RankedDocumentData] = []
            for doc in ranked_docs:
                metadata = RankedDocumentMetadata(score=doc.score if doc.score is not None else 0.0)
                response_docs.append(RankedDocumentData(content=doc.content, metadata=metadata))

            return RerankerResponse(documents=response_docs)

        metadata = reranker_action_metadata(name)

        return Action(
            kind=ActionKind.RERANKER,
            name=name,
            fn=wrapper,
            metadata=metadata.metadata,
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """Generate a list of available actions or models.

        Returns:
            list[ActionMetadata]: A list of ActionMetadata objects, each with the following attributes:
                - name (str): The name of the action or model.
                - kind (ActionKind): The type or category of the action.
                - info (dict): The metadata dictionary describing the model configuration and properties.
                - config_schema (type): The schema class used for validating the model's configuration.
        """
        genai_models = _list_genai_models(self._client, is_vertex=True)
        actions_list = []

        for name in genai_models.gemini:
            actions_list.append(
                model_action_metadata(
                    name=vertexai_name(name),
                    info=google_model_info(name).model_dump(by_alias=True),
                    config_schema=get_model_config_schema(name),
                )
            )

        for name in genai_models.imagen:
            actions_list.append(
                model_action_metadata(
                    name=vertexai_name(name),
                    info=vertexai_image_model_info(name).model_dump(by_alias=True),
                    config_schema=ImagenConfigSchema,
                )
            )

        for name in genai_models.veo:
            actions_list.append(
                model_action_metadata(
                    name=vertexai_name(name),
                    info=veo_model_info(name).model_dump(by_alias=True),
                    config_schema=VeoConfigSchema,
                )
            )

        for name in genai_models.embedders:
            embed_info = default_embedder_info(name)
            actions_list.append(
                embedder_action_metadata(
                    name=vertexai_name(name),
                    options=EmbedderOptions(
                        label=embed_info.get('label'),
                        supports=EmbedderSupports(input=embed_info.get('supports', {}).get('input')),
                        dimensions=embed_info.get('dimensions'),
                    ),
                )
            )

        for metric in VertexAIEvaluationMetricType:
            # create_vertex_evaluators handles namespacing but we only need metadata here.
            evaluator_name = vertexai_name(metric.lower())
            actions_list.append(
                ActionMetadata(
                    name=evaluator_name,
                    kind=ActionKind.EVALUATOR,
                    input_json_schema=to_json_schema(EvalRequest),
                    output_json_schema=to_json_schema(list[EvalFnResponse]),
                    metadata={'type': 'evaluator'},
                )
            )

        return actions_list


def _inject_attribution_headers(
    http_options: HttpOptions | HttpOptionsDict | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
) -> HttpOptions:
    """Adds genkit client info to the appropriate http headers."""
    if not http_options:
        opts = HttpOptions()
    elif isinstance(http_options, HttpOptions):
        opts = http_options
    else:
        opts = HttpOptions.model_validate(http_options)

    if base_url:
        opts.base_url = base_url
    if api_version:
        opts.api_version = api_version

    if not opts.headers:
        opts.headers = {}

    if 'x-goog-api-client' not in opts.headers:
        opts.headers['x-goog-api-client'] = GENKIT_CLIENT_HEADER
    else:
        opts.headers['x-goog-api-client'] += f' {GENKIT_CLIENT_HEADER}'

    if 'user-agent' not in opts.headers:
        opts.headers['user-agent'] = GENKIT_CLIENT_HEADER
    else:
        opts.headers['user-agent'] += f' {GENKIT_CLIENT_HEADER}'

    return opts
