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


"""OpenAI OpenAI API Compatible Plugin for Genkit."""

import enum
from typing import Any, TypeAlias

from openai import AsyncOpenAI
from openai.types import Model

from genkit.ai import Plugin
from genkit.blocks.embedding import EmbedderOptions, EmbedderSupports, embedder_action_metadata
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind
from genkit.core.schema import to_json_schema
from genkit.core.typing import GenerationCommonConfig
from genkit.plugins.compat_oai.models import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_IMAGE_MODELS,
    SUPPORTED_OPENAI_COMPAT_MODELS,
    SUPPORTED_OPENAI_MODELS,
    SUPPORTED_STT_MODELS,
    SUPPORTED_TTS_MODELS,
    OpenAIImageModel,
    OpenAIModel,
    OpenAIModelHandler,
    OpenAISTTModel,
    OpenAITTSModel,
)
from genkit.plugins.compat_oai.models.model_info import get_default_openai_model_info
from genkit.plugins.compat_oai.typing import OpenAIConfig
from genkit.types import Embedding, EmbedRequest, EmbedResponse, ModelInfo, Supports


def open_ai_name(name: str) -> str:
    """Create an OpenAI action name.

    Args:
        name: Base name for the action.

    Returns:
        The fully qualified OpenAI action name.
    """
    return f'openai/{name}'


class _ModelType(enum.Enum):
    """Classification of OpenAI model types based on name patterns."""

    EMBEDDER = 'embedder'
    IMAGE = 'image'
    TTS = 'tts'
    STT = 'stt'
    CHAT = 'chat'


def _classify_model(name: str) -> _ModelType:
    """Classify a model name into its type based on name patterns.

    Centralizes the name-matching logic used by both resolve() and
    list_actions() to avoid inconsistencies.

    Args:
        name: The model name (with or without 'openai/' prefix).

    Returns:
        The classified model type.
    """
    if 'embed' in name:
        return _ModelType.EMBEDDER
    if 'gpt-image' in name or 'dall-e' in name:
        return _ModelType.IMAGE
    if 'tts' in name:
        return _ModelType.TTS
    if 'whisper' in name or 'transcribe' in name:
        return _ModelType.STT
    return _ModelType.CHAT


# Default Supports for each multimodal model type, used as fallback when
# a model is not found in the registry.
_DEFAULT_SUPPORTS: dict[_ModelType, Supports] = {
    _ModelType.IMAGE: Supports(
        media=False,
        output=['media'],
        multiturn=False,
        system_role=False,
        tools=False,
    ),
    _ModelType.TTS: Supports(
        media=False,
        output=['media'],
        multiturn=False,
        system_role=False,
        tools=False,
    ),
    _ModelType.STT: Supports(
        media=True,
        output=['text', 'json'],
        multiturn=False,
        system_role=False,
        tools=False,
    ),
}

# Type alias for multimodal model classes.
_MultimodalModel: TypeAlias = OpenAIImageModel | OpenAITTSModel | OpenAISTTModel
_MultimodalModelConfig: TypeAlias = tuple[type[_MultimodalModel], dict[str, ModelInfo]]

# Maps multimodal model types to their class and registry.
_MULTIMODAL_CONFIG: dict[_ModelType, _MultimodalModelConfig] = {
    _ModelType.IMAGE: (OpenAIImageModel, SUPPORTED_IMAGE_MODELS),
    _ModelType.TTS: (OpenAITTSModel, SUPPORTED_TTS_MODELS),
    _ModelType.STT: (OpenAISTTModel, SUPPORTED_STT_MODELS),
}


def _get_multimodal_info_dict(
    name: str,
    model_type: _ModelType,
    supported_models: dict[str, ModelInfo],
) -> dict[str, object]:
    """Build the info dictionary for a multimodal model.

    Uses registry metadata when available, falls back to default supports.

    Args:
        name: The raw model name (without the 'openai/' prefix).
        model_type: The classified model type for default supports fallback.
        supported_models: Registry of known models and their metadata.

    Returns:
        A dictionary suitable for Action or ActionMetadata info field.
    """
    model_info = supported_models.get(name)
    if model_info:
        return model_info.model_dump(by_alias=True, exclude_none=True)

    default_supports = _DEFAULT_SUPPORTS.get(model_type)
    return {
        'label': f'OpenAI - {name}',
        'supports': default_supports.model_dump(by_alias=True, exclude_none=True) if default_supports else {},
    }


def _multimodal_action_metadata(
    name: str,
    supported_models: dict[str, ModelInfo],
    model_type: _ModelType,
) -> ActionMetadata:
    """Build ActionMetadata for a multimodal model.

    Args:
        name: The raw model name (without the 'openai/' prefix).
        supported_models: Registry of known models and their metadata.
        model_type: The classified model type for default supports fallback.

    Returns:
        ActionMetadata for the model.
    """
    return model_action_metadata(
        name=open_ai_name(name),
        config_schema=GenerationCommonConfig,
        info=_get_multimodal_info_dict(name, model_type, supported_models),
    )


def default_openai_metadata(name: str) -> dict[str, Any]:
    return {
        'model': {'label': f'OpenAI - {name}', 'supports': {'multiturn': True}},
    }


class OpenAI(Plugin):
    """A plugin for integrating OpenAI compatible models with the Genkit framework.

    This class registers OpenAI model handlers within a registry, allowing
    interaction with supported OpenAI models.
    """

    name = 'openai'

    def __init__(self, **openai_params: Any) -> None:  # noqa: ANN401
        """Initializes the OpenAI plugin with the specified parameters.

        Args:
            openai_params: Additional parameters that will be passed to the OpenAI client constructor.
                           These parameters may include API keys, timeouts, organization IDs, and
                           other configuration settings required by OpenAI's API.
        """
        self._openai_params = openai_params
        self._async_client = AsyncOpenAI(**openai_params)

    async def init(self) -> list[Action]:
        """Initialize plugin.

        Returns:
            Actions for built-in OpenAI models, embedders, image, TTS, and STT.
        """
        actions = []

        # Add known chat models.
        for name in SUPPORTED_OPENAI_MODELS:
            actions.append(self._create_model_action(open_ai_name(name)))

        # Add known embedders.
        for name in SUPPORTED_EMBEDDING_MODELS:
            actions.append(self._create_embedder_action(open_ai_name(name)))

        # Add multimodal models (Image, TTS, STT).
        for model_type, (model_class, supported_models) in _MULTIMODAL_CONFIG.items():
            for name in supported_models:
                actions.append(
                    self._create_multimodal_action(
                        open_ai_name(name),
                        model_class,
                        supported_models,
                        model_type,
                    )
                )

        return actions

    def get_model_info(self, name: str) -> dict[str, Any] | None:
        """Retrieves metadata and supported features for the specified model.

        This method looks up the model's information from a predefined list
        of supported OpenAI-compatible models or provides default information.

        Returns:
            A dictionary containing the model's 'name' and 'supports' features,
            or None if no information can be found (though typically, a default
            is provided). The 'supports' key contains a dictionary representing
            the model's capabilities (e.g., tools, streaming).
        """
        if model_supported := SUPPORTED_OPENAI_MODELS.get(name):
            supports = (
                model_supported.supports.model_dump(by_alias=True, exclude_none=True)
                if model_supported.supports
                else {}
            )
            return {
                'label': model_supported.label,
                'supports': supports,
            }

        model_info = SUPPORTED_OPENAI_COMPAT_MODELS.get(name, get_default_openai_model_info(name))
        supports = model_info.supports.model_dump(by_alias=True, exclude_none=True) if model_info.supports else {}
        return {
            'label': model_info.label,
            'supports': supports,
        }

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        Uses name-based pattern matching (mirroring JS implementation) to
        route to the correct model type: image, TTS, STT, embedder, or chat.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced name of the action to resolve.

        Returns:
            Action object if found, None otherwise.
        """
        if action_type == ActionKind.EMBEDDER:
            if _classify_model(name) != _ModelType.EMBEDDER:
                return None
            return self._create_embedder_action(name)

        if action_type == ActionKind.MODEL:
            model_type = _classify_model(name)
            if model_type == _ModelType.EMBEDDER:
                return None  # Embedders should not be resolved as models.
            if model_type in _MULTIMODAL_CONFIG:
                model_class, supported_models = _MULTIMODAL_CONFIG[model_type]
                return self._create_multimodal_action(name, model_class, supported_models, model_type)
            return self._create_model_action(name)

        return None

    def _create_model_action(self, name: str) -> Action:
        """Create an Action object for an OpenAI model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace('openai/', '') if name.startswith('openai/') else name

        # Create the model handler
        openai_model = OpenAIModelHandler(OpenAIModel(clean_name, self._async_client))
        model_info = self.get_model_info(clean_name) or {}

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=openai_model.generate,
            metadata={
                'model': {
                    **model_info,
                    'customOptions': to_json_schema(OpenAIConfig),
                },
            },
        )

    def _create_multimodal_action(
        self,
        name: str,
        model_class: type[_MultimodalModel],
        supported_models: dict[str, ModelInfo],
        model_type: _ModelType,
    ) -> Action:
        """Create an Action for a multimodal model (image, TTS, or STT).

        Args:
            name: The namespaced name of the model.
            model_class: The model class to instantiate.
            supported_models: Registry of known models and their metadata.
            model_type: The classified model type for default metadata fallback.

        Returns:
            Action object for the model.
        """
        clean_name = name.replace('openai/', '') if name.startswith('openai/') else name
        model_instance = model_class(clean_name, self._async_client)
        info_dict = _get_multimodal_info_dict(clean_name, model_type, supported_models)

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=model_instance.generate,
            metadata={'model': info_dict},
        )

    def _create_embedder_action(self, name: str) -> Action:
        """Create an Action object for an OpenAI embedder.

        Args:
            name: The namespaced name of the embedder.

        Returns:
            Action object for the embedder.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace('openai/', '') if name.startswith('openai/') else name

        # Get embedder info from known models or use default
        embedder_info = SUPPORTED_EMBEDDING_MODELS.get(
            clean_name,
            {
                'label': f'OpenAI Embedding - {clean_name}',
                'dimensions': 1536,
                'supports': {'input': ['text']},
            },
        )

        async def embed_fn(request: EmbedRequest) -> EmbedResponse:
            """Embedder function that calls OpenAI embeddings API."""
            # Extract text from document content
            texts = []
            for doc in request.input:
                doc_text = ''.join(  # type: ignore[arg-type]
                    part.root.text for part in doc.content if hasattr(part.root, 'text') and part.root.text
                )
                texts.append(doc_text)

            # Get optional parameters with proper types
            dimensions = None
            encoding_format = None
            if request.options:
                if dim_val := request.options.get('dimensions'):
                    dimensions = int(dim_val)
                if enc_val := request.options.get('encodingFormat'):
                    encoding_format = str(enc_val) if enc_val in ('float', 'base64') else None

            # Create embeddings for each document
            response = await self._async_client.embeddings.create(
                model=clean_name,
                input=texts,
                dimensions=dimensions,  # type: ignore[arg-type]
                encoding_format=encoding_format,  # type: ignore[arg-type]
            )

            # Convert OpenAI response to Genkit format
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
        """Generate a list of available actions or models.

        Uses pattern matching on model names (mirroring the JS implementation)
        to categorize models as embedders, image generators, TTS, STT, or chat.

        Returns:
            list[ActionMetadata]: A list of ActionMetadata objects.
        """
        actions: list[ActionMetadata] = []
        models_ = await self._async_client.models.list()
        models: list[Model] = models_.data
        for model in models:
            name = model.id
            model_type = _classify_model(name)
            if model_type == _ModelType.EMBEDDER:
                actions.append(
                    embedder_action_metadata(
                        name=open_ai_name(name),
                        options=EmbedderOptions(
                            label=f'OpenAI Embedding - {name}',
                            supports=EmbedderSupports(input=['text']),
                        ),
                    )
                )
            elif model_type in _DEFAULT_SUPPORTS:
                config = _MULTIMODAL_CONFIG[model_type]
                actions.append(_multimodal_action_metadata(name, config[1], model_type))
            else:
                actions.append(
                    model_action_metadata(
                        name=open_ai_name(name),
                        config_schema=GenerationCommonConfig,
                        info={
                            'label': f'OpenAI - {name}',
                            'supports': Supports(
                                multiturn=True,
                                system_role=True,
                                tools=False,
                            ).model_dump(by_alias=True, exclude_none=True),
                        },
                    )
                )
        return actions


def openai_model(name: str) -> str:
    """Returns a string representing the OpenAI model name to use with Genkit.

    Args:
        name: The name of the OpenAI model to use.

    Returns:
        A string representing the OpenAI model name to use with Genkit.
    """
    return f'openai/{name}'


__all__ = ['OpenAI', 'openai_model']
