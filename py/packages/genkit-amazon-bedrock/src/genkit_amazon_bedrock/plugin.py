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

"""Amazon Bedrock plugin for Genkit.

Registers Bedrock-hosted models (Anthropic Claude, Amazon Nova, Meta Llama,
Mistral, Cohere, and others) as Genkit model actions. Text generation uses the
Bedrock Converse and ConverseStream APIs; embedders and image generation use
InvokeModel. Reranking also uses InvokeModel but ships as the ``Bedrock.rerank``
helper: Genkit Python has no reranker primitive to register an action against.
"""

from typing import TYPE_CHECKING, Any, Literal

import structlog

from genkit import Document, ModelRequest, ModelResponse

# DocumentData has no public re-export yet; the rerank helper is built on it.
from genkit._core._typing import DocumentData
from genkit.embedder import EmbedRequest, EmbedResponse, embedder, embedder_action_metadata
from genkit.model import model as create_model, model_action_metadata
from genkit.plugin_api import (
    Action,
    ActionKind,
    ActionMetadata,
    ActionRunContext,
    GenkitError,
    Plugin,
    to_json_schema,
)
from genkit_amazon_bedrock.config import (
    DEFAULT_TOTAL_TIMEOUT,
    BedrockConfig,
    BedrockImageConfig,
    ModelDefinition,
)
from genkit_amazon_bedrock.embedders import (
    BedrockEmbedder,
    get_embedder_info,
    is_embedding_model,
    looks_like_embedding_model,
)
from genkit_amazon_bedrock.image import BedrockImageModel, is_image_model
from genkit_amazon_bedrock.model_info import get_model_info
from genkit_amazon_bedrock.models import BedrockModel
from genkit_amazon_bedrock.rerank import (
    BedrockReranker,
    BedrockRerankOptions,
    RerankerRequest,
    RerankerResponse,
    is_rerank_model,
)
from genkit_amazon_bedrock.transport import BedrockTransport

if TYPE_CHECKING:
    import boto3.session

logger = structlog.get_logger(__name__)

BEDROCK_PLUGIN_NAME = 'bedrock'


def bedrock_name(name: str) -> str:
    """Fully qualified Genkit action name for a Bedrock model.

    Args:
        name: Bedrock model ID.

    Returns:
        The namespaced action name, e.g. ``bedrock/anthropic.claude-...``.
    """
    return f'{BEDROCK_PLUGIN_NAME}/{name}'


class Bedrock(Plugin):
    """Amazon Bedrock plugin for Genkit."""

    name = BEDROCK_PLUGIN_NAME

    def __init__(
        self,
        region: str | None = None,
        max_retries: int | None = None,
        read_timeout: float | None = None,
        connect_timeout: float | None = None,
        max_pool_connections: int | None = None,
        total_timeout: float | None = DEFAULT_TOTAL_TIMEOUT,
        session: 'boto3.session.Session | None' = None,
        models: list[ModelDefinition] | None = None,
        embedders: list[str] | None = None,
    ) -> None:
        """Initializes the Bedrock plugin.

        The AWS client knobs all default to None, meaning "use the ambient AWS
        configuration" (``AWS_MAX_ATTEMPTS``, ``AWS_RETRY_MODE``,
        ``~/.aws/config``, a session's default client config), and fall back to
        the package defaults only when that configuration says nothing.

        Args:
            region: AWS region. Defaults to the SDK resolution chain
                (``AWS_REGION``, ``AWS_DEFAULT_REGION``, ``~/.aws/config``);
                initialization fails loudly when no region resolves rather
                than silently picking one.
            max_retries: Retry limit for Bedrock API calls.
            read_timeout: Socket read timeout in seconds. Resets on every byte
                received, so it caps silence, not the call.
            connect_timeout: Socket connect timeout in seconds.
            max_pool_connections: HTTP connection pool size.
            total_timeout: Whole-call deadline in seconds, covering retries and
                the slow-dribble case the read timeout cannot see. None removes
                the deadline, leaving only the socket timeouts.
            session: Optional pre-configured ``boto3.session.Session`` for custom
                credentials or advanced SDK wiring.
            models: Bedrock models to register. Models not listed can still be
                resolved dynamically by namespaced name.
            embedders: Bedrock embedding model IDs to register, e.g.
                ``amazon.titan-embed-text-v2:0``. As with models, unlisted IDs
                still resolve dynamically.
        """
        self.region = region
        self.max_retries = max_retries
        self.read_timeout = read_timeout
        self.connect_timeout = connect_timeout
        self.max_pool_connections = max_pool_connections
        self.total_timeout = total_timeout
        self._session = session
        self.models = models or []
        self.embedders = embedders or []
        self._transport = BedrockTransport(
            region=region,
            max_retries=max_retries,
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
            max_pool_connections=max_pool_connections,
            total_timeout=total_timeout,
            session=session,
        )

    async def init(self) -> list[Action]:
        """Initialize plugin.

        Builds the shared client so misconfiguration (e.g. no resolvable AWS
        region) fails at startup instead of on the first model call.

        Returns:
            Empty list (actions are lazily created via ``resolve``).
        """
        await self._transport.ensure_client()
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by namespaced name.

        Any model ID resolves. Nothing is discovered: listing the catalogue
        needs a second, control-plane ``bedrock`` client and the IAM actions
        that go with it, and this plugin opens only ``bedrock-runtime``.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced action name.

        Returns:
            Action object if resolvable, None otherwise.
        """
        prefix = f'{BEDROCK_PLUGIN_NAME}/'
        # Direct plugin.model() calls can pass any namespace; only ours resolves.
        if not name.startswith(prefix):
            return None
        model_id = name.removeprefix(prefix)
        if action_type == ActionKind.EMBEDDER:
            # An unroutable embedding ID still resolves, so embed() can name it
            # as an unsupported embedder instead of the registry saying 404.
            if not looks_like_embedding_model(model_id):
                logger.debug('Bedrock resolve declined', model=model_id, kind='embedder', reason='not_an_embedder')
                return None
            return self._create_embedder_action(model_id)
        if action_type != ActionKind.MODEL:
            logger.debug('Bedrock resolve declined', model=model_id, kind=str(action_type), reason='unsupported_kind')
            return None
        if looks_like_embedding_model(model_id):
            # Embedding models speak InvokeModel, not Converse; resolving one
            # as a chat model only defers the failure to call time.
            logger.debug('Bedrock resolve declined', model=model_id, kind='model', reason='embedding_model')
            return None
        if is_rerank_model(model_id):
            # Same story for rerank models; reranking is the Bedrock.rerank helper.
            logger.debug('Bedrock resolve declined', model=model_id, kind='model', reason='rerank_model')
            return None
        declared = self._declared_model_type(model_id)
        # Undeclared IDs are classified rather than assumed to be chat: resolve
        # is lazy, so otherwise bedrock/amazon.nova-canvas-v1:0 would take the
        # Converse path and fail at call time. Embedders classify the same way.
        model_type = declared if declared is not None else ('image' if is_image_model(model_id) else 'chat')
        logger.debug('Bedrock model resolved', model=model_id, model_type=model_type, declared=declared is not None)
        return self._create_model_action(model_id, model_type)

    def _declared_model_type(self, model_id: str) -> Literal['chat', 'text', 'image'] | None:
        for definition in self.models:
            if definition.name == model_id:
                return definition.type
        return None

    def _create_model_action(self, model_id: str, model_type: Literal['chat', 'text', 'image'] = 'chat') -> Action:
        model_info = get_model_info(model_id, model_type)
        is_image = model_type == 'image'

        async def _generate(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            if is_image:
                image_model = BedrockImageModel(model_id=model_id, transport=self._transport)
                return await image_model.generate(request, ctx)
            model = BedrockModel(model_id=model_id, transport=self._transport)
            return await model.generate(request, ctx)

        return create_model(
            bedrock_name(model_id),
            _generate,
            config_schema=BedrockImageConfig if is_image else BedrockConfig,
            metadata={
                'model': {
                    'label': model_info.label,
                    'stage': model_info.stage.value if model_info.stage else None,
                    'supports': (
                        model_info.supports.model_dump(by_alias=True, exclude_none=True) if model_info.supports else {}
                    ),
                    'customOptions': to_json_schema(BedrockImageConfig if is_image else BedrockConfig),
                },
            },
        )

    def _create_embedder_action(self, model_id: str) -> Action:
        async def _embed(request: EmbedRequest) -> EmbedResponse:
            embedder = BedrockEmbedder(model_id=model_id, transport=self._transport)
            return await embedder.embed(request)

        return embedder(
            bedrock_name(model_id),
            _embed,
            info=get_embedder_info(model_id),
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """List configured Bedrock models and embedders.

        Only explicitly configured entries are listed, and only those this
        plugin can actually serve: an ID in the wrong list, or a chat model
        declared ``type='image'``, would otherwise be advertised and then fail
        on use. Such a declaration still resolves, so the caller reads the
        image path's reason rather than a generic model-not-found. A bare
        ``Bedrock()`` therefore lists nothing; see ``resolve`` for why the
        catalogue is not read.

        Returns:
            ActionMetadata for each configured model and embedder.
        """
        actions: list[ActionMetadata] = [
            model_action_metadata(
                name=bedrock_name(definition.name),
                info=get_model_info(definition.name, definition.type).model_dump(by_alias=True, exclude_none=True),
                config_schema=BedrockImageConfig if definition.type == 'image' else BedrockConfig,
            )
            for definition in self.models
            if not looks_like_embedding_model(definition.name)
            and not is_rerank_model(definition.name)
            and (definition.type != 'image' or is_image_model(definition.name))
        ]
        models = len(actions)
        actions.extend(
            embedder_action_metadata(bedrock_name(model_id), get_embedder_info(model_id))
            for model_id in self.embedders
            if is_embedding_model(model_id)
        )
        logger.debug(
            'Bedrock actions listed',
            models=models,
            embedders=len(actions) - models,
            models_configured=len(self.models),
            embedders_configured=len(self.embedders),
        )
        return actions

    async def rerank(
        self,
        model_id: str,
        *,
        query: str | DocumentData,
        documents: list[DocumentData],
        options: BedrockRerankOptions | dict[str, Any] | None = None,
    ) -> RerankerResponse:
        """Rerank documents by relevance to a query.

        A helper rather than a registered action: Genkit Python has no
        first-class reranker primitive, so there is nothing to register
        against. Both the Cohere and Amazon rerank families are supported,
        and the request body is built from the model ID because they disagree
        over ``api_version``. The ID itself is sent to the service verbatim.

        Args:
            model_id: Bedrock rerank model ID, e.g. ``cohere.rerank-v3-5:0``
                or ``amazon.rerank-v1:0``.
            query: The query to rank against, as text or as a document.
            documents: The documents to rank.
            options: Per-call options, as ``BedrockRerankOptions`` or a mapping.

        Returns:
            The ranked documents in the order the service returned them, each
            carrying its relevance score.

        Raises:
            GenkitError: INVALID_ARGUMENT for a missing model ID or a query or
                document with no text, INTERNAL for a malformed response, and
                the mapped AWS status for a failed call.
        """
        if not model_id:
            raise GenkitError(message='bedrock rerank: model ID required', status='INVALID_ARGUMENT')
        reranker = BedrockReranker(model_id=model_id, transport=self._transport)
        return await reranker.rerank(
            RerankerRequest(
                query=Document.from_text(query) if isinstance(query, str) else query,
                documents=documents,
                options=options,
            )
        )
