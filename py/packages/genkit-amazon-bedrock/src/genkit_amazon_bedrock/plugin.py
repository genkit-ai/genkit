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
Mistral, Cohere, and others), embedders (Titan, Cohere, Nova), image
generators, and the Cohere reranker as Genkit actions. Text generation uses
the Bedrock Converse and ConverseStream APIs; embeddings, image generation,
and reranking use InvokeModel.

Ported from the Go plugin (genkit-ai/aws-bedrock-go-plugin).
"""

from typing import TYPE_CHECKING

from genkit import ModelRequest, ModelResponse
from genkit.model import model_action_metadata
from genkit.plugin_api import (
    Action,
    ActionKind,
    ActionMetadata,
    ActionRunContext,
    Plugin,
    to_json_schema,
)
from genkit_amazon_bedrock.config import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_MAX_POOL_CONNECTIONS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_READ_TIMEOUT,
    BedrockConfig,
    ModelDefinition,
)
from genkit_amazon_bedrock.model_info import get_model_info
from genkit_amazon_bedrock.models import BedrockModel
from genkit_amazon_bedrock.transport import BedrockTransport

if TYPE_CHECKING:
    import boto3.session

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
        max_retries: int = DEFAULT_MAX_RETRIES,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        max_pool_connections: int = DEFAULT_MAX_POOL_CONNECTIONS,
        session: 'boto3.session.Session | None' = None,
        models: list[ModelDefinition] | None = None,
        embedders: list[str] | None = None,
    ) -> None:
        """Initializes the Bedrock plugin.

        Args:
            region: AWS region. Defaults to the SDK resolution chain
                (``AWS_REGION``, ``AWS_DEFAULT_REGION``, ``~/.aws/config``);
                initialization fails loudly when no region resolves rather
                than silently picking one.
            max_retries: Retry limit for Bedrock API calls.
            read_timeout: Socket read timeout in seconds (not a whole-call
                deadline; long generations must not be killed mid-flight).
            connect_timeout: Socket connect timeout in seconds.
            max_pool_connections: HTTP connection pool size.
            session: Optional pre-configured ``boto3.session.Session`` for custom
                credentials or advanced SDK wiring.
            models: Bedrock models to register. Models not listed can still be
                resolved dynamically by namespaced name.
            embedders: Embedding model IDs to register (Titan, Cohere, Nova).
        """
        self.region = region
        self.max_retries = max_retries
        self.read_timeout = read_timeout
        self.connect_timeout = connect_timeout
        self.max_pool_connections = max_pool_connections
        self._session = session
        self.models = models or []
        self.embedders = embedders or []
        self._transport = BedrockTransport(
            region=region,
            max_retries=max_retries,
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
            max_pool_connections=max_pool_connections,
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

        Any model ID resolves — the Bedrock catalogue includes arbitrary
        inference profiles and ARNs and can never be fully enumerated.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced action name.

        Returns:
            Action object if resolvable, None otherwise.
        """
        if action_type != ActionKind.MODEL:
            return None
        model_id = name.removeprefix(f'{BEDROCK_PLUGIN_NAME}/')
        model_type = self._configured_model_type(model_id)
        if model_type not in ('chat', 'text'):
            # Image generation lands in a later slice.
            return None
        return self._create_model_action(name, model_id)

    def _configured_model_type(self, model_id: str) -> str:
        for definition in self.models:
            if definition.name == model_id:
                return definition.type
        return 'chat'

    def _create_model_action(self, name: str, model_id: str) -> Action:
        model_info = get_model_info(model_id)

        async def _generate(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            model = BedrockModel(model_id=model_id, transport=self._transport)
            return await model.generate(request, ctx)

        return Action(
            kind=ActionKind.MODEL,
            name=bedrock_name(model_id),
            fn=_generate,
            metadata={
                'model': {
                    'label': model_info.label,
                    'stage': model_info.stage.value if model_info.stage else None,
                    'supports': (
                        model_info.supports.model_dump(by_alias=True, exclude_none=True) if model_info.supports else {}
                    ),
                    'customOptions': to_json_schema(BedrockConfig),
                },
            },
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """List configured Bedrock models.

        Only explicitly configured models are listed; the catalogue itself is
        open-ended, and any model ID still resolves on demand.

        Returns:
            ActionMetadata for each configured chat model.
        """
        return [
            model_action_metadata(
                name=bedrock_name(definition.name),
                info=get_model_info(definition.name, definition.type).model_dump(by_alias=True, exclude_none=True),
                config_schema=BedrockConfig,
            )
            for definition in self.models
            if definition.type in ('chat', 'text')
        ]
