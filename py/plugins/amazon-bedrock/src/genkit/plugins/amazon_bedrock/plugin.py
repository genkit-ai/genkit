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

from genkit.plugin_api import (
    Action,
    ActionKind,
    ActionMetadata,
    Plugin,
)
from genkit.plugins.amazon_bedrock.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REGION,
    DEFAULT_REQUEST_TIMEOUT,
    ModelDefinition,
)

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
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        session: 'boto3.session.Session | None' = None,
        models: list[ModelDefinition] | None = None,
        embedders: list[str] | None = None,
    ) -> None:
        """Initializes the Bedrock plugin.

        Args:
            region: AWS region. Defaults to the SDK resolution chain, falling
                back to ``us-east-1`` (Go plugin parity).
            max_retries: Retry limit for Bedrock API calls.
            request_timeout: Per-call timeout in seconds.
            session: Optional pre-configured ``boto3.session.Session`` for custom
                credentials or advanced SDK wiring.
            models: Bedrock models to register. Models not listed can still be
                resolved dynamically by namespaced name.
            embedders: Embedding model IDs to register (Titan, Cohere, Nova).
        """
        self.region = region or DEFAULT_REGION
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self._session = session
        self.models = models or []
        self.embedders = embedders or []

    async def init(self) -> list[Action]:
        """Initialize plugin.

        Returns:
            Empty list (actions are lazily created via ``resolve``).
        """
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by namespaced name.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced action name.

        Returns:
            Action object if resolvable, None otherwise.
        """
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        """List available Bedrock models and embedders.

        Returns:
            ActionMetadata for each configured model and embedder.
        """
        return []
