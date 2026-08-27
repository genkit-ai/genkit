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

"""Embedding types and utilities for Genkit."""

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing_extensions import Never

from genkit._core._action import Action, ActionKind, get_func_description
from genkit._core._model import Document
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._typing import ActionMetadata, EmbedRequest, EmbedResponse


class EmbedderSupports(BaseModel):
    """Embedder capability support."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True)

    input: list[str] | None = None
    multilingual: bool | None = None


class EmbedderOptions(BaseModel):
    """Configuration options for an embedder."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True, alias_generator=to_camel)

    config_schema: dict[str, Any] | None = None
    label: str | None = None
    supports: EmbedderSupports | None = None
    dimensions: int | None = None


class EmbedderRef(BaseModel):
    """Reference to an embedder with configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True)

    name: str
    config: Any | None = None
    version: str | None = None


class Embedder:
    """Runtime embedder wrapper around an embedder Action."""

    def __init__(self, name: str, action: Action[EmbedRequest, EmbedResponse, Never]) -> None:
        """Initialize with embedder name and backing action."""
        self.name: str = name
        self._action: Action[EmbedRequest, EmbedResponse, Never] = action

    async def embed(
        self,
        documents: list[Document],
        options: dict[str, Any] | None = None,
    ) -> EmbedResponse:
        """Generate embeddings for a list of documents."""
        # Document veneer is compatible with DocumentData at runtime
        return (
            await self._action.run(EmbedRequest(input=documents, options=options))  # type: ignore[arg-type]
        ).response


EmbedderFn = Callable[[EmbedRequest], Awaitable[EmbedResponse]]


def embedder_action_metadata(
    name: str,
    options: EmbedderOptions | None = None,
) -> ActionMetadata:
    """Create ActionMetadata for an embedder action."""
    options = options if options is not None else EmbedderOptions()
    embedder_metadata_dict: dict[str, object] = {'embedder': {}}
    embedder_info = cast(dict[str, object], embedder_metadata_dict['embedder'])

    if options.label:
        embedder_info['label'] = options.label

    embedder_info['dimensions'] = options.dimensions

    if options.supports:
        embedder_info['supports'] = options.supports.model_dump(exclude_none=True, by_alias=True)

    embedder_info['customOptions'] = options.config_schema if options.config_schema else None

    return ActionMetadata(
        action_type=ActionKind.EMBEDDER,
        name=name,
        input_json_schema=to_json_schema(EmbedRequest),
        output_json_schema=to_json_schema(EmbedResponse),
        metadata=embedder_metadata_dict,
    )


def create_embedder_ref(name: str, config: dict[str, Any] | None = None, version: str | None = None) -> EmbedderRef:
    """Creates an EmbedderRef instance."""
    return EmbedderRef(name=name, config=config, version=version)


def embedder(
    name: str,
    fn: EmbedderFn,
    *,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    options: EmbedderOptions | None = None,
    description: str | None = None,
) -> Action:
    """Build an embedder action without registering it.

    Plugin ``init`` / ``resolve`` return this. ``define_embedder`` registers it.
    The config class stays on the action so a later isinstance check can see it.
    """
    embedder_info: dict[str, object] = {}

    if metadata and 'embedder' in metadata:
        existing = metadata['embedder']
        if isinstance(existing, dict):
            existing_dict = cast(dict[str, object], existing)
            for key, value in existing_dict.items():
                if isinstance(key, str) and key not in embedder_info:
                    embedder_info[key] = value

    if options:
        if options.label:
            embedder_info['label'] = options.label
        if options.dimensions:
            embedder_info['dimensions'] = options.dimensions
        if options.supports:
            embedder_info['supports'] = options.supports.model_dump(exclude_none=True, by_alias=True)
        if options.config_schema and config_schema is None:
            embedder_info['customOptions'] = to_json_schema(options.config_schema)

    if 'label' not in embedder_info or not embedder_info['label']:
        embedder_info['label'] = name

    if config_schema:
        embedder_info['customOptions'] = to_json_schema(config_schema)

    embedder_meta: dict[str, object] = metadata.copy() if metadata else {}
    embedder_meta['embedder'] = embedder_info

    return Action(
        kind=ActionKind.EMBEDDER,
        name=name,
        fn=fn,
        metadata=embedder_meta,
        description=get_func_description(fn, description),
        config_schema=config_schema,
    )


def define_embedder(
    registry: Registry,
    name: str,
    fn: EmbedderFn,
    options: EmbedderOptions | None = None,
    metadata: dict[str, object] | None = None,
    description: str | None = None,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
) -> Action:
    """Register a custom embedder action."""
    action = embedder(
        name,
        fn,
        config_schema=config_schema if config_schema is not None else (options.config_schema if options else None),
        metadata=metadata,
        options=options,
        description=description,
    )
    registry.register_action_from_instance(action)
    return action
