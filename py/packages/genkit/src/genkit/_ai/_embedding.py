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

from genkit._core._action import Action, ActionKind, ActionMetadata, get_func_description
from genkit._core._model import Document
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._typing import EmbedRequest, EmbedResponse


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
        kind=ActionKind.EMBEDDER,
        name=name,
        input_json_schema=to_json_schema(EmbedRequest),
        output_json_schema=to_json_schema(EmbedResponse),
        metadata=embedder_metadata_dict,
    )


def create_embedder_ref(name: str, config: dict[str, Any] | None = None, version: str | None = None) -> EmbedderRef:
    """Creates an EmbedderRef instance."""
    return EmbedderRef(name=name, config=config, version=version)


def define_embedder(
    registry: Registry,
    name: str,
    fn: EmbedderFn,
    options: EmbedderOptions | None = None,
    metadata: dict[str, object] | None = None,
    description: str | None = None,
) -> Action:
    """Register a custom embedder action."""
    embedder_meta: dict[str, object] = dict(metadata) if metadata else {}
    embedder_info: dict[str, object]
    existing_embedder = embedder_meta.get('embedder')
    if isinstance(existing_embedder, dict):
        embedder_info = {str(key): value for key, value in existing_embedder.items()}
    else:
        embedder_info = {}
    embedder_meta['embedder'] = embedder_info

    if options:
        if options.label:
            embedder_info['label'] = options.label
        if options.dimensions:
            embedder_info['dimensions'] = options.dimensions
        if options.supports:
            embedder_info['supports'] = options.supports.model_dump(exclude_none=True, by_alias=True)
        if options.config_schema:
            embedder_info['customOptions'] = to_json_schema(options.config_schema)

    embedder_description = get_func_description(fn, description)
    return registry.register_action(
        name=name,
        kind=ActionKind.EMBEDDER,
        fn=fn,
        metadata=embedder_meta,
        description=embedder_description,
    )
