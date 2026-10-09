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

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, cast

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing_extensions import Never, TypeVar

from genkit._core._action import Action, ActionKind, get_func_description, with_request_annotation
from genkit._core._model import Document, EmbedRequest, check_ref_config, check_ref_info
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._typing import ActionMetadata, EmbedResponse


class EmbedderSupports(BaseModel):
    """Embedder capability support."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True)

    input: list[str] | None = None
    multilingual: bool | None = None


class EmbedderInfo(BaseModel):
    """Catalog card for an embedder: label, vector width, and input kinds."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True, alias_generator=to_camel)

    config_schema: dict[str, Any] | None = None
    label: str | None = None
    supports: EmbedderSupports | None = None
    dimensions: int | None = None


# Covariant so EmbedderRef[CrmEmbedConfig] is assignable to EmbedderRef[BaseModel].
EmbedderRefConfigT = TypeVar('EmbedderRefConfigT', bound=BaseModel, covariant=True, default=BaseModel)


@dataclass(frozen=True, kw_only=True)
class EmbedderRef(Generic[EmbedderRefConfigT]):
    """Handle for an embedder, shaped like ModelRef.

    ``config_schema`` is optional: embedders have no generic base config the
    way models have ModelConfig. Without it, ``config`` is a plain mapping
    and the call is not checked against a class. With it, ``config`` must be
    an instance, and ``ai.embed`` requires the embedder to declare the same
    class. ``version`` is the lowest caller layer, below ``config`` and the
    call's config, the same as ModelRef.version.

    Fields cannot be rebound. config and info are copied at construction.
    """

    name: str
    config_schema: type[EmbedderRefConfigT] | None = None
    info: EmbedderInfo | None = None
    version: str | None = None
    config: EmbedderRefConfigT | Mapping[str, Any] | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        config = check_ref_config(name=self.name, schema=self.config_schema, config=self.config, schema_required=False)
        object.__setattr__(self, 'config', config)
        object.__setattr__(self, 'info', check_ref_info(name=self.name, info=self.info, info_type=EmbedderInfo))


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
        return (
            await self._action.run(EmbedRequest(input=documents, options=options))  # type: ignore[arg-type]
        ).response


EmbedderFn = Callable[[EmbedRequest[Any]], Awaitable[EmbedResponse]]


def embedder_action_metadata(
    name: str,
    info: EmbedderInfo | None = None,
) -> ActionMetadata:
    """Create ActionMetadata for an embedder action."""
    info = info if info is not None else EmbedderInfo()
    embedder_metadata_dict: dict[str, object] = {'embedder': {}}
    embedder_info = cast(dict[str, object], embedder_metadata_dict['embedder'])

    if info.label:
        embedder_info['label'] = info.label

    embedder_info['dimensions'] = info.dimensions

    if info.supports:
        embedder_info['supports'] = info.supports.model_dump(exclude_none=True, by_alias=True)

    embedder_info['customOptions'] = info.config_schema if info.config_schema else None

    return ActionMetadata(
        action_type=ActionKind.EMBEDDER,
        name=name,
        input_json_schema=to_json_schema(EmbedRequest),
        output_json_schema=to_json_schema(EmbedResponse),
        metadata=embedder_metadata_dict,
    )


def create_embedder_ref(
    name: str,
    *,
    config: EmbedderRefConfigT | Mapping[str, Any] | None = None,
    version: str | None = None,
    config_schema: type[EmbedderRefConfigT] | None = None,
    info: EmbedderInfo | None = None,
) -> EmbedderRef[EmbedderRefConfigT]:
    """Create an EmbedderRef. Settings and version are named.

    A version string in the second position used to be stored as config and
    silently dropped. Pass config= and version=.
    """
    return EmbedderRef[EmbedderRefConfigT](
        name=name, config_schema=config_schema, info=info, version=version, config=config
    )


def embedder(
    name: str,
    fn: EmbedderFn,
    *,
    metadata: dict[str, object] | None = None,
    info: EmbedderInfo | None = None,
    description: str | None = None,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
) -> Action:
    """Build an embedder action without registering it.

    Plugin ``init`` / ``resolve`` return this. ``define_embedder`` registers it.

    The options class comes from ``config_schema`` or the fn's
    ``EmbedRequest[Cfg]`` annotation; given both, they must be the same class.
    An unannotated fn still gets an ``EmbedRequest``.
    """
    embedder_info: dict[str, object] = {}

    if metadata and 'embedder' in metadata:
        existing = metadata['embedder']
        if isinstance(existing, dict):
            existing_dict = cast(dict[str, object], existing)
            for key, value in existing_dict.items():
                if isinstance(key, str) and key not in embedder_info:
                    embedder_info[key] = value

    if info:
        if info.label:
            embedder_info['label'] = info.label
        embedder_info['dimensions'] = info.dimensions
        if info.supports:
            embedder_info['supports'] = info.supports.model_dump(exclude_none=True, by_alias=True)
        embedder_info['customOptions'] = info.config_schema if info.config_schema else None

    if 'label' not in embedder_info or not embedder_info['label']:
        embedder_info['label'] = name

    if embedder_info.get('customOptions') is None and isinstance(config_schema, dict):
        embedder_info['customOptions'] = config_schema

    embedder_meta: dict[str, object] = metadata.copy() if metadata else {}
    embedder_meta['embedder'] = embedder_info

    action = Action(
        kind=ActionKind.EMBEDDER,
        name=name,
        fn=with_request_annotation(fn, EmbedRequest, kind=ActionKind.EMBEDDER),
        metadata=embedder_meta,
        description=get_func_description(fn, description),
        config_schema=config_schema,
    )
    # EmbedderInfo.config_schema is an explicit Dev UI override; else the class.
    if embedder_info.get('customOptions') is None and action.config_schema is not None:
        embedder_info['customOptions'] = to_json_schema(action.config_schema)
    return action


def define_embedder(
    registry: Registry,
    name: str,
    fn: EmbedderFn,
    info: EmbedderInfo | None = None,
    metadata: dict[str, object] | None = None,
    description: str | None = None,
    *,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
) -> Action:
    """Register a custom embedder action."""
    action = embedder(
        name,
        fn,
        metadata=metadata,
        info=info,
        description=description,
        config_schema=config_schema,
    )
    registry.register_action_from_instance(action)
    return action
