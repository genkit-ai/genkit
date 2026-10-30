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

from genkit._core._action import Action, ActionKind, get_func_description, with_request_annotation
from genkit._core._model import EmbedRequest
from genkit._core._registry import Registry
from genkit._core._schema import fixed_json_schema, to_json_schema
from genkit._core._typing import EmbedResponse


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


class EmbedderRef(BaseModel):
    """Reference to an embedder with configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True)

    name: str
    config: dict[str, Any] | None = None
    version: str | None = None


EmbedderFn = Callable[[EmbedRequest[Any]], Awaitable[EmbedResponse]]


class Embedder(Action[EmbedRequest, EmbedResponse, Never]):
    """An embedder action that converts documents into vector embeddings.

    Plugin ``init`` / ``resolve`` return this without registering it.
    ``define_embedder`` constructs and registers it on the registry.

    The options class comes from ``config_schema`` or the fn's
    ``EmbedRequest[Cfg]`` annotation; given both, they must be the same class.
    An unannotated fn still gets an ``EmbedRequest``.
    """

    def __init__(
        self,
        name: str,
        fn: EmbedderFn,
        *,
        config_schema: type[BaseModel] | dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        info: EmbedderInfo | None = None,
        description: str | None = None,
    ) -> None:
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

        if 'dimensions' not in embedder_info:
            embedder_info['dimensions'] = None
        if 'customOptions' not in embedder_info:
            embedder_info['customOptions'] = None

        embedder_meta: dict[str, object] = metadata.copy() if metadata else {}
        embedder_meta['embedder'] = embedder_info

        super().__init__(
            kind=ActionKind.EMBEDDER,
            name=name,
            fn=with_request_annotation(fn, EmbedRequest, kind=ActionKind.EMBEDDER),
            metadata=embedder_meta,
            description=get_func_description(fn, description),
            config_schema=config_schema,
            # Every embedder publishes EmbedRequest/EmbedResponse.
            _io_schemas=(fixed_json_schema(EmbedRequest), fixed_json_schema(EmbedResponse)),
        )
        # EmbedderInfo.config_schema is an explicit Dev UI override; else the class.
        if embedder_info.get('customOptions') is None and self.config_schema is not None:
            embedder_info['customOptions'] = to_json_schema(self.config_schema)


def define_embedder(
    registry: Registry,
    name: str,
    fn: EmbedderFn,
    info: EmbedderInfo | None = None,
    metadata: dict[str, object] | None = None,
    description: str | None = None,
    *,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
) -> Embedder:
    """Register a custom embedder action."""
    action = Embedder(
        name,
        fn,
        metadata=metadata,
        info=info,
        description=description,
        config_schema=config_schema,
    )
    registry.register_action_from_instance(action)
    return action
