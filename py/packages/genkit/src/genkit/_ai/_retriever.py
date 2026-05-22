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

"""Retriever and indexer types and utilities for Genkit."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit._core._action import Action, ActionKind, get_func_description
from genkit._core._base import GenkitModel
from genkit._core._model import Document
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema


class RetrieverRequest(GenkitModel):
    """Request payload for a retriever action."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, extra='forbid', populate_by_name=True
    )
    query: Document
    options: Any | None = None


class RetrieverResponse(GenkitModel):
    """Response payload from a retriever action."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, extra='forbid', populate_by_name=True
    )
    documents: list[Document]


class IndexerRequest(GenkitModel):
    """Request payload for an indexer action."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, extra='forbid', populate_by_name=True
    )
    documents: list[Document]
    options: Any | None = None


class IndexerResponse(GenkitModel):
    """Response payload from an indexer action (empty — indexing is side-effect only)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, extra='forbid', populate_by_name=True
    )


class RetrieverOptions(BaseModel):
    """Configuration options for a retriever."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True, alias_generator=to_camel)

    config_schema: dict[str, Any] | None = None
    label: str | None = None


class IndexerOptions(BaseModel):
    """Configuration options for an indexer."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True, alias_generator=to_camel)

    config_schema: dict[str, Any] | None = None
    label: str | None = None


class RetrieverRef(BaseModel):
    """Reference to a retriever with optional configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True)

    name: str
    config: Any | None = None


class IndexerRef(BaseModel):
    """Reference to an indexer with optional configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True)

    name: str
    config: Any | None = None


RetrieverFn = Callable[[RetrieverRequest], Awaitable[RetrieverResponse]]
IndexerFn = Callable[[IndexerRequest], Awaitable[IndexerResponse]]


def define_retriever(
    registry: Registry,
    name: str,
    fn: RetrieverFn,
    options: RetrieverOptions | None = None,
    metadata: dict[str, object] | None = None,
    description: str | None = None,
) -> Action:
    """Register a retriever action."""
    retriever_meta: dict[str, object] = dict(metadata) if metadata else {}
    retriever_info: dict[str, object]
    existing = retriever_meta.get('retriever')
    if isinstance(existing, dict):
        retriever_info = {str(k): v for k, v in existing.items()}
    else:
        retriever_info = {}
    retriever_meta['retriever'] = retriever_info

    if options:
        if options.label:
            retriever_info['label'] = options.label
        if options.config_schema:
            retriever_info['customOptions'] = to_json_schema(options.config_schema)

    retriever_description = get_func_description(fn, description)
    return registry.register_action(
        name=name,
        kind=ActionKind.RETRIEVER,
        fn=fn,
        metadata=retriever_meta,
        description=retriever_description,
    )


def define_indexer(
    registry: Registry,
    name: str,
    fn: IndexerFn,
    options: IndexerOptions | None = None,
    metadata: dict[str, object] | None = None,
    description: str | None = None,
) -> Action:
    """Register an indexer action."""
    indexer_meta: dict[str, object] = dict(metadata) if metadata else {}
    indexer_info: dict[str, object]
    existing = indexer_meta.get('indexer')
    if isinstance(existing, dict):
        indexer_info = {str(k): v for k, v in existing.items()}
    else:
        indexer_info = {}
    indexer_meta['indexer'] = indexer_info

    if options:
        if options.label:
            indexer_info['label'] = options.label
        if options.config_schema:
            indexer_info['customOptions'] = to_json_schema(options.config_schema)

    indexer_description = get_func_description(fn, description)
    return registry.register_action(
        name=name,
        kind=ActionKind.INDEXER,
        fn=fn,
        metadata=indexer_meta,
        description=indexer_description,
    )
