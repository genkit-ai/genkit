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

"""Valkey vector store plugin for Genkit.

Provides indexing (HSET with HNSW vector fields) and retrieval (FT.SEARCH KNN)
backed by Valkey with the valkey-search module.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from glide import (
    DataType,
    DistanceMetricType,
    FtCreateOptions,
    FtSearchOptions,
    GlideClient,
    GlideClientConfiguration,
    NodeAddress,
    NumericField,
    ReturnField,
    TagField,
    TextField,
    VectorAlgorithm,
    VectorField,
    VectorFieldAttributesHnsw,
    VectorType,
)
from glide import ft as glide_ft

from genkit._ai._retriever import (
    IndexerRef,
    IndexerRequest,
    IndexerResponse,
    RetrieverRef,
    RetrieverRequest,
    RetrieverResponse,
)
from genkit._core._action import Action, ActionKind
from genkit._core._model import Document
from genkit._core._plugin import Plugin
from genkit._core._typing import ActionMetadata, EmbedRequest


VALKEY_PLUGIN_NAME = 'valkey'


class MetadataFieldType(Enum):
    """Type of a metadata field for FT.CREATE schema."""

    TAG = 'TAG'
    NUMERIC = 'NUMERIC'


@dataclass
class MetadataField:
    """Declaration of a metadata field to index for filtering."""

    name: str
    field_type: MetadataFieldType = MetadataFieldType.TAG


@dataclass
class ValkeyRetrieverOptions:
    """Options passed at retrieval time."""

    k: int = 10
    # Raw FT.SEARCH pre-filter expression interpolated directly into the query.
    # Callers must ensure it does not contain untrusted user input.
    filter: str | None = None


@dataclass
class ValkeyConfig:
    """Configuration for a single Valkey index."""

    index_name: str
    embedder: str
    dimension: int
    host: str = 'localhost'
    port: int = 6379
    prefix: str | None = None
    embedder_options: dict[str, Any] | None = None
    distance_metric: DistanceMetricType = DistanceMetricType.COSINE
    metadata_fields: list[MetadataField] = field(default_factory=list)


def valkey_retriever_ref(index_name: str) -> RetrieverRef:
    """Return a typed retriever reference for a Valkey index."""
    return RetrieverRef(name=f'{VALKEY_PLUGIN_NAME}/{index_name}')


def valkey_indexer_ref(index_name: str) -> IndexerRef:
    """Return a typed indexer reference for a Valkey index."""
    return IndexerRef(name=f'{VALKEY_PLUGIN_NAME}/{index_name}')


def _float32_to_bytes(vec: list[float]) -> bytes:
    """Convert a list of floats to little-endian bytes (matches Go/JS encoding)."""
    return struct.pack(f'<{len(vec)}f', *vec)


def _doc_id(doc: Document) -> str:
    """Compute a deterministic document ID (MD5 of JSON serialization)."""
    serialized = json.dumps(doc.model_dump(by_alias=True), sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()


class Valkey(Plugin):
    """Valkey vector store plugin for Genkit.

    Registers an indexer and retriever for each configured index.
    Requires a Valkey instance with the valkey-search module loaded.
    """

    name = VALKEY_PLUGIN_NAME

    def __init__(self, configs: list[ValkeyConfig]) -> None:
        """Initialize with one or more index configurations."""
        self.configs = configs
        self._clients: dict[str, GlideClient] = {}

    async def init(self) -> list[Action]:
        """Connect to Valkey and ensure indexes exist."""
        actions: list[Action] = []

        for cfg in self.configs:
            prefix = cfg.prefix or cfg.index_name
            client = await GlideClient.create(
                GlideClientConfiguration(
                    addresses=[NodeAddress(cfg.host, cfg.port)]
                )
            )
            self._clients[cfg.index_name] = client

            await _ensure_index(
                client, cfg.index_name, cfg.dimension, prefix,
                cfg.distance_metric, cfg.metadata_fields,
            )

            indexer_action = Action(
                kind=ActionKind.INDEXER,
                name=f'{VALKEY_PLUGIN_NAME}/{cfg.index_name}',
                fn=self._make_index_fn(client, cfg, prefix),
                metadata={'indexer': {'label': f'Valkey - {cfg.index_name}'}},
            )
            actions.append(indexer_action)

            retriever_action = Action(
                kind=ActionKind.RETRIEVER,
                name=f'{VALKEY_PLUGIN_NAME}/{cfg.index_name}',
                fn=self._make_retrieve_fn(client, cfg, prefix),
                metadata={'retriever': {'label': f'Valkey - {cfg.index_name}'}},
            )
            actions.append(retriever_action)

        return actions

    async def close(self) -> None:
        """Close all Valkey client connections."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve is not needed — all actions are pre-registered in init."""
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        """List available indexer and retriever actions."""
        actions: list[ActionMetadata] = []
        for cfg in self.configs:
            actions.append(
                ActionMetadata(
                    action_type=ActionKind.INDEXER,
                    name=f'{VALKEY_PLUGIN_NAME}/{cfg.index_name}',
                )
            )
            actions.append(
                ActionMetadata(
                    action_type=ActionKind.RETRIEVER,
                    name=f'{VALKEY_PLUGIN_NAME}/{cfg.index_name}',
                )
            )
        return actions

    def _make_index_fn(self, client: GlideClient, cfg: ValkeyConfig, prefix: str):
        """Create the indexer function for a given config."""
        plugin = self

        async def index_fn(req: IndexerRequest) -> IndexerResponse:
            registry = plugin._registry
            if registry is None:
                raise RuntimeError(
                    'Valkey plugin not registered with a Genkit registry; '
                    'pass it to Genkit(plugins=[...]) before indexing'
                )
            embedder_action = await registry.resolve_embedder(cfg.embedder)
            if embedder_action is None:
                raise ValueError(f'Embedder "{cfg.embedder}" not found')

            embed_response = (
                await embedder_action.run(
                    EmbedRequest(input=req.documents, options=cfg.embedder_options)  # type: ignore[arg-type]
                )
            ).response

            if len(embed_response.embeddings) != len(req.documents):
                raise ValueError(
                    f'valkey: embedder returned {len(embed_response.embeddings)} embeddings for {len(req.documents)} docs'
                )
            for i, doc in enumerate(req.documents):
                embedding = embed_response.embeddings[i].embedding
                if len(embedding) != cfg.dimension:
                    raise ValueError(
                        f'valkey: embedder returned {len(embedding)}-dim vector, expected {cfg.dimension}'
                    )
                vec_bytes = _float32_to_bytes(embedding)
                doc_key = _doc_id(doc)
                content = doc.data
                data_type = doc.data_type or 'text'
                metadata_json = json.dumps(doc.metadata or {})

                key = f'{prefix}:{doc_key}'
                fields: dict[str, bytes | str] = {
                    'embedding': vec_bytes,
                    '_content': content,
                    '_metadata': metadata_json,
                    '_dataType': data_type,
                }

                # Store declared metadata fields as top-level HASH fields for filtering.
                if doc.metadata and cfg.metadata_fields:
                    for mf in cfg.metadata_fields:
                        val = doc.metadata.get(mf.name)
                        if val is not None:
                            fields[mf.name] = str(val)

                await client.hset(key, fields)

            return IndexerResponse()

        return index_fn

    def _make_retrieve_fn(self, client: GlideClient, cfg: ValkeyConfig, prefix: str):
        """Create the retriever function for a given config."""
        plugin = self

        async def retrieve_fn(req: RetrieverRequest) -> RetrieverResponse:
            k = 10
            filter_expr: str | None = None
            if req.options and isinstance(req.options, dict):
                k = req.options.get('k', 10)
                filter_expr = req.options.get('filter', None)

            registry = plugin._registry
            if registry is None:
                raise RuntimeError(
                    'Valkey plugin not registered with a Genkit registry; '
                    'pass it to Genkit(plugins=[...]) before retrieving'
                )
            embedder_action = await registry.resolve_embedder(cfg.embedder)
            if embedder_action is None:
                raise ValueError(f'Embedder "{cfg.embedder}" not found')

            embed_response = (
                await embedder_action.run(
                    EmbedRequest(input=[req.query], options=cfg.embedder_options)  # type: ignore[arg-type]
                )
            ).response

            if not embed_response.embeddings:
                raise ValueError('valkey: embedder returned no embeddings')
            query_vec = embed_response.embeddings[0].embedding
            query_bytes = _float32_to_bytes(query_vec)

            # Build KNN query, optionally with a pre-filter expression.
            if filter_expr:
                query_str = f'({filter_expr})=>[KNN $k @embedding $query_vec]'
            else:
                query_str = '*=>[KNN $k @embedding $query_vec]'

            search_options = FtSearchOptions(
                params={
                    'k': str(k),
                    'query_vec': query_bytes,
                },
                return_fields=[
                    ReturnField(field_identifier='_content'),
                    ReturnField(field_identifier='_metadata'),
                    ReturnField(field_identifier='_dataType'),
                ],
            )

            result = await glide_ft.search(client, cfg.index_name, query_str, search_options)

            documents: list[Document] = []
            if isinstance(result, list) and len(result) > 1:
                for entry in result[1:]:
                    if not isinstance(entry, dict):
                        continue
                    for _doc_key, fields in entry.items():
                        if not isinstance(fields, dict):
                            continue
                        content = ''
                        metadata_str = '{}'
                        data_type = 'text'
                        for field_name, field_val in fields.items():
                            fname = bytes(field_name).decode() if isinstance(field_name, (bytes, bytearray, memoryview)) else str(field_name)
                            fval = bytes(field_val).decode() if isinstance(field_val, (bytes, bytearray, memoryview)) else str(field_val)
                            if fname == '_content':
                                content = fval
                            elif fname == '_metadata':
                                metadata_str = fval
                            elif fname == '_dataType':
                                data_type = fval or 'text'

                        if not content:
                            continue

                        metadata = None
                        if metadata_str and metadata_str != '{}':
                            try:
                                metadata = json.loads(metadata_str)
                            except (json.JSONDecodeError, TypeError):
                                metadata = None

                        documents.append(Document.from_data(content, data_type, metadata))

            return RetrieverResponse(documents=documents)

        return retrieve_fn


async def _ensure_index(
    client: GlideClient,
    index_name: str,
    dimension: int,
    prefix: str,
    distance_metric: DistanceMetricType = DistanceMetricType.COSINE,
    metadata_fields: list[MetadataField] | None = None,
) -> None:
    """Create the FT index if it does not already exist."""
    schema = [
        VectorField(
            name='embedding',
            algorithm=VectorAlgorithm.HNSW,
            attributes=VectorFieldAttributesHnsw(
                dimensions=dimension,
                distance_metric=distance_metric,
                type=VectorType.FLOAT32,
            ),
        ),
        TextField(name='_content'),
        TextField(name='_metadata'),
        TextField(name='_dataType'),
    ]

    for mf in (metadata_fields or []):
        if mf.field_type == MetadataFieldType.NUMERIC:
            schema.append(NumericField(name=mf.name))
        else:
            schema.append(TagField(name=mf.name))

    options = FtCreateOptions(data_type=DataType.HASH, prefixes=[f'{prefix}:'])

    try:
        await glide_ft.create(client, index_name, schema, options)
    except Exception as e:
        if 'already exists' in str(e).lower():
            return
        raise
