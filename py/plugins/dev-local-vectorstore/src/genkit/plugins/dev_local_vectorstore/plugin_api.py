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

"""Local file-based vectorstore helper that provides retriever and indexer for Genkit."""

from typing import Any

from genkit.ai import Genkit
from genkit.blocks.retriever import (
    IndexerOptions,
    RetrieverOptions,
    indexer_action_metadata,
    retriever_action_metadata,
)
from genkit.core.registry import ActionKind
from genkit.core.schema import to_json_schema

from .indexer import DevLocalVectorStoreIndexer
from .retriever import DevLocalVectorStoreRetriever, RetrieverOptionsSchema


def define_dev_local_vector_store(
    ai: Genkit,
    *,
    name: str,
    embedder: str,
    embedder_options: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Define and register a dev local vector store retriever and indexer.

    NOT INTENDED FOR USE IN PRODUCTION

    Args:
        ai: The Genkit instance to register the retriever and indexer with.
        name: Name of the retriever and indexer.
        embedder: The embedder to use (e.g., 'vertexai/gemini-embedding-001').
        embedder_options: Optional configuration to pass to the embedder.

    Returns:
        Tuple of (retriever_name, indexer_name).
    """
    # Create and register retriever
    retriever = DevLocalVectorStoreRetriever(
        ai=ai,
        index_name=name,
        embedder=embedder,
        embedder_options=embedder_options,
    )

    ai.registry.register_action(
        kind=ActionKind.RETRIEVER,
        name=name,
        fn=retriever.retrieve,
        metadata=retriever_action_metadata(
            name=name,
            options=RetrieverOptions(
                label=name,
                config_schema=to_json_schema(RetrieverOptionsSchema),
            ),
        ).metadata,
    )

    # Create and register indexer
    indexer = DevLocalVectorStoreIndexer(
        ai=ai,
        index_name=name,
        embedder=embedder,
        embedder_options=embedder_options,
    )

    ai.registry.register_action(
        kind=ActionKind.INDEXER,
        name=name,
        fn=indexer.index,
        metadata=indexer_action_metadata(
            name=name,
            options=IndexerOptions(label=name),
        ).metadata,
    )

    return (name, name)


define_dev_local_vector_store_deprecated = define_dev_local_vector_store
