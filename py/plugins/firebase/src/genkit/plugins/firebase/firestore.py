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


"""Firestore vector store operations for Genkit."""

from collections.abc import Callable

from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentSnapshot
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

from genkit.ai import Genkit
from genkit.blocks.retriever import RetrieverOptions, retriever_action_metadata
from genkit.core.action.types import ActionKind
from genkit.core.typing import DocumentPart
from genkit.plugins.firebase.retriever import FirestoreRetriever

from .constant import MetadataTransformFn


def firestore_action_name(name: str) -> str:
    """Create a firestore action name.

    Args:
        name: Base name for the action

    Returns:
        str: Firestore action name.

    """
    return f'firestore/{name}'


def define_firestore_vector_store(
    ai: Genkit,
    *,
    name: str,
    embedder: str,
    embedder_options: dict[str, object] | None = None,
    collection: str,
    vector_field: str,
    content_field: str | Callable[[DocumentSnapshot], list['DocumentPart']],
    firestore_client: firestore.Client,
    distance_measure: DistanceMeasure = DistanceMeasure.COSINE,
    metadata_fields: list[str] | MetadataTransformFn | None = None,
) -> str:
    """Define and register a Firestore vector store retriever.

    Args:
        ai: The Genkit instance to register the retriever with.
        name: Name of the retriever.
        embedder: The embedder to use (e.g., 'vertexai/gemini-embedding-001').
        embedder_options: Optional configuration to pass to the embedder.
        collection: The name of the Firestore collection to query.
        vector_field: The name of the field containing the vector embeddings.
        content_field: The name of the field containing the document content, you wish to return.
        firestore_client: The Firestore database instance from which to query.
        distance_measure: The distance measure to use when comparing vectors. Defaults to 'COSINE'.
        metadata_fields: Optional list of metadata fields to include.

    Returns:
        The registered retriever name.
    """
    retriever = FirestoreRetriever(
        ai=ai,
        name=name,
        embedder=embedder,
        embedder_options=embedder_options,
        firestore_client=firestore_client,
        collection=collection,
        vector_field=vector_field,
        content_field=content_field,
        distance_measure=distance_measure,
        metadata_fields=metadata_fields,
    )

    action_name = firestore_action_name(name)

    ai.registry.register_action(
        kind=ActionKind.RETRIEVER,
        name=action_name,
        fn=retriever.retrieve,
        metadata=retriever_action_metadata(
            name=action_name,
            options=RetrieverOptions(label=name),
        ).metadata,
    )

    return action_name
