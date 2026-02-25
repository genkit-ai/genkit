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

"""Tests for Vertex AI plugin."""

from genkit.plugins.vertex_ai import (
    BigQueryRetriever,
    FirestoreRetriever,
    ModelGardenPlugin,
    RetrieverOptionsSchema,
    define_vertex_vector_search_big_query,
    define_vertex_vector_search_firestore,
    package_name,
)


def test_package_name() -> None:
    """Test package_name returns correct value."""
    assert package_name() == 'genkit.plugins.vertex_ai'


def test_model_garden_plugin_exported() -> None:
    """Test ModelGardenPlugin is exported."""
    assert ModelGardenPlugin is not None


def test_bigquery_retriever_exported() -> None:
    """Test BigQueryRetriever is exported."""
    assert BigQueryRetriever is not None


def test_firestore_retriever_exported() -> None:
    """Test FirestoreRetriever is exported."""
    assert FirestoreRetriever is not None


def test_retriever_options_schema_exported() -> None:
    """Test RetrieverOptionsSchema is exported."""
    assert RetrieverOptionsSchema is not None


def test_define_vertex_vector_search_big_query_callable() -> None:
    """Test define_vertex_vector_search_big_query is callable."""
    assert callable(define_vertex_vector_search_big_query)


def test_define_vertex_vector_search_firestore_callable() -> None:
    """Test define_vertex_vector_search_firestore is callable."""
    assert callable(define_vertex_vector_search_firestore)
