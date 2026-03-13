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

"""Tests for DevLocalVectorStoreRetriever."""

from genkit.plugins.dev_local_vectorstore.retriever import (
    DevLocalVectorStoreRetriever,
    RetrieverOptionsSchema,
    ScoredDocument,
)


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self) -> None:
        """Identical vectors have cosine similarity of 1.0."""
        a = [1.0, 2.0, 3.0]
        result = DevLocalVectorStoreRetriever.cosine_similarity(a, a)
        assert abs(result - 1.0) < 1e-9

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors have cosine similarity of 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = DevLocalVectorStoreRetriever.cosine_similarity(a, b)
        assert abs(result) < 1e-9

    def test_opposite_vectors(self) -> None:
        """Opposite vectors have cosine similarity of -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        result = DevLocalVectorStoreRetriever.cosine_similarity(a, b)
        assert abs(result - (-1.0)) < 1e-9

    def test_similar_vectors(self) -> None:
        """Similar vectors have high cosine similarity."""
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]
        result = DevLocalVectorStoreRetriever.cosine_similarity(a, b)
        assert result > 0.99

    def test_dissimilar_vectors(self) -> None:
        """Dissimilar vectors have lower cosine similarity."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 1.0]
        result = DevLocalVectorStoreRetriever.cosine_similarity(a, b)
        assert abs(result) < 0.01


class TestDotProduct:
    """Tests for dot product calculation."""

    def test_basic_dot_product(self) -> None:
        """Dot product of [1,2,3] and [4,5,6] = 32."""
        result = DevLocalVectorStoreRetriever.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert abs(result - 32.0) < 1e-9

    def test_zero_vector(self) -> None:
        """Dot product with zero vector is 0."""
        result = DevLocalVectorStoreRetriever.dot([1.0, 2.0], [0.0, 0.0])
        assert abs(result) < 1e-9

    def test_single_dimension(self) -> None:
        """Dot product of single-dimension vectors."""
        result = DevLocalVectorStoreRetriever.dot([3.0], [4.0])
        assert abs(result - 12.0) < 1e-9

    def test_negative_values(self) -> None:
        """Dot product with negative values."""
        result = DevLocalVectorStoreRetriever.dot([1.0, -2.0], [-3.0, 4.0])
        assert abs(result - (-11.0)) < 1e-9


class TestScoredDocument:
    """Tests for ScoredDocument model."""

    def test_create_scored_document(self) -> None:
        """ScoredDocument can be created with score and document."""
        from genkit.ai import Document

        doc = Document.from_text('test')
        scored = ScoredDocument(score=0.95, document=doc)
        assert scored.score == 0.95
        assert scored.document is doc

    def test_scored_document_ordering(self) -> None:
        """ScoredDocuments can be sorted by score."""
        from genkit.ai import Document

        docs = [
            ScoredDocument(score=0.3, document=Document.from_text('low')),
            ScoredDocument(score=0.9, document=Document.from_text('high')),
            ScoredDocument(score=0.6, document=Document.from_text('mid')),
        ]
        sorted_docs = sorted(docs, key=lambda d: d.score, reverse=True)
        assert sorted_docs[0].score == 0.9
        assert sorted_docs[1].score == 0.6
        assert sorted_docs[2].score == 0.3


class TestRetrieverOptionsSchema:
    """Tests for RetrieverOptionsSchema."""

    def test_default_limit_is_none(self) -> None:
        """Default limit is None."""
        opts = RetrieverOptionsSchema()
        assert opts.limit is None

    def test_custom_limit(self) -> None:
        """Custom limit value is preserved."""
        opts = RetrieverOptionsSchema(limit=5)
        assert opts.limit == 5

    def test_schema_serialization(self) -> None:
        """Schema can be serialized to JSON."""
        schema = RetrieverOptionsSchema.model_json_schema()
        assert 'limit' in schema.get('properties', {})
