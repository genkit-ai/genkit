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

"""Tests for FirestoreRetriever."""

from unittest.mock import MagicMock

import pytest

from genkit.core.typing import DocumentPart, TextPart
from genkit.plugins.firebase.retriever import FirestoreRetriever


def _make_retriever(**overrides: object) -> FirestoreRetriever:
    """Create a FirestoreRetriever with sensible defaults for testing."""
    defaults: dict[str, object] = {
        'ai': MagicMock(),
        'name': 'test-retriever',
        'embedder': 'test-embedder',
        'embedder_options': None,
        'firestore_client': MagicMock(),
        'collection': 'test-collection',
        'vector_field': 'embedding',
        'content_field': 'content',
    }
    defaults.update(overrides)
    return FirestoreRetriever(**defaults)  # type: ignore[arg-type]


class TestFirestoreRetrieverInit:
    """Tests for FirestoreRetriever initialization."""

    def test_init_stores_all_params(self) -> None:
        """Constructor stores all configuration parameters."""
        r = _make_retriever()
        assert r.name == 'test-retriever'
        assert r.embedder == 'test-embedder'
        assert r.collection == 'test-collection'
        assert r.vector_field == 'embedding'
        assert r.content_field == 'content'
        assert r.embedder_options is None


class TestFirestoreRetrieverValidation:
    """Tests for config validation."""

    def test_empty_collection_raises(self) -> None:
        """Empty collection name raises ValueError."""
        with pytest.raises(ValueError, match='collection'):
            _make_retriever(collection='')

    def test_empty_vector_field_raises(self) -> None:
        """Empty vector field name raises ValueError."""
        with pytest.raises(ValueError, match='vector field'):
            _make_retriever(vector_field='')

    def test_empty_embedder_raises(self) -> None:
        """Empty embedder name raises ValueError."""
        with pytest.raises(ValueError, match='embedder'):
            _make_retriever(embedder='')

    def test_none_firestore_client_raises(self) -> None:
        """None firestore client raises ValueError."""
        with pytest.raises(ValueError, match='firestore client'):
            _make_retriever(firestore_client=None)


class TestFirestoreRetrieverToContent:
    """Tests for _to_content conversion."""

    def test_string_content_field(self) -> None:
        """String content_field reads from doc snapshot field."""
        r = _make_retriever(content_field='body')
        doc_snapshot = MagicMock()
        doc_snapshot.get.return_value = 'Hello world'

        parts = r._to_content(doc_snapshot)

        assert len(parts) == 1
        doc_snapshot.get.assert_called_once_with('body')

    def test_string_content_field_empty(self) -> None:
        """Empty content returns empty list."""
        r = _make_retriever(content_field='body')
        doc_snapshot = MagicMock()
        doc_snapshot.get.return_value = None

        parts = r._to_content(doc_snapshot)
        assert parts == []

    def test_callable_content_field(self) -> None:
        """Callable content_field is invoked with doc snapshot."""
        custom_content = [DocumentPart(root=TextPart(text='custom'))]
        content_fn = MagicMock(return_value=custom_content)
        r = _make_retriever(content_field=content_fn)
        doc_snapshot = MagicMock()

        parts = r._to_content(doc_snapshot)

        content_fn.assert_called_once_with(doc_snapshot)
        assert parts == custom_content


class TestFirestoreRetrieverToMetadata:
    """Tests for _to_metadata conversion."""

    def test_no_metadata_fields_returns_dict_minus_vector_content(self) -> None:
        """Without metadata_fields config, returns all fields except vector and content."""
        r = _make_retriever(
            vector_field='vec',
            content_field='body',
            metadata_fields=None,
        )
        doc_snapshot = MagicMock()
        doc_snapshot.to_dict.return_value = {
            'vec': [0.1, 0.2],
            'body': 'text',
            'author': 'Alice',
            'date': '2025-01-01',
        }

        metadata = r._to_metadata(doc_snapshot)

        assert 'vec' not in metadata
        assert 'body' not in metadata
        assert metadata['author'] == 'Alice'
        assert metadata['date'] == '2025-01-01'

    def test_metadata_fields_list_filters(self) -> None:
        """List of metadata_fields only includes specified fields."""
        r = _make_retriever(metadata_fields=['author'])
        doc_snapshot = MagicMock()
        doc_snapshot.to_dict.return_value = {
            'author': 'Bob',
            'date': '2025-01-01',
            'hidden': 'secret',
        }

        metadata = r._to_metadata(doc_snapshot)

        assert metadata == {'author': 'Bob'}
        assert 'date' not in metadata
        assert 'hidden' not in metadata

    def test_metadata_fields_list_missing_field_skipped(self) -> None:
        """Missing fields in metadata_fields list are silently skipped."""
        r = _make_retriever(metadata_fields=['author', 'nonexistent'])
        doc_snapshot = MagicMock()
        doc_snapshot.to_dict.return_value = {'author': 'Carol'}

        metadata = r._to_metadata(doc_snapshot)

        assert metadata == {'author': 'Carol'}

    def test_callable_metadata_fields(self) -> None:
        """Callable metadata_fields is invoked with doc snapshot."""
        meta_fn = MagicMock(return_value={'custom_key': 'custom_val'})
        r = _make_retriever(metadata_fields=meta_fn)
        doc_snapshot = MagicMock()

        metadata = r._to_metadata(doc_snapshot)

        meta_fn.assert_called_once_with(doc_snapshot)
        assert metadata == {'custom_key': 'custom_val'}


class TestFirestoreRetrieverToDocument:
    """Tests for _to_document conversion."""

    def test_to_document_combines_content_and_metadata(self) -> None:
        """_to_document creates a Document with content and metadata."""
        r = _make_retriever(content_field='body', metadata_fields=['author'])
        doc_snapshot = MagicMock()
        doc_snapshot.get.return_value = 'Hello'
        doc_snapshot.to_dict.return_value = {'body': 'Hello', 'author': 'Dave'}

        doc = r._to_document(doc_snapshot)

        assert len(doc.content) == 1
        assert doc.metadata is not None
        assert doc.metadata['author'] == 'Dave'


class TestFirestoreActionName:
    """Tests for firestore_action_name helper."""

    def test_action_name_format(self) -> None:
        """Action name is prefixed with 'firestore/'."""
        from genkit.plugins.firebase.firestore import firestore_action_name

        assert firestore_action_name('my-store') == 'firestore/my-store'

    def test_action_name_preserves_special_chars(self) -> None:
        """Special characters in name are preserved."""
        from genkit.plugins.firebase.firestore import firestore_action_name

        assert firestore_action_name('my_store-v2') == 'firestore/my_store-v2'
