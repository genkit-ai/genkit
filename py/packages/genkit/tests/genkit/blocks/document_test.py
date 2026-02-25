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

"""Tests for Genkit document."""

from typing import cast

from genkit.blocks.document import Document
from genkit.core.typing import (
    DocumentData,
    DocumentPart,
    Embedding,
    Media,
    MediaPart,
    TextPart,
)


def test_makes_deep_copy() -> None:
    """Test that Document makes a deep copy of its content and metadata."""
    content = [DocumentPart(root=TextPart(text='some text'))]
    metadata = {'foo': 'bar'}
    doc = Document(content=content, metadata=metadata)

    text_part = cast(TextPart, content[0].root)
    text_part.text = 'other text'
    metadata['foo'] = 'faz'

    assert doc.content[0].root.text == 'some text'
    assert doc.metadata is not None
    assert doc.metadata['foo'] == 'bar'


def test_from_document_data() -> None:
    """Test creating a Document from DocumentData."""
    doc = Document.from_document_data(DocumentData(content=[DocumentPart(root=TextPart(text='some text'))]))

    assert doc.text() == 'some text'


def test_simple_text_document() -> None:
    """Test creating a simple text Document."""
    doc = Document.from_text('sample text')

    assert doc.text() == 'sample text'


def test_media_document() -> None:
    """Test creating a media Document."""
    doc = Document.from_media(url='data:one')

    assert doc.media() == [
        Media(url='data:one'),
    ]


def test_from_data_text_document() -> None:
    """Test creating a text Document using from_data."""
    data = 'foo'
    data_type = 'text'
    metadata = {'embedMetadata': {'embeddingType': 'text'}}
    doc = Document.from_data(data, data_type, metadata)

    assert doc.text() == data
    assert doc.metadata == metadata
    assert doc.data_type() == data_type


def test_from_data_media_document() -> None:
    """Test creating a media Document using from_data."""
    data = 'iVBORw0KGgoAAAANSUhEUgAAAAjCB0C8AAAAASUVORK5CYII='
    data_type = 'image/png'
    metadata = {'embedMetadata': {'embeddingType': 'image'}}
    doc = Document.from_data(data, data_type, metadata)

    assert doc.media() == [
        Media(url=data, content_type=data_type),
    ]
    assert doc.metadata == metadata
    assert doc.data_type() == data_type


def test_concatenates_text() -> None:
    """Test that text() concatenates multiple text parts."""
    content = [DocumentPart(root=TextPart(text='hello')), DocumentPart(root=TextPart(text='world'))]
    doc = Document(content=content)

    assert doc.text() == 'helloworld'


def test_multiple_media_document() -> None:
    """Test that media() returns all media parts."""
    content = [
        DocumentPart(root=MediaPart(media=Media(url='data:one'))),
        DocumentPart(root=MediaPart(media=Media(url='data:two'))),
    ]
    doc = Document(content=content)

    assert doc.media() == [
        Media(url='data:one'),
        Media(url='data:two'),
    ]


def test_data_with_text() -> None:
    """Test data() with a text document."""
    doc = Document.from_text('hello')

    assert doc.data() == 'hello'


def test_data_with_media() -> None:
    """Test data() with a media document."""
    doc = Document.from_media(url='gs://somebucket/someimage.png', content_type='image/png')

    assert doc.data() == 'gs://somebucket/someimage.png'


def test_data_type_with_text() -> None:
    """Test data_type() with a text document."""
    doc = Document.from_text('hello')

    assert doc.data_type() == 'text'


def test_data_type_with_media() -> None:
    """Test data_type() with a media document."""
    doc = Document.from_media(url='gs://somebucket/someimage.png', content_type='image/png')

    assert doc.data_type() == 'image/png'


def test_get_embedding_documents() -> None:
    """Test getting embedding documents for a single embedding."""
    doc = Document.from_text('foo')
    embeddings: list[Embedding] = [Embedding(embedding=[0.1, 0.2, 0.3])]
    docs = doc.get_embedding_documents(embeddings)

    assert docs == [doc]


def test_get_embedding_documents_multiple_embeddings() -> None:
    """Test getting embedding documents for multiple embeddings."""
    url = 'gs://somebucket/somevideo.mp4'
    content_type = 'video/mp4'
    metadata = {'start': 0, 'end': 60}
    doc = Document.from_media(url, content_type, metadata)
    embeddings: list[Embedding] = []

    for start in range(0, 60, 15):
        embeddings.append(make_test_embedding(start))
    docs = doc.get_embedding_documents(embeddings)

    assert len(docs) == len(embeddings)

    for i in range(len(docs)):
        assert docs[i].content == doc.content
        metadata = docs[i].metadata or {}
        assert metadata.get('embedMetadata') == embeddings[i].metadata
        orig_metadata = dict(metadata)
        orig_metadata.pop('embedMetadata', None)
        assert orig_metadata, doc.metadata


def make_test_embedding(start: int) -> Embedding:
    """Helper to create a test embedding with specific metadata."""
    return Embedding(
        embedding=[0.1, 0.2, 0.3],
        metadata={'embeddingType': 'video', 'start': start, 'end': start + 15},
    )
