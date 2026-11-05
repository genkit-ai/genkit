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

from genkit import Document, Part
from genkit._core._typing import (
    Media,
)


def test_makes_deep_copy() -> None:
    """Test that Document makes a deep copy of its content and metadata."""
    content = [Part.from_text('some text')]
    metadata = {'foo': 'bar'}
    doc = Document(content=content, metadata=metadata)

    content[0].text = 'other text'
    metadata['foo'] = 'faz'

    assert doc.content[0].text == 'some text'
    assert doc.metadata is not None
    assert doc.metadata['foo'] == 'bar'


def test_simple_text_document() -> None:
    """Test creating a simple text Document."""
    doc = Document.from_text('sample text')

    assert doc.text == 'sample text'


def test_media_document() -> None:
    """Test creating a media Document."""
    doc = Document.from_media(url='data:one')

    assert doc.media == [
        Media(url='data:one'),
    ]


def test_concatenates_text() -> None:
    """Test that text concatenates multiple text parts."""
    content = [Part.from_text('hello'), Part.from_text('world')]
    doc = Document(content=content)

    assert doc.text == 'helloworld'


def test_multiple_media_document() -> None:
    """Test that media returns all media parts."""
    content = [
        Part.from_media('data:one'),
        Part.from_media('data:two'),
    ]
    doc = Document(content=content)

    assert doc.media == [
        Media(url='data:one'),
        Media(url='data:two'),
    ]


def test_document_from_text_reads_back_through_text() -> None:
    """Document.from_text('hello') reads back as 'hello' through .text, with its metadata kept."""
    metadata = {'embedMetadata': {'embeddingType': 'text'}}
    doc = Document.from_text('hello', metadata)

    assert doc.text == 'hello'
    assert doc.metadata == metadata


def test_document_from_media_reads_back_through_media_url() -> None:
    """Document.from_media(url, 'image/png') reads back its url and content type through .media."""
    url = 'gs://somebucket/someimage.png'
    metadata = {'embedMetadata': {'embeddingType': 'image'}}
    doc = Document.from_media(url, 'image/png', metadata)

    assert doc.media[0].url == url
    assert doc.media[0].content_type == 'image/png'
    assert doc.metadata == metadata


def test_document_data_part_reads_back_through_content() -> None:
    """A data part reads back as the dict at doc.content[i].data, not through the document's text."""
    doc = Document(content=[Part.from_text('hello'), Part.from_data({'sku': 1})])

    assert doc.content[1].data == {'sku': 1}
    assert doc.text == 'hello'


def test_document_accepts_part_factories() -> None:
    """Document content accepts Part.from_text / Part.from_media."""
    doc = Document(
        content=[
            Part.from_text('Intro section'),
            Part.from_media('https://example.com/figure1.png', content_type='image/png'),
        ]
    )

    assert doc.text == 'Intro section'
    assert doc.media == [Media(url='https://example.com/figure1.png', content_type='image/png')]
