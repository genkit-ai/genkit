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

"""Document model for the Genkit framework.

This module provides types for creating and managing multi-modal documents in
Genkit.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

from genkit.core.typing import (
    DocumentData,
    DocumentPart,
    Embedding,
    Media,
    MediaPart,
    TextPart,
)

TEXT_DATA_TYPE: str = 'text'


class Document(DocumentData):
    """Represents document content along with its metadata.

    This object can be embedded, indexed or retrieved. Each document can contain
    multiple parts (for example text and an image).
    """

    def __init__(
        self,
        content: list[DocumentPart],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initializes a Document object.

        Performs a deep copy of the provided content and metadata to prevent
        unintended modifications to the original objects.

        Args:
            content: A list of DocumentPart objects representing the document's content.
            metadata: An optional dictionary containing metadata about the document.
        """
        doc_content = deepcopy(content)
        doc_metadata = deepcopy(metadata)
        super().__init__(content=doc_content, metadata=doc_metadata)

    @staticmethod
    def from_document_data(document_data: DocumentData) -> Document:
        """Constructs a Document instance from a DocumentData object.

        Args:
            document_data: The DocumentData object containing content and metadata.

        Returns:
            A new Document instance initialized with the provided data.
        """
        return Document(content=document_data.content, metadata=document_data.metadata)

    @staticmethod
    def from_text(text: str, metadata: dict[str, Any] | None = None) -> Document:
        """Constructs a Document instance from a single text string.

        Args:
            text: The text content for the document.
            metadata: Optional metadata for the document.

        Returns:
            A new Document instance containing a single text part.
        """
        # NOTE: DocumentPart is a RootModel requiring root=TextPart(...) syntax.
        return Document(content=[DocumentPart(root=TextPart(text=text))], metadata=metadata)

    @staticmethod
    def from_media(
        url: str,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Constructs a Document instance from a single media URL.

        Args:
            url: The URL of the media content.
            content_type: Optional MIME type of the media.
            metadata: Optional metadata for the document.

        Returns:
            A new Document instance containing a single media part.
        """
        return Document(
            # NOTE: DocumentPart is a RootModel requiring root=MediaPart(...) syntax.
            # Using contentType alias for ty type checker compatibility.
            content=[DocumentPart(root=MediaPart(media=Media(url=url, content_type=content_type)))],
            metadata=metadata,
        )

    @staticmethod
    def from_data(
        data: str,
        data_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Constructs a Document instance from a single data string, determining type.

        If `data_type` is 'text', creates a text document. Otherwise, assumes
        `data` is a URL and creates a media document.

        Args:
            data: The data string (either text content or a media URL).
            data_type: The type of the data ('text' or a media content type).
            metadata: Optional metadata for the document.

        Returns:
            A new Document instance.
        """
        if data_type == TEXT_DATA_TYPE:
            return Document.from_text(data, metadata)
        return Document.from_media(data, data_type, metadata)

    def text(self) -> str:
        """Concatenates all text parts of the document's content.

        Returns:
            A single string containing the text from all text parts, joined
            without delimiters.
        """
        texts = []
        for p in self.content:
            # Handle both TextPart objects and potential dict representations
            # p.root is the underlying TextPart or MediaPart
            part = p.root if hasattr(p, 'root') else p
            text_val = getattr(part, 'text', None)
            if isinstance(text_val, str):
                texts.append(text_val)
        return ''.join(texts)

    def media(self) -> list[Media]:
        """Retrieves all media parts from the document's content.

        Returns:
            A list of Media objects contained within the document.
        """
        return [
            part.root.media for part in self.content if isinstance(part.root, MediaPart) and part.root.media is not None
        ]

    def data(self) -> str:
        """Gets the primary data content of the document.

        Returns the concatenated text if available, otherwise the URL of the
        first media part. Returns an empty string if the document has neither
        text nor media.

        Returns:
            The primary data string (text or media URL).
        """
        if self.text():
            return self.text()

        if self.media():
            return self.media()[0].url

        return ''

    def data_type(self) -> str | None:
        """Gets the data type corresponding to the primary data content.

        Returns 'text' if the primary data is text. If the primary data is media,
        returns the content type of the first media part. Returns None if the
        document has no primary data or the media content type is not set.

        Returns:
            The data type string ('text' or a MIME type) or None.
        """
        if self.text():
            return TEXT_DATA_TYPE

        if self.media() and self.media()[0].content_type:
            return self.media()[0].content_type

        return None

    def get_embedding_documents(self, embeddings: list[Embedding]) -> list[Document]:
        """Creates multiple Document instances from a single document and its embeddings.

        Since embedders can return multiple embeddings for one input document,
        but storage often requires a 1:1 document-to-embedding relationship,
        this method duplicates the original document for each embedding.
        Embedding metadata is added to the corresponding document's metadata
        under the 'embedMetadata' key.

        Args:
            embeddings: A list of Embedding objects generated for this document.

        Returns:
            A list of Document objects, one for each provided embedding.
        """
        documents = []
        for embedding in embeddings:
            content = deepcopy(self.content)
            metadata = deepcopy(self.metadata)
            if embedding.metadata:
                if not metadata:
                    metadata = {}
                metadata['embedMetadata'] = embedding.metadata
            documents.append(Document(content=content, metadata=metadata))
        _ = check_unique_documents(documents)
        return documents


def check_unique_documents(documents: list[Document]) -> bool:
    """Checks if a list of documents contains duplicates based on their JSON representation.

    Prints a warning if duplicates are found, as this can cause issues with
    vector storage systems that key documents by their content hash.
    Duplicate documents might arise if embeddings are generated without unique
    metadata for different parts or aspects of the same original document.

    Args:
        documents: A list of Document objects to check.

    Returns:
        True if all documents are unique, False otherwise (primarily for testing).
    """
    seen = set()
    for doc in documents:
        if doc.model_dump_json() in seen:
            warnings.warn(
                'Embedding documents are not unique. Are you missing embed metadata?',
                stacklevel=2,
            )
            return False
        seen.add(doc.model_dump_json())
    return True
