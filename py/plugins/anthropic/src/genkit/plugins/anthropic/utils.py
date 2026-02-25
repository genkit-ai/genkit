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

"""Anthropic plugin utility functions.

Pure-function helpers for converting Genkit content parts to Anthropic API
format. Extracted from the model module for independent unit testing.

See:
    - Cache control: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    - Document input: https://docs.anthropic.com/en/docs/build-with-claude/pdf-support
"""

import re
from typing import Any

from genkit.core.logging import get_logger
from genkit.types import GenerateRequest, GenerationUsage, MediaPart, Part, TextPart

logger = get_logger(__name__)

# PDF MIME type for document handling.
PDF_MIME_TYPE = 'application/pdf'

# Plain text MIME type for document handling.
TEXT_MIME_TYPE = 'text/plain'

# MIME types supported by Anthropic's DocumentBlockParam.
DOCUMENT_MIME_TYPES = frozenset({PDF_MIME_TYPE, TEXT_MIME_TYPE})

__all__ = [
    'DOCUMENT_MIME_TYPES',
    'PDF_MIME_TYPE',
    'TEXT_MIME_TYPE',
    'build_cache_usage',
    'get_cache_control',
    'maybe_strip_fences',
    'strip_markdown_fences',
    'to_anthropic_document',
    'to_anthropic_image',
    'to_anthropic_media',
]


def strip_markdown_fences(text: str) -> str:
    r"""Strip markdown code fences from a JSON response.

    Models sometimes wrap JSON output in markdown fences like
    ``\`\`\`json ... \`\`\``` even when instructed to output raw
    JSON.  This helper removes the fences.

    Args:
        text: The response text, possibly wrapped in fences.

    Returns:
        The text with markdown fences removed, or the original
        text if no fences are found.
    """
    stripped = text.strip()
    match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?\s*```$', stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def maybe_strip_fences(request: GenerateRequest, parts: list[Part]) -> list[Part]:
    """Strip markdown fences from text parts when JSON output is expected.

    Args:
        request: The original generate request.
        parts: The response content parts.

    Returns:
        Parts with fences stripped from text if JSON was requested.
    """
    if not request.output or request.output.format != 'json':
        return parts

    cleaned: list[Part] = []
    changed = False
    for part in parts:
        if isinstance(part.root, TextPart) and part.root.text:
            cleaned_text = strip_markdown_fences(part.root.text)
            if cleaned_text != part.root.text:
                cleaned.append(Part(root=TextPart(text=cleaned_text)))
                changed = True
            else:
                cleaned.append(part)
        else:
            cleaned.append(part)
    return cleaned if changed else parts


def get_cache_control(part: Any) -> dict[str, str] | None:  # noqa: ANN401
    """Extract cache_control metadata from a content part.

    Genkit parts can carry arbitrary metadata. If a part has
    ``metadata.cache_control``, it is passed through to the
    Anthropic API as cache control configuration.

    ``Metadata`` is a Pydantic ``RootModel[dict[str, Any]]``, so the
    underlying dict is accessed via ``.root``.

    Supported format::

        Part(root=TextPart(text='...', metadata={'cache_control': {'type': 'ephemeral'}}))

    Args:
        part: The actual (unwrapped) content part (e.g. TextPart, MediaPart).

    Returns:
        Cache control dict (e.g. ``{'type': 'ephemeral'}``) or None.
    """
    metadata = getattr(part, 'metadata', None)
    if metadata is None:
        return None

    # Metadata is a RootModel[dict[str, Any]] — unwrap the root dict.
    meta_dict = metadata.root if hasattr(metadata, 'root') else metadata
    if not isinstance(meta_dict, dict):
        return None

    cache_ctrl = meta_dict.get('cache_control')
    if cache_ctrl and isinstance(cache_ctrl, dict):
        return cache_ctrl
    return None


def to_anthropic_document(url: str, content_type: str) -> dict[str, Any]:
    """Convert a media URL to Anthropic DocumentBlockParam.

    Supports base64-encoded and URL-based document sources for
    ``application/pdf`` and ``text/plain`` types.

    See: https://docs.anthropic.com/en/docs/build-with-claude/pdf-support

    Args:
        url: The document URL or data URI.
        content_type: The MIME type of the document.

    Returns:
        Anthropic document block dict.
    """
    if url.startswith('data:'):
        _, base64_data = url.split(',', 1)
        return {
            'type': 'document',
            'source': {
                'type': 'base64',
                'media_type': content_type,
                'data': base64_data,
            },
        }

    # URL-based document source — only PDF supports URL source.
    if content_type == PDF_MIME_TYPE:
        return {
            'type': 'document',
            'source': {'type': 'url', 'url': url},
        }

    # Plain text from URL: fall back to text block since Anthropic's
    # URL source only supports PDFs.
    logger.warning(
        'Plain text URL documents are not supported by Anthropic DocumentBlockParam; falling back to text block.'
    )
    return {'type': 'text', 'text': f'[Document: {url}]'}


def to_anthropic_image(url: str, content_type: str) -> dict[str, Any]:
    """Convert to Anthropic image block.

    Args:
        url: The image URL or data URI.
        content_type: The MIME type of the image.

    Returns:
        Anthropic image block dict.
    """
    if url.startswith('data:'):
        _, base64_data = url.split(',', 1)
        img_content_type = content_type or url.split(':')[1].split(';')[0]
        return {
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': img_content_type,
                'data': base64_data,
            },
        }
    return {'type': 'image', 'source': {'type': 'url', 'url': url}}


def to_anthropic_media(media_part: MediaPart) -> dict[str, Any]:
    """Convert a MediaPart to the appropriate Anthropic format.

    Routes to ``document`` block for PDF/plain-text MIME types,
    and ``image`` block for image MIME types.

    Args:
        media_part: The Genkit MediaPart to convert.

    Returns:
        Anthropic content block dict (document or image).
    """
    url = media_part.media.url
    content_type = media_part.media.content_type or ''

    # Infer MIME type from data URI if not explicitly set.
    if not content_type and url.startswith('data:'):
        content_type = url.split(':')[1].split(';')[0]

    # Route PDFs and plain text to DocumentBlockParam.
    if content_type in DOCUMENT_MIME_TYPES:
        return to_anthropic_document(url, content_type)

    # Default: image handling.
    return to_anthropic_image(url, content_type)


def build_cache_usage(
    input_tokens: int,
    output_tokens: int,
    basic_usage: GenerationUsage,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> GenerationUsage:
    """Build GenerationUsage with cache-aware token counts.

    Args:
        input_tokens: Number of input tokens from the API response.
        output_tokens: Number of output tokens from the API response.
        basic_usage: Basic character/image usage from message content.
        cache_creation_input_tokens: Tokens for newly created cache entries.
        cache_read_input_tokens: Tokens read from existing cache entries.

    Returns:
        GenerationUsage with token, character, and cache counts.
    """
    custom: dict[str, float] = {}
    if cache_creation_input_tokens:
        custom['cache_creation_input_tokens'] = cache_creation_input_tokens
    if cache_read_input_tokens:
        custom['cache_read_input_tokens'] = cache_read_input_tokens

    return GenerationUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_characters=basic_usage.input_characters,
        output_characters=basic_usage.output_characters,
        input_images=basic_usage.input_images,
        output_images=basic_usage.output_images,
        custom=custom if custom else None,
    )
