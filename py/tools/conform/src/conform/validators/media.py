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

"""Media validator: valid-media.

Canonical JS source (keep in sync):
    genkit-tools/cli/src/commands/dev-test-model.ts — VALIDATORS['valid-media']
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from conform.validators import ValidationError, register
from conform.validators.helpers import get_media_part


@register('valid-media')
def valid_media(
    response: dict[str, Any],
    arg: str | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Check that the response contains a valid media part.

    Args:
        response: The parsed model response dict.
        arg: Expected media type prefix (e.g. 'image', 'audio').
        chunks: Streaming chunks (unused by this validator).
    """
    media_part = get_media_part(response)
    if not media_part:
        raise ValidationError(f'Model did not return {arg or "media"} part.')

    media = media_part.get('media', {})

    if arg:
        content_type = media.get('contentType', '')
        if content_type and not content_type.startswith(f'{arg}/'):
            raise ValidationError(f'Expected {arg} content type, got {content_type}')

    if arg == 'image':
        url = media.get('url', '')
        if not url:
            raise ValidationError('Media part missing URL')
        if url.startswith('data:'):
            if not url.startswith('data:image/'):
                raise ValidationError('Invalid data URL content type for image')
        elif url.startswith('http'):
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    raise ValidationError(f'Invalid URL: {url}')
            except Exception as exc:
                raise ValidationError(f'Invalid URL: {url}') from exc
        else:
            raise ValidationError(f'Unknown URL format: {url}')
