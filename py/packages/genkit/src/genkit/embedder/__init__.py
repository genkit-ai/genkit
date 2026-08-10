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

"""Embedder namespace module for Genkit.

This module provides embedder-related types and utilities for plugin authors
and advanced users who need access to the embedder protocol types.

Example:
    from genkit.embedder import (
        EmbedRequest,
        EmbedResponse,
        embedder_action_metadata,
        EmbedderRef,
    )
"""

from genkit._ai._embedding import (
    EmbedderOptions,
    EmbedderRef,
    EmbedderSupports,
    create_embedder_ref as embedder_ref,
    embedder_action_metadata,
)
from genkit._core._typing import (
    Embedding,
    EmbedRequest,
    EmbedResponse,
)

__all__ = [
    # Request/Response types
    'EmbedRequest',
    'EmbedResponse',
    'Embedding',
    # Factory functions and metadata
    'embedder_action_metadata',
    'embedder_ref',
    # Reference types
    'EmbedderRef',
    # Options and capabilities
    'EmbedderSupports',
    'EmbedderOptions',
]
