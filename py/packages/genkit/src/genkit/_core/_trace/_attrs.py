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

"""Canonical Genkit span attribute keys and value enums.

``Attr`` members are the wire names Dev UI / exporters read. ``State`` /
``Subtype`` hold allowed values for those keys so callers don't mix a key
(``Attr.STATE``) with a value (``State.ERROR``).
"""

from typing import Final

from genkit._core._compat import StrEnum

PREFIX: Final[str] = 'genkit'
METADATA_PREFIX: Final[str] = f'{PREFIX}:metadata:'


class Attr(StrEnum):
    """Span attribute keys."""

    NAME = f'{PREFIX}:name'
    PATH = f'{PREFIX}:path'
    QUALIFIED_PATH = f'{PREFIX}:qualifiedPath'
    TYPE = f'{PREFIX}:type'
    INPUT = f'{PREFIX}:input'
    OUTPUT = f'{PREFIX}:output'
    STATE = f'{PREFIX}:state'
    ERROR = f'{PREFIX}:error'
    IS_ROOT = f'{PREFIX}:isRoot'
    IS_FAILURE_SOURCE = f'{PREFIX}:isFailureSource'
    FAILED_SPAN = f'{PREFIX}:failedSpan'
    FAILED_PATH = f'{PREFIX}:failedPath'
    FEATURE = f'{PREFIX}:feature'
    MODEL = f'{PREFIX}:model'
    SUBTYPE = f'{METADATA_PREFIX}subtype'


class State(StrEnum):
    """Values for :attr:`Attr.STATE`."""

    SUCCESS = 'success'
    ERROR = 'error'


class Subtype(StrEnum):
    """Common values for :attr:`Attr.SUBTYPE` (not exhaustive)."""

    MODEL = 'model'
    PROMPT = 'prompt'
    TOOL = 'tool'
    FLOW = 'flow'


def metadata_key(key: str) -> str:
    """Prefix a short metadata key: ``flow:name`` → ``genkit:metadata:flow:name``."""
    return f'{METADATA_PREFIX}{key}'
