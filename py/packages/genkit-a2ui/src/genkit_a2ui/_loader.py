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

"""Register A2UI catalogs so generate and the Developer UI can find them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from genkit import Genkit, GenkitError, RuntimeErrorReason, get_logger

from ._catalog import BASIC_CATALOG, A2uiCatalog
from ._types import A2UI_CATALOG_VALUE_TYPE, BASIC_CATALOG_ID, DEFAULT_CATALOG_ID

logger = get_logger(__name__)


class ValueLookup(Protocol):
    """Anything catalogs can be read from: the app, or ``ctx.ai`` inside middleware."""

    def lookup_value(self, *, kind: str, name: str) -> object | None: ...


class A2uiCatalogError(GenkitError):
    """Raised when a catalog cannot be read, registered, or resolved.

    `load_catalog` and `load_catalog_file` raise this at startup, so a bad
    catalog file fails loudly before any model call. `resolve_catalog` runs
    inside the model call instead, so an unregistered catalog id comes back as a
    failed `ModelResponse` with the reason on `finish_message`.
    """

    def __init__(self, message: str) -> None:
        # A catalog you did not register, or a file that does not parse, is a
        # bad argument, not a bad model answer. Mirrors Go's
        # ErrInvalidInput = ErrInvalidArgument.Subtype("invalid input").
        super().__init__(
            status='INVALID_ARGUMENT',
            message=message,
            reason=RuntimeErrorReason.INVALID_INPUT,
        )


def load_catalog(ai: Genkit, catalog: A2uiCatalog) -> A2uiCatalog:
    if not catalog.id:
        raise A2uiCatalogError('a2ui: load_catalog: catalog has no id')
    existing = ai.lookup_value(kind=A2UI_CATALOG_VALUE_TYPE, name=catalog.id)
    if existing is not None:
        current = A2uiCatalog.from_value(existing)
        if current is None:
            raise A2uiCatalogError(f'a2ui: load_catalog: registry value {catalog.id!r} is not a catalog')
        if current != catalog:
            logger.warning(
                'a2ui: load_catalog: a different catalog is already registered under this id; keeping the existing one'
            )
        return current
    ai.define_value(kind=A2UI_CATALOG_VALUE_TYPE, name=catalog.id, value=catalog.as_value())
    return catalog


def load_catalog_file(ai: Genkit, path: str) -> A2uiCatalog:
    return load_catalog(ai, read_catalog_file(path=path))


def register_basic_catalog(ai: Genkit) -> A2uiCatalog:
    return load_catalog(ai, BASIC_CATALOG)


def read_catalog_file(*, path: str) -> A2uiCatalog:
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError) as exc:
        raise A2uiCatalogError(f'a2ui: failed to read catalog file {path!r}: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise A2uiCatalogError(f'a2ui: catalog file {path!r} is not valid JSON: {exc}') from exc
    catalog = A2uiCatalog.from_value(raw)
    if catalog is None:
        raise A2uiCatalogError(
            f'a2ui: catalog file {path!r} is not a catalog '
            '(need an id, a components array, and a name on every component)'
        )
    return catalog


def resolve_catalog(*, ai: ValueLookup, catalog: str | None) -> A2uiCatalog:
    lookup = catalog or DEFAULT_CATALOG_ID
    found = ai.lookup_value(kind=A2UI_CATALOG_VALUE_TYPE, name=lookup)
    if found is not None:
        resolved = A2uiCatalog.from_value(found)
        if resolved is None:
            raise A2uiCatalogError(f'a2ui: registry value {lookup!r} is not a catalog')
        return resolved
    if lookup in {DEFAULT_CATALOG_ID, BASIC_CATALOG_ID}:
        return BASIC_CATALOG
    raise A2uiCatalogError(
        f'a2ui: no catalog registered under id {lookup!r}; '
        f'register one with load_catalog(ai, catalog) or use the default {DEFAULT_CATALOG_ID!r} catalog'
    )
