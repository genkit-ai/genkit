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

from genkit._core._logger import get_logger
from genkit._core._protocols import GenkitLike, RegistryLike

from ._catalog import BASIC_CATALOG, A2uiCatalog
from ._types import A2UI_CATALOG_VALUE_TYPE, BASIC_CATALOG_ID, DEFAULT_CATALOG_ID

logger = get_logger(__name__)


def load_catalog(ai: GenkitLike, catalog: A2uiCatalog) -> A2uiCatalog:
    if not catalog.id:
        raise ValueError('a2ui: load_catalog: catalog has no id')
    existing = ai.registry.lookup_value(A2UI_CATALOG_VALUE_TYPE, catalog.id)
    if existing is not None:
        current = A2uiCatalog.from_value(existing)
        if current is None:
            raise ValueError(f'a2ui: load_catalog: registry value {catalog.id!r} is not a catalog')
        if current != catalog:
            logger.warning(
                'a2ui: load_catalog: a different catalog is already registered under this id; keeping the existing one'
            )
        return current
    ai.registry.register_value(A2UI_CATALOG_VALUE_TYPE, catalog.id, catalog.as_value())
    return catalog


def load_catalog_file(ai: GenkitLike, path: str) -> A2uiCatalog:
    return load_catalog(ai, read_catalog_file(path=path))


def register_basic_catalog(ai: GenkitLike) -> A2uiCatalog:
    return load_catalog(ai, BASIC_CATALOG)


def read_catalog_file(*, path: str) -> A2uiCatalog:
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f'a2ui: failed to read catalog file {path!r}: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'a2ui: catalog file {path!r} is not valid JSON: {exc}') from exc
    catalog = A2uiCatalog.from_value(raw)
    if catalog is None:
        raise ValueError(
            f'a2ui: catalog file {path!r} is not a catalog '
            '(need an id, a components array, and a name on every component)'
        )
    return catalog


def resolve_catalog(*, registry: RegistryLike, catalog: str | None) -> A2uiCatalog:
    lookup = catalog or DEFAULT_CATALOG_ID
    found = registry.lookup_value(A2UI_CATALOG_VALUE_TYPE, lookup)
    if found is not None:
        resolved = A2uiCatalog.from_value(found)
        if resolved is None:
            raise ValueError(f'a2ui: registry value {lookup!r} is not a catalog')
        return resolved
    if lookup in {DEFAULT_CATALOG_ID, BASIC_CATALOG_ID}:
        return BASIC_CATALOG
    raise ValueError(
        f'a2ui: no catalog registered under id {lookup!r}; '
        f'register one with load_catalog(ai, catalog) or use the default {DEFAULT_CATALOG_ID!r} catalog'
    )
