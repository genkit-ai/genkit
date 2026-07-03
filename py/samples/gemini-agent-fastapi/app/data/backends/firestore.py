# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Firestore backend — the only file in the app that imports ``google.cloud``.

Each item is a document at ``{root}/{collection}/{owner}/{item_id}``, so a user's
records are grouped under their uid and can't collide with another user's. To move
to another database, add a sibling module implementing ``Storage`` and point
``get_storage`` at it — this file is all the database-specific code there is.
"""

from __future__ import annotations

import asyncio
import builtins
from functools import lru_cache
from typing import Any

from google.cloud import firestore

from app.core.config import get_settings


@lru_cache
def _client() -> firestore.Client:
    return firestore.Client()


def _items(collection: str, owner: str) -> Any:  # noqa: ANN401
    return _client().collection(get_settings().firestore_collection).document(collection).collection(owner)


class FirestoreStorage:
    """Async wrapper around sync firestore client operations."""

    async def read(self, collection: str, owner: str, item_id: str) -> dict[str, Any] | None:
        def _read() -> dict[str, Any] | None:
            snap = _items(collection, owner).document(item_id).get()
            return snap.to_dict() if snap.exists else None

        return await asyncio.to_thread(_read)

    async def write(self, collection: str, owner: str, item_id: str, doc: dict[str, Any]) -> None:
        await asyncio.to_thread(lambda: _items(collection, owner).document(item_id).set(doc))

    async def list(self, collection: str, owner: str) -> builtins.list[dict[str, Any]]:
        def _list() -> builtins.list[dict[str, Any]]:
            return [doc.to_dict() for doc in _items(collection, owner).stream()]

        return await asyncio.to_thread(_list)

    async def delete(self, collection: str, owner: str, item_id: str) -> None:
        await asyncio.to_thread(lambda: _items(collection, owner).document(item_id).delete())
