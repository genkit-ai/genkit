# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The storage engine: one tiny interface, two typed front doors.

Everything the app stores goes through here. ``Storage`` is the only thing that
knows how to talk to a database; ``user_doc`` and ``user_collection`` are the
typed, per-user helpers you actually use in features. Every read and write is
scoped to an owner (a uid), so app code can't accidentally touch another user's
data — the API won't let you leave the owner out.
"""

from __future__ import annotations

import builtins
from functools import lru_cache
from typing import Any, Generic, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from app.core.config import get_settings
from app.data.backends.firestore import FirestoreStorage

T = TypeVar('T', bound=BaseModel)

# Item id for a one-per-user document (``user_doc``); collections use real ids.
_SINGLE = '_'


class Storage(Protocol):
    """Raw per-owner document store — the one thing that knows what a database is.

    A backend implements these four methods and nothing else. Every typed store in
    the app is built on top of them, so a new database is one new implementation.
    """

    async def read(self, collection: str, owner: str, item_id: str) -> dict[str, Any] | None: ...
    async def write(self, collection: str, owner: str, item_id: str, doc: dict[str, Any]) -> None: ...
    async def list(self, collection: str, owner: str) -> builtins.list[dict[str, Any]]: ...
    async def delete(self, collection: str, owner: str, item_id: str) -> None: ...


@lru_cache
def get_storage() -> Storage:
    """Pick the datastore once, for the whole app.

    There's no in-memory fallback on purpose: a store that silently forgets on
    restart is a data-loss trap in production. You must point at a real datastore
    — a project in the cloud, or the Firestore emulator locally.
    """
    if get_settings().use_firestore:
        return FirestoreStorage()
    raise RuntimeError(
        'No datastore configured. Set FIREBASE_PROJECT_ID (deployed) or '
        'FIRESTORE_EMULATOR_HOST (local dev) — see the README quickstart.'
    )


class UserDoc(Generic[T]):
    """One document per user — settings, profile, onboarding state."""

    def __init__(self, name: str, model: type[T]) -> None:
        self._name = name
        self._model = model

    async def get(self, uid: str) -> T | None:
        raw = await get_storage().read(self._name, uid, _SINGLE)
        return self._model.model_validate(raw) if raw else None

    async def save(self, uid: str, value: T) -> None:
        await get_storage().write(self._name, uid, _SINGLE, value.model_dump(mode='json', by_alias=True))

    async def delete(self, uid: str) -> None:
        await get_storage().delete(self._name, uid, _SINGLE)


class UserCollection(Generic[T]):
    """Many documents per user — notes, projects, usage records."""

    def __init__(self, name: str, model: type[T]) -> None:
        self._name = name
        self._model = model

    async def list(self, uid: str) -> builtins.list[T]:
        return [self._model.model_validate(d) for d in await get_storage().list(self._name, uid)]

    async def add(self, uid: str, value: T) -> str:
        """Store a new item under a generated id and return it."""
        item_id = uuid4().hex
        await get_storage().write(self._name, uid, item_id, value.model_dump(mode='json', by_alias=True))
        return item_id

    async def get(self, uid: str, item_id: str) -> T | None:
        raw = await get_storage().read(self._name, uid, item_id)
        return self._model.model_validate(raw) if raw else None

    async def save(self, uid: str, item_id: str, value: T) -> None:
        """Store an item under a caller-chosen id (upsert)."""
        await get_storage().write(self._name, uid, item_id, value.model_dump(mode='json', by_alias=True))

    async def delete(self, uid: str, item_id: str) -> None:
        await get_storage().delete(self._name, uid, item_id)


def user_doc(name: str, model: type[T]) -> UserDoc[T]:
    """Declare a one-per-user store for ``model`` under collection ``name``."""
    return UserDoc(name, model)


def user_collection(name: str, model: type[T]) -> UserCollection[T]:
    """Declare a many-per-user store for ``model`` under collection ``name``."""
    return UserCollection(name, model)
