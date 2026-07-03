# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The signed-in user for the current request, carried as ambient context.

The agent's session store is a single shared instance, but on every read and write
it needs to know *whose* namespace to use — that's what stops one user from reaching
another's conversations even if they guess a session id. The store can't take the
uid as an argument because the agent calls into it deep inside a turn, so it reads
this value lazily instead (``FirestoreSessionStore(snapshot_path_prefix=get_tenant_uid)``).

It's a ``ContextVar`` rather than a module global because the server handles many
requests at once: a global would let one request's user overwrite another's
mid-flight and leak data across tenants. A ``ContextVar`` is isolated per async
request, so each turn only ever sees its own user.
"""

from __future__ import annotations

from contextvars import ContextVar

_tenant_uid: ContextVar[str] = ContextVar('tenant_uid', default='global')


def set_tenant_uid(uid: str) -> None:
    """Pin the current request to a user.

    Called once at the auth boundary — ``with_tenant`` for REST routes, the chat
    ``context_provider`` for the agent — before anything touches the store.
    """
    _tenant_uid.set(uid)


def get_tenant_uid() -> str:
    """The current request's user, read by the session store when it builds a path.

    Defaults to ``'global'`` when no user has been set, e.g. local runs without auth.
    """
    return _tenant_uid.get()
