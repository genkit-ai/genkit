# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Where the agent persists conversation snapshots.

This is genkit's own session store — distinct from ``session_index`` (the
lightweight list for the UI). The agent reads and writes full conversation state
here so a user can resume or rewind a chat. It's namespaced per user via the
tenant prefix (see ``app.core.tenant``), so one user can't reach another's
conversations even by guessing a session id.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.core.tenant import get_tenant_uid
from genkit.plugins.google_cloud.session_store import FirestoreSessionStore


@lru_cache
def get_agent_session_store() -> FirestoreSessionStore:
    """The agent's snapshot store, bound to the configured datastore.

    Like the rest of the data layer there's no in-memory fallback: conversations
    that vanish on restart are a trap. Point at a real datastore — a cloud project
    or the Firestore emulator locally.
    """
    if not get_settings().use_firestore:
        raise RuntimeError(
            'No datastore configured. Set FIREBASE_PROJECT_ID (deployed) or '
            'FIRESTORE_EMULATOR_HOST (local dev) — see the README quickstart.'
        )
    return FirestoreSessionStore(snapshot_path_prefix=get_tenant_uid)
