# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The session list shown in the UI — one row per conversation, per user.

Updated after each turn. The messages themselves live in the agent snapshot store
(``agent_sessions``); this is only the lightweight index used to list and resume.
It's a plain module over ``user_collection`` — the merge logic below is all there
is to it.
"""

from __future__ import annotations

from app.data.store import user_collection
from app.models import SessionSummary, utcnow_iso

_rows = user_collection('session_index', SessionSummary)


async def upsert(
    uid: str,
    session_id: str,
    *,
    title: str,
    snapshot_id: str | None,
    message_count: int,
) -> None:
    """Record or update one session's summary, keeping its first title and created time."""
    prev = await _rows.get(uid, session_id)
    await _rows.save(
        uid,
        session_id,
        SessionSummary(
            sessionId=session_id,
            title=prev.title if prev else title[:80],
            createdAt=prev.created_at if prev else utcnow_iso(),
            updatedAt=utcnow_iso(),
            messageCount=message_count,
            snapshotId=snapshot_id,
        ),
    )


async def list_for(uid: str) -> list[SessionSummary]:
    """A user's sessions, newest first."""
    rows = await _rows.list(uid)
    return sorted(rows, key=lambda row: row.updated_at or '', reverse=True)


async def get_one(uid: str, session_id: str) -> SessionSummary | None:
    return await _rows.get(uid, session_id)
