# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""List and load agent sessions."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import Field

from app.ai.agents.copilot import copilot_agent
from app.ai.messages import message_text
from app.api.deps import with_tenant
from app.auth import AuthUser
from app.data import session_index
from app.models import SessionSummary

router = APIRouter()


class SessionDetail(SessionSummary):
    messages: list[dict[str, object]] = Field(default_factory=list)


@router.get('', response_model=list[SessionSummary])
async def list_sessions(user: Annotated[AuthUser, Depends(with_tenant)]) -> list[SessionSummary]:
    """Return session summaries for the signed-in user.

    Summaries come from a listing index that's kept fresh after each turn. That
    post-turn update is paused until the agent framework grows a turn lifecycle
    hook (see ``app/api/chat.py``), so this currently returns an empty list.
    Message history still loads per session via ``GET /sessions/{session_id}``.
    """
    return await session_index.list_for(user.uid)


@router.get('/{session_id}', response_model=SessionDetail)
async def get_session(
    session_id: str,
    user: Annotated[AuthUser, Depends(with_tenant)],
) -> SessionDetail:
    """Load one session with full message history from the agent snapshot."""
    snapshot = await copilot_agent.get_snapshot(session_id=session_id)
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Session not found')

    message_count = len(snapshot.state.messages or []) if snapshot.state else 0
    summary = await session_index.get_one(user.uid, session_id) or SessionSummary(
        sessionId=session_id,
        title='Conversation',
        messageCount=message_count,
        snapshotId=snapshot.snapshot_id,
    )

    messages: list[dict[str, object]] = []
    if snapshot.state is not None and snapshot.state.messages:
        for msg in snapshot.state.messages:
            messages.append({'role': msg.role, 'content': message_text(msg)})

    return SessionDetail(**summary.model_dump(by_alias=True), messages=messages)
