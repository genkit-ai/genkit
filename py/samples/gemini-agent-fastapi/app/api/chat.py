# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Chat routes — the copilot agent served over its client-facing HTTP routes.

``serve_agent`` builds the run-turn / getSnapshot / abort routes the
``AgentChat`` client speaks and hands back a router to mount. We supply the one
app-specific piece: who the caller is (``_auth_context``).

NOTE: keeping the session list (``app/data/session_index``) fresh needs a
post-turn seam — the finished turn is the only place the server-assigned
session id exists. That belongs in the agent framework as a turn lifecycle hook,
not in this transport, so it's intentionally not wired here for now. Until that
lands in core, ``GET /api/sessions`` returns an empty list; conversation history
still loads per session via ``GET /api/sessions/{id}``.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.ai.agents.copilot import copilot_agent
from app.auth import AuthError, get_auth_provider
from app.core.tenant import set_tenant_uid
from genkit.plugin_api import RequestData
from genkit.plugins.fastapi import serve_agent


async def _auth_context(request_data: RequestData[Request]) -> dict[str, object]:
    """Authenticate the request and scope the session store to the signed-in user."""
    header = request_data.request.headers.get('authorization') or ''
    token = header[7:] if header[:7].lower() == 'bearer ' else None
    try:
        user = await get_auth_provider().verify_token(token)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.detail) from exc
    set_tenant_uid(user.uid)
    return {'auth': user.model_dump(), 'uid': user.uid}


router = serve_agent(
    copilot_agent,
    base_path='/chat',
    context_provider=_auth_context,
)
