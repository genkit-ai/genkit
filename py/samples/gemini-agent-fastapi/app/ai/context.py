# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Reading the signed-in user inside a tool.

The chat ``context_provider`` (see ``app.api.chat``) verifies the caller and puts
their uid on the tool context before the turn runs. Tools call ``current_uid`` to
get it, instead of reaching into the raw context dict — so per-user tools all look
the same and can't disagree on where the uid lives.
"""

from __future__ import annotations

from genkit import ToolRunContext


def current_uid(ctx: ToolRunContext) -> str | None:
    """The signed-in user's id for this turn, or ``None`` if unauthenticated."""
    uid = ctx.context.get('uid') if ctx.context else None
    return str(uid) if uid else None
