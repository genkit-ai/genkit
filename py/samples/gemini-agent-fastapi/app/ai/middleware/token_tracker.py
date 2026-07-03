# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Token accounting middleware — logs model usage per user for billing."""

from __future__ import annotations

from app.ai import ai
from app.data import usage
from app.models import TokenUsageRecord
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext


@ai.middleware(name='token_tracker')
class TokenAccountingMiddleware(BaseMiddleware):
    """Record input/output token burn after each model call."""

    async def wrap_model(self, params, ctx: GenerateMiddlewareContext, next_fn):
        response = await next_fn(params, ctx)
        auth = ctx.custom_context.get('auth') or {}
        uid = auth.get('uid') if isinstance(auth, dict) else ctx.custom_context.get('uid')
        if uid and response.usage:
            await usage.add(
                str(uid),
                TokenUsageRecord.create(
                    input_tokens=response.usage.input_tokens or 0,
                    output_tokens=response.usage.output_tokens or 0,
                    model=params.model,
                    session_id=str(ctx.custom_context.get('session_id') or '') or None,
                ),
            )
        return response
