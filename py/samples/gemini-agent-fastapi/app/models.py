# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The app's domain types, in one place.

Everything the app stores or returns is one of these Pydantic models. Routes,
tools, and the data layer all speak in these types — open this file to see the
whole data model at a glance.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.core.identity import uid_from_email


def utcnow_iso() -> str:
    """A sortable UTC timestamp — used wherever a record needs a time."""
    return datetime.now(timezone.utc).isoformat()


class UserProfile(BaseModel):
    """Fields support and the copilot may repeat back to the signed-in user."""

    uid: str
    display_name: str = Field(alias='displayName')
    email: str
    plan: str
    member_since: str = Field(alias='memberSince')
    company: str | None = None
    timezone: str | None = None
    support_tier: str = Field(alias='supportTier')

    model_config = {'populate_by_name': True}


class TokenUsageRecord(BaseModel):
    """One model call's token burn — appended to a user's usage log for billing."""

    input_tokens: int = Field(alias='inputTokens')
    output_tokens: int = Field(alias='outputTokens')
    total_tokens: int = Field(alias='totalTokens')
    model: str | None = None
    session_id: str | None = Field(default=None, alias='sessionId')
    recorded_at: str = Field(alias='recordedAt')

    model_config = {'populate_by_name': True}

    @classmethod
    def create(
        cls,
        *,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
        session_id: str | None = None,
    ) -> TokenUsageRecord:
        """Build a record with the total and timestamp filled in."""
        return cls(
            inputTokens=input_tokens,
            outputTokens=output_tokens,
            totalTokens=input_tokens + output_tokens,
            model=model,
            sessionId=session_id,
            recordedAt=utcnow_iso(),
        )


class Order(BaseModel):
    """A purchase belonging to a user — the kind of business data a tool looks up."""

    id: str
    item: str
    status: str  # active | fulfilled | cancelled


class SessionSummary(BaseModel):
    """A row in the session list — enough to show and resume a conversation.

    The messages themselves live in the agent's snapshot store; this is the
    lightweight index used to list and open a conversation.
    """

    session_id: str = Field(alias='sessionId')
    title: str
    created_at: str | None = Field(default=None, alias='createdAt')
    updated_at: str | None = Field(default=None, alias='updatedAt')
    message_count: int = Field(default=0, alias='messageCount')
    snapshot_id: str | None = Field(default=None, alias='snapshotId')

    model_config = {'populate_by_name': True}


# The signed-in demo user, seeded into the datastore by scripts/seed_demo.py so
# the quickstart shows real per-user personalization.
DEMO_PROFILE = UserProfile(
    uid=uid_from_email('demo@example.com'),
    displayName='Demo User',
    email='demo@example.com',
    plan='Pro',
    memberSince='2025-11-01',
    company='Acme Analytics',
    timezone='America/Los_Angeles',
    supportTier='standard',
)

# Demo orders for the same user, seeded alongside the profile so listMyOrders
# returns real rows from the datastore in the quickstart.
DEMO_ORDERS = [
    Order(id='ord_1001', item='Pro Plan (monthly)', status='active'),
    Order(id='ord_1002', item='Token pack (500k)', status='fulfilled'),
]
