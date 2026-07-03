# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Load the signed-in user's profile from the datastore."""

from __future__ import annotations

from pydantic import BaseModel

from app.ai import ai
from app.ai.context import current_uid
from app.data import profiles
from app.models import UserProfile
from genkit import ToolRunContext


class GetMyProfileInput(BaseModel):
    """No arguments — the signed-in user is taken from request auth."""


class GetMyProfileOutput(BaseModel):
    found: bool = False
    profile: UserProfile | None = None


@ai.tool(
    name='getMyProfile',
    description="Load the signed-in user's profile (name, plan, company, support tier).",
)
async def get_my_profile(_input: GetMyProfileInput, ctx: ToolRunContext) -> GetMyProfileOutput:
    uid = current_uid(ctx)
    if not uid:
        return GetMyProfileOutput()

    profile = await profiles.get(uid)
    if profile is None:
        return GetMyProfileOutput(found=False)

    return GetMyProfileOutput(found=True, profile=profile)
