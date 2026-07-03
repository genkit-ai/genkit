# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Login route — delegates to the configured auth provider.

Verification and vendor specifics live in ``app.auth``; this file is just the HTTP
surface for the dev convenience login. Real deployments sign users in with their
provider's client SDK and send the ID token on every call, so this route reports
501 when a server-side password login isn't available.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.auth import AuthError, get_auth_provider

router = APIRouter()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=4)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    uid: str
    email: str | None = None


@router.post('/login', response_model=LoginResponse)
async def login(body: LoginRequest) -> LoginResponse:
    """Exchange demo credentials for a bearer token (local dev only)."""
    try:
        issued = await get_auth_provider().password_login(body.email, body.password)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.detail) from exc

    if issued is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail='Server-side login is disabled for this provider; sign in with its client SDK.',
        )

    return LoginResponse(access_token=issued.access_token, uid=issued.user.uid, email=issued.user.email)
