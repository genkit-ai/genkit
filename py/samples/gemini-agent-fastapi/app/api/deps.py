# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Shared FastAPI dependencies — auth and tenant scoping for plain REST routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth import AuthError, AuthUser, get_auth_provider
from app.core.tenant import set_tenant_uid

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AuthUser:
    """Resolve the signed-in user from the bearer token, via the auth provider."""
    token = credentials.credentials if credentials else None
    try:
        return await get_auth_provider().verify_token(token)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.detail) from exc


async def with_tenant(user: Annotated[AuthUser, Depends(get_current_user)]) -> AuthUser:
    """Authenticate and scope the session store to this user for the request."""
    set_tenant_uid(user.uid)
    return user
