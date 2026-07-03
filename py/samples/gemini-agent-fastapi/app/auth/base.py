# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Auth interface — the contract every identity provider implements.

The app only ever holds an ``AuthUser`` and an ``AuthProvider``; it never knows
whether the token came from Firebase, a signed dev JWT, Clerk, or Supabase. A new
provider implements ``verify_token`` (and, if it supports server-side password
login for local dev, ``password_login``) and nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class AuthUser(BaseModel):
    """The signed-in caller — the only identity shape the rest of the app sees."""

    uid: str
    email: str | None = None
    token: dict[str, Any] | None = None


class IssuedToken(BaseModel):
    """A freshly minted token plus the user it authenticates."""

    access_token: str
    user: AuthUser


class AuthError(Exception):
    """Provider-level auth failure. The API layer maps this to HTTP 401."""

    def __init__(self, detail: str = 'Invalid or missing credentials') -> None:
        self.detail = detail
        super().__init__(detail)


class AuthProvider(ABC):
    """How the app turns a bearer token into a user, independent of the vendor."""

    @abstractmethod
    async def verify_token(self, raw_token: str | None) -> AuthUser:
        """Validate a bearer token and return the caller, or raise ``AuthError``."""

    async def password_login(self, email: str, password: str) -> IssuedToken | None:
        """Exchange credentials for a token, for zero-config local dev.

        Hosted providers sign users in from the client SDK, so they return ``None``
        here and the login route reports that server-side login isn't available.
        """
        return None
