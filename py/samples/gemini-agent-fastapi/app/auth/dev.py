# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Dev backend — signed HS256 tokens and demo password login.

The zero-config default so quickstarts can sign in and stream a turn without any
identity vendor set up. Not for production: credentials are hardcoded and tokens
are signed with a shared secret.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt

from app.auth.base import AuthError, AuthProvider, AuthUser, IssuedToken
from app.core.config import get_settings
from app.core.identity import uid_from_email

# Demo credentials for local dev — replace with a real provider in production.
_DEV_USERS: dict[str, str] = {
    'demo@example.com': 'demo1234',
}


class DevJwtProvider(AuthProvider):
    async def verify_token(self, raw_token: str | None) -> AuthUser:
        if not raw_token:
            raise AuthError('Missing bearer token')
        try:
            decoded = jwt.decode(raw_token, get_settings().jwt_secret, algorithms=['HS256'])
        except jwt.PyJWTError as exc:
            raise AuthError('Invalid token') from exc
        uid = decoded.get('sub')
        if not uid:
            raise AuthError('Invalid token payload')
        return AuthUser(uid=str(uid), email=decoded.get('email'), token=decoded)

    async def password_login(self, email: str, password: str) -> IssuedToken | None:
        email = email.lower()
        expected = _DEV_USERS.get(email)
        if expected is None or expected != password:
            raise AuthError('Invalid email or password')

        uid = uid_from_email(email)
        now = datetime.now(timezone.utc)
        access_token = jwt.encode(
            {'sub': uid, 'email': email, 'iat': now, 'exp': now + timedelta(days=7)},
            get_settings().jwt_secret,
            algorithm='HS256',
        )
        return IssuedToken(access_token=access_token, user=AuthUser(uid=uid, email=email))
