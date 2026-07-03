# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Firebase Auth backend.

All Firebase knowledge lives here. Users sign in with the Firebase client SDK and
send the resulting ID token on every call; this provider verifies it. To move to
another vendor, add a sibling module implementing ``AuthProvider`` and point the
factory in ``__init__.py`` at it — deleting this file is the whole job.
"""

from __future__ import annotations

import os

from app.auth.base import AuthError, AuthProvider, AuthUser
from app.core.config import get_settings


def _ensure_app() -> None:
    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._apps:
        return
    project = get_settings().firebase_project_id
    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    cred = credentials.Certificate(cred_path) if cred_path else credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {'projectId': project})


class FirebaseProvider(AuthProvider):
    async def verify_token(self, raw_token: str | None) -> AuthUser:
        if not raw_token:
            raise AuthError('Missing bearer token')
        _ensure_app()
        from firebase_admin import auth

        try:
            decoded = auth.verify_id_token(raw_token)
        except Exception as exc:  # noqa: BLE001
            raise AuthError('Invalid Firebase token') from exc
        return AuthUser(uid=decoded['uid'], email=decoded.get('email'), token=decoded)
