# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Auth: a swappable identity provider behind one interface.

The app asks this factory for a provider and gets back an ``AuthProvider`` — never
a specific vendor. Choosing the backend happens here and nowhere else, so switching
identity providers is a change to this file plus one module implementing ``base.py``.
"""

from __future__ import annotations

import os
from functools import lru_cache

from app.auth.base import AuthError, AuthProvider, AuthUser, IssuedToken
from app.core.config import get_settings


def _use_firebase() -> bool:
    return bool(get_settings().firebase_project_id) and not os.environ.get('AUTH_DEV_MODE')


@lru_cache
def get_auth_provider() -> AuthProvider:
    if _use_firebase():
        from app.auth.firebase import FirebaseProvider

        return FirebaseProvider()
    from app.auth.dev import DevJwtProvider

    return DevJwtProvider()


__all__ = [
    'AuthError',
    'AuthProvider',
    'AuthUser',
    'IssuedToken',
    'get_auth_provider',
]
