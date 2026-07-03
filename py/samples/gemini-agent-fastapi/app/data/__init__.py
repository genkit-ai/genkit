# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The data layer, at a glance.

Everything the app stores is declared here in one line each, so you can see the
whole data model on one screen. Two shapes: ``user_doc`` for one-per-user records,
``user_collection`` for many-per-user. Both are typed and scoped to the signed-in
user; the actual database lives in ``store.py`` + ``backends/``.

Add a feature by adding a model in ``app.models`` and one line below.
"""

from __future__ import annotations

from app.data.agent_sessions import get_agent_session_store
from app.data.store import user_collection, user_doc
from app.models import Order, TokenUsageRecord, UserProfile

# One document per user.
profiles = user_doc('profiles', UserProfile)

# Many documents per user.
orders = user_collection('orders', Order)
usage = user_collection('usage', TokenUsageRecord)

__all__ = [
    'get_agent_session_store',
    'orders',
    'profiles',
    'usage',
]
