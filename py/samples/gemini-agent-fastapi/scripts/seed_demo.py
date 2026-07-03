# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Seed the demo user's data into the configured datastore.

Run once after starting the Firestore emulator (or against a real project) so the
quickstart shows real per-user data — the copilot can greet the demo user by name
and list their orders. Idempotent: rerunning overwrites the same documents.

    uv run python scripts/seed_demo.py
"""

from __future__ import annotations

import asyncio

from app.data import orders, profiles
from app.models import DEMO_ORDERS, DEMO_PROFILE


async def main() -> None:
    uid = DEMO_PROFILE.uid
    await profiles.save(uid, DEMO_PROFILE)
    for order in DEMO_ORDERS:
        await orders.save(uid, order.id, order)
    print(f'Seeded profile + {len(DEMO_ORDERS)} orders for {DEMO_PROFILE.email} (uid={uid}).')


if __name__ == '__main__':
    asyncio.run(main())
