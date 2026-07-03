# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Stable user id helpers shared across auth, tools, and data stores."""


def uid_from_email(email: str) -> str:
    """Map an email to a stable uid for dev JWT and demo data."""
    return email.strip().lower().replace('@', '_at_').replace('.', '_')
