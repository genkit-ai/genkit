# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The copilot — a prompt-backed agent with session persistence."""

from __future__ import annotations

from app.ai import ai
from app.data import get_agent_session_store

copilot_agent = ai.define_prompt_agent(
    name='copilot',
    store=get_agent_session_store(),
)
