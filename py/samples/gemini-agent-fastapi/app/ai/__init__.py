# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Shared Genkit runtime and registered extensions."""

from __future__ import annotations

from pathlib import Path

from app.core.config import bootstrap_env
from genkit import Genkit
from genkit.plugins.google_genai import GoogleAI

# Make the Gemini key available before the client boots (see core.config).
bootstrap_env()

_PROMPT_DIR = Path(__file__).resolve().parent / 'prompts'

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-flash-latest',
    prompt_dir=_PROMPT_DIR,
)


def _register_extensions() -> None:
    from app.ai.middleware import token_tracker  # noqa: F401
    from app.ai.tools import user_orders, user_profile  # noqa: F401


_register_extensions()

__all__ = ['ai']
