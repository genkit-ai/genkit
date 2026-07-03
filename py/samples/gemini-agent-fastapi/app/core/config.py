# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Application settings and the one-time runtime bootstrap.

This is the single source of truth for env: everything else reads settings from
here instead of touching ``os.environ`` or calling ``load_dotenv`` on its own.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the API."""

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    gemini_api_key: str | None = None
    firebase_project_id: str | None = None
    jwt_secret: str = 'dev-only-change-me'
    cors_origins: str = '*'
    firestore_collection: str = 'gemini-agent'
    port: int = 8000

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == '*':
            return ['*']
        return [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]

    @property
    def use_firestore(self) -> bool:
        return bool(self.firebase_project_id or os.environ.get('FIRESTORE_EMULATOR_HOST'))


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def bootstrap_env() -> None:
    """Load ``.env`` and make the Gemini key visible to the SDK, exactly once.

    The Genkit runtime reads ``GEMINI_API_KEY`` from the environment when it boots,
    so if the key came from settings (a ``.env`` entry or a mounted secret) we copy
    it across before anything constructs the client.
    """
    load_dotenv()
    settings = get_settings()
    if settings.gemini_api_key:
        os.environ.setdefault('GEMINI_API_KEY', settings.gemini_api_key)
