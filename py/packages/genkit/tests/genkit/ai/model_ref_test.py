#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for model_ref and ModelRef."""

import pytest
from pydantic import BaseModel, ValidationError

from genkit._ai._prompt import PromptConfig
from genkit.model import ModelConfig, model_ref


class CustomConfig(BaseModel):
    """Plugin-specific config for typing tests."""

    temperature: float | None = None
    safety_settings: dict[str, str] | None = None


def test_model_ref_stamps_namespace_and_schema() -> None:
    """Namespace prefixing and config_schema are stamped on the ref."""
    ref = model_ref('gemini-pro-latest', namespace='googleai', config_schema=ModelConfig)
    assert ref.name == 'googleai/gemini-pro-latest'
    assert ref.config_schema is ModelConfig


def test_model_ref_idempotent_namespace() -> None:
    """Already-prefixed names are not double-namespaced."""
    ref = model_ref('googleai/gemini-pro-latest', namespace='googleai')
    assert ref.name == 'googleai/gemini-pro-latest'


def test_model_ref_is_frozen() -> None:
    """ModelRef instances cannot be mutated after creation."""
    ref = model_ref('x', namespace='ns')
    with pytest.raises(ValidationError):
        ref.name = 'y'  # pyrefly: ignore[read-only]


def test_model_ref_stamps_typed_config() -> None:
    """Default config on the ref keeps the plugin config type."""
    config = CustomConfig(temperature=0.7, safety_settings={'HARM': 'BLOCK'})
    ref = model_ref('gemini-pro-latest', namespace='googleai', config_schema=CustomConfig, config=config)
    assert ref.config is config
    assert ref.config_schema is CustomConfig
    assert ref.config is not None
    assert ref.config.temperature == 0.7


def test_prompt_config_keeps_plugin_specific_fields() -> None:
    pc = PromptConfig(config=CustomConfig(temperature=0.7, safety_settings={'HARM': 'BLOCK'}))
    dumped = pc.model_dump()['config']
    # safety_settings isn't part of the common config; without SerializeAsAny it
    # would be dropped when a config flows through an executable prompt.
    assert dumped['safety_settings'] == {'HARM': 'BLOCK'}
    assert dumped['temperature'] == 0.7
