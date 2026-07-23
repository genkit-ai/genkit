#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for model_ref and ModelRef."""

import pytest
from pydantic import BaseModel

from genkit._ai._prompt import PromptConfig, normalize_config, resolve_model_arg
from genkit.model import model_ref
from genkit.plugin_api import ModelConfig


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
    ref = model_ref('googleai/gemini-pro-latest', namespace='googleai', config_schema=ModelConfig)
    assert ref.name == 'googleai/gemini-pro-latest'


def test_model_ref_is_frozen() -> None:
    """ModelRef instances cannot be mutated after creation."""
    from dataclasses import FrozenInstanceError
    from typing import Any, cast

    ref = model_ref('x', namespace='ns', config_schema=ModelConfig)
    with pytest.raises(FrozenInstanceError):
        cast(Any, ref).name = 'y'


def test_model_ref_requires_config_schema() -> None:
    """model_ref rejects calls that omit config_schema."""
    with pytest.raises(TypeError):
        model_ref('gemini-pro-latest', namespace='googleai')  # type: ignore  # pyrefly: ignore  # pyright: ignore


def test_model_ref_stamps_typed_config() -> None:
    """Default config on the ref keeps the plugin config type."""
    config = CustomConfig(temperature=0.7, safety_settings={'HARM': 'BLOCK'})
    ref = model_ref('gemini-pro-latest', namespace='googleai', config_schema=CustomConfig, config=config)
    assert ref.config is config
    assert ref.config_schema is CustomConfig
    assert ref.config is not None
    assert ref.config.temperature == 0.7


def test_model_ref_config_merging() -> None:
    """Call-time config merges over ModelRef default config instead of overwriting."""
    ref = model_ref(
        'gemini-pro-latest',
        namespace='googleai',
        config_schema=CustomConfig,
        config=CustomConfig(temperature=0.7, safety_settings={'HARM': 'BLOCK'}),
    )
    name, resolved_cfg = resolve_model_arg(ref, CustomConfig(temperature=0.2))
    assert name == 'googleai/gemini-pro-latest'
    assert isinstance(resolved_cfg, dict)
    assert resolved_cfg['temperature'] == 0.2
    assert resolved_cfg['safety_settings'] == {'HARM': 'BLOCK'}


def test_prompt_config_keeps_plugin_specific_fields() -> None:
    pc = PromptConfig(config=normalize_config(CustomConfig(temperature=0.7, safety_settings={'HARM': 'BLOCK'})))
    dumped = pc.model_dump()['config']
    assert dumped['safety_settings'] == {'HARM': 'BLOCK'}
    assert dumped['temperature'] == 0.7


def test_define_prompt_with_typed_model_ref() -> None:
    from genkit import Genkit

    ai = Genkit()
    ref = model_ref('gemini-pro-latest', config_schema=CustomConfig)
    prompt = ai.define_prompt(
        'test_prompt',
        model=ref,
        config=CustomConfig(temperature=0.5),
        prompt='Hello',
    )
    assert prompt is not None
