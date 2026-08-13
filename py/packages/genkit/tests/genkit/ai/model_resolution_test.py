#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for veneer model resolution helpers.

Covers normalize_config, resolve_model_name, and resolve_model_ref as pure
functions, independent of generate()/prompt wiring.
"""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel, Field

from genkit._ai._model import (
    ModelConfig,
    ResolvedModel,
    normalize_config,
    resolve_model_name,
    resolve_model_ref,
)
from genkit._core._error import GenkitError
from genkit._core._registry import Registry
from genkit.model import model_ref


class CustomConfig(BaseModel):
    """Plugin-style config used for merge tests."""

    temperature: float | None = None
    top_k: float | None = None
    safety_settings: dict[str, str] | None = None


class ExcludedKeyConfig(ModelConfig):
    """ModelConfig whose api_key is omitted from model_dump."""

    api_key: str | None = Field(None, exclude=True)


def test_resolve_model_ref_merges_without_overwrite() -> None:
    """resolve_model_ref keeps ref-only keys and lets call-time keys win."""
    ref = model_ref(
        'gemini-pro-latest',
        namespace='googleai',
        config_schema=CustomConfig,
        config=CustomConfig(temperature=0.7, safety_settings={'HARM': 'BLOCK'}),
    )

    resolved = resolve_model_ref(model=ref, config={'temperature': 0.2})

    assert resolved.name == 'googleai/gemini-pro-latest'
    assert resolved.config['temperature'] == 0.2
    assert resolved.config['safety_settings'] == {'HARM': 'BLOCK'}


def test_normalize_config_dumps_pydantic() -> None:
    """normalize_config turns configs into plain dicts, using {} for None."""
    assert normalize_config(config=ModelConfig(temperature=0.5)) == {'temperature': 0.5}
    assert normalize_config(config=None) == {}


def test_resolve_model_ref_strips_explicit_none() -> None:
    """Post-merge None-strip: cleared keys are absent from the resolved config."""
    ref = model_ref(
        'gemini-pro-latest',
        namespace='googleai',
        config_schema=CustomConfig,
        config=CustomConfig(temperature=0.7, top_k=40),
    )

    resolved = resolve_model_ref(model=ref, config={'temperature': None})

    assert 'temperature' not in resolved.config
    assert resolved.config['top_k'] == 40


def test_resolve_model_name_prefers_explicit() -> None:
    """An explicit model name wins over any registry default."""
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', 'default-model')
    assert resolve_model_name(model='explicit', registry=registry) == 'explicit'


def test_resolve_model_name_falls_back_to_registry_default() -> None:
    """With no explicit name, the registry defaultModel value is used."""
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', 'default-model')
    assert resolve_model_name(model=None, registry=registry) == 'default-model'


def test_resolve_model_name_raises_with_custom_message() -> None:
    """No explicit name and no default raises INVALID_ARGUMENT with the given message."""
    with pytest.raises(GenkitError, match='No model specified for generate_operation.'):
        resolve_model_name(
            model=None,
            registry=Registry(),
            message='No model specified for generate_operation.',
        )


def test_normalize_config_excludes_unset_fields() -> None:
    """Pydantic fields the caller never set stay out of the merge entirely."""
    assert normalize_config(config=CustomConfig(temperature=0.7)) == {'temperature': 0.7}


def test_normalize_config_preserves_explicit_none() -> None:
    """An explicitly-set None survives normalization so it can clear lower layers."""
    assert normalize_config(config=CustomConfig(temperature=None)) == {'temperature': None}


def test_resolve_model_ref_version_lowest_precedence() -> None:
    """ref.version seeds the config but is overridden by ref config and call config."""
    ref = model_ref('m1', config_schema=CustomConfig, version='001')
    assert resolve_model_ref(model=ref, config={}).config == {'version': '001'}
    assert resolve_model_ref(model=ref, config={'version': '002'}).config == {'version': '002'}


def test_resolved_model_is_frozen() -> None:
    """ResolvedModel is immutable once constructed."""
    resolved = ResolvedModel(name='m', config={})
    with pytest.raises(FrozenInstanceError):
        resolved.name = 'other'  # type: ignore[misc]


def test_normalize_config_raises_type_error_for_unsupported_type() -> None:
    """normalize_config raises TypeError if the config is not None, BaseModel, or Mapping."""
    with pytest.raises(TypeError, match='Unsupported config type'):
        normalize_config(config=123)


def test_resolve_model_name_raises_when_default_is_not_string() -> None:
    """resolve_model_name raises GenkitError if the default model is not a string."""
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', 123)
    with pytest.raises(GenkitError, match='No model configured.'):
        resolve_model_name(model=None, registry=registry)


def test_normalize_config_preserves_explicit_none_on_model_config() -> None:
    """GenkitModel dump must keep an explicit None so merge can clear defaults."""
    assert normalize_config(config=ModelConfig(temperature=None)) == {'temperature': None}


def test_normalize_config_keeps_python_field_names() -> None:
    """Aliased fields dump as snake_case so a later snake_case override hits the same key."""
    assert normalize_config(config=ModelConfig(max_output_tokens=100)) == {'max_output_tokens': 100}


def test_resolve_model_ref_model_config_none_clears_default() -> None:
    """ModelConfig(temperature=None) clears a ref default, not just a dict None."""
    ref = model_ref('m', config_schema=ModelConfig, config=ModelConfig(temperature=0.7))
    resolved = resolve_model_ref(
        model=ref,
        config=normalize_config(config=ModelConfig(temperature=None)),
    )
    assert 'temperature' not in resolved.config


def test_resolve_model_ref_same_key_override_on_aliased_field() -> None:
    """Call-time max_output_tokens replaces the ref's, rather than sitting beside maxOutputTokens."""
    ref = model_ref('m', config_schema=ModelConfig, config=ModelConfig(max_output_tokens=100))
    resolved = resolve_model_ref(model=ref, config={'max_output_tokens': 200})
    assert resolved.config == {'max_output_tokens': 200}


def test_normalize_config_restores_excluded_fields() -> None:
    """Fields marked exclude=True still reach the plugin (per-request api_key)."""
    assert normalize_config(config=ExcludedKeyConfig(api_key='secret')) == {'api_key': 'secret'}
