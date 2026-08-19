#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for veneer model resolution helpers.

Covers normalize_config, resolve_model_arg, resolve_model_name, and
resolve_model_ref as pure functions, independent of generate()/prompt wiring.
"""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel, Field

from genkit._ai._model import (
    ModelConfig,
    ResolvedModel,
    normalize_config,
    resolve_call_model,
    resolve_model_arg,
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


class OtherFamilyConfig(BaseModel):
    """A second family's knobs, used to pin leftover keys across a hop."""

    temperature: float | None = None
    frequency_penalty: float | None = None


class NestedConfig(BaseModel):
    """Nested bag used to pin shallow replace."""

    thinking: dict[str, object] | None = None


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


def test_normalize_config_rejects_unsupported_type() -> None:
    """A leftover int is INVALID_ARGUMENT, same class of reject as model=123."""
    with pytest.raises(GenkitError, match='config is int, expected Mapping or BaseModel'):
        normalize_config(config=123)


def test_resolve_model_name_raises_when_default_is_not_string() -> None:
    """A configured default of the wrong type says so, rather than 'not configured'."""
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', 123)
    with pytest.raises(GenkitError, match='defaultModel is int, expected str or ModelRef'):
        resolve_model_name(model=None, registry=registry)


def test_resolve_model_name_empty_string_falls_back_to_default() -> None:
    """model='' is omitted, so a constructor default still applies."""
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', 'default-model')
    assert resolve_model_name(model='', registry=registry) == 'default-model'


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


def test_normalize_config_passes_through_camel_case_keys() -> None:
    """Dict keys stay as written; the plugin config schema decides spelling."""
    assert normalize_config(config={'maxOutputTokens': 100}) == {'maxOutputTokens': 100}


def test_normalize_config_accepts_snake_case_keys() -> None:
    """snake_case dict keys pass through unchanged."""
    assert normalize_config(config={'max_output_tokens': 100}) == {'max_output_tokens': 100}


def test_resolve_model_name_unwraps_model_ref_default() -> None:
    """A constructor ModelRef stored as defaultModel resolves to its wire name."""
    registry = Registry()
    ref = model_ref('echo-model', config_schema=CustomConfig, config=CustomConfig(temperature=0.7))
    registry.register_value('defaultModel', 'defaultModel', ref)
    assert resolve_model_name(model=None, registry=registry) == 'echo-model'


def test_resolve_model_name_unwraps_explicit_model_ref() -> None:
    """An explicit ModelRef argument resolves to its wire name."""
    ref = model_ref('echo-model', namespace='googleai', config_schema=CustomConfig)
    assert resolve_model_name(model=ref, registry=Registry()) == 'googleai/echo-model'


def test_resolve_model_arg_returns_default_model_ref() -> None:
    """resolve_model_arg keeps the stored ModelRef so callers can merge its config."""
    registry = Registry()
    ref = model_ref('echo-model', config_schema=CustomConfig, config=CustomConfig(temperature=0.7))
    registry.register_value('defaultModel', 'defaultModel', ref)
    assert resolve_model_arg(model=None, registry=registry) is ref


def test_resolve_call_model_merges_constructor_model_ref() -> None:
    """A stored constructor ref contributes version and config when model= is omitted."""
    registry = Registry()
    ref = model_ref(
        'echo-model',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(temperature=0.7),
    )
    registry.register_value('defaultModel', 'defaultModel', ref)
    resolved = resolve_call_model(model=None, config={}, registry=registry)
    assert resolved.name == 'echo-model'
    assert resolved.config == {'version': '001', 'temperature': 0.7}


def test_resolve_call_model_call_time_config_wins() -> None:
    """Call-time config overlays the constructor ref per key."""
    registry = Registry()
    ref = model_ref(
        'echo-model',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(temperature=0.7),
    )
    registry.register_value('defaultModel', 'defaultModel', ref)
    resolved = resolve_call_model(model=None, config={'temperature': 0.2}, registry=registry)
    assert resolved.config == {'version': '001', 'temperature': 0.2}


def test_resolve_model_ref_keeps_explicit_empty_values() -> None:
    """``0``, ``''``, ``[]``, ``{}``, and ``False`` are values the caller set.

    Only ``None`` means omit the field. Clearing stop sequences is ``[]``,
    not dropping the key; a blank version is ``''``.
    """
    ref = model_ref(
        'm',
        config_schema=ModelConfig,
        config=ModelConfig(temperature=0.7, stop_sequences=['STOP'], version='001'),
    )
    resolved = resolve_model_ref(
        model=ref,
        config={
            'temperature': 0,
            'max_output_tokens': 0,
            'version': '',
            'stop_sequences': [],
            'thinking': {},
            'stream': False,
        },
    )
    assert resolved.config == {
        'temperature': 0,
        'max_output_tokens': 0,
        'version': '',
        'stop_sequences': [],
        'thinking': {},
        'stream': False,
    }


def test_resolve_model_ref_config_version_beats_ref_version() -> None:
    """ref.config.version overlays the constructor version= field."""
    ref = model_ref(
        'm',
        config_schema=ModelConfig,
        version='001',
        config=ModelConfig(version='002'),
    )
    assert resolve_model_ref(model=ref, config={}).config == {'version': '002'}


def test_normalize_config_passes_through_nested_camel_case_keys() -> None:
    """Nested dict keys stay as written, same as top-level ones."""
    assert normalize_config(config={'thinking': {'budgetTokens': 1024}}) == {'thinking': {'budgetTokens': 1024}}


def test_resolve_model_arg_rejects_non_name_explicit_model() -> None:
    """A leftover int must not silently run the constructor default."""
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', 'echo-model')
    with pytest.raises(GenkitError, match='model is int, expected str or ModelRef'):
        resolve_model_arg(model=123, registry=registry)


def test_resolve_call_model_string_path_omits_none() -> None:
    """None means omit on a name, same as after a ref merge."""
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', 'echo')
    resolved = resolve_call_model(
        model='echo',
        config={'temperature': None, 'top_k': 40},
        registry=registry,
    )
    assert resolved.name == 'echo'
    assert 'temperature' not in resolved.config
    assert resolved.config['top_k'] == 40


def test_resolve_call_model_explicit_string_ignores_default_ref_config() -> None:
    """An explicit name is a different model; constructor knobs stay off it."""
    registry = Registry()
    default = model_ref(
        'flash',
        config_schema=CustomConfig,
        config=CustomConfig(temperature=0.7, safety_settings={'HARM': 'BLOCK'}),
    )
    registry.register_value('defaultModel', 'defaultModel', default)
    resolved = resolve_call_model(model='openai/gpt', config={'temperature': 0.2}, registry=registry)
    assert resolved.name == 'openai/gpt'
    assert resolved.config == {'temperature': 0.2}


def test_resolve_model_ref_keeps_other_family_keys() -> None:
    """Switching models does not drop keys the new schema does not know.

    A prompt can still be holding a Gemini bag when the call picks OpenAI.
    Those leftovers stay; the helper does not know families.
    """
    openai = model_ref(
        'gpt',
        namespace='openai',
        config_schema=OtherFamilyConfig,
        config=OtherFamilyConfig(frequency_penalty=0.5),
    )
    resolved = resolve_model_ref(
        model=openai,
        config={'temperature': 0.7, 'safety_settings': {'HARM': 'BLOCK'}},
    )
    assert resolved.name == 'openai/gpt'
    assert resolved.config == {
        'frequency_penalty': 0.5,
        'temperature': 0.7,
        'safety_settings': {'HARM': 'BLOCK'},
    }


def test_resolve_model_ref_replaces_nested_bag() -> None:
    """A nested dict replaces the whole object. Inner keys are not deep-merged.

    Call-time ``thinking={'budget': 256}`` drops the ref's ``level``.
    """
    ref = model_ref(
        'm',
        config_schema=NestedConfig,
        config=NestedConfig(thinking={'budget': 1024, 'level': 'low'}),
    )
    resolved = resolve_model_ref(model=ref, config={'thinking': {'budget': 256}})
    assert resolved.config == {'thinking': {'budget': 256}}
