#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for veneer model resolution helpers.

Covers dump, fold, overlay, and resolve as pure functions, independent
of generate()/prompt wiring.
"""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from genkit._ai._model import (
    ModelConfig,
    ResolvedModel,
    fold_config_aliases,
    get_request_api_key,
    normalize_config,
    overlay_config,
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

    api_key: str | None = Field(default=None, exclude=True)


class OtherFamilyConfig(BaseModel):
    """A second family's knobs, used to pin leftover keys across a hop."""

    temperature: float | None = None
    frequency_penalty: float | None = None


class NestedConfig(BaseModel):
    """Nested bag used to pin shallow replace."""

    thinking: dict[str, object] | None = None


class AliasedNestedConfig(BaseModel):
    """Schema with a generated camel alias, used to pin alias fold."""

    model_config = ConfigDict(alias_generator=to_camel, extra='allow', populate_by_name=True)
    thinking_config: dict[str, object] | None = None


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


def test_resolve_model_name_empty_default_means_not_configured() -> None:
    """An empty constructor default is the same as having no default.

    A common setup is ``Genkit(model=os.getenv('MODEL') or '')``. When
    ``MODEL`` is unset, that stores an empty string as the instance
    default. A later ``generate(model='')`` (or ``generate()`` with no
    model) treats the call as omitted and looks up that default.

    An empty default is not a model name. The error is the same
    ``No model configured.`` you get when nothing was registered, not
    ``defaultModel is str``.
    """
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', '')
    with pytest.raises(GenkitError, match='No model configured'):
        resolve_model_name(model='', registry=registry)
    with pytest.raises(GenkitError, match='No model configured'):
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


def test_normalize_config_passes_through_camel_case_keys() -> None:
    """Dump does not fold. A dict's keys stay as written."""
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


def test_resolve_call_model_string_path_does_not_fold() -> None:
    """A name has no schema, so camel keys are not rewritten."""
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', 'echo')
    resolved = resolve_call_model(
        model='echo',
        config={'maxOutputTokens': 5},
        registry=registry,
    )
    assert resolved.config == {'maxOutputTokens': 5}


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


def test_fold_config_aliases_renames_to_field_name() -> None:
    """A camel dict key becomes the Python field name."""
    assert fold_config_aliases(config={'maxOutputTokens': 5}, schema=ModelConfig) == {'max_output_tokens': 5}


def test_fold_config_aliases_unknown_keys_pass_through() -> None:
    """Fold is a rename, not validation — extras keep their spelling."""
    assert fold_config_aliases(
        config={'maxOutputTokens': 5, 'fooBar': 1},
        schema=ModelConfig,
    ) == {'max_output_tokens': 5, 'fooBar': 1}


def test_overlay_config_folds_and_last_layer_wins() -> None:
    """Same slot across layers: later spelling replaces the earlier one."""
    assert overlay_config(
        layers=[{'max_output_tokens': 100}, {'maxOutputTokens': 5}],
        schema=ModelConfig,
    ) == {'max_output_tokens': 5}


def test_overlay_config_none_clears() -> None:
    """None after fold drops the field instead of leaving a sibling alias."""
    assert (
        overlay_config(
            layers=[{'max_output_tokens': 100}, {'maxOutputTokens': None}],
            schema=ModelConfig,
        )
        == {}
    )


def test_overlay_config_unknown_keys_pass_through() -> None:
    """Extras keep their spelling and still last-write-win."""
    assert overlay_config(
        layers=[{'fooBar': 1}, {'fooBar': 2, 'maxOutputTokens': 5}],
        schema=ModelConfig,
    ) == {'fooBar': 2, 'max_output_tokens': 5}


def test_resolve_model_ref_alias_overlay_replaces_field() -> None:
    """Call-time maxOutputTokens replaces the dumped max_output_tokens."""
    ref = model_ref('m', config_schema=ModelConfig, config=ModelConfig(max_output_tokens=100))
    resolved = resolve_model_ref(model=ref, config={'maxOutputTokens': 5})
    assert resolved.config == {'max_output_tokens': 5}


def test_resolve_model_ref_alias_none_clears_field() -> None:
    """Call-time maxOutputTokens=None clears the dumped ref default."""
    ref = model_ref('m', config_schema=ModelConfig, config=ModelConfig(max_output_tokens=100))
    resolved = resolve_model_ref(model=ref, config={'maxOutputTokens': None})
    assert 'max_output_tokens' not in resolved.config
    assert 'maxOutputTokens' not in resolved.config


def test_resolve_model_ref_alias_replaces_nested_bag() -> None:
    """thinkingConfig is the thinking_config slot. The call's bag replaces it whole.

    ``thinking_level`` and ``include_thoughts`` from the ref are gone. Inner
    keys stay as written — fold does not walk into the nested dict.
    """
    ref = model_ref(
        'm',
        config_schema=AliasedNestedConfig,
        config=AliasedNestedConfig(
            thinking_config={
                'include_thoughts': True,
                'thinking_budget': 1024,
                'thinking_level': 'low',
            }
        ),
    )
    resolved = resolve_model_ref(model=ref, config={'thinkingConfig': {'thinkingBudget': 256}})
    assert resolved.config == {'thinking_config': {'thinkingBudget': 256}}


def test_resolve_model_ref_alias_none_clears_nested_bag() -> None:
    """thinkingConfig=None clears the whole thinking_config slot."""
    ref = model_ref(
        'm',
        config_schema=AliasedNestedConfig,
        config=AliasedNestedConfig(thinking_config={'thinking_budget': 1024, 'thinking_level': 'low'}),
    )
    resolved = resolve_model_ref(model=ref, config={'thinkingConfig': None})
    assert 'thinking_config' not in resolved.config
    assert 'thinkingConfig' not in resolved.config


def test_resolve_model_ref_does_not_fold_inner_keys() -> None:
    """Both spellings inside a nested bag stay. Only the outer key folds."""
    ref = model_ref('m', config_schema=AliasedNestedConfig)
    resolved = resolve_model_ref(
        model=ref,
        config={'thinkingConfig': {'thinking_budget': 1, 'thinkingBudget': 256}},
    )
    assert resolved.config == {'thinking_config': {'thinking_budget': 1, 'thinkingBudget': 256}}


def test_resolve_model_ref_both_spellings_last_write_wins() -> None:
    """Both keys in one dict fold onto one field; later key wins."""
    ref = model_ref('m', config_schema=ModelConfig)
    camel_last = resolve_model_ref(
        model=ref,
        config={'max_output_tokens': 1, 'maxOutputTokens': 5},
    )
    snake_last = resolve_model_ref(
        model=ref,
        config={'maxOutputTokens': 5, 'max_output_tokens': 1},
    )
    assert camel_last.config == {'max_output_tokens': 5}
    assert snake_last.config == {'max_output_tokens': 1}


def test_get_request_api_key_reads_camel_dict() -> None:
    """A wire-shaped dict still exposes the per-request key."""
    assert get_request_api_key({'apiKey': 'secret'}) == 'secret'
    assert get_request_api_key({'api_key': 'secret'}) == 'secret'
