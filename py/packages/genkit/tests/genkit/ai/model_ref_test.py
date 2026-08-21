#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelRef and model_ref()."""

import ast
import importlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import BaseModel

from genkit._core._error import GenkitError
from genkit.model import ModelConfigDict, ModelInfo, ModelRef, Supports, model_ref
from genkit.plugin_api import ModelConfig


def _from_import(module: str, name: str) -> object:
    """``from module import name`` — ImportError if the name is gone."""
    try:
        return getattr(importlib.import_module(module), name)
    except AttributeError as exc:
        raise ImportError(f'cannot import name {name!r} from {module!r}') from exc


class CustomConfig(BaseModel):
    """Plugin-specific configuration schema for testing typed ModelRef parameterization."""

    temperature: float | None = None
    top_p: float | None = None
    safety_settings: dict[str, str] | None = None


def test_model_ref_with_custom_pydantic_schema() -> None:
    """ModelRef parameterized with a custom Pydantic schema retains typed config_schema and config."""
    config = CustomConfig(temperature=0.7, top_p=0.9, safety_settings={'HARM': 'BLOCK_NONE'})
    ref = model_ref(
        'gemini-pro-latest',
        namespace='googleai',
        config_schema=CustomConfig,
        config=config,
    )

    assert isinstance(ref, ModelRef)
    assert ref.name == 'googleai/gemini-pro-latest'
    assert ref.config_schema is CustomConfig
    assert ref.config is not config
    assert ref.config == config
    assert ref.config is not None
    assert ref.config.temperature == 0.7
    assert ref.config.top_p == 0.9


def test_typed_ref_assignable_to_model_ref_base_model() -> None:
    """A ref built with a plugin config class is still usable as ModelRef[BaseModel]."""
    ref = model_ref('echo', config_schema=CustomConfig)
    as_base: ModelRef[BaseModel] = ref
    assert as_base is ref


def test_model_ref_with_bare_base_model_schema() -> None:
    """model_ref() accepts arbitrary BaseModel schemas, including bare BaseModel itself."""
    ref = model_ref('generic-model', config_schema=BaseModel)

    assert isinstance(ref, ModelRef)
    assert ref.name == 'generic-model'
    assert ref.config_schema is BaseModel
    assert ref.config is None


def test_model_ref_namespace_prefixing() -> None:
    """model_ref() prefixes namespace on names and is idempotent for already-prefixed names."""
    ref1 = model_ref('gemini-pro-latest', namespace='googleai', config_schema=ModelConfig)
    assert ref1.name == 'googleai/gemini-pro-latest'

    # Already prefixed: should not duplicate namespace
    ref2 = model_ref('googleai/gemini-pro-latest', namespace='googleai', config_schema=ModelConfig)
    assert ref2.name == 'googleai/gemini-pro-latest'


def test_model_ref_requires_explicit_config_schema() -> None:
    """model_ref() raises TypeError if config_schema keyword argument is missing."""
    with pytest.raises(TypeError):
        model_ref('gemini-pro-latest', namespace='googleai')  # type: ignore[call-arg]


def test_model_ref_immutability() -> None:
    """ModelRef is a frozen dataclass and disallows mutating attributes after creation."""
    ref = model_ref('custom-model', config_schema=CustomConfig)

    with pytest.raises(FrozenInstanceError):
        ref.name = 'changed'  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        ref.config = CustomConfig(temperature=0.1)  # type: ignore[misc]


def test_model_ref_dataclass_value_equality() -> None:
    """ModelRef instances support value-based equality comparison."""
    ref1 = model_ref('m1', config_schema=CustomConfig, config=CustomConfig(temperature=0.5))
    ref2 = model_ref('m1', config_schema=CustomConfig, config=CustomConfig(temperature=0.5))
    ref3 = model_ref('m1', config_schema=CustomConfig, config=CustomConfig(temperature=0.9))

    assert ref1 == ref2
    assert ref1 != ref3
    assert ref1 != 'm1'


def test_model_ref_version_equality() -> None:
    """ModelRef instances with different versions compare as not equal."""
    v1 = model_ref('m1', config_schema=CustomConfig, version='001')
    v2 = model_ref('m1', config_schema=CustomConfig, version='001')
    v3 = model_ref('m1', config_schema=CustomConfig, version='002')

    assert v1 == v2
    assert v1 != v3


def test_model_ref_is_unhashable() -> None:
    """ModelRef opts out of hashing so set/dict use doesn't fail only when config is set."""
    ref = model_ref('m1', config_schema=CustomConfig, config=CustomConfig(temperature=0.5))
    with pytest.raises(TypeError, match='unhashable type'):
        hash(ref)


def test_model_ref_invalid_config_type_raises() -> None:
    """ModelRef raises GenkitError(INVALID_ARGUMENT) when config is not an instance of config_schema."""
    expected = f'm1: config must be an instance of {CustomConfig.__module__}.CustomConfig, got builtins.dict'
    with pytest.raises(GenkitError, match=expected):
        model_ref('m1', config_schema=CustomConfig, config={'temperature': 0.7})  # type: ignore[arg-type]


def test_model_ref_invalid_config_schema_raises() -> None:
    """ModelRef raises GenkitError when config_schema is not a BaseModel subclass."""
    expected = 'm1: config_schema must be a BaseModel subclass, got builtins.dict'
    with pytest.raises(GenkitError, match=expected):
        model_ref('m1', config_schema={'type': 'object'})  # type: ignore[arg-type]


def test_model_ref_invalid_config_schema_does_not_leak_isinstance() -> None:
    """A dict schema fails as config_schema, not as isinstance() arg 2."""
    with pytest.raises(GenkitError, match='config_schema must be a BaseModel subclass'):
        model_ref(
            'm1',
            config_schema={'type': 'object'},  # type: ignore[arg-type]
            config=CustomConfig(temperature=0.7),
        )


def test_model_ref_invalid_info_type_raises() -> None:
    """ModelRef raises GenkitError when info is a dict instead of ModelInfo."""
    expected = f'm1: info must be an instance of {ModelInfo.__module__}.ModelInfo, got builtins.dict'
    with pytest.raises(GenkitError, match=expected):
        model_ref('m1', config_schema=CustomConfig, info={'not': 'a ModelInfo'})  # type: ignore[arg-type]


def test_model_ref_preserves_version_and_info_metadata() -> None:
    """model_ref() stamps version and ModelInfo metadata on the ModelRef instance."""
    info = ModelInfo(supports=Supports(multiturn=True, media=True))
    ref = model_ref(
        'veo-2',
        config_schema=BaseModel,
        namespace='googleai',
        version='001',
        info=info,
    )

    assert ref.name == 'googleai/veo-2'
    assert ref.version == '001'
    assert ref.info is not info
    assert ref.info == info
    assert ref.info is not None
    assert ref.info.supports is not None
    assert ref.info.supports.multiturn is True


def test_model_config_dict_accepts_common_knobs() -> None:
    """ModelConfigDict is the typed shape for config={...} literals."""
    config: ModelConfigDict = {
        'temperature': 0.5,
        'max_output_tokens': 256.0,
        'top_k': 40,
        'top_p': 0.9,
        'stop_sequences': ['END'],
        'version': '001',
        'api_key': 'test-key',
    }

    assert config['temperature'] == 0.5
    assert config['max_output_tokens'] == 256.0


def test_model_config_dict_accepts_none_clear() -> None:
    """None on a common knob is a valid literal (clears a ModelRef default)."""
    config: ModelConfigDict = {'temperature': None}

    assert config['temperature'] is None


def test_model_config_dict_keys_match_generation_common_config() -> None:
    """Autocomplete knobs stay in lockstep with the runtime schema."""
    assert set(ModelConfigDict.__annotations__) == set(ModelConfig.model_fields)


def test_public_config_import_contract() -> None:
    """App code gets ModelConfigDict; ModelConfig lives on plugin_api."""
    from genkit import ModelConfigDict as VeneerModelConfigDict, PromptGenerateOptions
    from genkit.plugin_api import ModelConfig as PluginModelConfig

    assert VeneerModelConfigDict is ModelConfigDict
    assert PluginModelConfig is ModelConfig

    opts: PromptGenerateOptions = {'config': {'temperature': None, 'banana': True}}
    assert opts['config'] == {'temperature': None, 'banana': True}

    with pytest.raises(ImportError, match='ModelConfig'):
        _from_import('genkit', 'ModelConfig')
    with pytest.raises(ImportError, match='ModelConfig'):
        _from_import('genkit.model', 'ModelConfig')


def test_in_tree_sources_do_not_import_model_config_from_veneer() -> None:
    """Plugin packages take ModelConfig from plugin_api, not genkit / genkit.model."""
    packages = Path(__file__).resolve().parents[4]
    forbidden = {('genkit', 'ModelConfig'), ('genkit.model', 'ModelConfig')}
    offenders: list[str] = []
    for path in packages.rglob('*.py'):
        if path.resolve() == Path(__file__).resolve():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            for alias in node.names:
                if (node.module, alias.name) in forbidden:
                    offenders.append(f'{path.relative_to(packages)}:{node.lineno}')
    assert offenders == []


def test_model_ref_isolates_caller_config_and_info() -> None:
    """Mutating the caller's config or info after construction does not change the ref."""
    config = CustomConfig(temperature=0.5, safety_settings={'HARM': 'BLOCK_NONE'})
    info = ModelInfo(label='before', supports=Supports(multiturn=True))
    ref = model_ref('m1', config_schema=CustomConfig, config=config, info=info)

    config.temperature = 0.9
    assert config.safety_settings is not None
    config.safety_settings['HARM'] = 'BLOCK_ALL'
    info.label = 'after'
    assert info.supports is not None
    info.supports.multiturn = False

    assert ref.config is not None
    assert ref.config.temperature == 0.5
    assert ref.config.safety_settings == {'HARM': 'BLOCK_NONE'}
    assert ref.info is not None
    assert ref.info.label == 'before'
    assert ref.info.supports is not None
    assert ref.info.supports.multiturn is True
