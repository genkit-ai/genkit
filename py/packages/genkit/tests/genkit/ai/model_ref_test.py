"""Unit tests for ModelRef, model_ref(), and ConfigT TypeVar scenarios."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel

from genkit._ai._model import ModelConfig
from genkit._core._typing import ModelInfo, Supports
from genkit.model import ModelRef, model_ref


class CustomConfig(BaseModel):
    """Plugin-specific configuration schema for testing ConfigT parameterization."""

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
    assert ref.config is config
    assert ref.config is not None
    assert ref.config.temperature == 0.7
    assert ref.config.top_p == 0.9


def test_model_ref_with_bare_base_model_schema() -> None:
    """model_ref() accepts bare BaseModel when no model-specific Pydantic config class is provided."""
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
    assert ref.info is info
    assert ref.info.supports is not None
    assert ref.info.supports.multiturn is True


def test_prompt_config_keeps_plugin_specific_fields() -> None:
    """PromptConfig preserves custom plugin-specific fields when normalizing config."""
    from genkit._ai._prompt import PromptConfig, normalize_config

    pc = PromptConfig(config=normalize_config(CustomConfig(temperature=0.7, safety_settings={'HARM': 'BLOCK'})))
    dumped = pc.model_dump()['config']
    assert dumped['safety_settings'] == {'HARM': 'BLOCK'}
    assert dumped['temperature'] == 0.7


def test_define_prompt_with_typed_model_ref() -> None:
    """ai.define_prompt accepts a typed ModelRef instance."""
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
