#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Dumping a ModelRequest and reading it back keeps output settings and config."""

import pytest
from pydantic import BaseModel, ValidationError

from genkit._core._model import ModelRequest, OutputConfig, config_type_path


class CarrierCfg(BaseModel):
    """Stand-in for a different plugin's config class."""

    model_config = {'extra': 'allow'}


class PluginCfg(BaseModel):
    """Stand-in for a plugin config schema."""

    temperature: float | None = None


def test_wire_roundtrip_preserves_output_fields() -> None:
    """JSON mode written as OutputConfig comes back as output_format / schema on the wire."""
    req = ModelRequest[CarrierCfg](
        messages=[],
        config={'temperature': 0.5},
        output=OutputConfig(
            format='json',
            constrained=True,
            content_type='application/json',
            json_schema={'type': 'object'},
        ),
    )
    dumped = req.model_dump(mode='python')
    assert dumped['output'] == {
        'format': 'json',
        'constrained': True,
        'contentType': 'application/json',
        'schema': {'type': 'object'},
    }
    reparsed = ModelRequest[CarrierCfg].model_validate(dumped)
    assert reparsed.output_format == 'json'
    assert reparsed.output_constrained is True
    assert reparsed.output_content_type == 'application/json'
    assert reparsed.output_schema == {'type': 'object'}


def test_output_always_present_on_wire() -> None:
    """A built request always has an output object, even when nobody asked for JSON."""
    req = ModelRequest[CarrierCfg](messages=[])
    assert req.model_dump(mode='python')['output'] == {}


def test_cross_config_revalidation_preserves_output() -> None:
    """model_validate of a dumped request rebuilds config as the target plugin class."""
    req = ModelRequest[CarrierCfg](
        messages=[],
        config={'temperature': 0.5},
        output=OutputConfig(format='json', constrained=True),
    )
    dumped = req.model_dump(mode='python')
    reparsed = ModelRequest[PluginCfg].model_validate(dumped)
    assert isinstance(reparsed.config, PluginCfg)
    assert reparsed.config.temperature == 0.5
    assert reparsed.output_format == 'json'
    assert reparsed.output_constrained is True


def test_flat_properties_read_and_write_nested_storage() -> None:
    """Assigning request.output_format writes the nested output object."""
    req = ModelRequest[CarrierCfg](messages=[])
    req.output_format = 'json'
    req.output_schema = {'type': 'integer'}
    assert req.output.format == 'json'
    assert req.output.json_schema == {'type': 'integer'}
    assert req.model_dump(mode='python')['output']['schema'] == {'type': 'integer'}


def test_bad_config_type_raises_validation_error() -> None:
    """config=5 on the constructor is a ValidationError."""
    with pytest.raises(ValidationError):
        ModelRequest[CarrierCfg](messages=[], config=5)  # type: ignore[arg-type]


def test_foreign_config_class_raises_validation_error() -> None:
    """ModelRequest[PluginCfg](config=CarrierCfg()) is a ValidationError. Pass a dict."""
    with pytest.raises(ValidationError, match=r'config must be .+\.PluginCfg or a mapping, got .+\.CarrierCfg'):
        ModelRequest[PluginCfg](messages=[], config=CarrierCfg())


def test_config_type_path_uses_plugin_package_export() -> None:
    """Error strings say genkit_openai.OpenAIConfig, not genkit_openai.typing.OpenAIConfig."""
    from genkit_google_genai import GeminiConfigSchema
    from genkit_openai import OpenAIConfig

    assert config_type_path(GeminiConfigSchema) == 'genkit_google_genai.GeminiConfigSchema'
    assert config_type_path(OpenAIConfig) == 'genkit_openai.OpenAIConfig'
