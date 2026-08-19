#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""ModelRequest wire-format round-trip and config type-contract tests."""

import pytest
from pydantic import BaseModel, ValidationError

from genkit._core._model import ModelRequest, OutputConfig


class CarrierCfg(BaseModel):
    """Permissive stand-in for the sending side of a cross-typed handoff."""

    model_config = {'extra': 'allow'}


class PluginCfg(BaseModel):
    """Stand-in for a plugin config schema."""

    temperature: float | None = None


def test_wire_roundtrip_preserves_output_fields() -> None:
    """dump -> validate must be an identity for the output settings."""
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
    """A built request always includes an output object, even when empty.

    This matches how the JS SDK's generate path builds requests: an output
    object is always present (empty when no output options are set), even
    though the wire schema marks the field optional for hand-built requests.
    """
    req = ModelRequest[CarrierCfg](messages=[])
    assert req.model_dump(mode='python')['output'] == {}


def test_cross_config_revalidation_preserves_output() -> None:
    """The _validate_input fallback scenario: ModelRequest[CarrierCfg] -> ModelRequest[PluginCfg]."""
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
    """The flat accessors are a live view over output (plugin tests mutate them)."""
    req = ModelRequest[CarrierCfg](messages=[])
    req.output_format = 'json'
    req.output_schema = {'type': 'integer'}
    assert req.output.format == 'json'
    assert req.output.json_schema == {'type': 'integer'}
    assert req.model_dump(mode='python')['output']['schema'] == {'type': 'integer'}


def test_bad_config_type_raises_validation_error() -> None:
    """Wrong config type must surface as ValidationError, never bare TypeError."""
    with pytest.raises(ValidationError):
        ModelRequest[CarrierCfg](messages=[], config=5)  # type: ignore[arg-type]
