# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GenkitConfig base class."""

import pytest
from pydantic import ValidationError

from genkit._core._base import GenkitModel
from genkit._core._typing import GenerationCommonConfig
from genkit.model import ModelConfig, ModelRequest
from genkit.plugin_api import GenkitConfig


class SampleConfig(GenkitConfig):
    """Test configuration model subclassing GenkitConfig."""

    sample_rate: float
    max_retries: int = 3
    api_key_name: str | None = None


def test_plugin_config_snake_case_kwargs() -> None:
    """GenkitConfig accepts snake_case kwargs."""
    cfg = SampleConfig(sample_rate=0.5, max_retries=5, api_key_name='KEY')
    assert cfg.sample_rate == 0.5
    assert cfg.max_retries == 5
    assert cfg.api_key_name == 'KEY'


def test_plugin_config_accepts_camel_case_wire_dict() -> None:
    """GenkitConfig parses camelCase wire dicts via alias_generator."""
    cfg = SampleConfig.model_validate({'sampleRate': 0.8, 'maxRetries': 2, 'apiKeyName': 'CUSTOM'})
    assert cfg.sample_rate == 0.8
    assert cfg.max_retries == 2
    assert cfg.api_key_name == 'CUSTOM'


def test_plugin_config_model_dump_preserves_snake_case_by_default() -> None:
    """GenkitConfig does not hijack model_dump; default is by_alias=False."""
    cfg = SampleConfig(sample_rate=0.5, max_retries=5)
    assert cfg.model_dump(exclude_none=True) == {
        'sample_rate': 0.5,
        'max_retries': 5,
    }


def test_plugin_config_model_dump_by_alias_emits_camel_case() -> None:
    """GenkitConfig dumps camelCase wire format when by_alias=True is passed."""
    cfg = SampleConfig(sample_rate=0.5, max_retries=5)
    assert cfg.model_dump(by_alias=True, exclude_none=True) == {
        'sampleRate': 0.5,
        'maxRetries': 5,
    }


def test_plugin_config_forbids_extra_kwargs() -> None:
    """Unknown kwargs raise ValidationError at creation time."""
    with pytest.raises(ValidationError):
        SampleConfig(sample_rate=0.5, unknown_field='bad')  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        SampleConfig.model_validate({'sampleRate': 0.5, 'unknownField': 'bad'})


def test_model_config_is_plugin_config_not_genkit_model() -> None:
    """Model configs follow the upstream dump contract, not GenkitModel's camelCase default."""
    assert issubclass(ModelConfig, GenerationCommonConfig)
    assert issubclass(GenerationCommonConfig, GenkitConfig)
    assert not issubclass(ModelConfig, GenkitModel)


def test_generated_common_config_rejects_unknown_keys() -> None:
    """The schema marks GenerationCommonConfig open; the generated class still forbids extras."""
    with pytest.raises(ValidationError):
        GenerationCommonConfig.model_validate({'temprature': 0.2})


def test_model_config_default_dump_is_sdk_shape() -> None:
    """The default dump returns field names, ready for a provider SDK."""
    config = ModelConfig(max_output_tokens=512, stop_sequences=['END'])
    assert config.model_dump(exclude_none=True) == {'max_output_tokens': 512, 'stop_sequences': ['END']}
    assert config.model_dump(by_alias=True, exclude_none=True) == {'maxOutputTokens': 512, 'stopSequences': ['END']}


def test_model_config_nested_in_request_keeps_wire_shape() -> None:
    """Inside a GenkitModel parent, the parent's dump settings apply: camelCase, no nulls."""
    request = ModelRequest(messages=[], config=ModelConfig(max_output_tokens=512))
    assert request.model_dump()['config'] == {'maxOutputTokens': 512}


def test_genkit_config_rejects_both_spellings_of_one_setting() -> None:
    """Two spellings of one key in one input raise; neither silently wins."""
    with pytest.raises(ValidationError):
        SampleConfig.model_validate({'sampleRate': 0.5, 'sample_rate': 0.5})
    with pytest.raises(ValidationError):
        SampleConfig(sample_rate=0.5, sampleRate=0.6)  # type: ignore[call-arg]
