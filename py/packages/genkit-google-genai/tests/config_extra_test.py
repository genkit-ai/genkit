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

"""`GeminiConfig` rejects typos; `extra` is merged over the request body."""

from unittest.mock import MagicMock

import pytest
from genkit_google_genai._models._gemini import GeminiConfig, GeminiModel
from genkit_google_genai._models._sdk_config import attach_config_extra, attach_leftovers
from google.genai import types as genai_types
from pydantic import ValidationError

from genkit import GenkitError, Message, Part, Role
from genkit.model import ModelRequest


async def _cfg(config: GeminiConfig) -> genai_types.GenerateContentConfig:
    model = GeminiModel('gemini-2.5-flash', MagicMock())
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])], config=config)
    cfg = await model._genkit_to_googleai_cfg(request=request)
    assert cfg is not None
    return cfg


def _extra_body(cfg: genai_types.GenerateContentConfig) -> dict:
    assert cfg.http_options is not None
    assert cfg.http_options.extra_body is not None
    return cfg.http_options.extra_body


def test_unknown_key_raises() -> None:
    """`{'temprature': 0.2}` fails instead of riding to the wire."""
    with pytest.raises(ValidationError, match='temprature'):
        GeminiConfig.model_validate({'temprature': 0.2})


@pytest.mark.asyncio
async def test_extra_lands_at_top_of_request_body() -> None:
    """`extra={'labels': {...}}` is sent as a top-level body field."""
    cfg = await _cfg(GeminiConfig.model_validate({'temperature': 0.2, 'extra': {'labels': {'team': 'search'}}}))

    assert _extra_body(cfg) == {'labels': {'team': 'search'}}
    assert cfg.temperature == 0.2


@pytest.mark.asyncio
async def test_extra_alone_still_builds_a_config() -> None:
    """A config holding only `extra` still reaches the request."""
    cfg = await _cfg(GeminiConfig.model_validate({'extra': {'labels': {'team': 'search'}}}))

    assert _extra_body(cfg) == {'labels': {'team': 'search'}}


@pytest.mark.asyncio
async def test_extra_nested_block_adds_keys_instead_of_replacing() -> None:
    """`{'generationConfig': {...}}` goes on extra_body, which google-genai merges recursively."""
    cfg = await _cfg(GeminiConfig.model_validate({'temperature': 0.2, 'extra': {'generationConfig': {'newKnob': 1}}}))

    assert _extra_body(cfg) == {'generationConfig': {'newKnob': 1}}
    assert cfg.temperature == 0.2


@pytest.mark.parametrize(
    'field',
    ['contents', 'Contents', 'systemInstruction', 'SYSTEM_INSTRUCTION', 'tools', 'toolConfig', 'cachedContent'],
)
@pytest.mark.asyncio
async def test_extra_cannot_set_genkit_built_fields(field: str) -> None:
    """Fields Genkit builds from the request are rejected, not overwritten."""
    with pytest.raises(GenkitError) as err:
        await _cfg(GeminiConfig.model_validate({'extra': {field: []}}))

    assert err.value.status == 'INVALID_ARGUMENT'
    assert repr(field) in str(err.value)


@pytest.mark.parametrize('field', ['responseSchema', 'response_json_schema', 'responseMimeType'])
@pytest.mark.asyncio
async def test_extra_cannot_set_structured_output_fields(field: str) -> None:
    """generationConfig fields Genkit sets from the output config are rejected too."""
    with pytest.raises(GenkitError) as err:
        await _cfg(GeminiConfig.model_validate({'extra': {'generationConfig': {field: 'x'}}}))

    assert f"'generationConfig.{field}'" in str(err.value)


def test_declared_sampling_knobs_stay_flat_and_typed() -> None:
    """seed and the penalties are declared, so they validate flat in either spelling."""
    config = GeminiConfig.model_validate({'seed': 7, 'presencePenalty': 0.5, 'frequency_penalty': 0.1})

    assert (config.seed, config.presence_penalty, config.frequency_penalty) == (7, 0.5, 0.1)


@pytest.mark.asyncio
async def test_extra_keeps_plugin_level_extra_body() -> None:
    """A request's `extra` layers over the plugin's extra_body instead of replacing it."""
    plugin_http = genai_types.HttpOptions(extra_body={'labels': {'env': 'prod'}, 'keep': 1})
    model = GeminiModel('gemini-2.5-flash', MagicMock(), client_kwargs={'http_options': plugin_http})
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiConfig.model_validate({'extra': {'labels': {'team': 'search'}}}),
    )

    cfg = await model._genkit_to_googleai_cfg(request=request)

    assert cfg is not None
    assert _extra_body(cfg) == {'labels': {'env': 'prod', 'team': 'search'}, 'keep': 1}


@pytest.mark.asyncio
async def test_generate_gemini_extra_snake_case_generation_config_keeps_other_generation_fields() -> None:
    """`extra={'generation_config': {'newKnob': 2}}` merges into generationConfig instead of replacing it."""
    cfg = genai_types.GenerateContentConfig()
    cfg = attach_leftovers(cfg, {'futureKnob': 1}, nest='generationConfig')
    cfg = attach_config_extra(cfg, {'generation_config': {'newKnob': 2}}, action_name='gemini-2.5-flash')

    assert _extra_body(cfg)['generationConfig'] == {'futureKnob': 1, 'newKnob': 2}
    assert 'generation_config' not in _extra_body(cfg)


@pytest.mark.asyncio
async def test_generate_gemini_extra_keeps_plugin_extra_body_in_other_casing() -> None:
    """Plugin `extra_body={'generation_config': {...}}` plus request `generationConfig` keeps both sets of keys."""
    plugin_http = genai_types.HttpOptions(extra_body={'generation_config': {'pluginKnob': 1}})
    model = GeminiModel('gemini-2.5-flash', MagicMock(), client_kwargs={'http_options': plugin_http})
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiConfig.model_validate({'extra': {'generationConfig': {'newKnob': 2}}}),
    )

    cfg = await model._genkit_to_googleai_cfg(request=request)

    assert cfg is not None
    assert _extra_body(cfg) == {'generation_config': {'pluginKnob': 1, 'newKnob': 2}}
