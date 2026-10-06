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

"""`GeminiConfigSchema` rejects typos; `extra` is merged over the request body."""

from unittest.mock import MagicMock

import pytest
from genkit_google_genai.models._secrets import reject_request_config_api_key
from genkit_google_genai.models.gemini import GeminiConfigSchema, GeminiModel
from google.genai import types as genai_types
from pydantic import ValidationError

from genkit import GenkitError, Message, Part, Role
from genkit.model import ModelRequest


async def _cfg(config: GeminiConfigSchema) -> genai_types.GenerateContentConfig:
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
        GeminiConfigSchema.model_validate({'temprature': 0.2})


@pytest.mark.asyncio
async def test_extra_lands_at_top_of_request_body() -> None:
    """`extra={'labels': {...}}` is sent as a top-level body field."""
    cfg = await _cfg(GeminiConfigSchema.model_validate({'temperature': 0.2, 'extra': {'labels': {'team': 'search'}}}))

    assert _extra_body(cfg) == {'labels': {'team': 'search'}}
    assert cfg.temperature == 0.2


@pytest.mark.asyncio
async def test_extra_alone_still_builds_a_config() -> None:
    """A config holding only `extra` still reaches the request."""
    cfg = await _cfg(GeminiConfigSchema.model_validate({'extra': {'labels': {'team': 'search'}}}))

    assert _extra_body(cfg) == {'labels': {'team': 'search'}}


@pytest.mark.asyncio
async def test_extra_nested_block_adds_keys_instead_of_replacing() -> None:
    """`{'generationConfig': {...}}` goes on extra_body, which google-genai merges recursively."""
    cfg = await _cfg(
        GeminiConfigSchema.model_validate({'temperature': 0.2, 'extra': {'generationConfig': {'newKnob': 1}}})
    )

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
        await _cfg(GeminiConfigSchema.model_validate({'extra': {field: []}}))

    assert err.value.status == 'INVALID_ARGUMENT'
    assert repr(field) in str(err.value)


@pytest.mark.parametrize('field', ['responseSchema', 'response_json_schema', 'responseMimeType'])
@pytest.mark.asyncio
async def test_extra_cannot_set_structured_output_fields(field: str) -> None:
    """generationConfig fields Genkit sets from the output config are rejected too."""
    with pytest.raises(GenkitError) as err:
        await _cfg(GeminiConfigSchema.model_validate({'extra': {'generationConfig': {field: 'x'}}}))

    assert f"'generationConfig.{field}'" in str(err.value)


def test_api_key_in_extra_is_rejected() -> None:
    """A key in `extra` would ride the wire and land in traces; it belongs in context.secrets."""
    with pytest.raises(GenkitError, match='context.secrets'):
        reject_request_config_api_key(GeminiConfigSchema.model_validate({'extra': {'api_key': 'sk'}}))


def test_declared_sampling_knobs_stay_flat_and_typed() -> None:
    """seed and the penalties are declared, so they validate flat in either spelling."""
    config = GeminiConfigSchema.model_validate({'seed': 7, 'presencePenalty': 0.5, 'frequency_penalty': 0.1})

    assert (config.seed, config.presence_penalty, config.frequency_penalty) == (7, 0.5, 0.1)


@pytest.mark.asyncio
async def test_extra_keeps_plugin_level_extra_body() -> None:
    """A request's `extra` layers over the plugin's extra_body instead of replacing it."""
    plugin_http = genai_types.HttpOptions(extra_body={'labels': {'env': 'prod'}, 'keep': 1})
    model = GeminiModel('gemini-2.5-flash', MagicMock(), client_kwargs={'http_options': plugin_http})
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiConfigSchema.model_validate({'extra': {'labels': {'team': 'search'}}}),
    )

    cfg = await model._genkit_to_googleai_cfg(request=request)

    assert cfg is not None
    assert _extra_body(cfg) == {'labels': {'env': 'prod', 'team': 'search'}, 'keep': 1}
