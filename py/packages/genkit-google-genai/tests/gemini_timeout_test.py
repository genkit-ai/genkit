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

"""GeminiConfig.timeout: per-request seconds, sent as this request's http_options.timeout in ms."""

from unittest.mock import MagicMock

import pytest
from genkit_google_genai import GeminiConfig
from genkit_google_genai._models._gemini import GeminiModel
from google.genai import types as genai_types
from google.genai._api_client import patch_http_options
from pydantic import ValidationError

from genkit import Message, Part, Role
from genkit.model import ModelRequest


async def _cfg(config: GeminiConfig, client_kwargs: dict | None = None) -> genai_types.GenerateContentConfig:
    model = GeminiModel('gemini-2.5-flash', MagicMock(), client_kwargs=client_kwargs)
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])], config=config)
    cfg = await model._genkit_to_googleai_cfg(request=request)
    assert cfg is not None
    return cfg


@pytest.mark.asyncio
@pytest.mark.parametrize(('seconds', 'ms'), [(30, 30000), (2.007, 2007), (0.0001, 1)])
async def test_timeout_seconds_become_request_http_options_ms(seconds: float, ms: int) -> None:
    """`timeout=30` alone builds a config whose http_options.timeout is 30000 ms."""
    cfg = await _cfg(GeminiConfig(timeout=seconds))

    assert cfg.http_options is not None
    assert cfg.http_options.timeout == ms


@pytest.mark.asyncio
async def test_timeout_is_not_sent_in_the_body() -> None:
    """timeout is transport only: nothing named timeout reaches the generation config or extra_body."""
    cfg = await _cfg(GeminiConfig(timeout=30, temperature=0.2))

    dumped = cfg.model_dump(exclude_none=True, exclude={'http_options'})
    assert 'timeout' not in dumped
    assert cfg.http_options is not None
    assert cfg.http_options.extra_body is None


@pytest.mark.asyncio
async def test_timeout_and_extra_share_the_request_http_options() -> None:
    """`extra` rides on http_options.extra_body; a timeout on the same request keeps it."""
    cfg = await _cfg(GeminiConfig.model_validate({'timeout': 5, 'extra': {'labels': {'team': 'search'}}}))

    assert cfg.http_options is not None
    assert cfg.http_options.timeout == 5000
    assert cfg.http_options.extra_body == {'labels': {'team': 'search'}}


@pytest.mark.asyncio
async def test_request_timeout_overrides_only_the_plugin_timeout() -> None:
    """google-genai layers request http_options over the client's, so base_url and headers stay."""
    plugin_http = genai_types.HttpOptions(timeout=60000, base_url='https://proxy.example', headers={'x-team': 'a'})
    cfg = await _cfg(GeminiConfig(timeout=5), client_kwargs={'http_options': plugin_http})

    assert cfg.http_options is not None
    effective = patch_http_options(plugin_http, cfg.http_options)
    assert effective.timeout == 5000
    assert effective.base_url == 'https://proxy.example'
    assert effective.headers is not None
    assert effective.headers['x-team'] == 'a'


@pytest.mark.asyncio
async def test_no_timeout_leaves_http_options_alone() -> None:
    """Without timeout or extra, the request carries no http_options, so the plugin's apply unchanged."""
    cfg = await _cfg(GeminiConfig(temperature=0.2))

    assert cfg.http_options is None


@pytest.mark.parametrize('bad', [0, -1, float('inf'), float('nan'), True, 'soon'])
def test_timeout_must_be_positive_finite_seconds(bad: object) -> None:
    """0, negative, inf, NaN, bool and non-numbers raise at validation."""
    with pytest.raises(ValidationError, match='timeout'):
        GeminiConfig.model_validate({'timeout': bad})


def test_timeout_is_in_the_dev_ui_schema() -> None:
    """The Dev UI form shows timeout with its seconds description."""
    schema = GeminiConfig.model_json_schema(by_alias=True)
    assert 'seconds' in schema['properties']['timeout']['description']
