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

"""AnthropicConfig sampling bounds and cache usage mapping."""

import pytest
from genkit_anthropic import AnthropicConfig
from genkit_anthropic._utils import build_cache_usage
from pydantic import ValidationError

from genkit.model import GenerationUsage


@pytest.mark.parametrize(
    ('field', 'value'),
    [('temperature', 1.5), ('temperature', -0.1), ('top_p', 1.1), ('top_p', -0.5)],
)
def test_out_of_range_sampling_values_fail_validation(field: str, value: float) -> None:
    """Anthropic accepts temperature and top_p in [0, 1]; anything else fails before a request."""
    with pytest.raises(ValidationError, match=f'{field} must be between 0.0 and 1.0'):
        AnthropicConfig.model_validate({field: value})


@pytest.mark.parametrize('value', [0.0, 0.7, 1.0])
def test_in_range_sampling_values_pass(value: float) -> None:
    """Bounds are inclusive."""
    config = AnthropicConfig(temperature=value, top_p=value)

    assert (config.temperature, config.top_p) == (value, value)


def test_bounds_show_in_the_dev_ui_schema() -> None:
    """The JSON schema the Dev UI reads advertises the range."""
    props = AnthropicConfig.model_json_schema()['properties']

    assert (props['temperature']['minimum'], props['temperature']['maximum']) == (0.0, 1.0)
    assert (props['topP']['minimum'], props['topP']['maximum']) == (0.0, 1.0)


def test_cache_tokens_map_to_the_shared_usage_fields() -> None:
    """Cache reads and writes fill the GenerationUsage fields OpenAI and Gemini fill."""
    usage = build_cache_usage(
        input_tokens=40,
        output_tokens=20,
        basic_usage=GenerationUsage(),
        cache_creation_input_tokens=10,
        cache_read_input_tokens=50,
    )

    assert (usage.cached_content_tokens, usage.cache_write_tokens) == (50, 10)
    assert (usage.input_tokens, usage.total_tokens) == (40, 60)
    assert usage.custom == {'cache_creation_input_tokens': 10, 'cache_read_input_tokens': 50}


def test_no_cache_tokens_leaves_the_fields_unset() -> None:
    """A response without cache activity reports None, not 0."""
    usage = build_cache_usage(input_tokens=40, output_tokens=20, basic_usage=GenerationUsage())

    assert (usage.cached_content_tokens, usage.cache_write_tokens) == (None, None)
