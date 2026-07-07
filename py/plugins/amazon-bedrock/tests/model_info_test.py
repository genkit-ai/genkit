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

"""Tests for the Bedrock model capability registry."""

import pytest

from genkit import Stage
from genkit.plugins.amazon_bedrock.model_info import (
    INFERENCE_PROFILE_PREFIXES,
    MODEL_CAPABILITIES,
    get_model_info,
    strip_inference_profile_prefix,
)


@pytest.mark.parametrize('prefix', INFERENCE_PROFILE_PREFIXES)
def test_strip_inference_profile_prefix(prefix: str) -> None:
    model_id = f'{prefix}anthropic.claude-sonnet-4-5-20250929-v1:0'
    assert strip_inference_profile_prefix(model_id) == 'anthropic.claude-sonnet-4-5-20250929-v1:0'


def test_strip_leaves_bare_model_id_untouched() -> None:
    assert strip_inference_profile_prefix('amazon.nova-lite-v1:0') == 'amazon.nova-lite-v1:0'


def test_strip_only_removes_first_matching_prefix() -> None:
    assert strip_inference_profile_prefix('us.us.model') == 'us.model'


def test_strip_handles_inference_profile_arn() -> None:
    arn = 'arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0'
    assert strip_inference_profile_prefix(arn) == 'anthropic.claude-3-5-sonnet-20241022-v2:0'


def test_strip_handles_foundation_model_arn() -> None:
    arn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
    assert strip_inference_profile_prefix(arn) == 'anthropic.claude-3-haiku-20240307-v1:0'


def test_strip_handles_govcloud_partition_arn() -> None:
    arn = (
        'arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:'
        'inference-profile/us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0'
    )
    assert strip_inference_profile_prefix(arn) == 'anthropic.claude-3-5-sonnet-20240620-v1:0'


def test_application_inference_profile_arn_falls_back_to_unstable() -> None:
    arn = 'arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123opaque'
    info = get_model_info(arn)
    assert info.stage == Stage.UNSTABLE
    assert info.supports.tools is True


def test_us_gov_wins_over_us() -> None:
    assert strip_inference_profile_prefix('us-gov.anthropic.claude-3-opus-20240229-v1:0') == (
        'anthropic.claude-3-opus-20240229-v1:0'
    )


def test_known_model_is_stable_with_registry_capabilities() -> None:
    info = get_model_info('anthropic.claude-3-5-haiku-20241022-v1:0')
    assert info.stage == Stage.STABLE
    assert info.supports.multiturn is True
    assert info.supports.system_role is True
    assert info.supports.tools is True
    assert info.supports.tool_choice is True
    assert info.supports.media is False


def test_inference_profile_id_resolves_registry_entry() -> None:
    info = get_model_info('eu.amazon.nova-micro-v1:0')
    assert info.stage == Stage.STABLE
    assert info.supports.media is False
    assert info.label == 'eu.amazon.nova-micro-v1:0'


def test_unknown_model_defaults_to_unstable_converse_capabilities() -> None:
    info = get_model_info('vendor.brand-new-model-v1:0')
    assert info.stage == Stage.UNSTABLE
    assert info.supports.multiturn is True
    assert info.supports.tools is True
    assert info.supports.media is True


def test_image_model_supports_media_output_only() -> None:
    info = get_model_info('amazon.titan-image-generator-v1', model_type='image')
    assert info.stage == Stage.STABLE
    assert info.supports.media is True
    assert info.supports.multiturn is False
    assert info.supports.tools is False
    assert info.supports.system_role is False


def test_embedding_model_supports_nothing() -> None:
    info = get_model_info('amazon.titan-embed-text-v2:0', model_type='embedding')
    assert info.stage == Stage.STABLE
    assert info.supports.media is False
    assert info.supports.multiturn is False
    assert info.supports.tools is False


def test_registry_matches_go_plugin_size() -> None:
    assert len(MODEL_CAPABILITIES) == 47


def test_all_registry_keys_are_base_ids() -> None:
    for model_id in MODEL_CAPABILITIES:
        assert strip_inference_profile_prefix(model_id) == model_id
