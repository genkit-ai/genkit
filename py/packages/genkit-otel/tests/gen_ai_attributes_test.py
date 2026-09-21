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

from genkit_otel._gen_ai_attributes import (
    ContentCapturingMode,
    as_double,
    as_int,
    as_string_list,
    derive_output_type,
    derive_provider_name,
    map_finish_reason,
    parse_content_capturing_mode,
    split_model_name,
)


def test_split_prefixed_model_name() -> None:
    prefix, model = split_model_name('googleai/gemini-flash-latest')
    assert prefix == 'googleai'
    assert model == 'gemini-flash-latest'


def test_split_keeps_only_first_slash() -> None:
    prefix, model = split_model_name('vertexai/publishers/google/models/x')
    assert prefix == 'vertexai'
    assert model == 'publishers/google/models/x'


def test_split_no_slash() -> None:
    prefix, model = split_model_name('some-model')
    assert prefix is None
    assert model == 'some-model'


def test_derive_known_providers() -> None:
    assert derive_provider_name('googleai') == 'gcp.gemini'
    assert derive_provider_name('google-genai') == 'gcp.gemini'
    assert derive_provider_name('vertexai') == 'gcp.vertex_ai'
    assert derive_provider_name('openai') == 'openai'
    assert derive_provider_name('anthropic') == 'anthropic'


def test_derive_provider_is_case_insensitive() -> None:
    assert derive_provider_name('GoogleAI') == 'gcp.gemini'


def test_derive_unknown_prefix_lowercased() -> None:
    assert derive_provider_name('MyPlugin') == 'myplugin'


def test_derive_null_or_empty_prefix() -> None:
    assert derive_provider_name(None) is None
    assert derive_provider_name('') is None


def test_map_direct_finish_reasons() -> None:
    assert map_finish_reason('stop', failed=False) == 'stop'
    assert map_finish_reason('length', failed=False) == 'length'
    assert map_finish_reason('blocked', failed=False) == 'content_filter'


def test_map_interrupted_to_stop() -> None:
    assert map_finish_reason('interrupted', failed=False) == 'stop'


def test_map_ambiguous_finish_reasons_by_failure() -> None:
    assert map_finish_reason('other', failed=False) == 'stop'
    assert map_finish_reason('other', failed=True) == 'error'
    assert map_finish_reason('unknown', failed=True) == 'error'
    assert map_finish_reason(None, failed=True) == 'error'


def test_derive_output_type_json() -> None:
    assert derive_output_type(format='json') == 'json'
    assert derive_output_type(content_type='application/json') == 'json'


def test_derive_output_type_text() -> None:
    assert derive_output_type(format='text') == 'text'
    assert derive_output_type(content_type='text/plain') == 'text'


def test_derive_output_type_unknown() -> None:
    assert derive_output_type() is None
    assert derive_output_type(format='media') is None


def test_as_int() -> None:
    assert as_int(3) == 3
    assert as_int(3.9) == 3
    assert as_int('5') == 5
    assert as_int('x') is None
    assert as_int(None) is None
    assert as_int(True) is None


def test_as_double() -> None:
    assert as_double(3) == 3.0
    assert as_double(3.5) == 3.5
    assert as_double('5.5') == 5.5
    assert as_double('x') is None
    assert as_double(None) is None


def test_as_string_list() -> None:
    assert as_string_list(['a', 'b']) == ['a', 'b']
    assert as_string_list([1, 2]) == ['1', '2']
    assert as_string_list('a') == ['a']
    assert as_string_list(None) is None


def test_parse_known_content_tokens() -> None:
    assert parse_content_capturing_mode('NO_CONTENT') is ContentCapturingMode.NO_CONTENT
    assert parse_content_capturing_mode('span_only') is ContentCapturingMode.SPAN_ONLY
    assert parse_content_capturing_mode('Event_Only') is ContentCapturingMode.EVENT_ONLY
    assert parse_content_capturing_mode('  SPAN_AND_EVENT  ') is ContentCapturingMode.SPAN_AND_EVENT


def test_parse_null_or_empty_is_no_content() -> None:
    assert parse_content_capturing_mode(None) is ContentCapturingMode.NO_CONTENT
    assert parse_content_capturing_mode('') is ContentCapturingMode.NO_CONTENT
    assert parse_content_capturing_mode('   ') is ContentCapturingMode.NO_CONTENT


def test_parse_unknown_token_is_none() -> None:
    assert parse_content_capturing_mode('true') is None
    assert parse_content_capturing_mode('bogus') is None
