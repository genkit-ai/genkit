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

"""Tests for the Converse request/response converters.

The expectations encode Bedrock wire-format truths, not incidental structure.
"""

import base64

import pytest
from genkit_amazon_bedrock.config import BedrockConfig
from genkit_amazon_bedrock.converters import (
    REASONING_SIGNATURE_METADATA_KEY,
    REDACTED_CONTENT_METADATA_KEY,
    build_converse_request,
    build_inference_config,
    cache_point_part,
    content_blocks_to_parts,
    map_finish_reason,
    normalize_config,
    to_bedrock_role,
    to_bedrock_tool,
    to_model_response,
    usage_from_response,
)

from genkit import (
    FinishReason,
    Message,
    ModelRequest,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolDefinition,
)
from genkit.plugin_api import GenkitError, ModelConfig

PNG_BYTES = b'\x89PNG\r\n\x1a\nfakeimagedata'
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


def user_text_request(text: str = 'hello', **kwargs) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text=text))])],
        **kwargs,
    )


# --- Stop reasons ---------------------------------------------------------


@pytest.mark.parametrize(
    'stop_reason,expected',
    [
        ('end_turn', FinishReason.STOP),
        ('stop_sequence', FinishReason.STOP),
        ('tool_use', FinishReason.STOP),
        ('max_tokens', FinishReason.LENGTH),
        ('model_context_window_exceeded', FinishReason.LENGTH),
        ('content_filtered', FinishReason.BLOCKED),
        ('guardrail_intervened', FinishReason.BLOCKED),
        ('malformed_model_output', FinishReason.OTHER),
        ('malformed_tool_use', FinishReason.OTHER),
        ('some_future_reason', FinishReason.OTHER),
        ('', FinishReason.OTHER),
        (None, FinishReason.OTHER),
    ],
)
def test_map_finish_reason(stop_reason, expected) -> None:
    assert map_finish_reason(stop_reason) == expected


# --- Roles ----------------------------------------------------------------


def test_roles_map_to_converse_vocabulary() -> None:
    assert to_bedrock_role(Role.USER) == 'user'
    assert to_bedrock_role(Role.TOOL) == 'user'
    assert to_bedrock_role(Role.MODEL) == 'assistant'


def test_unsupported_role_raises() -> None:
    with pytest.raises(GenkitError):
        to_bedrock_role('reviewer')


def test_unsupported_role_errors_even_for_empty_message() -> None:
    # The role is validated before parts are converted.
    request = ModelRequest(messages=[Message(role='reviewer', content=[])])
    with pytest.raises(GenkitError, match='unsupported role'):
        build_converse_request('amazon.nova-lite-v1:0', request)


# --- Config normalization -------------------------------------------------


def test_normalize_config_none_and_passthrough() -> None:
    assert normalize_config(None) is None
    config = BedrockConfig(tool_choice='auto')
    assert normalize_config(config) is config


def test_normalize_config_from_model_config() -> None:
    config = normalize_config(ModelConfig(temperature=0.5, max_output_tokens=100, top_p=0.9))
    assert config is not None
    assert config.temperature == 0.5
    assert config.max_output_tokens == 100
    assert config.top_p == 0.9


@pytest.mark.parametrize(
    'raw,expected',
    [
        ({'maxOutputTokens': 50}, 50),
        ({'max_output_tokens': 60}, 60),
    ],
)
def test_normalize_config_from_dict_accepts_both_spellings(raw, expected) -> None:
    config = normalize_config(raw)
    assert config is not None
    assert config.max_output_tokens == expected


def test_normalize_config_rejects_unsupported_type() -> None:
    with pytest.raises(GenkitError):
        normalize_config(42)


def test_normalize_config_rejects_unknown_keys() -> None:
    # 'maxTokens' is the Converse wire name, not a config field; tolerating it
    # would run the call with no token cap at all.
    with pytest.raises(GenkitError, match='unknown config key') as excinfo:
        normalize_config({'maxTokens': 4096})
    assert excinfo.value.status == 'INVALID_ARGUMENT'
    assert 'maxTokens' in str(excinfo.value)


def test_normalize_config_rejects_invalid_values() -> None:
    with pytest.raises(GenkitError, match='invalid config') as excinfo:
        normalize_config({'temperature': 'hot'})
    assert excinfo.value.status == 'INVALID_ARGUMENT'


def test_build_inference_config_empty_is_none() -> None:
    assert build_inference_config(None) is None
    assert build_inference_config(BedrockConfig()) is None
    assert build_inference_config(BedrockConfig(tool_choice='auto')) is None


def test_build_inference_config_fields() -> None:
    config = BedrockConfig(max_output_tokens=256, temperature=0.7, top_p=0.9, stop_sequences=['END'])
    assert build_inference_config(config) == {
        'maxTokens': 256,
        'temperature': 0.7,
        'topP': 0.9,
        'stopSequences': ['END'],
    }


def test_non_positive_max_output_tokens_is_dropped() -> None:
    # Bedrock rejects maxTokens below 1; treat it as unset rather than fail.
    assert build_inference_config(BedrockConfig(max_output_tokens=0)) is None


def test_explicit_zero_temperature_is_sent() -> None:
    # None means unset; an explicit 0.0 is a real setting and must be sent.
    inference_config = build_inference_config(BedrockConfig(temperature=0.0))
    assert inference_config == {'temperature': 0.0}


def test_top_k_and_version_are_accepted_but_ignored() -> None:
    # Converse has no first-class topK or version, so both are dropped.
    request = user_text_request(config=BedrockConfig(top_k=40, version='v9'))
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert 'inferenceConfig' not in kwargs
    assert 'additionalModelRequestFields' not in kwargs


# --- maxTokens --------------------------------------------------------------


def test_configured_max_tokens_is_sent() -> None:
    request = user_text_request(config=BedrockConfig(max_output_tokens=32))
    kwargs = build_converse_request('anthropic.claude-3-haiku-20240307-v1:0', request)
    assert kwargs['inferenceConfig'] == {'maxTokens': 32}


def test_dict_config_max_output_tokens_lands_in_inference_config() -> None:
    # The shape the samples pass: a camelCase dict straight into generate().
    request = user_text_request(config={'maxOutputTokens': 4096})
    kwargs = build_converse_request('us.anthropic.claude-sonnet-4-5-20250929-v1:0', request)
    assert kwargs['inferenceConfig'] == {'maxTokens': 4096}


@pytest.mark.parametrize('model_id', ['us.anthropic.claude-sonnet-4-5-20250929-v1:0', 'amazon.nova-lite-v1:0'])
def test_no_max_tokens_is_injected_for_any_model(model_id) -> None:
    kwargs = build_converse_request(model_id, user_text_request())
    assert 'inferenceConfig' not in kwargs


# --- Request assembly -------------------------------------------------------


def test_model_id_sent_verbatim_with_inference_profile_prefix() -> None:
    kwargs = build_converse_request('us.amazon.nova-lite-v1:0', user_text_request())
    assert kwargs['modelId'] == 'us.amazon.nova-lite-v1:0'


def test_simple_text_round_trip_shape() -> None:
    kwargs = build_converse_request('amazon.nova-lite-v1:0', user_text_request('hi'))
    assert kwargs['messages'] == [{'role': 'user', 'content': [{'text': 'hi'}]}]
    assert 'system' not in kwargs
    assert 'toolConfig' not in kwargs


def test_system_message_becomes_top_level_system() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='be terse'))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert kwargs['system'] == [{'text': 'be terse'}]
    assert kwargs['messages'] == [{'role': 'user', 'content': [{'text': 'hi'}]}]


def test_empty_system_text_is_dropped() -> None:
    # Bedrock rejects empty system text, so a blank rendered prompt is dropped
    # rather than sent; regular message text has no such floor.
    request = ModelRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text=''))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text=''))]),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert 'system' not in kwargs
    assert kwargs['messages'] == [{'role': 'user', 'content': [{'text': ''}]}]


def test_cache_point_in_system_and_messages() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='rules')), cache_point_part()]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi')), cache_point_part()]),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert kwargs['system'] == [{'text': 'rules'}, {'cachePoint': {'type': 'default'}}]
    assert kwargs['messages'][0]['content'] == [{'text': 'hi'}, {'cachePoint': {'type': 'default'}}]


def test_multi_turn_roles() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='q'))]),
            Message(role=Role.MODEL, content=[Part(root=TextPart(text='a'))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='q2'))]),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert [m['role'] for m in kwargs['messages']] == ['user', 'assistant', 'user']


def test_empty_messages_are_dropped() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
            Message(role=Role.MODEL, content=[]),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert len(kwargs['messages']) == 1


def test_tool_role_message_becomes_user_tool_result() -> None:
    request = ModelRequest(
        messages=[
            Message(
                role=Role.TOOL,
                content=[
                    Part.model_validate({'toolResponse': {'ref': 'call-1', 'name': 'weather', 'output': {'temp': 21}}})
                ],
            ),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    message = kwargs['messages'][0]
    assert message['role'] == 'user'
    tool_result = message['content'][0]['toolResult']
    assert tool_result['toolUseId'] == 'call-1'
    assert tool_result['status'] == 'success'
    # Non-string outputs ride as a JSON string in a text content block.
    assert tool_result['content'] == [{'text': '{"temp": 21}'}]


def test_string_tool_output_is_verbatim() -> None:
    request = ModelRequest(
        messages=[
            Message(
                role=Role.TOOL,
                content=[
                    Part.model_validate({'toolResponse': {'ref': 'call-2', 'name': 'weather', 'output': 'sunny'}})
                ],
            ),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    tool_result = kwargs['messages'][0]['content'][0]['toolResult']
    assert tool_result['content'] == [{'text': 'sunny'}]
    assert tool_result['toolUseId'] == 'call-2'


@pytest.mark.parametrize(
    'part',
    [
        {'toolRequest': {'name': 'weather', 'input': {}}},
        {'toolResponse': {'name': 'weather', 'output': 'sunny'}},
    ],
    ids=['tool_request', 'tool_response'],
)
def test_tool_part_without_ref_errors(part: dict[str, object]) -> None:
    # Bedrock's toolUseId has a one-character floor, so '' cannot be sent.
    request = ModelRequest(messages=[Message(role=Role.TOOL, content=[Part.model_validate(part)])])
    with pytest.raises(GenkitError, match='requires a ref to send as toolUseId') as excinfo:
        build_converse_request('amazon.nova-lite-v1:0', request)
    assert excinfo.value.status == 'INVALID_ARGUMENT'


def test_tool_request_part_becomes_tool_use_block() -> None:
    request = ModelRequest(
        messages=[
            Message(
                role=Role.MODEL,
                content=[
                    Part.model_validate({
                        'toolRequest': {'ref': 'call-9', 'name': 'weather', 'input': {'city': 'Lagos'}}
                    })
                ],
            ),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert kwargs['messages'][0]['content'][0]['toolUse'] == {
        'toolUseId': 'call-9',
        'name': 'weather',
        'input': {'city': 'Lagos'},
    }


# --- Media ------------------------------------------------------------------


def test_image_data_url_decodes_to_raw_bytes() -> None:
    part = Part.model_validate({'media': {'url': f'data:image/png;base64,{PNG_B64}'}})
    request = ModelRequest(messages=[Message(role=Role.USER, content=[part])])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    image = kwargs['messages'][0]['content'][0]['image']
    assert image['format'] == 'png'
    assert image['source']['bytes'] == PNG_BYTES


def test_explicit_content_type_beats_data_url_header() -> None:
    part = Part.model_validate({'media': {'url': f'data:image/png;base64,{PNG_B64}', 'contentType': 'image/jpeg'}})
    request = ModelRequest(messages=[Message(role=Role.USER, content=[part])])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert kwargs['messages'][0]['content'][0]['image']['format'] == 'jpeg'


def test_jpg_alias_normalizes_to_jpeg() -> None:
    part = Part.model_validate({'media': {'url': PNG_B64, 'contentType': 'image/jpg'}})
    request = ModelRequest(messages=[Message(role=Role.USER, content=[part])])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert kwargs['messages'][0]['content'][0]['image']['format'] == 'jpeg'


def test_document_mime_maps_to_document_block() -> None:
    part = Part.model_validate({'media': {'url': PNG_B64, 'contentType': 'application/pdf'}})
    request = ModelRequest(messages=[Message(role=Role.USER, content=[part])])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    document = kwargs['messages'][0]['content'][0]['document']
    assert document['format'] == 'pdf'
    assert document['name'] == 'document'
    assert document['source']['bytes'] == PNG_BYTES


def test_html_is_document_not_image() -> None:
    part = Part.model_validate({'media': {'url': PNG_B64, 'contentType': 'text/html'}})
    request = ModelRequest(messages=[Message(role=Role.USER, content=[part])])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert 'document' in kwargs['messages'][0]['content'][0]


@pytest.mark.parametrize(
    'mime,block_kind,expected_format',
    [
        ('image/gif', 'image', 'gif'),
        ('image/webp', 'image', 'webp'),
        ('text/csv', 'document', 'csv'),
        ('text/markdown', 'document', 'md'),
        ('text/plain', 'document', 'txt'),
        ('application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'document', 'docx'),
        ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'document', 'xlsx'),
    ],
)
def test_media_mime_format_matrix(mime, block_kind, expected_format) -> None:
    part = Part.model_validate({'media': {'url': PNG_B64, 'contentType': mime}})
    request = ModelRequest(messages=[Message(role=Role.USER, content=[part])])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    block = kwargs['messages'][0]['content'][0]
    assert block[block_kind]['format'] == expected_format


@pytest.mark.parametrize(
    'url,match',
    [
        ('https://example.com/cat.png', 'remote URLs are not supported'),
        ('data:image/png,rawdata', 'must be base64-encoded'),
        ('   ', 'empty data'),
        ('!!!not-base64!!!', 'decode base64 media'),
    ],
)
def test_media_payload_validation_errors(url, match) -> None:
    part = Part.model_validate({'media': {'url': url, 'contentType': 'image/png'}})
    request = ModelRequest(messages=[Message(role=Role.USER, content=[part])])
    with pytest.raises(GenkitError, match=match):
        build_converse_request('amazon.nova-lite-v1:0', request)


def test_unsupported_mime_type_errors() -> None:
    part = Part.model_validate({'media': {'url': PNG_B64, 'contentType': 'video/mp4'}})
    request = ModelRequest(messages=[Message(role=Role.USER, content=[part])])
    with pytest.raises(GenkitError, match='unsupported media MIME type'):
        build_converse_request('amazon.nova-lite-v1:0', request)


def test_media_without_content_type_errors() -> None:
    part = Part.model_validate({'media': {'url': PNG_B64}})
    request = ModelRequest(messages=[Message(role=Role.USER, content=[part])])
    with pytest.raises(GenkitError, match='no content type'):
        build_converse_request('amazon.nova-lite-v1:0', request)


# --- Tools and tool choice ---------------------------------------------------


WEATHER_TOOL = ToolDefinition(
    name='weather',
    description='Get the weather',
    input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
)


def test_tools_become_tool_specs() -> None:
    request = user_text_request(tools=[WEATHER_TOOL])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    tool_spec = kwargs['toolConfig']['tools'][0]['toolSpec']
    assert tool_spec['name'] == 'weather'
    assert tool_spec['description'] == 'Get the weather'
    assert tool_spec['inputSchema']['json']['properties'] == {'city': {'type': 'string'}}
    # No toolChoice unless requested; Bedrock defaults to auto.
    assert 'toolChoice' not in kwargs['toolConfig']


def test_tool_schema_defaults_injected() -> None:
    tool = ToolDefinition(name='noop', description='')
    schema = to_bedrock_tool(tool)['toolSpec']['inputSchema']['json']
    assert schema['type'] == 'object'
    assert schema['properties'] == {}
    assert schema['$schema'] == 'http://json-schema.org/draft-07/schema#'


def test_empty_tool_description_is_omitted() -> None:
    # Bedrock rejects an empty description, and a tool declared without a
    # docstring reaches the plugin with description ''.
    tool_spec = to_bedrock_tool(ToolDefinition(name='noop', description=''))['toolSpec']
    assert 'description' not in tool_spec
    assert tool_spec['name'] == 'noop'


def test_tool_without_name_errors() -> None:
    with pytest.raises(GenkitError, match='tool name required'):
        to_bedrock_tool(ToolDefinition(name='', description=''))


@pytest.mark.parametrize(
    'tool_choice,expected',
    [
        ('auto', {'auto': {}}),
        ('required', {'any': {}}),
        ('any', {'any': {}}),
        ('weather', {'tool': {'name': 'weather'}}),
    ],
)
def test_tool_choice_mapping(tool_choice, expected) -> None:
    request = user_text_request(tools=[WEATHER_TOOL], config=BedrockConfig(tool_choice=tool_choice))
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert kwargs['toolConfig']['toolChoice'] == expected


def test_tool_choice_none_omits_tool_config_entirely() -> None:
    request = user_text_request(tools=[WEATHER_TOOL], config=BedrockConfig(tool_choice='none'))
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert 'toolConfig' not in kwargs


def test_request_tool_choice_used_when_config_silent() -> None:
    request = user_text_request(tools=[WEATHER_TOOL], tool_choice='required')
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert kwargs['toolConfig']['toolChoice'] == {'any': {}}


def test_unknown_named_tool_choice_errors() -> None:
    request = user_text_request(tools=[WEATHER_TOOL], config=BedrockConfig(tool_choice='no-such-tool'))
    with pytest.raises(GenkitError, match='does not match any declared tool'):
        build_converse_request('amazon.nova-lite-v1:0', request)


def test_tool_choice_without_tools_is_ignored() -> None:
    request = user_text_request(config=BedrockConfig(tool_choice='weather'))
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert 'toolConfig' not in kwargs


def test_trailing_assistant_message_rejected_when_tools_present() -> None:
    # Bedrock would reject this request anyway; failing here names the
    # constraint instead of silently dropping the caller's prefill.
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='q'))]),
            Message(role=Role.MODEL, content=[Part(root=TextPart(text='thinking...'))]),
        ],
        tools=[WEATHER_TOOL],
    )
    with pytest.raises(GenkitError, match='ending with an assistant message') as excinfo:
        build_converse_request('amazon.nova-lite-v1:0', request)
    assert excinfo.value.status == 'INVALID_ARGUMENT'


def test_trailing_assistant_message_kept_without_tools() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='q'))]),
            Message(role=Role.MODEL, content=[Part(root=TextPart(text='a'))]),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert [m['role'] for m in kwargs['messages']] == ['user', 'assistant']


def test_trailing_assistant_kept_under_tool_choice_none() -> None:
    # No toolConfig is sent under "none", so the prefill has nothing to violate.
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='q'))]),
            Message(role=Role.MODEL, content=[Part(root=TextPart(text='a'))]),
        ],
        tools=[WEATHER_TOOL],
        config=BedrockConfig(tool_choice='none'),
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert [m['role'] for m in kwargs['messages']] == ['user', 'assistant']
    assert 'toolConfig' not in kwargs


def test_additional_model_request_fields_forwarded_verbatim() -> None:
    thinking = {'thinking': {'type': 'enabled', 'budget_tokens': 2048}}
    request = user_text_request(config=BedrockConfig(additional_model_request_fields=thinking))
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert kwargs['additionalModelRequestFields'] == thinking


# --- Reasoning round-trip (request side) -------------------------------------


def test_bedrock_reasoning_part_round_trips() -> None:
    # The signature is a string on the Converse wire; it must replay verbatim.
    part = Part(
        root=ReasoningPart(
            reasoning='step by step',
            metadata={REASONING_SIGNATURE_METADATA_KEY: 'sig-abc123=='},
        )
    )
    request = ModelRequest(
        messages=[
            Message(role=Role.MODEL, content=[part]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='go on'))]),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert kwargs['messages'][0]['content'] == [
        {'reasoningContent': {'reasoningText': {'text': 'step by step', 'signature': 'sig-abc123=='}}}
    ]


def test_bytes_signature_form_is_tolerated() -> None:
    part = Part(
        root=ReasoningPart(
            reasoning='step by step',
            metadata={REASONING_SIGNATURE_METADATA_KEY: b'sig-abc123=='},
        )
    )
    request = ModelRequest(messages=[Message(role=Role.MODEL, content=[part])])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    block = kwargs['messages'][0]['content'][0]['reasoningContent']['reasoningText']
    assert block['signature'] == 'sig-abc123=='


def test_redacted_content_emitted_before_signed_text() -> None:
    part = Part(
        root=ReasoningPart(
            reasoning='visible part',
            metadata={
                REASONING_SIGNATURE_METADATA_KEY: 'sig',
                REDACTED_CONTENT_METADATA_KEY: base64.b64encode(b'redacted-blob').decode(),
            },
        )
    )
    request = ModelRequest(messages=[Message(role=Role.MODEL, content=[part])])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    blocks = kwargs['messages'][0]['content']
    assert blocks[0] == {'reasoningContent': {'redactedContent': b'redacted-blob'}}
    assert blocks[1]['reasoningContent']['reasoningText']['text'] == 'visible part'


def test_redacted_only_reasoning_part_still_replays() -> None:
    # Redacted-only parts have reasoning == '' and must not be dropped.
    part = Part(
        root=ReasoningPart(
            reasoning='',
            metadata={REDACTED_CONTENT_METADATA_KEY: base64.b64encode(b'blob').decode()},
        )
    )
    request = ModelRequest(messages=[Message(role=Role.MODEL, content=[part])])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert kwargs['messages'][0]['content'] == [{'reasoningContent': {'redactedContent': b'blob'}}]


def test_generic_reasoning_part_is_not_replayed() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='q'))]),
            Message(role=Role.MODEL, content=[Part(root=ReasoningPart(reasoning='foreign thoughts'))]),
        ]
    )
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    # The reasoning-only assistant message converts to zero blocks and drops.
    assert [m['role'] for m in kwargs['messages']] == ['user']


# --- Response conversion ------------------------------------------------------


def converse_response(**overrides) -> dict:
    response = {
        'output': {'message': {'role': 'assistant', 'content': [{'text': 'hello'}]}},
        'stopReason': 'end_turn',
        'usage': {'inputTokens': 10, 'outputTokens': 5, 'totalTokens': 15},
    }
    response.update(overrides)
    return response


def test_text_response_round_trip() -> None:
    request = user_text_request()
    response = to_model_response(converse_response(), request)
    assert response.message is not None
    assert response.message.role == Role.MODEL
    assert response.message.content[0].root.text == 'hello'
    assert response.finish_reason == FinishReason.STOP
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 15
    assert response.request is request


def test_tool_use_block_becomes_tool_request_part() -> None:
    blocks = [{'toolUse': {'toolUseId': 'call-1', 'name': 'weather', 'input': {'city': 'Lagos'}}}]
    parts = content_blocks_to_parts(blocks)
    tool_request = parts[0].root.tool_request
    assert tool_request is not None
    assert tool_request.ref == 'call-1'
    assert tool_request.name == 'weather'
    assert tool_request.input == {'city': 'Lagos'}


def test_tool_use_with_missing_input_gets_empty_object() -> None:
    parts = content_blocks_to_parts([{'toolUse': {'toolUseId': 'x', 'name': 'noop'}}])
    tool_request = parts[0].root.tool_request
    assert tool_request is not None
    assert tool_request.input == {}


def test_tool_input_coerced_toward_schema() -> None:
    tool = ToolDefinition(
        name='calc',
        description='',
        input_schema={
            'type': 'object',
            'properties': {
                'count': {'type': 'integer'},
                'ratio': {'type': 'number'},
                'enabled': {'type': 'boolean'},
                'note': {'type': 'string'},
            },
        },
    )
    blocks = [
        {
            'toolUse': {
                'toolUseId': 'c1',
                'name': 'calc',
                'input': {'count': '7', 'ratio': '0.5', 'enabled': 'true', 'note': 'hi', 'extra': '1'},
            }
        }
    ]
    parts = content_blocks_to_parts(blocks, [tool])
    tool_request = parts[0].root.tool_request
    assert tool_request is not None
    assert tool_request.input == {'count': 7, 'ratio': 0.5, 'enabled': True, 'note': 'hi', 'extra': '1'}


@pytest.mark.parametrize('value,expected', [(7, '7'), (7.5, '7.5'), (True, True)], ids=['int', 'float', 'bool'])
def test_number_coerced_to_string_schema(value: object, expected: object) -> None:
    # A wire number against a string field fails pydantic validation unless
    # it is coerced. Booleans are left alone.
    tool = ToolDefinition(
        name='calc',
        description='',
        input_schema={'type': 'object', 'properties': {'note': {'type': 'string'}}},
    )
    blocks = [{'toolUse': {'toolUseId': 'c1', 'name': 'calc', 'input': {'note': value}}}]
    parts = content_blocks_to_parts(blocks, [tool])
    tool_request = parts[0].root.tool_request
    assert tool_request is not None
    assert tool_request.input == {'note': expected}


def test_tool_input_float_truncates_for_integer_schema() -> None:
    tool = ToolDefinition(
        name='calc', description='', input_schema={'type': 'object', 'properties': {'n': {'type': 'integer'}}}
    )
    parts = content_blocks_to_parts([{'toolUse': {'toolUseId': 'c', 'name': 'calc', 'input': {'n': 7.9}}}], [tool])
    tool_request = parts[0].root.tool_request
    assert tool_request is not None
    assert tool_request.input == {'n': 7}


def test_reasoning_text_block_becomes_reasoning_part_with_both_keys() -> None:
    blocks = [{'reasoningContent': {'reasoningText': {'text': 'because', 'signature': 'sig'}}}]
    parts = content_blocks_to_parts(blocks)
    root = parts[0].root
    assert root.reasoning == 'because'
    assert root.metadata is not None
    assert root.metadata['signature'] == 'sig'
    assert root.metadata[REASONING_SIGNATURE_METADATA_KEY] == 'sig'


def test_redacted_content_block_becomes_reasoning_part() -> None:
    parts = content_blocks_to_parts([{'reasoningContent': {'redactedContent': b'blob'}}])
    root = parts[0].root
    assert root.reasoning == ''
    assert root.metadata is not None
    # Stored as a base64 string so the part survives JSON serialization.
    assert root.metadata[REDACTED_CONTENT_METADATA_KEY] == base64.b64encode(b'blob').decode()


def test_reasoning_survives_response_to_request_round_trip() -> None:
    # The full circle: wire response blocks -> Genkit parts -> wire request
    # blocks, byte-identical, including non-UTF8 redacted content.
    redacted_blob = b'\x89\xff\x00binary'
    response_blocks = [
        {'reasoningContent': {'reasoningText': {'text': 'because', 'signature': 'sig-abc123=='}}},
        {'reasoningContent': {'redactedContent': redacted_blob}},
    ]
    parts = content_blocks_to_parts(response_blocks)
    request = ModelRequest(messages=[Message(role=Role.MODEL, content=parts)])
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    replayed = kwargs['messages'][0]['content']
    assert replayed[0] == {'reasoningContent': {'reasoningText': {'text': 'because', 'signature': 'sig-abc123=='}}}
    assert replayed[1] == {'reasoningContent': {'redactedContent': redacted_blob}}


def test_reasoning_parts_serialize_to_json() -> None:
    # Tracing serializes action output with model_dump_json; raw bytes in
    # metadata would crash it for non-UTF8 redacted blobs.
    parts = content_blocks_to_parts([
        {'reasoningContent': {'reasoningText': {'text': 'because', 'signature': 'sig'}}},
        {'reasoningContent': {'redactedContent': b'\x89\xff\x00binary'}},
    ])
    message = Message(role=Role.MODEL, content=parts)
    assert message.model_dump_json()


def test_empty_reasoning_block_is_skipped() -> None:
    assert content_blocks_to_parts([{'reasoningContent': {'reasoningText': {}}}]) == []


def test_unknown_reasoning_variant_errors() -> None:
    with pytest.raises(GenkitError, match='unhandled reasoning content variant'):
        content_blocks_to_parts([{'reasoningContent': {'futureThing': 1}}])


def test_unknown_response_block_errors() -> None:
    with pytest.raises(GenkitError, match='unhandled response content variant'):
        content_blocks_to_parts([{'video': {'format': 'mp4'}}])


def test_empty_response_content_yields_placeholder_text_part() -> None:
    response = converse_response(
        output={'message': {'role': 'assistant', 'content': []}},
        stopReason='guardrail_intervened',
    )
    model_response = to_model_response(response, user_text_request())
    assert model_response.message is not None
    assert model_response.message.content[0].root.text == ''
    assert model_response.finish_reason == FinishReason.BLOCKED


def test_missing_output_tolerated() -> None:
    response = converse_response(output={})
    model_response = to_model_response(response, user_text_request())
    assert model_response.message is not None
    assert model_response.message.content[0].root.text == ''


def test_usage_maps_cache_read_tokens_only() -> None:
    usage = usage_from_response({
        'inputTokens': 4,
        'outputTokens': 6,
        'totalTokens': 110,
        'cacheReadInputTokens': 100,
        'cacheWriteInputTokens': 50,
    })
    assert usage is not None
    assert usage.input_tokens == 4
    assert usage.output_tokens == 6
    # Totals are trusted verbatim from AWS, never recomputed.
    assert usage.total_tokens == 110
    assert usage.cached_content_tokens == 100


def test_usage_none_when_absent() -> None:
    assert usage_from_response(None) is None
