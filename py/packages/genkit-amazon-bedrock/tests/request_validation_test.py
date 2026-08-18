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

"""Validates built requests against the bedrock-runtime service model.

Several Converse fields are ``NonEmptyString`` (``min: 1``) and botocore
enforces that client-side, so an empty value fails the call before it reaches
AWS. These tests run the real validator over the request builder's output, which
catches that whole class of defect without credentials or network access.

Every request is validated against both Converse and ConverseStream: one
builder feeds both operations, so a field that only one of them models would
otherwise fail client-side on the streaming path alone.
"""

import base64

import botocore.session
import pytest
from botocore.validate import ParamValidator
from genkit_amazon_bedrock.config import BedrockConfig
from genkit_amazon_bedrock.converters import build_converse_request, cache_point_part

from genkit import (
    Message,
    ModelRequest,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolDefinition,
)

_SERVICE_MODEL = botocore.session.get_session().get_service_model('bedrock-runtime')
CONVERSE_INPUT_SHAPES = {
    operation: _SERVICE_MODEL.operation_model(operation).input_shape for operation in ('Converse', 'ConverseStream')
}

PNG_B64 = base64.b64encode(b'\x89PNG\r\n\x1a\nfakeimagedata').decode()


def assert_valid_request(kwargs: dict) -> None:
    """Asserts botocore accepts every parameter, for both operations."""
    for operation, shape in CONVERSE_INPUT_SHAPES.items():
        report = ParamValidator().validate(kwargs, shape)
        assert not report.has_errors(), f'{operation}: {report.generate_report()}'


def assert_valid_converse_request(request: ModelRequest) -> dict:
    """Builds the request and asserts botocore accepts every parameter."""
    kwargs = build_converse_request('amazon.nova-lite-v1:0', request)
    assert_valid_request(kwargs)
    return kwargs


def test_undocumented_tool_passes_validation() -> None:
    # A tool declared without a docstring reaches the plugin with description
    # '', which the NonEmptyString floor on toolSpec.description rejects.
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
        tools=[ToolDefinition(name='noop', description='', input_schema={'type': 'object', 'properties': {}})],
    )
    kwargs = assert_valid_converse_request(request)
    assert 'description' not in kwargs['toolConfig']['tools'][0]['toolSpec']


def test_blank_system_prompt_passes_validation() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text=''))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
        ]
    )
    assert_valid_converse_request(request)


def test_empty_assistant_text_passes_validation() -> None:
    # to_model_response emits a '' text part for guardrail-blocked responses;
    # replaying it must stay valid (ContentBlock.text has no floor).
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
            Message(role=Role.MODEL, content=[Part(root=TextPart(text=''))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='again'))]),
        ]
    )
    assert_valid_converse_request(request)


def test_tool_round_trip_passes_validation() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='weather?'))]),
            Message(
                role=Role.MODEL,
                content=[
                    Part.model_validate({
                        'toolRequest': {'ref': 'call-1', 'name': 'weather', 'input': {'city': 'Lagos'}}
                    })
                ],
            ),
            Message(
                role=Role.TOOL,
                content=[
                    Part.model_validate({'toolResponse': {'ref': 'call-1', 'name': 'weather', 'output': {'c': 21}}})
                ],
            ),
        ],
        tools=[
            ToolDefinition(
                name='weather',
                description='Get the weather',
                input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
            )
        ],
    )
    assert_valid_converse_request(request)


def test_media_and_cache_points_pass_validation() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='rules')), cache_point_part()]),
            Message(
                role=Role.USER,
                content=[
                    Part.model_validate({'media': {'url': f'data:image/png;base64,{PNG_B64}'}}),
                    Part.model_validate({'media': {'url': f'data:application/pdf;base64,{PNG_B64}'}}),
                    Part(root=TextPart(text='what is this?')),
                    cache_point_part(),
                ],
            ),
        ]
    )
    assert_valid_converse_request(request)


def test_reasoning_replay_passes_validation() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='think'))]),
            Message(
                role=Role.MODEL,
                content=[
                    Part(
                        root=ReasoningPart(
                            reasoning='step one',
                            metadata={'bedrockReasoningSignature': 'sig-abc', 'signature': 'sig-abc'},
                        )
                    ),
                    Part(root=TextPart(text='done')),
                ],
            ),
            Message(role=Role.USER, content=[Part(root=TextPart(text='continue'))]),
        ]
    )
    assert_valid_converse_request(request)


@pytest.mark.parametrize('model_id', ['anthropic.claude-sonnet-4-5-20250929-v1:0', 'amazon.nova-lite-v1:0'])
def test_inference_config_passes_validation(model_id: str) -> None:
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
        config=BedrockConfig(temperature=0.0, top_p=0.9, max_output_tokens=256, stop_sequences=['STOP']),
    )
    assert_valid_request(build_converse_request(model_id, request))
