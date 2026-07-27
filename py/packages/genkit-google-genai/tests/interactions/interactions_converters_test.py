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

"""Tests for Interactions API converters."""

from __future__ import annotations

import pytest
from genkit_google_genai._interactions.converters import (
    ensure_tool_ids,
    from_interaction,
    from_interaction_content,
    from_interaction_step,
    from_interaction_sync,
    from_thought_step,
    parts_from_steps,
    to_interaction_content,
    to_interaction_role,
    to_interaction_steps,
    to_interaction_tool,
)
from genkit_google_genai._interactions.options import ClientOptions
from google.genai.interactions import Content, Interaction, Step, ThoughtStep
from pydantic import TypeAdapter

from genkit import (
    CustomPart,
    Media,
    MediaPart,
    Part,
    ReasoningPart,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)
from genkit.model import Message, ToolDefinition

ContentAdapter: TypeAdapter[Content] = TypeAdapter(Content)
StepAdapter: TypeAdapter[Step] = TypeAdapter(Step)


def part_dict(part: Part) -> dict:
    return part.root.model_dump(by_alias=True, exclude_none=True)


class TestEnsureToolIds:
    def test_assigns_ids_to_tool_requests_without_refs(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(ToolRequestPart(tool_request=ToolRequest(name='tool1', input={}))),
                    Part(ToolRequestPart(tool_request=ToolRequest(name='tool2', input={}))),
                ],
            )
        ]
        result = ensure_tool_ids(messages)
        req1 = result[0].content[0].root.tool_request
        req2 = result[0].content[1].root.tool_request
        assert req1 is not None and req1.ref and req1.ref.startswith('genkit-auto-id-')
        assert req2 is not None and req2.ref and req2.ref.startswith('genkit-auto-id-')
        assert req1.ref != req2.ref

    def test_assigns_matching_ids_to_tool_responses(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(ToolRequestPart(tool_request=ToolRequest(name='tool1', input={}))),
                    Part(ToolRequestPart(tool_request=ToolRequest(name='tool2', input={}))),
                ],
            ),
            Message(
                role='tool',
                content=[
                    Part(ToolResponsePart(tool_response=ToolResponse(name='tool1', output={}))),
                    Part(ToolResponsePart(tool_response=ToolResponse(name='tool2', output={}))),
                ],
            ),
        ]
        result = ensure_tool_ids(messages)
        req1 = result[0].content[0].root.tool_request
        req2 = result[0].content[1].root.tool_request
        res1 = result[1].content[0].root.tool_response
        res2 = result[1].content[1].root.tool_response
        assert req1 and req1.ref and res1 and res1.ref == req1.ref
        assert req2 and req2.ref and res2 and res2.ref == req2.ref

    def test_assigns_orphan_id_without_matching_request(self) -> None:
        messages = [
            Message(
                role='tool',
                content=[Part(ToolResponsePart(tool_response=ToolResponse(name='tool1', output={})))],
            )
        ]
        result = ensure_tool_ids(messages)
        res1 = result[0].content[0].root.tool_response
        assert res1 and res1.ref and res1.ref.startswith('genkit-orphan-id-')

    def test_preserves_existing_refs(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part(ToolRequestPart(tool_request=ToolRequest(name='tool1', input={}, ref='existing-id')))],
            )
        ]
        result = ensure_tool_ids(messages)
        req1 = result[0].content[0].root.tool_request
        assert req1 and req1.ref == 'existing-id'


class TestToInteractionRole:
    def test_user(self) -> None:
        assert to_interaction_role('user') == 'user'

    def test_model(self) -> None:
        assert to_interaction_role('model') == 'model'

    def test_tool_maps_to_user(self) -> None:
        assert to_interaction_role('tool') == 'user'

    def test_system_raises(self) -> None:
        with pytest.raises(ValueError, match='system_instruction'):
            to_interaction_role('system')


class TestToInteractionTool:
    def test_converts_tool_definition(self) -> None:
        tool = ToolDefinition(
            name='myFunc',
            description='desc',
            input_schema={'type': 'object', 'properties': {'arg': {'type': 'string'}}},
        )
        result = to_interaction_tool(tool)
        assert result == {
            'type': 'function',
            'name': 'myFunc',
            'description': 'desc',
            'parameters': {'type': 'object', 'properties': {'arg': {'type': 'string'}}},
        }


class TestToInteractionContent:
    def test_text(self) -> None:
        result = to_interaction_content(Part(TextPart(text='Hello')))
        assert result == {'type': 'text', 'text': 'Hello'}

    def test_image_data(self) -> None:
        result = to_interaction_content(
            Part(MediaPart(media=Media(url='data:image/png;base64,DATA', content_type='image/png')))
        )
        assert result == {'type': 'image', 'data': 'DATA', 'mime_type': 'image/png'}

    def test_image_uri(self) -> None:
        result = to_interaction_content(
            Part(MediaPart(media=Media(url='gs://bucket/image.png', content_type='image/png')))
        )
        assert result == {'type': 'image', 'uri': 'gs://bucket/image.png', 'mime_type': 'image/png'}

    def test_audio(self) -> None:
        result = to_interaction_content(
            Part(MediaPart(media=Media(url='data:audio/mp3;base64,DATA', content_type='audio/mp3')))
        )
        assert result == {'type': 'audio', 'data': 'DATA', 'mime_type': 'audio/mp3'}

    def test_document(self) -> None:
        result = to_interaction_content(
            Part(MediaPart(media=Media(url='gs://bucket/doc.pdf', content_type='application/pdf')))
        )
        assert result == {'type': 'document', 'uri': 'gs://bucket/doc.pdf', 'mime_type': 'application/pdf'}

    def test_unsupported_media_raises(self) -> None:
        with pytest.raises(ValueError, match='Unsupported media type'):
            to_interaction_content(Part(MediaPart(media=Media(url='https://example.com/x', content_type='text/plain'))))


class TestToInteractionSteps:
    def test_tool_request(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part(ToolRequestPart(tool_request=ToolRequest(name='func', input={'a': 1}, ref='ref1')))],
            )
        ]
        assert to_interaction_steps(messages) == [
            {'type': 'function_call', 'name': 'func', 'arguments': {'a': 1}, 'id': 'ref1'}
        ]

    def test_tool_response(self) -> None:
        messages = [
            Message(
                role='tool',
                content=[
                    Part(
                        root=ToolResponsePart(
                            tool_response=ToolResponse(name='func', output={'result': 'ok'}, ref='ref1')
                        )
                    )
                ],
            )
        ]
        assert to_interaction_steps(messages) == [
            {'type': 'function_result', 'name': 'func', 'result': {'result': 'ok'}, 'call_id': 'ref1'}
        ]

    def test_model_output_grouping(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part(TextPart(text='Thinking')), Part(TextPart(text='Done'))],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'model_output',
                'content': [{'type': 'text', 'text': 'Thinking'}, {'type': 'text', 'text': 'Done'}],
            }
        ]

    def test_system_role_rejected(self) -> None:
        messages = [Message(role='system', content=[Part(TextPart(text='be terse'))])]
        with pytest.raises(ValueError, match='system_instruction'):
            to_interaction_steps(messages)

    def test_code_execution_call_always_sends_python(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(
                        root=CustomPart(
                            custom={'executableCode': {'code': 'print(1)', 'language': 'PYTHON'}},
                            metadata={'callId': 'c1'},
                        )
                    )
                ],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'code_execution_call',
                'id': 'c1',
                'arguments': {'code': 'print(1)', 'language': 'python'},
            }
        ]

    def test_google_search_call(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(
                        root=CustomPart(
                            custom={'googleSearchCall': {'id': 'gs1', 'arguments': {'queries': ['genkit']}}},
                            metadata={'thoughtSignature': 'sig'},
                        )
                    )
                ],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'google_search_call',
                'id': 'gs1',
                'arguments': {'queries': ['genkit']},
                'signature': 'sig',
            }
        ]

    def test_thought_becomes_its_own_step(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(
                        root=ReasoningPart(
                            reasoning='plan the answer',
                            metadata={'thoughtSignature': 'sig-t'},
                        )
                    )
                ],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'thought',
                'summary': [{'type': 'text', 'text': 'plan the answer'}],
                'signature': 'sig-t',
            }
        ]

    def test_mixed_model_turn_flushes_standalone_steps_before_inline_text(self) -> None:
        """Standalone steps (thought/tool) emit immediately; text waits until end of turn.

        That means [thought, text, tool_request] becomes thought, function_call,
        then model_output(text) — not thought, model_output, function_call.
        """
        messages = [
            Message(
                role='model',
                content=[
                    Part(root=ReasoningPart(reasoning='think', metadata={'thoughtSignature': 's'})),
                    Part(TextPart(text='calling tool')),
                    Part(ToolRequestPart(tool_request=ToolRequest(name='lookup', input={'q': 1}, ref='c1'))),
                ],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'thought',
                'summary': [{'type': 'text', 'text': 'think'}],
                'signature': 's',
            },
            {'type': 'function_call', 'name': 'lookup', 'arguments': {'q': 1}, 'id': 'c1'},
            {
                'type': 'model_output',
                'content': [{'type': 'text', 'text': 'calling tool'}],
            },
        ]


class TestFromInteractionContent:
    def test_text(self) -> None:
        result = from_interaction_content(
            ContentAdapter.validate_python({
                'type': 'text',
                'text': 'Hello world',
                'annotations': [{'start_index': 0, 'end_index': 5, 'source': 'source'}],
            })
        )
        assert part_dict(result) == {
            'text': 'Hello world',
            'metadata': {
                'annotations': [{'start_index': 0, 'end_index': 5, 'source': 'source', 'type': 'file_citation'}]
            },
        }

    def test_image_data(self) -> None:
        result = from_interaction_content(
            ContentAdapter.validate_python({'type': 'image', 'data': 'BASE64DATA', 'mime_type': 'image/png'})
        )
        assert part_dict(result) == {'media': {'url': 'data:image/png;base64,BASE64DATA', 'contentType': 'image/png'}}

    def test_image_resolution(self) -> None:
        result = from_interaction_content(
            ContentAdapter.validate_python({
                'type': 'image',
                'uri': 'gs://bucket/image.png',
                'mime_type': 'image/png',
                'resolution': 'high',
            })
        )
        assert part_dict(result) == {
            'media': {'url': 'gs://bucket/image.png', 'contentType': 'image/png'},
            'metadata': {'resolution': 'high'},
        }

    def test_thought(self) -> None:
        step = ThoughtStep.model_validate({
            'type': 'thought',
            'signature': 'SIG',
            'summary': [{'type': 'text', 'text': 'Thinking...'}],
        })
        result = from_thought_step(step)
        assert part_dict(result) == {
            'reasoning': 'Thinking...',
            'metadata': {'thoughtSignature': 'SIG'},
            'custom': {'thought': step.model_dump(mode='python')},
        }


class TestFromInteractionStep:
    def test_model_output_includes_empty_annotations(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({'type': 'model_output', 'content': [{'type': 'text', 'text': 'Hello'}]})
        )
        root = result[0].root
        assert isinstance(root, TextPart)
        assert root.text == 'Hello'
        assert root.metadata is not None
        assert 'annotations' in root.metadata

    def test_user_input_dropped(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({'type': 'user_input', 'content': [{'type': 'text', 'text': 'Hello'}]})
        )
        assert result == []

    def test_function_call_is_tool_request(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({
                'type': 'function_call',
                'id': 'c1',
                'name': 'get_weather',
                'arguments': {'city': 'Austin'},
            })
        )
        root = result[0].root
        assert isinstance(root, ToolRequestPart)
        assert root.tool_request is not None
        assert root.tool_request.name == 'get_weather'
        assert root.tool_request.input == {'city': 'Austin'}
        assert root.tool_request.ref == 'c1'

    def test_function_result_is_tool_response(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({
                'type': 'function_result',
                'call_id': 'c1',
                'name': 'get_weather',
                'result': {'temp': 92},
                'is_error': False,
            })
        )
        root = result[0].root
        assert isinstance(root, ToolResponsePart)
        assert root.tool_response is not None
        assert root.tool_response.name == 'get_weather'
        assert root.tool_response.output == {'temp': 92}
        assert root.tool_response.ref == 'c1'
        assert root.metadata == {'isError': False}

    def test_google_search_call_is_custom_part(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({
                'type': 'google_search_call',
                'id': 'gs1',
                'arguments': {'queries': ['genkit']},
                'signature': 'sig',
            })
        )
        root = result[0].root
        assert isinstance(root, CustomPart)
        assert root.custom == {'googleSearchCall': {'id': 'gs1', 'arguments': {'queries': ['genkit']}}}
        assert root.metadata == {'thoughtSignature': 'sig'}

    def test_code_execution_call_is_custom_part(self) -> None:
        result = from_interaction_step(
            StepAdapter.validate_python({
                'type': 'code_execution_call',
                'id': 'ce1',
                'arguments': {'code': 'print(1)', 'language': 'python'},
            })
        )
        root = result[0].root
        assert isinstance(root, CustomPart)
        assert root.custom == {'executableCode': {'code': 'print(1)', 'language': 'python'}}
        assert root.metadata == {'callId': 'ce1'}


class TestInboundFlattensToSingleModelMessage:
    """Every Interaction response becomes one role=model Message of flattened parts."""

    def test_mixed_tape_drops_user_input_and_keeps_order(self) -> None:
        interaction = Interaction.model_validate({
            'id': 'ix-1',
            'status': 'completed',
            'environment_id': 'env-1',
            'steps': [
                {'type': 'user_input', 'content': [{'type': 'text', 'text': 'weather?'}]},
                {
                    'type': 'thought',
                    'signature': 'sig',
                    'summary': [{'type': 'text', 'text': 'need a tool'}],
                },
                {'type': 'model_output', 'content': [{'type': 'text', 'text': 'checking'}]},
                {
                    'type': 'function_call',
                    'id': 'c1',
                    'name': 'get_weather',
                    'arguments': {'city': 'Austin'},
                },
            ],
        })
        op = from_interaction(interaction)
        assert op.done is True
        assert op.output is not None
        message = op.output.message
        assert message is not None
        assert message.role == 'model'
        assert message.metadata == {
            'interactionId': 'ix-1',
            'environmentId': 'env-1',
            'interactionStatus': 'completed',
        }
        assert [type(part.root).__name__ for part in message.content] == [
            'ReasoningPart',
            'TextPart',
            'ToolRequestPart',
        ]
        assert message.content[0].root.reasoning == 'need a tool'
        assert message.content[1].root.text == 'checking'
        assert message.content[2].root.tool_request.ref == 'c1'

    def test_already_resolved_tool_pair_stays_inside_model_message(self) -> None:
        """A tape that already has function_result still flattens into one model turn."""
        steps = [
            StepAdapter.validate_python({
                'type': 'function_call',
                'id': 'c1',
                'name': 'get_weather',
                'arguments': {'city': 'Austin'},
            }),
            StepAdapter.validate_python({
                'type': 'function_result',
                'call_id': 'c1',
                'name': 'get_weather',
                'result': {'temp': 92},
            }),
            StepAdapter.validate_python({
                'type': 'model_output',
                'content': [{'type': 'text', 'text': '92F'}],
            }),
        ]
        parts = parts_from_steps(steps)
        assert [type(part.root).__name__ for part in parts] == [
            'ToolRequestPart',
            'ToolResponsePart',
            'TextPart',
        ]


class TestFunctionCallRoundTrip:
    """The bug we hit: function_call must be ToolRequestPart, not CustomPart."""

    def test_outbound_then_inbound_preserves_tool_request(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(
                        ToolRequestPart(
                            tool_request=ToolRequest(name='get_weather', input={'city': 'Austin'}, ref='c1')
                        )
                    )
                ],
            )
        ]
        steps = to_interaction_steps(messages)
        assert steps == [{'type': 'function_call', 'name': 'get_weather', 'arguments': {'city': 'Austin'}, 'id': 'c1'}]
        inbound = from_interaction_step(StepAdapter.validate_python(steps[0]))
        root = inbound[0].root
        assert isinstance(root, ToolRequestPart)
        assert root.tool_request is not None
        assert root.tool_request.name == 'get_weather'
        assert root.tool_request.input == {'city': 'Austin'}
        assert root.tool_request.ref == 'c1'

    def test_inbound_then_outbound_preserves_function_call(self) -> None:
        step = StepAdapter.validate_python({
            'type': 'function_call',
            'id': 'c1',
            'name': 'get_weather',
            'arguments': {'city': 'Austin'},
        })
        part = from_interaction_step(step)[0]
        assert isinstance(part.root, ToolRequestPart)
        again = to_interaction_steps([Message(role='model', content=[part])])
        assert again == [{'type': 'function_call', 'name': 'get_weather', 'arguments': {'city': 'Austin'}, 'id': 'c1'}]

    def test_thought_round_trip_keeps_signature(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(
                        root=ReasoningPart(
                            reasoning='plan',
                            metadata={'thoughtSignature': 'sig'},
                        )
                    )
                ],
            )
        ]
        steps = to_interaction_steps(messages)
        inbound = from_interaction_step(StepAdapter.validate_python(steps[0]))
        root = inbound[0].root
        assert isinstance(root, ReasoningPart)
        assert root.reasoning == 'plan'
        assert root.metadata == {'thoughtSignature': 'sig'}


class TestClientOptionsWireFormat:
    def test_operation_metadata_uses_camel_case_keys(self) -> None:
        op = from_interaction(
            Interaction.model_validate({'id': '123', 'status': 'in_progress'}),
            ClientOptions(api_key='k', base_url='https://example.test', api_version='v1beta'),
        )
        assert op.metadata == {
            'clientOptions': {
                'apiKey': 'k',
                'baseUrl': 'https://example.test',
                'apiVersion': 'v1beta',
            }
        }


class TestFromInteractionStatusMapping:
    def test_cancelled(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'cancelled'}))
        assert result.done is True
        assert result.output is not None
        assert result.output.finish_reason == 'aborted'
        assert result.output.finish_message == 'Operation cancelled'
        assert part_dict(result.output.message.content[0]) == {'text': 'Operation cancelled.'}

    def test_failed_exits_poll_loop(self) -> None:
        result = from_interaction(
            Interaction.model_validate({
                'id': '123',
                'status': 'failed',
                'steps': [{'type': 'model_output', 'error': {'code': 3, 'message': 'boom'}}],
            })
        )
        assert result.done is True
        assert result.error is not None
        assert result.error.message == 'boom'

    def test_failed_without_step_error_falls_back(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'failed'}))
        assert result.done is True
        assert result.error is not None
        assert result.error.message == 'Interaction failed'

    def test_requires_action_leaves_done_unset(self) -> None:
        interaction = Interaction.model_validate({
            'id': '123',
            'status': 'requires_action',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'approve plan'}]}],
        })
        result = from_interaction(interaction)
        assert result.id == '123'
        assert result.done is None
        assert result.output is None
        assert not (result.metadata or {}).get('interaction_status')

    def test_in_progress(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'in_progress'}))
        assert result.done is False

    def test_queued_keeps_polling(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'queued'}))
        assert result.done is False

    def test_incomplete_surfaces_partial_steps(self) -> None:
        result = from_interaction(
            Interaction.model_validate({
                'id': '123',
                'status': 'incomplete',
                'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'partial'}]}],
            })
        )
        assert result.done is True
        assert result.error is None
        assert result.output is not None
        assert result.output.finish_reason == 'length'
        assert result.output.finish_message == 'Interaction incomplete (truncated output)'
        assert result.output.message is not None
        assert result.output.message.content[0].root.text == 'partial'
        assert result.output.message.metadata is not None
        assert result.output.message.metadata.get('interactionStatus') == 'incomplete'

    def test_incomplete_without_steps_errors(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'incomplete'}))
        assert result.done is True
        assert result.output is None
        assert result.error is not None

    def test_budget_exceeded_surfaces_partial_steps(self) -> None:
        result = from_interaction(
            Interaction.model_validate({
                'id': '123',
                'status': 'budget_exceeded',
                'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'draft'}]}],
            })
        )
        assert result.done is True
        assert result.error is None
        assert result.output is not None
        assert result.output.finish_reason == 'aborted'
        assert result.output.finish_message == 'Interaction exceeded its budget'
        assert result.output.message is not None
        assert result.output.message.content[0].root.text == 'draft'
        assert result.output.message.metadata is not None
        assert result.output.message.metadata.get('interactionStatus') == 'budget_exceeded'

    def test_budget_exceeded_without_steps_errors(self) -> None:
        result = from_interaction(Interaction.model_validate({'id': '123', 'status': 'budget_exceeded'}))
        assert result.done is True
        assert result.output is None
        assert result.error is not None
        assert 'budget' in (result.error.message or '').lower()


class TestFromInteractionSync:
    def test_failed_raises(self) -> None:
        with pytest.raises(ValueError, match='Interaction failed'):
            from_interaction_sync(Interaction.model_validate({'status': 'failed'}))
