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

from typing import cast

import pytest
from genkit_google_genai._interactions.converters import (
    ensure_tool_ids,
    from_interaction,
    from_interaction_content,
    from_interaction_step,
    from_interaction_sync,
    to_interaction_content,
    to_interaction_role,
    to_interaction_steps,
    to_interaction_tool,
)
from genkit_google_genai._interactions.types import Content, GeminiInteraction

from genkit import (
    CustomPart,
    Media,
    MediaPart,
    Part,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)
from genkit.model import Message, ToolDefinition


def _part_dict(part: Part) -> dict:
    return part.root.model_dump(by_alias=True, exclude_none=True)


class TestEnsureToolIds:
    def test_assigns_ids_to_tool_requests_without_refs(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='tool1', input={}))),
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='tool2', input={}))),
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
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='tool1', input={}))),
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='tool2', input={}))),
                ],
            ),
            Message(
                role='tool',
                content=[
                    Part(root=ToolResponsePart(tool_response=ToolResponse(name='tool1', output={}))),
                    Part(root=ToolResponsePart(tool_response=ToolResponse(name='tool2', output={}))),
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
                content=[Part(root=ToolResponsePart(tool_response=ToolResponse(name='tool1', output={})))],
            )
        ]
        result = ensure_tool_ids(messages)
        res1 = result[0].content[0].root.tool_response
        assert res1 and res1.ref and res1.ref.startswith('genkit-orphan-id-')

    def test_preserves_existing_refs(self) -> None:
        messages = [
            Message(
                role='model',
                content=[
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='tool1', input={}, ref='existing-id')))
                ],
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
        result = to_interaction_content(Part(root=TextPart(text='Hello')))
        assert result == {'type': 'text', 'text': 'Hello'}

    def test_image_data(self) -> None:
        result = to_interaction_content(
            Part(root=MediaPart(media=Media(url='data:image/png;base64,DATA', content_type='image/png')))
        )
        assert result == {'type': 'image', 'data': 'DATA', 'mime_type': 'image/png'}

    def test_image_uri(self) -> None:
        result = to_interaction_content(
            Part(root=MediaPart(media=Media(url='gs://bucket/image.png', content_type='image/png')))
        )
        assert result == {'type': 'image', 'uri': 'gs://bucket/image.png', 'mime_type': 'image/png'}

    def test_audio(self) -> None:
        result = to_interaction_content(
            Part(root=MediaPart(media=Media(url='data:audio/mp3;base64,DATA', content_type='audio/mp3')))
        )
        assert result == {'type': 'audio', 'data': 'DATA', 'mime_type': 'audio/mp3'}

    def test_document(self) -> None:
        result = to_interaction_content(
            Part(root=MediaPart(media=Media(url='gs://bucket/doc.pdf', content_type='application/pdf')))
        )
        assert result == {'type': 'document', 'uri': 'gs://bucket/doc.pdf', 'mime_type': 'application/pdf'}

    def test_unsupported_media_raises(self) -> None:
        with pytest.raises(ValueError, match='Unsupported media type'):
            to_interaction_content(
                Part(root=MediaPart(media=Media(url='https://example.com/x', content_type='text/plain')))
            )


class TestToInteractionSteps:
    def test_tool_request(self) -> None:
        messages = [
            Message(
                role='model',
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='func', input={'a': 1}, ref='ref1')))],
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
                content=[Part(root=TextPart(text='Thinking')), Part(root=TextPart(text='Done'))],
            )
        ]
        assert to_interaction_steps(messages) == [
            {
                'type': 'model_output',
                'content': [{'type': 'text', 'text': 'Thinking'}, {'type': 'text', 'text': 'Done'}],
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


class TestFromInteractionContent:
    def test_text(self) -> None:
        result = from_interaction_content({
            'type': 'text',
            'text': 'Hello world',
            'annotations': [{'start_index': 0, 'end_index': 5, 'source': 'source'}],
        })
        assert _part_dict(result) == {
            'text': 'Hello world',
            'metadata': {'annotations': [{'start_index': 0, 'end_index': 5, 'source': 'source'}]},
        }

    def test_image_data(self) -> None:
        result = from_interaction_content({'type': 'image', 'data': 'BASE64DATA', 'mime_type': 'image/png'})
        assert _part_dict(result) == {'media': {'url': 'data:image/png;base64,BASE64DATA', 'contentType': 'image/png'}}

    def test_image_resolution(self) -> None:
        result = from_interaction_content({
            'type': 'image',
            'uri': 'gs://bucket/image.png',
            'mime_type': 'image/png',
            'resolution': 'high',
        })
        assert _part_dict(result) == {
            'media': {'url': 'gs://bucket/image.png', 'contentType': 'image/png'},
            'metadata': {'resolution': 'high'},
        }

    def test_thought(self) -> None:
        content = {'type': 'thought', 'signature': 'SIG', 'summary': [{'type': 'text', 'text': 'Thinking...'}]}
        result = from_interaction_content(cast(Content, content))
        assert _part_dict(result) == {
            'reasoning': 'Thinking...',
            'metadata': {'thoughtSignature': 'SIG'},
            'custom': {'thought': content},
        }


class TestFromInteractionStep:
    def test_model_output_includes_empty_annotations(self) -> None:
        result = from_interaction_step({'type': 'model_output', 'content': [{'type': 'text', 'text': 'Hello'}]})
        root = result[0].root
        assert isinstance(root, TextPart)
        assert root.text == 'Hello'
        assert root.metadata is not None
        assert 'annotations' in root.metadata

    def test_user_input_dropped(self) -> None:
        result = from_interaction_step({'type': 'user_input', 'content': [{'type': 'text', 'text': 'Hello'}]})
        assert result == []


class TestFromInteractionStatusMapping:
    def test_cancelled(self) -> None:
        result = from_interaction({'id': '123', 'status': 'cancelled'})
        assert result.done is True
        assert result.output is not None
        assert result.output.finish_reason == 'aborted'
        assert result.output.finish_message == 'Operation cancelled'
        assert _part_dict(result.output.message.content[0]) == {'text': 'Operation cancelled.'}

    def test_failed_exits_poll_loop(self) -> None:
        result = from_interaction({'id': '123', 'status': 'failed', 'error': {'message': 'boom'}})
        assert result.done is True
        assert result.error is not None
        assert result.error.message == 'boom'

    def test_requires_action_done_with_metadata(self) -> None:
        interaction: GeminiInteraction = {
            'id': '123',
            'status': 'requires_action',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'approve plan'}]}],
        }
        result = from_interaction(interaction)
        assert result.done is True
        assert result.output is not None
        assert result.metadata is not None
        assert result.metadata['interaction_status'] == 'requires_action'

    def test_in_progress(self) -> None:
        result = from_interaction({'id': '123', 'status': 'in_progress'})
        assert result.done is False


class TestFromInteractionSync:
    def test_failed_raises(self) -> None:
        with pytest.raises(ValueError, match='Interaction failed'):
            from_interaction_sync({'status': 'failed'})
