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

"""A Genkit type inside your own pydantic model dumps the same as on its own."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer

from genkit import Genkit
from genkit._core._model import (
    Document,
    GenerateActionOptions,
    Message,
    ModelConfig,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    OutputConfig,
    Part,
    SessionSnapshot,
    SessionState,
)
from genkit._core._typing import FinishReason, Media, ToolRequest
from genkit.exp.agent import FileSessionStore
from genkit.model import model_ref
from genkit.testing import EchoModel, define_echo_model


class Chat(BaseModel):
    history: list[Message]


class Held(BaseModel):
    value: Any


def _tool_request_message() -> Message:
    return Message(role='model', content=[Part.from_tool_request('lookup', {'q': 1}, ref='r1')])


def _schema_def(name: str) -> dict[str, Any]:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'genkit-tools' / 'genkit-schema.json'
        if candidate.exists():
            schema = json.loads(candidate.read_text())
            return {**schema, '$ref': f'#/$defs/{name}'}
    raise AssertionError('genkit-tools/genkit-schema.json not found')


def test_message_inside_user_model_dumps_same_as_top_level_message() -> None:
    """A Message held in a user's pydantic model dumps to exactly what message.model_dump() gives."""
    msg = _tool_request_message()

    dumped = Chat(history=[msg]).model_dump()

    assert dumped['history'][0] == msg.model_dump()
    assert dumped == {
        'history': [{'role': 'model', 'content': [{'toolRequest': {'ref': 'r1', 'name': 'lookup', 'input': {'q': 1}}}]}]
    }


def test_message_inside_user_model_dump_json_uses_camel_case_without_nulls() -> None:
    """model_dump_json() of a user model holding a tool-request message has toolRequest and no null keys."""
    raw = Chat(history=[_tool_request_message()]).model_dump_json()

    assert '"toolRequest"' in raw
    assert 'tool_request' not in raw
    assert 'null' not in raw
    assert json.loads(raw) == {
        'history': [{'role': 'model', 'content': [{'toolRequest': {'ref': 'r1', 'name': 'lookup', 'input': {'q': 1}}}]}]
    }


def test_message_inside_user_model_round_trips_through_json() -> None:
    """Chat.model_validate_json(chat.model_dump_json()) gives back the same chat."""
    chat = Chat(history=[_tool_request_message(), Message(role='user', content=[Part.from_data(None)])])

    assert Chat.model_validate_json(chat.model_dump_json()) == chat


def test_part_with_null_data_inside_user_model_keeps_data_null() -> None:
    """A Part(data=None) nested in a user model still dumps {"data": null}, so it reloads as a data part."""
    chat = Chat(history=[Message(role='tool', content=[Part.from_data(None)])])

    assert chat.model_dump()['history'][0]['content'] == [{'data': None}]
    assert json.loads(chat.model_dump_json())['history'][0]['content'] == [{'data': None}]
    reloaded = Chat.model_validate(chat.model_dump())
    assert reloaded.history[0].content[0].data is None
    assert 'data' in reloaded.history[0].content[0].model_fields_set


def test_part_with_null_data_keeps_data_null_in_message_chunk_and_snapshot() -> None:
    """data: null survives in a top-level Message, a stream chunk, and a saved snapshot."""
    part = Part.from_data(None)
    msg = Message(role='tool', content=[part])
    chunk = ModelResponseChunk(role='model', index=0, content=[part])
    snapshot = SessionSnapshot(snapshot_id='s1', created_at='t', state=SessionState(messages=[msg]))

    assert part.model_dump() == {'data': None}
    assert json.loads(part.model_dump_json()) == {'data': None}
    assert msg.model_dump()['content'] == [{'data': None}]
    assert json.loads(chunk.model_dump_json())['content'] == [{'data': None}]
    reloaded = SessionSnapshot.model_validate_json(snapshot.model_dump_json())
    assert reloaded.state is not None and reloaded.state.messages is not None
    assert reloaded.state.messages[0].content[0].model_dump() == {'data': None}


def test_dump_with_by_alias_false_still_returns_python_field_names() -> None:
    """part.model_dump(by_alias=False) returns tool_request, not toolRequest."""
    part = Part.from_tool_request('lookup', {'q': 1})

    assert part.model_dump(by_alias=False) == {'tool_request': {'name': 'lookup', 'input': {'q': 1}}}
    assert Chat(history=[_tool_request_message()]).model_dump(by_alias=False)['history'][0]['content'] == [
        {'tool_request': {'ref': 'r1', 'name': 'lookup', 'input': {'q': 1}}}
    ]


def test_model_config_dump_by_alias_false_keeps_field_names_and_nulls() -> None:
    """ModelConfig(...).model_dump(by_alias=False, exclude_none=False) still returns max_output_tokens and None."""
    config = ModelConfig(max_output_tokens=10, temperature=None)

    dumped = config.model_dump(by_alias=False, exclude_none=False, exclude_unset=True)

    assert dumped == {'max_output_tokens': 10, 'temperature': None}


@pytest.mark.asyncio
async def test_model_config_temperature_none_still_clears_ref_default() -> None:
    """config=ModelConfig(temperature=None) over a ref with a temperature sends no temperature."""
    ai = Genkit()
    echo: EchoModel
    echo, _ = define_echo_model(ai, name='testEcho')
    ref = model_ref('testEcho', config_schema=ModelConfig, config=ModelConfig(temperature=0.2, top_k=3))

    await ai.generate(model=ref, prompt='hi', config=ModelConfig(temperature=None))

    assert echo.last_request is not None
    sent = echo.last_request.config
    sent_dict = sent.model_dump() if isinstance(sent, BaseModel) else dict(sent or {})
    assert 'temperature' not in sent_dict
    assert sent_dict.get('topK', sent_dict.get('top_k')) == 3


def test_bytes_in_genkit_part_dump_json_is_base64() -> None:
    """A Genkit type holding bytes dumps them as base64 in JSON, and model_dump() keeps bytes."""
    part = Part.from_data(b'\x00\xff')

    assert json.loads(part.model_dump_json()) == {'data': 'AP8='}
    assert json.loads(Message(role='tool', content=[part]).model_dump_json())['content'] == [{'data': 'AP8='}]
    assert part.model_dump() == {'data': b'\x00\xff'}


def test_bytes_in_part_inside_user_model_follow_the_user_models_bytes_setting() -> None:
    """Bytes held in a Part's data inside a user model use the user model's JSON bytes setting (utf-8 by default)."""
    chat = Chat(history=[Message(role='tool', content=[Part.from_data(b'\x00\x01')])])

    assert json.loads(chat.model_dump_json())['history'][0]['content'] == [{'data': '\x00\x01'}]


def test_user_bytes_field_next_to_message_is_unchanged() -> None:
    """A user model's own bytes field serializes the way pydantic does by default."""

    class Upload(BaseModel):
        blob: bytes
        note: Message

    upload = Upload(blob=b'\x00\x01', note=Message(role='user', content=[Part.from_text('hi')]))

    assert json.loads(upload.model_dump_json()) == {
        'blob': '\x00\x01',
        'note': {'role': 'user', 'content': [{'text': 'hi'}]},
    }
    assert upload.model_dump()['blob'] == b'\x00\x01'


def test_model_response_round_trips_through_user_model_json() -> None:
    """A ModelResponse nested in a user model survives dump-JSON then validate unchanged."""

    class Saved(BaseModel):
        response: ModelResponse

    response = ModelResponse(message=_tool_request_message(), finish_reason=FinishReason.STOP)
    saved = Saved(response=response)

    assert saved.model_dump()['response'] == response.model_dump()
    reloaded = Saved.model_validate_json(saved.model_dump_json())
    assert reloaded.response.model_dump() == response.model_dump()
    assert reloaded.response.message is not None
    assert reloaded.response.message.tool_requests[0].tool_request == ToolRequest(
        name='lookup', input={'q': 1}, ref='r1'
    )


def test_message_subclass_with_its_own_serializer_dumps_with_it() -> None:
    """A user subclass of Message with its own model_serializer gets its serializer, nested or not."""

    class TaggedMessage(Message):
        @model_serializer(mode='wrap')
        def _tag(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
            return {**handler(self), 'tag': 'mine'}

    msg = TaggedMessage(role='user', content=[Part.from_text('hi')])

    assert msg.model_dump() == {'role': 'user', 'content': [{'text': 'hi'}], 'tag': 'mine'}
    assert Held(value=msg).model_dump()['value']['tag'] == 'mine'


@pytest.mark.asyncio
async def test_session_snapshot_saved_by_previous_release_still_loads(tmp_path: Path) -> None:
    """A snapshot file with snake_case keys and nulls loads through FileSessionStore unchanged."""
    old_file = {
        'snapshot_id': 'snap-1',
        'session_id': 'sess-1',
        'parent_id': None,
        'created_at': '2026-07-01T00:00:00Z',
        'status': 'completed',
        'finish_reason': None,
        'error': None,
        'state': {
            'session_id': 'sess-1',
            'custom': None,
            'artifacts': None,
            'messages': [
                {
                    'role': 'model',
                    'metadata': None,
                    'content': [
                        {
                            'text': None,
                            'media': None,
                            'tool_request': {'ref': 'r1', 'name': 'lookup', 'input': {'q': 1}, 'partial': None},
                            'tool_response': None,
                            'data': None,
                            'metadata': None,
                            'custom': None,
                            'reasoning': None,
                        }
                    ],
                }
            ],
        },
    }
    (tmp_path / 'snap-1.json').write_text(json.dumps(old_file))
    store = FileSessionStore(str(tmp_path))

    loaded = await store.get_snapshot(snapshot_id='snap-1')

    assert loaded is not None
    assert loaded.state is not None and loaded.state.messages is not None
    assert loaded.state.messages[0].model_dump() == {
        'role': 'model',
        'content': [{'toolRequest': {'ref': 'r1', 'name': 'lookup', 'input': {'q': 1}}}],
    }


def _every_part_kind() -> list[Part]:
    return [
        Part.from_text('hi'),
        Part.from_media('https://example.com/a.png', 'image/png'),
        Part.from_tool_request('lookup', {'q': 1}, ref='r1'),
        Part.from_tool_response('lookup', {'a': 2}, ref='r1'),
        Part.from_data({'x': 1}),
        Part.from_data(None),
        Part.from_custom({'vendor': 1}),
        Part.from_reasoning('thinking'),
    ]


def _message_with_every_part_kind() -> Message:
    return Message(role='model', content=_every_part_kind(), metadata={'turn': 1})


def _request() -> ModelRequest:
    return ModelRequest(
        messages=[_message_with_every_part_kind()],
        config={'temperature': 0.2},
        tools=[],
        output=OutputConfig(format='json', json_schema={'type': 'object'}),
        docs=[Document.from_text('doc')],
    )


WIRE_SAMPLES: dict[str, tuple[str, Any]] = {
    'message': ('Message', _message_with_every_part_kind),
    'part_media': ('Part', lambda: Part(media=Media(url='https://example.com/a.png'))),
    'document': ('DocumentData', lambda: Document(content=[Part.from_text('d')], metadata={'id': 1})),
    'model_request': ('ModelRequest', _request),
    'model_response': (
        'ModelResponse',
        lambda: ModelResponse(
            message=_message_with_every_part_kind(), finish_reason=FinishReason.STOP, request=_request()
        ),
    ),
    'model_response_chunk': (
        'ModelResponseChunk',
        lambda: ModelResponseChunk(role='model', index=0, content=_every_part_kind()),
    ),
    'generate_action_options': (
        'GenerateActionOptions',
        lambda: GenerateActionOptions(model='m', messages=[_message_with_every_part_kind()], max_turns=2),
    ),
    'session_snapshot': (
        'SessionSnapshot',
        lambda: SessionSnapshot(
            snapshot_id='s1',
            created_at='t',
            state=SessionState(session_id='x', messages=[_message_with_every_part_kind()]),
        ),
    ),
}


@pytest.mark.parametrize('sample', WIRE_SAMPLES, ids=list(WIRE_SAMPLES))
def test_genkit_type_inside_user_model_dumps_wire_shape_same_as_top_level(sample: str) -> None:
    """Each wire type dumps the same nested in a user model as on its own, and that JSON fits genkit-schema.json."""
    schema_name, build = WIRE_SAMPLES[sample]
    value = build()

    nested = json.loads(Held(value=value).model_dump_json())['value']

    assert nested == json.loads(value.model_dump_json())
    assert Held(value=value).model_dump()['value'] == value.model_dump()
    jsonschema.validate(nested, _schema_def(schema_name))
