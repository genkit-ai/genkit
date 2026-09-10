# Copyright 2025 Google LLC
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

"""Unit tests for Part factory methods and property getters."""

import importlib
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from genkit import Document, Media, Message, Part, ToolRequest, ToolResponse, respond_to_interrupt, restart_tool
from genkit._ai._agents._client import (
    SessionSnapshot as ClientSessionSnapshot,
    SessionState as ClientSessionState,
    to_agent_input,
)
from genkit._ai._generate import require_model_response
from genkit._core._model import (
    PART_KIND_FIELDS,
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentResult,
    AgentStreamChunk,
    Artifact,
    Candidate,
    GenerateActionOptions,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    Resume,
    SessionSnapshot,
    SessionState,
    as_artifact,
    as_candidate,
    as_document,
    as_message,
    as_model_request,
    as_model_response_chunk,
    as_part,
)
from genkit._core._typing import (
    Artifact as ArtifactData,
    DocumentData,
    FinishReason,
    MessageData,
    PartData,
    Resource,
    TextPart,
)
from genkit.middleware import ToolHookParams
from genkit.model import Document as ModelDocument


def _from_import(module: str, name: str) -> object:
    """``from module import name`` — ImportError if the name is gone."""
    try:
        return getattr(importlib.import_module(module), name)
    except AttributeError as exc:
        raise ImportError(f'cannot import name {name!r} from {module!r}') from exc


_GETTERS = ('text', 'media', 'tool_request', 'tool_response', 'data', 'reasoning', 'custom')

_PART_OF = {
    'text': lambda: Part.from_text('hi'),
    'media': lambda: Part.from_media('https://x'),
    'tool_request': lambda: Part.from_tool_request(name='lookup'),
    'tool_response': lambda: Part.from_tool_response(name='lookup'),
    'reasoning': lambda: Part.from_reasoning('think'),
    'resource': lambda: Part.model_validate({'resource': {'uri': 'file://x'}}),
    'data': lambda: Part.from_data({'payload': 1}),
    'custom': lambda: Part.from_custom({'vendor': True}),
}
_KIND_VALUE = {
    'text': 'other',
    'media': Media(url='https://y'),
    'tool_request': ToolRequest(name='other'),
    'tool_response': ToolResponse(name='other'),
    'reasoning': 'other think',
    'resource': Resource(uri='file://y'),
    'data': {'payload': 2},
    'custom': {'vendor': False},
}
_ALL_KINDS = (*PART_KIND_FIELDS, 'custom')
assert set(_PART_OF) == set(_ALL_KINDS) == set(_KIND_VALUE)

_SECOND_KIND = [
    pytest.param(base, other, id=f'{base}+{other}')
    for base in PART_KIND_FIELDS
    for other in PART_KIND_FIELDS
    if base != other
]
_TWO_KINDS = [
    pytest.param(base, other, id=f'{base}+{other}')
    for i, base in enumerate(PART_KIND_FIELDS)
    for other in PART_KIND_FIELDS[i + 1 :]
]


def assert_inactive(part: Part, *, active: str) -> None:
    for name in _GETTERS:
        if name != active:
            assert getattr(part, name) is None


def test_part_from_text() -> None:
    """Part.from_text creates a text part with direct property access."""
    p = Part.from_text('hello world', metadata={'source': 'user'})
    assert p.text is not None
    assert p.text == 'hello world'
    assert p.metadata == {'source': 'user'}
    assert_inactive(p, active='text')


def test_part_from_text_defaults() -> None:
    """from_text with no metadata leaves metadata unset."""
    p = Part.from_text('hi')
    assert p.text == 'hi'
    assert p.metadata is None


def test_part_from_text_empty() -> None:
    """Empty text is still a text part."""
    p = Part.from_text('')
    assert p.text == ''
    assert_inactive(p, active='text')


def test_part_from_media() -> None:
    """Part.from_media creates a media part with direct property access."""
    p = Part.from_media('https://example.com/image.png', content_type='image/png', metadata={'alt': 'dish'})
    assert p.media is not None
    assert p.media.url == 'https://example.com/image.png'
    assert p.media.content_type == 'image/png'
    assert p.metadata == {'alt': 'dish'}
    assert_inactive(p, active='media')


def test_part_from_media_without_content_type() -> None:
    """content_type is optional."""
    p = Part.from_media('https://example.com/image.png')
    assert p.media is not None
    assert p.media.url == 'https://example.com/image.png'
    assert p.media.content_type is None
    assert p.metadata is None


def test_part_from_tool_request() -> None:
    """Part.from_tool_request creates a tool request part."""
    p = Part.from_tool_request(name='get_weather', input={'city': 'Paris'}, ref='call-123', metadata={'src': 'model'})
    assert p.tool_request is not None
    assert p.tool_request.name == 'get_weather'
    assert p.tool_request.input == {'city': 'Paris'}
    assert p.tool_request.ref == 'call-123'
    assert p.metadata == {'src': 'model'}
    assert_inactive(p, active='tool_request')


def test_part_from_tool_request_defaults() -> None:
    """name-only tool request leaves input and ref unset."""
    p = Part.from_tool_request(name='lookup')
    assert p.tool_request is not None
    assert p.tool_request.name == 'lookup'
    assert p.tool_request.input is None
    assert p.tool_request.ref is None
    assert p.metadata is None


def test_part_from_tool_response() -> None:
    """Part.from_tool_response creates a tool response part."""
    p = Part.from_tool_response(name='get_weather', output={'temp': 22}, ref='call-123')
    assert p.tool_response is not None
    assert p.tool_response.name == 'get_weather'
    assert p.tool_response.output == {'temp': 22}
    assert p.tool_response.ref == 'call-123'
    assert_inactive(p, active='tool_response')


def test_part_from_tool_response_defaults() -> None:
    """name-only tool response leaves output and ref unset."""
    p = Part.from_tool_response(name='lookup')
    assert p.tool_response is not None
    assert p.tool_response.name == 'lookup'
    assert p.tool_response.output is None
    assert p.tool_response.ref is None


def test_part_from_reasoning() -> None:
    """Part.from_reasoning creates a reasoning part."""
    p = Part.from_reasoning('thinking step by step', metadata={'step': 1})
    assert p.reasoning == 'thinking step by step'
    assert p.metadata == {'step': 1}
    assert_inactive(p, active='reasoning')


def test_part_from_data() -> None:
    """Part.from_data creates a data part."""
    p = Part.from_data({'custom': 'payload'})
    assert p.data == {'custom': 'payload'}
    assert_inactive(p, active='data')


def test_part_from_data_string() -> None:
    """data can be a string."""
    p = Part.from_data('just a string')
    assert p.data == 'just a string'


def test_part_from_data_list() -> None:
    """data can be a list."""
    p = Part.from_data([1, 2])
    assert p.data == [1, 2]


def test_part_from_custom() -> None:
    """Part.from_custom creates a custom part."""
    p = Part.from_custom({'vendor_field': True})
    assert p.custom == {'vendor_field': True}
    assert_inactive(p, active='custom')


def test_part_from_custom_empty() -> None:
    """Empty custom payload is still a custom part."""
    p = Part.from_custom({})
    assert p.custom == {}


def test_from_text_reads_text() -> None:
    """Part.from_text('hi') still reads .text."""
    p = Part.from_text('hi')
    assert p.text == 'hi'
    assert p.media is None


def test_part_text_kwarg_reads_text() -> None:
    """Part(text='hi') still reads .text."""
    p = Part(text='hi')
    assert p.text == 'hi'
    assert p.media is None


def test_part_root_constructor_raises() -> None:
    """Part(root=TextPart(...)) is gone; they use Part.from_text or Part(text=...)."""
    with pytest.raises(ValidationError, match=r'root'):
        Part(**{'root': TextPart(text='hi')})


def test_part_has_no_root() -> None:
    """The part they pass next has no public root."""
    assert hasattr(Part.from_text('hi'), 'root') is False
    assert hasattr(Part(text='hi'), 'root') is False


def test_part_text_kwarg_keeps_vendor_custom() -> None:
    """custom may ride on a text part."""
    p = Part(text='hi', custom={'vendor': True})
    assert p.text == 'hi'
    assert p.custom == {'vendor': True}


def test_message_text_part_keeps_vendor_custom() -> None:
    """custom may ride on a text part in a message."""
    msg = Message(role='user', content=[{'text': 'hi', 'custom': {'vendor': True}}])
    assert msg.content[0].text == 'hi'
    assert msg.content[0].custom == {'vendor': True}


def test_reasoning_part_keeps_signed_thought() -> None:
    """A reasoning part may carry the vendor thought blob in custom."""
    thought = {'sig': 'abc'}
    p = Part(reasoning='think', custom={'thought': thought})
    assert p.reasoning == 'think'
    assert p.custom == {'thought': thought}


def test_message_content_items_are_part() -> None:
    msg = Message(role='user', content=[Part.from_text('hi')])
    assert type(msg.content[0]) is Part


def test_document_content_items_are_part() -> None:
    doc = Document(content=[Part.from_text('hi')])
    assert type(doc.content[0]) is Part


def test_message_is_not_message_data() -> None:
    msg = Message(role='user', content=[Part.from_text('hi')])
    assert type(msg) is Message
    assert isinstance(msg, MessageData) is False
    assert msg.model_dump() == {'role': 'user', 'content': [{'text': 'hi'}]}


def test_message_from_wire_dict() -> None:
    msg = Message.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})
    assert type(msg) is Message
    assert msg.text == 'hi'


def test_as_message_unwraps_message_data() -> None:
    data = MessageData.model_validate({'role': 'user', 'content': [{'text': 'hi'}]})
    msg = as_message(data)
    assert type(msg) is Message
    assert isinstance(msg, MessageData) is False
    assert msg.text == 'hi'


def test_empty_message_has_empty_text() -> None:
    msg = Message(role='user', content=[])
    assert msg.text == ''
    assert msg.tool_requests == []


def test_document_is_not_document_data() -> None:
    doc = Document(content=[Part.from_text('hi')])
    assert type(doc) is Document
    assert isinstance(doc, DocumentData) is False
    assert doc.model_dump() == {'content': [{'text': 'hi'}]}


def test_as_document_unwraps_document_data() -> None:
    data = DocumentData.model_validate({'content': [{'text': 'hi'}]})
    doc = as_document(data)
    assert type(doc) is Document
    assert isinstance(doc, DocumentData) is False
    assert type(doc.content[0]) is Part
    assert doc.text == 'hi'


def test_as_part_unwraps_part_data() -> None:
    data = PartData.model_validate({'text': 'hi'})
    part = as_part(data)
    assert type(part) is Part
    assert part.text == 'hi'


def test_as_artifact_unwraps_artifact_data() -> None:
    data = ArtifactData.model_validate({'parts': [{'text': 'hi'}]})
    art = as_artifact(data)
    assert type(art) is Artifact
    assert type(art.parts[0]) is Part
    assert art.parts[0].text == 'hi'


def test_message_from_another_message_raises() -> None:
    msg = Message(role='user', content=[Part.from_text('hi')])
    with pytest.raises(TypeError, match='as_message'):
        Message(msg)  # type: ignore[misc]


def test_document_from_another_document_raises() -> None:
    doc = Document(content=[Part.from_text('hi')])
    with pytest.raises(TypeError, match='as_document'):
        Document(doc)  # type: ignore[arg-type]


def test_text_part_is_not_exported_from_genkit() -> None:
    with pytest.raises(ImportError, match='TextPart'):
        _from_import('genkit', 'TextPart')


def test_media_part_is_not_exported_from_genkit() -> None:
    with pytest.raises(ImportError, match='MediaPart'):
        _from_import('genkit', 'MediaPart')


def test_document_imports_from_genkit_model() -> None:
    doc = ModelDocument.from_text('hi')
    assert type(doc) is Document
    assert doc.text == 'hi'


def test_reasoning_part_is_not_exported_from_genkit() -> None:
    with pytest.raises(ImportError, match='ReasoningPart'):
        _from_import('genkit', 'ReasoningPart')


def test_tool_request_part_is_not_exported_from_genkit() -> None:
    with pytest.raises(ImportError, match='ToolRequestPart'):
        _from_import('genkit', 'ToolRequestPart')


def test_tool_response_part_is_not_exported_from_genkit() -> None:
    with pytest.raises(ImportError, match='ToolResponsePart'):
        _from_import('genkit', 'ToolResponsePart')


def test_respond_to_interrupt_returns_part() -> None:
    interrupt = Part.from_tool_request(name='ask', input={'q': 'ok?'}, ref='r1')
    reply = respond_to_interrupt('yes', interrupt=interrupt)
    assert type(reply) is Part
    assert reply.tool_response is not None
    assert reply.tool_response.name == 'ask'
    assert reply.tool_response.output == 'yes'
    assert reply.metadata == {'interruptResponse': True}


def test_restart_tool_returns_part() -> None:
    interrupt = Part.from_tool_request(name='pay', input={'amount': 10}, ref='r1')
    restart = restart_tool(interrupt=interrupt, resumed_metadata={'k': 'v'})
    assert type(restart) is Part
    assert restart.tool_request is not None
    assert restart.tool_request.name == 'pay'
    assert restart.metadata is not None
    assert restart.metadata.get('resumed') == {'k': 'v'}


def test_resume_respond_text_part_raises() -> None:
    with pytest.raises(ValueError, match='resume_respond needs a tool response part'):
        Resume(respond=[Part.from_text('hi')])


def test_tool_hook_params_text_part_raises() -> None:
    """wrap_tool only accepts a tool-request Part."""
    with pytest.raises(ValidationError, match='wrap_tool needs a tool request part'):
        ToolHookParams(tool_request_part=Part.from_text('hi'), tool=object())


def test_agent_stream_chunk_model_chunk_is_veneer() -> None:
    chunk = AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part.from_text('hi')]))
    assert type(chunk.model_chunk) is ModelResponseChunk
    assert type(chunk.model_chunk.content[0]) is Part
    assert chunk.model_chunk.content[0].text == 'hi'


def test_candidate_message_is_message() -> None:
    cand = Candidate(
        index=0,
        message=Message(role='model', content=[Part.from_text('hi')]),
        finish_reason=FinishReason.STOP,
    )
    assert type(cand.message) is Message
    assert type(cand.message.content[0]) is Part
    assert cand.message.text == 'hi'


def test_part_from_text_round_trip() -> None:
    """Dumping and re-parsing a factory-built part keeps the payload."""
    p = Part.from_text('hello', metadata={'source': 'user'})
    again = Part.model_validate(p.model_dump())
    assert again.text == 'hello'
    assert again.metadata == {'source': 'user'}
    assert_inactive(again, active='text')


def test_message_empty_part_raises() -> None:
    with pytest.raises(ValidationError):
        Message(role='user', content=[{}])


@pytest.mark.parametrize(('base', 'other'), _SECOND_KIND)
def test_assigning_a_second_kind_raises(base: str, other: str) -> None:
    """Stamping a second kind onto any part raises so generate never sends two keys."""
    part = _PART_OF[base]()
    with pytest.raises(ValidationError, match='exactly one'):
        setattr(part, other, _KIND_VALUE[other])
    assert getattr(part, base) is not None
    assert getattr(part, other) is None


@pytest.mark.parametrize('kind', _ALL_KINDS)
def test_assigning_metadata_keeps_the_kind(kind: str) -> None:
    """Metadata may ride; assigning it does not change the kind."""
    part = _PART_OF[kind]()
    part.metadata = {'source': 'user'}
    assert part.metadata == {'source': 'user'}
    assert getattr(part, kind) is not None
    assert_inactive(part, active=kind)


@pytest.mark.parametrize('kind', _ALL_KINDS)
def test_replacing_the_same_kind_stays_one_kind(kind: str) -> None:
    """Updating the same kind stays that kind."""
    part = _PART_OF[kind]()
    setattr(part, kind, _KIND_VALUE[kind])
    assert getattr(part, kind) is not None
    assert_inactive(part, active=kind)


@pytest.mark.parametrize(('base', 'other'), _TWO_KINDS)
def test_message_rejects_a_two_kind_part(base: str, other: str) -> None:
    """A Message will not carry a part that already has two kinds."""
    part = Part.model_construct(**{base: _KIND_VALUE[base], other: _KIND_VALUE[other]})
    with pytest.raises(ValidationError, match='exactly one'):
        Message(role='user', content=[part])


def _two_kind_on_message() -> Message:
    msg = Message(role='user', content=[Part.from_text('hi')])
    msg.content[0] = Part.model_construct(text='caption', media=Media(url='https://y'))
    return msg


def _two_kind_on_artifact() -> Artifact:
    art = Artifact(name='note', parts=[Part.from_text('hi')])
    art.parts[0] = Part.model_construct(text='caption', media=Media(url='https://y'))
    return art


def _two_kind_on_resume() -> Resume:
    resume = Resume(respond=[Part.from_tool_response(name='lookup')])
    assert resume.respond is not None
    resume.respond[0] = Part.model_construct(text='caption', media=Media(url='https://y'))
    return resume


_MESSAGE_WRAPS: dict[str, Callable[[Message], object]] = {
    'as_message': as_message,
    'GenerateActionOptions': lambda m: GenerateActionOptions(model='programmableModel', messages=[m]),
    'AgentInput': lambda m: AgentInput(message=m),
    'AgentOutput': lambda m: AgentOutput(message=m),
    'AgentResult': lambda m: AgentResult(message=m),
    'SessionState': lambda m: SessionState(messages=[m]),
    'ModelResponse': lambda m: ModelResponse(message=m, finish_reason=FinishReason.STOP),
}

_ARTIFACT_WRAPS: dict[str, Callable[[Artifact], object]] = {
    'as_artifact': as_artifact,
    'AgentOutput.artifacts': lambda a: AgentOutput(artifacts=[a]),
    'SessionState.artifacts': lambda a: SessionState(artifacts=[a]),
}

_RESUME_WRAPS: dict[str, Callable[[Resume], object]] = {
    'GenerateActionOptions.resume': lambda r: GenerateActionOptions(model='programmableModel', resume=r),
    'AgentInput.resume': lambda r: AgentInput(resume=r),
}

_STATE_WRAPS: dict[str, Callable[[SessionState], object]] = {
    'SessionSnapshot': lambda s: SessionSnapshot(snapshot_id='s', created_at='t', state=s),
    'AgentInit': lambda s: AgentInit(state=s),
    'AgentOutput.state': lambda s: AgentOutput(state=s),
}


def _two_kind_on_chunk() -> ModelResponseChunk:
    chunk = ModelResponseChunk(content=[Part.from_text('hi')])
    chunk.content[0] = Part.model_construct(text='caption', media=Media(url='https://y'))
    return chunk


_CHUNK_WRAPS: dict[str, Callable[[ModelResponseChunk], object]] = {
    'AgentStreamChunk.model_chunk': lambda c: AgentStreamChunk(model_chunk=c),
}


@pytest.mark.parametrize('wrap', _MESSAGE_WRAPS.values(), ids=_MESSAGE_WRAPS.keys())
def test_wrapper_rejects_a_two_kind_part_already_on_content(
    wrap: Callable[[Message], object],
) -> None:
    """A Message already holding two kinds does not dump them through a wrapper."""
    with pytest.raises(ValidationError, match='exactly one'):
        wrap(_two_kind_on_message())


@pytest.mark.parametrize('wrap', _ARTIFACT_WRAPS.values(), ids=_ARTIFACT_WRAPS.keys())
def test_wrapper_rejects_a_two_kind_part_already_on_artifact(
    wrap: Callable[[Artifact], object],
) -> None:
    """An Artifact already holding two kinds does not persist them."""
    with pytest.raises(ValidationError, match='exactly one'):
        wrap(_two_kind_on_artifact())


@pytest.mark.parametrize('wrap', _RESUME_WRAPS.values(), ids=_RESUME_WRAPS.keys())
def test_wrapper_rejects_a_two_kind_part_already_on_resume(
    wrap: Callable[[Resume], object],
) -> None:
    """Resume respond/restart already holding two kinds does not reach the model."""
    with pytest.raises(ValidationError, match='exactly one'):
        wrap(_two_kind_on_resume())


@pytest.mark.parametrize('wrap', _STATE_WRAPS.values(), ids=_STATE_WRAPS.keys())
def test_wrapper_rejects_a_two_kind_part_already_on_session_state(
    wrap: Callable[[SessionState], object],
) -> None:
    """A session already holding two kinds does not persist them."""
    state = SessionState(messages=[Message(role='user', content=[Part.from_text('hi')])])
    assert state.messages is not None
    state.messages[0].content[0] = Part.model_construct(text='caption', media=Media(url='https://y'))
    with pytest.raises(ValidationError, match='exactly one'):
        wrap(state)


@pytest.mark.parametrize('wrap', _CHUNK_WRAPS.values(), ids=_CHUNK_WRAPS.keys())
def test_wrapper_rejects_a_two_kind_part_already_on_chunk(
    wrap: Callable[[ModelResponseChunk], object],
) -> None:
    """A stream chunk already holding two kinds does not persist them."""
    with pytest.raises(ValidationError, match='exactly one'):
        wrap(_two_kind_on_chunk())


def test_stream_chunk_wrap_keeps_index_zero() -> None:
    """index=0 is a real chunk position and must survive the wrap."""
    chunk = ModelResponseChunk(content=[Part.from_text('hi')])
    chunk.index = 0
    wrapped = AgentStreamChunk(model_chunk=chunk)
    assert wrapped.model_chunk is not None
    assert wrapped.model_chunk.index == 0
    assert wrapped.model_chunk.content[0].text == 'hi'


def test_model_response_chunk_ctor_keeps_index_zero() -> None:
    chunk = ModelResponseChunk(content=[Part.from_text('hi')], index=0)
    assert chunk.index == 0
    walked = as_model_response_chunk(chunk)
    assert walked.index == 0
    assert walked is not chunk
    assert as_model_response_chunk({'content': [{'text': 'hi'}], 'index': 0}).index == 0


def test_model_response_rejects_a_two_kind_part_already_on_candidate() -> None:
    cand = Candidate(
        index=0,
        message=Message(role='model', content=[Part.from_text('hi')]),
        finish_reason=FinishReason.STOP,
    )
    cand.message.content[0] = Part.model_construct(text='caption', media=Media(url='https://y'))
    with pytest.raises(ValidationError, match='exactly one'):
        ModelResponse(
            message=Message(role='model', content=[Part.from_text('ok')]),
            finish_reason=FinishReason.STOP,
            candidates=[cand],
        )


def test_model_response_rejects_a_two_kind_part_already_on_request() -> None:
    req = ModelRequest(messages=[Message(role='user', content=[Part.from_text('hi')])])
    req.messages[0].content[0] = Part.model_construct(text='caption', media=Media(url='https://y'))
    with pytest.raises(ValidationError, match='exactly one'):
        ModelResponse(
            message=Message(role='model', content=[Part.from_text('ok')]),
            finish_reason=FinishReason.STOP,
            request=req,
        )


def test_require_model_response_rejects_a_two_kind_part_already_on_message() -> None:
    resp = ModelResponse(message=Message(role='model', content=[Part.from_text('ok')]), finish_reason=FinishReason.STOP)
    assert resp.message is not None
    resp.message.content[0] = Part.model_construct(text='caption', media=Media(url='https://y'))
    with pytest.raises(ValidationError, match='exactly one'):
        require_model_response(raw=resp, name='programmableModel')


def test_as_candidate_rebuilds_and_keeps_message_text() -> None:
    cand = Candidate(
        index=0,
        message=Message(role='model', content=[Part.from_text('hi')]),
        finish_reason=FinishReason.STOP,
    )
    walked = as_candidate(cand)
    assert walked is not cand
    assert walked.message.content[0].text == 'hi'


def test_as_model_request_rebuilds_and_keeps_message_text() -> None:
    req = ModelRequest(messages=[Message(role='user', content=[Part.from_text('hi')])])
    walked = as_model_request(req)
    assert walked is not req
    assert walked.messages[0].content[0].text == 'hi'


def test_require_model_response_rebuilds_and_keeps_message_text() -> None:
    resp = ModelResponse(message=Message(role='model', content=[Part.from_text('ok')]), finish_reason=FinishReason.STOP)
    walked = require_model_response(raw=resp, name='programmableModel')
    assert walked is not resp
    assert walked.message is not None
    assert walked.message.content[0].text == 'ok'


def test_to_agent_input_rejects_a_two_kind_part_already_on_content() -> None:
    """send() reconstructs AgentInput so a later plant does not go on the wire."""
    inp = AgentInput(message=Message(role='user', content=[Part.from_text('hi')]))
    assert inp.message is not None
    inp.message.content[0] = Part.model_construct(text='caption', media=Media(url='https://y'))
    with pytest.raises(ValidationError, match='exactly one'):
        to_agent_input(inp)


def test_client_session_snapshot_rejects_a_two_kind_part_already_on_state() -> None:
    """The typed client snapshot walks the same way as the veneer snapshot."""
    state = ClientSessionState(messages=[Message(role='user', content=[Part.from_text('hi')])])
    assert state.messages is not None
    state.messages[0].content[0] = Part.model_construct(text='caption', media=Media(url='https://y'))
    with pytest.raises(ValidationError, match='exactly one'):
        ClientSessionSnapshot(snapshot_id='s', created_at='t', state=state)


def test_message_text_part_from_factory_still_works() -> None:
    msg = Message(role='user', content=[Part.from_text('hi')])
    assert msg.content[0].text == 'hi'
    assert_inactive(msg.content[0], active='text')


def test_from_text_none_raises() -> None:
    with pytest.raises(ValidationError):
        Part.from_text(None)  # type: ignore[arg-type]


def test_from_media_empty_url_is_still_media() -> None:
    p = Part.from_media('')
    assert p.media is not None
    assert p.media.url == ''
    assert p.text is None
    assert_inactive(p, active='media')


def test_empty_metadata_dict_is_present_on_dump() -> None:
    msg = Message(role='user', content=[Part.from_text('hi', metadata={})])
    assert msg.model_dump()['content'][0]['metadata'] == {}


def test_from_text_dump_matches_wire_format() -> None:
    msg = Message(role='user', content=[Part.from_text('hello')])
    dumped = msg.model_dump()['content'][0]
    assert dumped == {'text': 'hello'}
    assert 'root' not in dumped


def test_from_media_dump_matches_wire_format() -> None:
    msg = Message(role='user', content=[Part.from_media('https://x', content_type='image/png')])
    dumped = msg.model_dump()['content'][0]
    assert dumped == {'media': {'url': 'https://x', 'contentType': 'image/png'}}
    assert 'root' not in dumped
    assert dumped['media'].get('contentType') == 'image/png'


def test_from_tool_request_dump_omits_unset_input() -> None:
    msg = Message(role='model', content=[Part.from_tool_request(name='lookup')])
    dumped = msg.model_dump()['content'][0]
    assert dumped == {'toolRequest': {'name': 'lookup'}}
    assert 'input' not in dumped['toolRequest']


def test_from_reasoning_dump_is_reasoning_not_text() -> None:
    msg = Message(role='model', content=[Part.from_reasoning('think step by step')])
    dumped = msg.model_dump()['content'][0]
    assert dumped == {'reasoning': 'think step by step'}
    again = Part.model_validate(dumped)
    assert again.reasoning == 'think step by step'
    assert again.text is None


def test_from_custom_and_from_data_dump_different_keys() -> None:
    custom_msg = Message(role='user', content=[Part.from_custom({'a': 1})])
    data_msg = Message(role='user', content=[Part.from_data({'a': 1})])
    assert custom_msg.model_dump()['content'][0] == {'custom': {'a': 1}}
    assert data_msg.model_dump()['content'][0] == {'data': {'a': 1}}


def test_media_part_text_is_none_url_is_on_media() -> None:
    p = Part.from_media('https://example.com/image.png')
    assert p.text is None
    assert p.media is not None
    assert p.media.url == 'https://example.com/image.png'


def test_message_text_skips_reasoning_and_media() -> None:
    msg = Message(
        role='model',
        content=[
            Part.from_reasoning('think'),
            Part.from_text('hello'),
            Part.from_media('https://x'),
        ],
    )
    assert msg.text == 'hello'


def test_resource_wire_part_has_no_public_getters() -> None:
    p = Part.model_validate({'resource': {'uri': 'test://x'}})
    for name in _GETTERS:
        assert getattr(p, name) is None
