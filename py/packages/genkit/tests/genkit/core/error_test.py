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

"""Unit tests for the error module."""

from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import genkit
from genkit._core import _error as error_mod
from genkit._core._error import (
    GenkitError,
    GenkitRuntimeError,
    PublicError,
    ReflectionError,
    RuntimeErrorReason,
    format_validation_error,
    get_callable_json,
    get_error_stack,
    get_http_status,
    get_reflection_json,
    mark_request_error,
    parse_retry_after_ms,
    wrap_http_error,
)
from genkit._core._model import AgentOutput, SessionSnapshot
from genkit._core._typing import GenkitRuntimeError as WireError
from genkit.plugin_api import ErrorResponseMetadata


def test_runtime_error_reasons_are_the_ones_helpers_write() -> None:
    assert {reason.value for reason in RuntimeErrorReason} == {
        'INVALID_SCHEMA',
        'INVALID_INPUT',
        'INVALID_OUTPUT',
        'ACTION_NOT_FOUND',
        'MODEL_NOT_FOUND',
        'TOOL_NOT_FOUND',
        'MAX_TURNS_EXCEEDED',
        'TOOL_FAILED',
        'UNSUPPORTED_BY_MODEL',
        'INVALID_PART',
        'UNRESOLVED_TOOL_REQUEST',
        'INVALID_RESUME',
        'SNAPSHOT_NOT_FOUND',
        'SNAPSHOT_NOT_RESUMABLE',
        'SESSION_STORE_NOT_CONFIGURED',
        'SESSION_ID_REQUIRED',
        'INVALID_SESSION_ID',
        'INVALID_SNAPSHOT_ID',
        'CONNECTION_CLOSED',
    }


def test_runtime_error_reason_accessor_keeps_reason_nested() -> None:
    error = GenkitRuntimeError(
        status='ABORTED',
        message='stopped',
        details={'reason': 'MAX_TURNS_EXCEEDED', 'attempt': 5},
    )

    assert error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert error.model_dump(exclude_none=True) == {
        'status': 'ABORTED',
        'message': 'stopped',
        'details': {'reason': 'MAX_TURNS_EXCEEDED', 'attempt': 5},
    }
    assert GenkitRuntimeError(message='bad', details={'reason': 5}).reason is None
    assert GenkitRuntimeError(message='bad', details={'reason': 'not-valid'}).reason is None
    with pytest.raises(ValidationError):
        error.reason = RuntimeErrorReason.TOOL_FAILED  # type: ignore[misc]


def test_genkit_runtime_error_fields_are_read_only() -> None:
    """Assigning ``error.message = 'x'`` raises a validation error."""
    error = GenkitRuntimeError(status='INTERNAL', message='bad')
    with pytest.raises(ValidationError):
        error.message = 'x'
    assert error.message == 'bad'


def test_snapshot_and_agent_output_decode_the_same_error_type() -> None:
    """A persisted turn and a live response expose the same ``.reason``."""
    wire = {'status': 'ABORTED', 'message': 'stopped', 'details': {'reason': 'MAX_TURNS_EXCEEDED'}}
    snapshot = SessionSnapshot.model_validate({'snapshotId': 's1', 'createdAt': '2026-10-06T00:00:00Z', 'error': wire})
    output = AgentOutput.model_validate({'error': wire})

    assert isinstance(snapshot.error, GenkitRuntimeError)
    assert isinstance(output.error, GenkitRuntimeError)
    assert snapshot.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert output.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED


def test_snapshot_accepts_generated_wire_error() -> None:
    """A store holding the generated wire class still builds a snapshot with ``.reason``."""
    wire = WireError(status='NOT_FOUND', message='gone', details={'reason': 'TOOL_NOT_FOUND'})
    snapshot = SessionSnapshot.model_validate({'snapshotId': 's1', 'createdAt': '2026-10-06T00:00:00Z', 'error': wire})

    assert isinstance(snapshot.error, GenkitRuntimeError)
    assert snapshot.error.reason is RuntimeErrorReason.TOOL_NOT_FOUND


def test_agent_output_error_rejects_object_that_only_looks_like_an_error() -> None:
    """AgentOutput(error=Obj()) with only message/status/details attributes raises ValidationError."""

    class Obj:
        message = 'm'
        status = 'INTERNAL'
        details = {'reason': 'TOOL_FAILED'}

    with pytest.raises(ValidationError):
        AgentOutput(error=Obj())  # type: ignore[arg-type]


def test_genkit_error_reason_stays_in_details() -> None:
    error = GenkitError(
        status='NOT_FOUND',
        message="Failed to resolve model 'nope/ghost'.",
        reason=RuntimeErrorReason.MODEL_NOT_FOUND,
    )

    assert error.reason is RuntimeErrorReason.MODEL_NOT_FOUND
    assert error.details['reason'] == 'MODEL_NOT_FOUND'
    assert 'MODEL_NOT_FOUND' not in error.message
    assert GenkitError(status='NOT_FOUND', message='missing').reason is None
    with pytest.raises(AttributeError):
        error.reason = RuntimeErrorReason.TOOL_NOT_FOUND  # type: ignore[misc]


def test_genkit_error() -> None:
    error = GenkitError(
        status='INVALID_ARGUMENT',
        message='Test message',
        details={'extra_msg': 'Test detail'},
        source='test_source',
    )
    assert error.message == 'Test message'
    assert error.http_code == 400
    assert error.status == 'INVALID_ARGUMENT'
    assert error.details['extra_msg'] == 'Test detail'
    assert error.source == 'test_source'
    assert str(error) == 'test_source: INVALID_ARGUMENT: Test message'

    error_no_source = GenkitError(status='INTERNAL', message='Test message 2')
    assert str(error_no_source) == 'INTERNAL: Test message 2'

    # When wrapping another exception the cause should appear in str(...) too,
    # so the model and any plain ``f"{e}"`` log line see the real reason.
    wrapped = GenkitError(
        status='INTERNAL',
        message='Error while running action read_file',
        cause=ValueError("File not found: 'workspace/foo.py'"),
    )
    assert str(wrapped) == ("INTERNAL: Error while running action read_file: File not found: 'workspace/foo.py'")
    assert wrapped.message == 'Error while running action read_file'


def test_genkit_error_to_json() -> None:
    # NOT_FOUND is a valid gRPC-style status (maps to HTTP 404).
    error = GenkitError(status='NOT_FOUND', message='Resource not found', details={'id': 123})
    serializable = error.to_serializable()
    assert isinstance(serializable, ReflectionError)
    assert serializable.code == 5
    assert serializable.message == 'Resource not found'
    assert serializable.details is not None
    assert serializable.details.model_dump()['id'] == 123


def test_genkit_error_response_metadata_is_in_process_only() -> None:
    response_metadata: ErrorResponseMetadata = {
        'retry_after_ms': 1500.5,
        'headers': {'retry-after': '1.5005'},
    }
    error = GenkitError(
        status='RESOURCE_EXHAUSTED',
        message='Rate limited',
        response_metadata=response_metadata,
    )

    assert error.response_metadata == response_metadata
    assert 'response_metadata' not in error.to_callable_serializable().model_dump()
    assert 'response_metadata' not in error.to_serializable().model_dump()


def test_public_error() -> None:
    error = PublicError(
        status='UNAUTHENTICATED',
        message='Please log in',
        details={'extra_msg': 'Session expired'},
    )
    assert error.status == 'UNAUTHENTICATED'
    assert error.message == 'Please log in'
    assert error.details['extra_msg'] == 'Session expired'


def test_genkit_error_message_is_the_sentence_they_passed() -> None:
    """err.message is the text passed as message=, without the status prefix."""
    err = genkit.GenkitError(status='NOT_FOUND', message='missing')
    assert err.message == 'missing'


def test_genkit_error_str_still_leads_with_the_status() -> None:
    """str(err) still reads 'NOT_FOUND: missing'."""
    err = genkit.GenkitError(status='NOT_FOUND', message='missing')
    assert str(err) == 'NOT_FOUND: missing'


def test_genkit_error_message_with_cause_is_the_sentence_they_passed() -> None:
    """A wrapped cause stays on str(err); .message is still the sentence they passed."""
    err = genkit.GenkitError(status='NOT_FOUND', message='missing', cause=ValueError('disk'))
    assert err.message == 'missing'
    assert str(err) == 'NOT_FOUND: missing: disk'


def test_public_error_message_is_the_sentence_they_passed() -> None:
    """A PublicError's .message is its sentence too."""
    err = genkit.PublicError(status='UNAUTHENTICATED', message='Please log in')
    assert err.message == 'Please log in'


@pytest.mark.asyncio
async def test_flow_error_message_is_the_sentence_the_flow_raised() -> None:
    """Catching a GenkitError raised from a flow, .message is the flow's sentence."""
    ai = genkit.Genkit()

    @ai.flow()
    async def lookup_account(account_id: str) -> str:
        raise genkit.GenkitError(status='NOT_FOUND', message='no such account')

    with pytest.raises(genkit.GenkitError) as raised:
        await lookup_account('acct-1')
    err = raised.value
    assert err.message == 'no such account'
    assert err.status == 'NOT_FOUND'


def test_get_http_status() -> None:
    genkit_error = GenkitError(status='PERMISSION_DENIED', message='No access')
    assert get_http_status(genkit_error) == 500

    non_genkit_error = ValueError('Some other error')
    assert get_http_status(non_genkit_error) == 500

    wrapped = GenkitError(
        status='INTERNAL',
        message='Error while running action boom',
        cause=ValueError('secret'),
    )
    assert get_http_status(wrapped) == 500


def test_get_callable_json() -> None:
    genkit_error = GenkitError(status='INVALID_ARGUMENT', message='bad id 12345')
    json_data = get_callable_json(genkit_error)
    assert json_data == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert '12345' not in str(json_data)

    non_genkit_error = TypeError('Type error')
    json_data = get_callable_json(non_genkit_error)
    assert json_data == {'message': 'Internal Error', 'status': 'INTERNAL'}

    wrapped = GenkitError(
        status='INTERNAL',
        message='Error while running action boom',
        cause=ValueError('secret'),
    )
    json_data = get_callable_json(wrapped)
    assert json_data == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert 'secret' not in str(json_data)

    public = PublicError(status='NOT_FOUND', message='missing recipe')
    json_data = get_callable_json(public)
    assert json_data['message'] == 'missing recipe'
    assert json_data['status'] == 'NOT_FOUND'
    assert 'stack' not in json_data.get('details', {})
    assert get_http_status(public) == 404


def test_served_error_body_for_internal_wrapper_around_wrapped_raw_error_is_internal_error() -> None:
    """An INTERNAL wrapper around another wrapped raw raise sends neither wrapper's text."""
    nested = GenkitError(
        status='INTERNAL',
        message='outer secret',
        cause=GenkitError(status='INTERNAL', message='inner secret', cause=ValueError('raw secret')),
    )

    assert get_callable_json(nested) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert get_http_status(nested) == 500


def test_served_error_body_for_wrapped_public_error_is_internal_error() -> None:
    """A hand-built INTERNAL wrapper around a PublicError is redacted on the served body."""
    wrapped = GenkitError(
        status='INTERNAL',
        message='hide this',
        cause=PublicError(status='NOT_FOUND', message='no order 99'),
    )

    assert get_callable_json(wrapped) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert get_http_status(wrapped) == 500


def test_dev_ui_error_body_for_genkit_error_keeps_its_real_message() -> None:
    """The Dev UI error body still shows a GenkitError's own message; only served flows redact it."""
    error = GenkitError(status='INVALID_ARGUMENT', message='bad id 12345')

    assert 'bad id 12345' in get_reflection_json(error).message


def test_get_error_stack() -> None:
    try:
        raise ValueError('Example Error')
    except ValueError as e:
        tb = get_error_stack(e)
        assert tb == ''


def test_wrap_http_error_classifies_status() -> None:
    cause = RuntimeError('bad request')
    error = wrap_http_error(cause, status_code=400)
    assert error.status == 'INVALID_ARGUMENT'
    assert error.cause is cause
    assert error.message == 'bad request'


def test_wrap_http_error_marks_503_unavailable() -> None:
    """A 503 must stay retryable — not collapse to INTERNAL."""
    cause = RuntimeError('overloaded')
    error = wrap_http_error(cause, status_code=503)
    assert error.status == 'UNAVAILABLE'
    assert error.cause is cause


def test_wrap_http_error_coerces_string_status_code() -> None:
    """Some SDKs leave the code as a string; still classify a real 503."""
    cause = RuntimeError('overloaded')
    error = wrap_http_error(cause, status_code='503')
    assert error.status == 'UNAVAILABLE'


@pytest.mark.parametrize('status_code', [None, 'nope', 0, -1, 200, 301, 402, 413, 418])
def test_wrap_http_error_leaves_missing_status_unclassified(status_code: object) -> None:
    """No HTTP failure status, or a 4xx with no canonical status, means retry still sees the raw error."""
    cause = RuntimeError('model failed')
    with pytest.raises(RuntimeError) as raised:
        wrap_http_error(cause, status_code=status_code)
    assert raised.value is cause


def test_wrap_http_error_marks_408_deadline_exceeded() -> None:
    """A request timeout is transient — retry should wait and try again."""
    cause = RuntimeError('request timeout')
    error = wrap_http_error(cause, status_code=408)
    assert error.status == 'DEADLINE_EXCEEDED'
    assert error.cause is cause


def test_wrap_http_error_reads_retry_after() -> None:
    """Retry should wait the provider delay, not come back in a second."""

    class FakeResponse:
        headers = {'retry-after': '60'}

    class FakeError(RuntimeError):
        def __init__(self) -> None:
            super().__init__('rate limited')
            self.response = FakeResponse()

    error = wrap_http_error(FakeError(), status_code=429, message='rate limited')
    assert error.status == 'RESOURCE_EXHAUSTED'
    assert error.response_metadata == {'retry_after_ms': 60000.0}
    assert error.message == 'rate limited'
    assert error.to_callable_serializable().model_dump(exclude_none=True) == {
        'message': 'Internal Error',
        'status': 'INTERNAL',
    }


def test_served_error_body_for_provider_401_is_internal_error() -> None:
    """A plugin error built from a provider 401 serves as 500 Internal Error, with no provider text."""
    error = wrap_http_error(RuntimeError('API key not valid'), status_code=401)

    assert get_callable_json(error) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert get_http_status(error) == 500
    assert 'API key not valid' not in str(get_callable_json(error))


def test_in_process_provider_error_keeps_unauthenticated() -> None:
    """wrap_http_error still classifies a 401 as UNAUTHENTICATED for Retry and Fallback."""
    error = wrap_http_error(RuntimeError('API key not valid'), status_code=401)

    assert error.status == 'UNAUTHENTICATED'
    assert error.message == 'API key not valid'


def test_wrap_http_error_keeps_provider_status_in_process() -> None:
    """A provider 429 stays RESOURCE_EXHAUSTED in-process so Retry still sees it."""
    error = wrap_http_error(RuntimeError('quota'), status_code=429)

    assert error.status == 'RESOURCE_EXHAUSTED'
    assert get_callable_json(error) == {'message': 'Internal Error', 'status': 'INTERNAL'}


def test_served_error_body_for_unmarked_genkit_error_is_internal_error() -> None:
    """A GenkitError built by hand, not a PublicError, serves as 500 Internal Error."""
    error = GenkitError(
        status='UNAVAILABLE',
        message='overloaded',
        cause=RuntimeError('APIError(503 UNAVAILABLE)'),
    )

    assert get_callable_json(error) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert get_http_status(error) == 500


def test_served_error_body_for_action_input_error_keeps_400() -> None:
    """The served action's own input check stays 400 with a generic sentence."""
    error = mark_request_error(error=GenkitError(status='INVALID_ARGUMENT', message='expected str, got dict'))

    assert get_callable_json(error) == {'message': 'Invalid argument', 'status': 'INVALID_ARGUMENT'}
    assert get_http_status(error) == 400


def test_served_error_body_for_public_input_error_keeps_400() -> None:
    """A PublicError for a bad request body stays 400 with its sentence."""
    error = PublicError(
        'INVALID_ARGUMENT',
        'Action request must be wrapped in {"data": ...} object',
    )

    assert get_callable_json(error) == {
        'message': 'Action request must be wrapped in {"data": ...} object',
        'status': 'INVALID_ARGUMENT',
    }
    assert get_http_status(error) == 400


def test_served_error_body_omits_details_on_non_public_genkit_error() -> None:
    """A provider dump in details does not leave the process on a non-PublicError."""
    error = GenkitError(
        status='INVALID_ARGUMENT',
        message='bad key',
        details={'error': {'message': 'API key expired'}},
    )

    assert get_callable_json(error) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert 'API key expired' not in str(get_callable_json(error))


def test_served_error_body_includes_public_error_details_without_stack() -> None:
    """A PublicError's details go on the wire; stack does not."""
    error = PublicError('NOT_FOUND', 'no order 99', details={'id': '99', 'stack': 'trace'})

    assert get_callable_json(error) == {
        'message': 'no order 99',
        'status': 'NOT_FOUND',
        'details': {'id': '99'},
    }


def test_served_error_body_for_not_found_wrapping_unavailable_is_internal_error() -> None:
    """A NOT_FOUND that wraps UNAVAILABLE is still 500; only PublicError keeps 404."""
    error = GenkitError(
        status='NOT_FOUND',
        message='no order',
        cause=GenkitError(status='UNAVAILABLE', message='store down'),
    )

    assert get_callable_json(error) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert get_http_status(error) == 500


def test_served_error_body_dumps_pydantic_details_on_public_error() -> None:
    """A PublicError whose details hold a model still JSON-encodes."""
    import json

    from pydantic import BaseModel

    class Extra(BaseModel):
        id: str

    error = PublicError('NOT_FOUND', 'no order 99', details={'m': Extra(id='99')})
    body = get_callable_json(error)

    assert body == {
        'message': 'no order 99',
        'status': 'NOT_FOUND',
        'details': {'m': {'id': '99'}},
    }
    json.dumps(body)


def test_served_error_body_dumps_models_nested_in_lists_on_public_error() -> None:
    """A PublicError with models nested in a list still JSON-encodes those details."""
    import json

    from pydantic import BaseModel

    class FieldViolation(BaseModel):
        field: str

    error = PublicError(
        'INVALID_ARGUMENT',
        'bad',
        details={'violations': [FieldViolation(field='a')]},
    )
    body = get_callable_json(error)

    assert body == {
        'message': 'bad',
        'status': 'INVALID_ARGUMENT',
        'details': {'violations': [{'field': 'a'}]},
    }
    json.dumps(body)


def test_to_callable_serializable_redacts_like_get_callable_json() -> None:
    """A non-public error's wire body drops the message and details, same as get_callable_json."""
    error = GenkitError(
        status='INVALID_ARGUMENT',
        message='bad id 12345',
        details={'secret': 'ssn'},
    )

    body = error.to_callable_serializable()
    assert body.model_dump(exclude_none=True) == get_callable_json(error)
    # => {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert error.message == 'bad id 12345'


def test_to_callable_serializable_keeps_public_error_text() -> None:
    """A PublicError keeps its message and details; stack is stripped."""
    error = PublicError('NOT_FOUND', 'no order 99', details={'reason': 'MISSING', 'stack': 'trace'})

    body = error.to_callable_serializable()
    assert body.model_dump(exclude_none=True) == {
        'message': 'no order 99',
        'status': 'NOT_FOUND',
        'details': {'reason': 'MISSING'},
    }


@pytest.mark.parametrize(
    ('value', 'expected_ms'),
    [
        ('2', 2000.0),
        (' 1.5 ', 1500.0),
        ('0', 0.0),
    ],
)
def test_parse_retry_after_delay_seconds(value: str, expected_ms: float) -> None:
    """Parse whole, fractional, and zero delay-seconds values."""
    assert parse_retry_after_ms(value) == expected_ms


@pytest.mark.parametrize('value', ['', '   ', 'not-a-delay'])
def test_parse_retry_after_rejects_blank_and_malformed_values(value: str) -> None:
    """Do not attach metadata for blank or malformed header values."""
    assert parse_retry_after_ms(value) is None


@pytest.mark.parametrize('value', ['inf', 'Infinity', 'nan', '1e999', '1e307'])
def test_parse_retry_after_rejects_non_finite_delays(value: str) -> None:
    """Reject delays that are, or scale to, non-finite milliseconds."""
    assert parse_retry_after_ms(value) is None


def test_parse_retry_after_future_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert a future HTTP-date to a relative millisecond delay."""
    monkeypatch.setattr(error_mod.time, 'time', lambda: 1_700_000_000.0)

    assert parse_retry_after_ms('Tue, 14 Nov 2023 22:13:25 GMT') == 5000.0


def test_parse_retry_after_past_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clamp a past HTTP-date delay to zero."""
    monkeypatch.setattr(error_mod.time, 'time', lambda: 1_700_000_000.0)

    assert parse_retry_after_ms('Tue, 14 Nov 2023 22:13:15 GMT') == 0.0


def test_parse_retry_after_returns_none_on_timestamp_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore platform timestamp failures for parseable dates."""
    retry_at = MagicMock()
    retry_at.timestamp.side_effect = OSError
    monkeypatch.setattr(error_mod, 'parsedate_to_datetime', lambda _: retry_at)

    assert parse_retry_after_ms('Thu, 01 Jan 1601 00:00:00') is None


class _Item(BaseModel):
    dish: str
    qty: int


class _Order(BaseModel):
    table: int
    items: list[_Item]


class _StrictItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dish: str


def _validation_error(schema: Any, value: object) -> ValidationError:  # noqa: ANN401
    with pytest.raises(ValidationError) as exc:
        TypeAdapter(schema).validate_python(value)
    return exc.value


@pytest.mark.parametrize(
    ('schema', 'value', 'want'),
    [
        pytest.param(str, None, 'Input should be a valid string, got None', id='str'),
        pytest.param(
            int, 'abc', "Input should be a valid integer, unable to parse string as an integer, got 'abc'", id='int'
        ),
        pytest.param(
            _Item, None, 'Input should be a valid dictionary or instance of _Item, got None', id='model given None'
        ),
        pytest.param(
            _Order, {'table': 4, 'items': [{'dish': 'pad thai'}]}, 'items[0].qty: Field required', id='nested missing'
        ),
        pytest.param(
            dict[str, int],
            {'tip': 'x'},
            "tip: Input should be a valid integer, unable to parse string as an integer, got 'x'",
            id='dict value',
        ),
        pytest.param(
            Literal['small', 'large'], 'medium', "Input should be 'small' or 'large', got 'medium'", id='literal'
        ),
        pytest.param(_StrictItem, {'dish': 'x', 'tip': 5}, 'tip: Extra inputs are not permitted, got 5', id='extra'),
        pytest.param(
            int | str,
            [1],
            'int: Input should be a valid integer, got [1]; str: Input should be a valid string, got [1]',
            id='union keeps branch names',
        ),
        pytest.param(
            _Order,
            {'table': 'x', 'items': [{}, {}]},
            "table: Input should be a valid integer, unable to parse string as an integer, got 'x'; "
            'items[0].dish: Field required; items[0].qty: Field required; and 2 more',
            id='caps at three',
        ),
        pytest.param(
            _Item,
            {'dish': 'pad thai', 'qty': 'y' * 200},
            'qty: Input should be a valid integer, unable to parse string as an integer, '
            "got 'yyyyyyyyyyyyyyyyy...yyyyyyyyyyyyyyyyyy'",
            id='long value is shortened',
        ),
    ],
)
def test_format_validation_error_is_one_line_per_problem(schema: Any, value: object, want: str) -> None:  # noqa: ANN401
    """Each Pydantic error becomes `path: message, got <value>`, with no header, bracket, or docs URL."""
    assert format_validation_error(_validation_error(schema, value)) == want


def test_genkit_error_wrapping_validation_error_shows_the_short_form_once() -> None:
    """`GenkitError(cause=ValidationError)` reads `status: message: <short form>`, not Pydantic's dump."""
    cause = _validation_error(_Item, {'dish': 'pad thai'})

    error = GenkitError(status='INVALID_ARGUMENT', message="Invalid input for flow 'order'", cause=cause)

    assert str(error) == "INVALID_ARGUMENT: Invalid input for flow 'order': qty: Field required"
    assert error.cause is cause


def test_reflection_json_adds_run_trace_id_when_error_has_none() -> None:
    """`get_reflection_json(ValueError('x'), trace_id='abc')` puts the run id on details."""
    ref = get_reflection_json(ValueError('x'), trace_id='abc')

    assert ref.details is not None
    assert ref.details.trace_id == 'abc'
    assert ref.message == 'x'


def test_reflection_json_keeps_error_trace_id_over_run_trace_id() -> None:
    """A GenkitError that already has a trace id keeps it when the run supplies another."""
    error = GenkitError(status='FAILED_PRECONDITION', message='not paid', trace_id='keep-me')
    ref = get_reflection_json(error, trace_id='run-id')

    assert ref.details is not None
    assert ref.details.trace_id == 'keep-me'


def test_reflection_json_without_trace_id_is_unchanged() -> None:
    """No `trace_id` argument means no `details.trace_id` is added."""
    ref = get_reflection_json(ValueError('x'))

    assert ref.details is None or ref.details.trace_id is None


def test_genkit_error_with_empty_validation_error_has_no_trailing_colon() -> None:
    """An empty ValidationError adds nothing after the message."""
    error = GenkitError(
        status='INVALID_ARGUMENT',
        message='title missing',
        cause=ValidationError.from_exception_data('Recipe', []),
    )

    assert str(error) == 'INVALID_ARGUMENT: title missing'
