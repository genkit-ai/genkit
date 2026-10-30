#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Genkit extra API methods."""

import json
import os
import signal
import socket
import subprocess  # noqa: S404
import sys
import threading
from typing import TypeVar
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

import genkit.plugin_api
from genkit import Genkit, Message, Part, Role, get_logger
from genkit._core._action import ActionKind, _action_context
from genkit._core._error import GenkitError, RuntimeErrorReason
from genkit._core._model import ModelRequest, ModelResponse
from genkit._core._operation import Operation
from genkit._core._telemetry._log_exporter import build_log_record
from genkit.evaluator import BaseDataPoint, EvaluatorResponse, Score
from genkit.model import ModelContext
from genkit.telemetry import (
    SpanMetadata,
    SpanNext,
    configure_instrumentation,
    reset_instrumentation,
)

T = TypeVar('T')

_RUN_MAIN_SCRIPT = """
import sys

from genkit import Genkit

ai = Genkit()


async def main() -> str:
    if sys.argv[1] == 'fail':
        raise ValueError('boom')
    return 'done'


print('RESULT', ai.run_main(main()), flush=True)
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _run_main_then_signal(outcome: str, sig: signal.Signals) -> tuple[int, str]:
    """Start a reflection-enabled run_main in a child, send sig once it waits, return (exit code, output)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('GENKIT_')}
    # Reflection on outside dev, so no runtime file is written.
    env |= {'GENKIT_REFLECTION_ENABLED': 'true', 'GENKIT_REFLECTION_PORT': str(_free_port())}
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, '-c', _RUN_MAIN_SCRIPT, outcome],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    # Kill a child that never reaches the ready line instead of stalling the run.
    watchdog = threading.Timer(30, proc.kill)
    watchdog.start()
    try:
        assert proc.stdout is not None
        output: list[str] = []
        for line in proc.stdout:
            output.append(line)
            if 'Press Ctrl+C to stop' in line:
                break
    finally:
        watchdog.cancel()
    proc.send_signal(sig)
    try:
        rest, _ = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        rest, _ = proc.communicate()
        pytest.fail(f'process kept running after {sig.name}:\n{"".join(output)}{rest}')
    output.append(rest)
    return proc.returncode, ''.join(output)


def _printed_result(output: str) -> str | None:
    """The line the child printed with run_main's return value, if any.

    Matched by line start: Python 3.13+ tracebacks quote the ``print('RESULT', ...)``
    source line, so a substring check would find it in a traceback too.
    """
    return next((line for line in output.splitlines() if line.startswith('RESULT ')), None)


posix_signals = pytest.mark.skipif(sys.platform == 'win32', reason='sends POSIX signals to a child process')


@posix_signals
@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM], ids=['ctrl_c', 'sigterm'])
def test_run_main_raises_the_coroutine_error_when_stopped(sig: signal.Signals) -> None:
    """A failing main keeps reflection up; stopping it (Ctrl+C or SIGTERM) raises the main's error."""
    returncode, output = _run_main_then_signal('fail', sig)

    assert returncode != 0, output
    assert 'ValueError: boom' in output
    assert 'KeyboardInterrupt' not in output
    assert 'during asyncio.run() shutdown' not in output
    assert _printed_result(output) is None, output


@posix_signals
def test_run_main_returns_the_coroutine_result_on_sigterm() -> None:
    """SIGTERM is a clean stop: run_main hands back what main returned and the process exits 0."""
    returncode, output = _run_main_then_signal('ok', signal.SIGTERM)

    assert returncode == 0, output
    assert _printed_result(output) == 'RESULT done', output


@posix_signals
def test_run_main_ctrl_c_after_a_clean_main_exits() -> None:
    """Ctrl+C after a clean main still raises KeyboardInterrupt, and the process exits instead of hanging."""
    returncode, output = _run_main_then_signal('ok', signal.SIGINT)

    assert returncode != 0, output
    assert 'KeyboardInterrupt' in output
    assert _printed_result(output) is None, output


@pytest.mark.asyncio
async def test_genkit_run() -> None:
    """Test Genkit.run method."""
    ai = Genkit()

    async def async_fn() -> str:
        return 'world'

    res1 = await ai.run(name='test1', fn=async_fn)
    assert res1 == 'world'

    # Test with metadata
    res2 = await ai.run(name='test2', fn=async_fn, metadata={'foo': 'bar'})
    assert res2 == 'world'

    # Test that sync functions raise TypeError
    def sync_fn() -> str:
        return 'hello'

    with pytest.raises(TypeError, match='fn must be a coroutine function'):
        await ai.run(name='test3', fn=sync_fn)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_genkit_run_tags_flow_step_action_type() -> None:
    """ai.run tells the provider its span is a flow step, so traces can label it."""

    class Recording:
        last: SpanMetadata | None = None

        async def run_in_new_span(self, metadata: SpanMetadata, next: SpanNext[T]) -> T:
            self.last = metadata
            return await next()

    recording = Recording()
    reset_instrumentation()
    configure_instrumentation(recording)
    try:
        ai = Genkit()

        async def step() -> str:
            return 'ok'

        assert await ai.run(name='lookup_account', fn=step) == 'ok'
        assert recording.last is not None
        assert recording.last.name == 'lookup_account'
        assert recording.last.action_type == 'flowStep'
    finally:
        reset_instrumentation()


def test_get_logger_lives_only_in_genkit() -> None:
    """`get_logger` is app-facing: exported from `genkit`, not from `genkit.plugin_api`."""
    assert 'get_logger' in genkit.__all__
    assert 'get_logger' not in genkit.plugin_api.__all__
    assert not hasattr(genkit.plugin_api, 'get_logger')


@pytest.mark.asyncio
async def test_get_logger_in_flow_attaches_trace_id(hex_ids: None) -> None:
    """get_logger() lines inside a flow attach the flow's trace ID to the log record."""
    ai = Genkit()
    captured: list[dict[str, object]] = []

    def capture_log(*, level: int, event: str, attrs: dict[str, object] | None = None) -> None:
        captured.append(build_log_record(level=level, event=event, attrs=attrs or {}))

    with mock.patch('genkit._core._telemetry._log_exporter.emit_log', side_effect=capture_log):

        @ai.flow()
        async def cart_flow() -> str:
            get_logger(__name__).info('looked up cart')
            return 'ok'

        assert await cart_flow() == 'ok'

    assert len(captured) == 1
    assert captured[0]['body'] == {'stringValue': 'looked up cart'}
    trace_id = captured[0].get('traceId')
    assert isinstance(trace_id, str) and len(trace_id) == 32


@pytest.mark.asyncio
async def test_genkit_check_operation() -> None:
    """Test Genkit.check_operation method."""
    ai = Genkit()

    op = Operation(id='123', done=False, action='/background-model/test_action')

    # Create mock background model with check method
    mock_background_model = MagicMock()
    finished = ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('result')]))
    mock_background_model.check = AsyncMock(return_value=Operation(id='123', done=True, output=finished))

    # Patch lookup_background_model to return our mock
    with mock.patch(
        'genkit._ai._background_model.lookup_background_model',
        new=AsyncMock(return_value=mock_background_model),
    ) as mock_lookup:
        updated_op = await ai.check_operation(op)

        assert updated_op.done is True
        assert updated_op.output is not None
        assert updated_op.output.text == 'result'
        mock_lookup.assert_called_once()


@pytest.mark.asyncio
async def test_genkit_check_operation_no_action() -> None:
    """Test Genkit.check_operation method with no action."""
    ai = Genkit()
    op = Operation(id='123', done=False)  # action is None

    with pytest.raises(GenkitError, match='Provided operation is missing original request information') as exc_info:
        await ai.check_operation(op)
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_genkit_check_operation_malformed_key_is_invalid_argument() -> None:
    """A mangled action key on a reloaded handle is the caller's bad argument."""
    ai = Genkit()
    op = Operation(id='123', done=False, action='missing')

    with pytest.raises(
        GenkitError, match='Failed to resolve background model from original request: missing'
    ) as exc_info:
        await ai.check_operation(op)
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_genkit_check_operation_not_found() -> None:
    """Test Genkit.check_operation method with action not found."""
    ai = Genkit()
    op = Operation(id='123', done=False, action='/background-model/nope')

    with pytest.raises(
        GenkitError, match='Failed to resolve background model from original request: /background-model/nope'
    ) as exc_info:
        await ai.check_operation(op)
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_check_operation_round_trips_persisted_dump() -> None:
    """model_dump(by_alias=True) -> model_validate still works next to passing the dict."""
    ai = Genkit()

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation:
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ModelContext) -> Operation:
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-rt', start=start, check=check)
    op = Operation(id='job-1', done=False, action='/background-model/bg-rt')

    reloaded = Operation.model_validate(op.model_dump(by_alias=True))
    updated = await ai.check_operation(reloaded)

    assert updated.done is True


def _text_response(text: str) -> ModelResponse:
    return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(text)]))


@pytest.mark.asyncio
async def test_check_operation_accepts_stored_dict() -> None:
    """generate_operation -> JSON -> json.loads -> check_operation, no model_validate step."""
    ai = Genkit()

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation:
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ModelContext) -> Operation:
        return Operation(id=op.id, done=True, output=_text_response('rendered'))

    ai.define_background_model(name='bg-dict', start=start, check=check)
    started = await ai.generate_operation(model='bg-dict', prompt='a cat video')
    saved = started.model_dump_json()

    updated = await ai.check_operation(json.loads(saved))

    assert type(updated) is Operation[ModelResponse]
    assert updated.done is True
    assert updated.action == '/background-model/bg-dict'
    assert isinstance(updated.output, ModelResponse)
    assert updated.output.text == 'rendered'


@pytest.mark.asyncio
async def test_check_operation_stored_dict_output_reaches_check_as_model_response() -> None:
    """A finished handle's dict output is validated before the check fn sees it."""
    ai = Genkit()
    seen: list[object] = []

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation:
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ModelContext) -> Operation:
        seen.append(op.output)
        return op

    ai.define_background_model(name='bg-done', start=start, check=check)
    finished = Operation[ModelResponse](
        id='job-1', done=True, action='/background-model/bg-done', output=_text_response('done')
    )

    updated = await ai.check_operation(json.loads(finished.model_dump_json()))

    assert isinstance(seen[0], ModelResponse)
    assert isinstance(updated.output, ModelResponse)
    assert updated.output.text == 'done'


@pytest.mark.asyncio
async def test_check_operation_typod_key_is_invalid_argument() -> None:
    """A typo'd key fails INVALID_ARGUMENT with the field named instead of loading as done=None."""
    ai = Genkit()

    with pytest.raises(GenkitError, match='invalid Operation: Done: Extra inputs are not permitted') as exc_info:
        await ai.check_operation({'id': 'x', 'action': '/background-model/test_action', 'Done': True})
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_check_operation_dict_without_id_is_invalid_argument() -> None:
    ai = Genkit()

    with pytest.raises(GenkitError, match='invalid Operation: id: Field required') as exc_info:
        await ai.check_operation({'action': '/background-model/test_action', 'done': False})
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT


@pytest.mark.asyncio
async def test_check_operation_model_response_dump_is_invalid_argument() -> None:
    """A stored generate() response is not the handle; point at response.operation."""
    ai = Genkit()
    boxed = ModelResponse(operation=Operation(id='123', action='/background-model/test_action'))

    with pytest.raises(GenkitError, match='got a ModelResponse dump; pass response.operation') as exc_info:
        await ai.check_operation(json.loads(boxed.model_dump_json()))
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT


@pytest.mark.asyncio
async def test_check_action_input_schema_stays_closed() -> None:
    """Dicts are validated before the action runs, so the published input schema is still a closed Operation."""
    ai = Genkit()

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation:
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ModelContext) -> Operation:
        return op

    ai.define_background_model(name='bg-schema', start=start, check=check, cancel=check)

    for kind, name in (
        (ActionKind.CHECK_OPERATION, 'bg-schema/check'),
        (ActionKind.CANCEL_OPERATION, 'bg-schema/cancel'),
    ):
        action = await ai._registry.resolve_action(kind, name)
        assert action is not None
        schema = action.input_schema
        assert schema['additionalProperties'] is False
        assert schema['required'] == ['id']


@pytest.mark.asyncio
async def test_check_operation_boxed_response_is_invalid_argument() -> None:
    """generate() returns a ModelResponse; the handle is response.operation."""
    ai = Genkit()
    boxed = ModelResponse(operation=Operation(id='123', action='/background-model/test_action'))

    with pytest.raises(GenkitError, match='got ModelResponse; pass response.operation') as exc_info:
        await ai.check_operation(boxed)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_check_operation_str_is_invalid_argument() -> None:
    ai = Genkit()

    with pytest.raises(GenkitError, match='got str, expected Operation') as exc_info:
        await ai.check_operation('not-an-op')  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert 'INVALID_INPUT' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_cancel_operation_round_trips_persisted_dump() -> None:
    """Cancel accepts the same save/reload path as check."""
    ai = Genkit()

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation:
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ModelContext) -> Operation:
        return op

    async def cancel(op: Operation, _ctx: ModelContext) -> Operation:
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-cancel-rt', start=start, check=check, cancel=cancel)
    op = Operation(id='job-1', done=False, action='/background-model/bg-cancel-rt')

    reloaded = Operation.model_validate(op.model_dump(by_alias=True))
    updated = await ai.cancel_operation(reloaded)

    assert updated.done is True


@pytest.mark.asyncio
async def test_cancel_operation_accepts_stored_dict() -> None:
    """The dict a started handle dumped to cancels without a model_validate step."""
    ai = Genkit()

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation:
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ModelContext) -> Operation:
        return op

    async def cancel(op: Operation, _ctx: ModelContext) -> Operation:
        return Operation(id=op.id, done=True, output=_text_response('cancelled'))

    ai.define_background_model(name='bg-cancel-dict', start=start, check=check, cancel=cancel)
    started = await ai.generate_operation(model='bg-cancel-dict', prompt='a cat video')

    updated = await ai.cancel_operation(json.loads(started.model_dump_json()))

    assert type(updated) is Operation[ModelResponse]
    assert updated.done is True
    assert isinstance(updated.output, ModelResponse)
    assert updated.output.text == 'cancelled'


@pytest.mark.asyncio
async def test_cancel_operation_without_cancel_is_unimplemented() -> None:
    """The wrapper's UNIMPLEMENTED propagates through the veneer unchanged."""

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation:
        return Operation(id='123', done=False)

    async def check(op: Operation, _ctx: ModelContext) -> Operation:
        return op

    ai = Genkit()
    ai.define_background_model(name='veneer-no-cancel', start=start, check=check)
    op = Operation(id='123', done=False, action='/background-model/veneer-no-cancel')

    with pytest.raises(GenkitError, match='does not support cancellation') as exc_info:
        await ai.cancel_operation(op)
    assert exc_info.value.status == 'UNIMPLEMENTED'
    assert exc_info.value.reason is RuntimeErrorReason.UNSUPPORTED_BY_MODEL
    assert 'UNSUPPORTED_BY_MODEL' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_background_model_cancel_without_fn_is_unimplemented() -> None:
    """A real no-cancel BackgroundModel raises UNIMPLEMENTED from .cancel."""

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation:
        return Operation(id='1', done=False)

    async def check(op: Operation, _ctx: ModelContext) -> Operation:
        return op

    ai = Genkit()
    action = ai.define_background_model(name='no-cancel', start=start, check=check)
    op = Operation(id='1', action='/background-model/no-cancel')

    with pytest.raises(GenkitError, match='does not support cancellation') as exc_info:
        await action.cancel(op)
    assert exc_info.value.status == 'UNIMPLEMENTED'
    assert exc_info.value.reason is RuntimeErrorReason.UNSUPPORTED_BY_MODEL
    assert 'UNSUPPORTED_BY_MODEL' not in exc_info.value.original_message


@pytest.mark.asyncio
async def test_background_model_check_gate() -> None:
    """BackgroundModel.check uses the same require_operation gate as the veneer: dict in, boxes and strings out."""

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation:
        return Operation(id='1', done=False)

    async def check(op: Operation, _ctx: ModelContext) -> Operation:
        return Operation(id=op.id, done=True)

    ai = Genkit()
    action = ai.define_background_model(name='bg-check', start=start, check=check)
    stored = {'id': '1', 'action': '/background-model/bg-check'}
    boxed = ModelResponse(operation=Operation(id='1', action='/background-model/bg-check'))

    updated = await action.check(stored)
    assert isinstance(updated, Operation)
    assert updated.done is True

    with pytest.raises(GenkitError, match='invalid Operation: Done') as typo_exc:
        await action.check({'id': '1', 'action': '/background-model/bg-check', 'Done': True})
    assert typo_exc.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError, match='got ModelResponse; pass response.operation') as box_exc:
        await action.check(boxed)  # type: ignore[arg-type]
    assert box_exc.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError, match='got str, expected Operation') as str_exc:
        await action.check('not-an-op')  # type: ignore[arg-type]
    assert str_exc.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_background_model_cancel_validates_dict_before_unimplemented() -> None:
    """A stored dict is validated first, so a bad one is INVALID_ARGUMENT and a good one reaches UNIMPLEMENTED."""

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation:
        return Operation(id='1', done=False)

    async def check(op: Operation, _ctx: ModelContext) -> Operation:
        return op

    ai = Genkit()
    action = ai.define_background_model(name='no-cancel', start=start, check=check)

    with pytest.raises(GenkitError, match='invalid Operation: id') as bad_exc:
        await action.cancel({'action': '/background-model/no-cancel'})
    assert bad_exc.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError, match='does not support cancellation') as exc_info:
        await action.cancel({'id': '1', 'action': '/background-model/no-cancel'})
    assert exc_info.value.status == 'UNIMPLEMENTED'


@pytest.mark.asyncio
async def test_current_context() -> None:
    """Test Genkit.current_context method."""
    # current_context is a static method
    assert Genkit.current_context() is None

    context: dict[str, object] = {'auth': {'uid': '123'}}

    # Simulate being inside an action run using ModelContext internal mechanism
    token = _action_context.set(context)
    try:
        assert Genkit.current_context() == context
    finally:
        _action_context.reset(token)

    assert Genkit.current_context() is None


def test_genkit_positional_argument_raises_type_error() -> None:
    with pytest.raises(
        TypeError,
        match=(
            r'Genkit\(\) takes no positional arguments, got 1\. '
            r'Pass keyword arguments instead, e\.g\. '
            r"Genkit\(model='googleai/gemini-flash-latest'\)\."
        ),
    ):
        Genkit('googleai/gemini-flash-latest')  # type: ignore[reportCallIssue,too-many-positional-arguments]
    with pytest.raises(
        TypeError,
        match=(
            r'Genkit\(\) takes no positional arguments, got 1\. '
            r'Pass keyword arguments instead, e\.g\. '
            r'Genkit\(plugins=\[...\], model="..."\)\.'
        ),
    ):
        Genkit([])  # type: ignore[reportCallIssue,too-many-positional-arguments]


def test_genkit_path_string_does_not_suggest_model_kwarg() -> None:
    with pytest.raises(TypeError) as exc_info:
        Genkit('./prompts')  # type: ignore[reportCallIssue,too-many-positional-arguments]
    message = str(exc_info.value)
    assert "model='./prompts'" not in message
    assert 'Genkit(plugins=[...], model="...")' in message


def test_genkit_two_positional_args_says_got_2() -> None:
    with pytest.raises(
        TypeError,
        match=r'Genkit\(\) takes no positional arguments, got 2\.',
    ):
        Genkit('googleai/gemini-flash-latest', [])  # type: ignore[reportCallIssue,too-many-positional-arguments]


def test_genkit_init_has_docstring() -> None:
    """Genkit.__init__.__doc__ documents keyword arguments and lazy model resolution."""
    doc = Genkit.__init__.__doc__
    assert doc is not None
    assert 'plugins:' in doc
    assert 'model:' in doc
    assert 'prompt_dir:' in doc
    assert 'GoogleAI.gemini_model' in doc


def test_define_evaluator_takes_name_and_fn_positionally() -> None:
    """define_evaluator(name, fn, *, ...) matches define_model; options stay keyword-only."""
    ai = Genkit()

    async def exact(row: BaseDataPoint, options: object | None) -> EvaluatorResponse:
        return EvaluatorResponse(test_case_id=row.test_case_id or '', evaluation=[Score(score=True)])

    action = ai.define_evaluator('exact', exact, display_name='Exact', definition='Matches exactly.')
    assert action.name == 'exact'

    with pytest.raises(TypeError):
        ai.define_evaluator('exact2', exact, 'Exact', 'Matches exactly.')  # type: ignore[misc]  # pyright: ignore[reportCallIssue]
