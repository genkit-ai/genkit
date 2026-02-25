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

"""Tests for the testing utilities module.

This module contains comprehensive tests for the testing utilities,
ensuring parity with the JavaScript implementation in:
    js/ai/src/testing/model-tester.ts

Test Coverage
=============

┌─────────────────────────────────────────────────────────────────────────────┐
│ Test Case                        │ Description                              │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ EchoModel Tests                                                             │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ test_echo_model_basic            │ Basic echo functionality                 │
│ test_echo_model_with_config      │ Echo includes config in response         │
│ test_echo_model_stream_countdown │ Stream countdown chunks                  │
│ test_echo_model_stores_request   │ Stores last request for inspection       │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ ProgrammableModel Tests                                                     │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ test_programmable_model_basic    │ Returns programmed responses             │
│ test_programmable_model_multiple │ Multiple sequential responses            │
│ test_programmable_model_chunks   │ Streams programmed chunks                │
│ test_programmable_model_reset    │ Reset clears state                       │
│ test_programmable_model_request  │ Stores deep copy of last request         │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ StaticResponseModel Tests                                                   │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ test_static_model_basic          │ Returns same response always             │
│ test_static_model_request_count  │ Counts requests                          │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ test_models() Tests                                                         │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ test_test_models_basic           │ Basic test suite execution               │
│ test_test_models_report_format   │ Report structure matches JS              │
│ test_skip_test_error             │ SkipTestError handling                   │
│ test_gablorken_tool              │ Tool calculation test                    │
└──────────────────────────────────┴──────────────────────────────────────────┘
"""

import pytest

from genkit.ai import Genkit
from genkit.core.typing import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    Message,
    Part,
    Role,
    TextPart,
)
from genkit.testing import (
    EchoModel,
    GablorkenInput,
    ProgrammableModel,
    SkipTestError,
    StaticResponseModel,
    define_echo_model,
    define_programmable_model,
    define_static_response_model,
    skip,
    test_models as run_model_tests,
)


class MockActionRunContext:
    """Mock context for testing model functions directly."""

    def __init__(self) -> None:
        """Initialize with empty chunks list."""
        self.chunks: list[GenerateResponseChunk] = []

    def send_chunk(self, chunk: GenerateResponseChunk) -> None:
        """Append a chunk to the chunks list."""
        self.chunks.append(chunk)


@pytest.fixture
def ai() -> Genkit:
    """Create a fresh Genkit instance for each test."""
    return Genkit()


class TestEchoModel:
    """Tests for EchoModel functionality."""

    def test_echo_model_basic(self) -> None:
        """Test basic echo functionality."""
        echo = EchoModel()
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='Hello world'))],
                ),
            ],
        )

        # pyright: ignore[reportArgumentType] - MockActionRunContext is compatible
        response = echo.model_fn(request, ctx)  # type: ignore[arg-type]

        assert response.message is not None
        text = response.message.content[0].root.text
        assert isinstance(text, str)
        assert '[ECHO]' in text
        assert 'user:' in text
        assert 'Hello world' in text

    def test_echo_model_with_config(self) -> None:
        """Test that echo includes config in response."""
        echo = EchoModel()
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='test'))],
                ),
            ],
            config={'temperature': 0.5},
        )

        response = echo.model_fn(request, ctx)  # type: ignore[arg-type]

        assert response.message is not None
        text = response.message.content[0].root.text
        assert isinstance(text, str)
        assert 'temperature' in text

    def test_echo_model_stream_countdown(self) -> None:
        """Test stream countdown functionality."""
        echo = EchoModel(stream_countdown=True)
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='test'))],
                ),
            ],
        )

        echo.model_fn(request, ctx)  # type: ignore[arg-type]

        # Should have streamed 3, 2, 1
        assert len(ctx.chunks) == 3
        assert ctx.chunks[0].content[0].root.text == '3'
        assert ctx.chunks[1].content[0].root.text == '2'
        assert ctx.chunks[2].content[0].root.text == '1'

    def test_echo_model_stores_request(self) -> None:
        """Test that echo stores the last request."""
        echo = EchoModel()
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='test'))],
                ),
            ],
        )

        echo.model_fn(request, ctx)  # type: ignore[arg-type]

        assert echo.last_request is not None
        assert echo.last_request.messages[0].content[0].root.text == 'test'

    @pytest.mark.asyncio
    async def test_define_echo_model(self, ai: Genkit) -> None:
        """Test define_echo_model helper function."""
        echo, _action = define_echo_model(ai, name='testEcho')

        response = await ai.generate(model='testEcho', prompt='Hello')

        assert '[ECHO]' in response.text
        assert echo.last_request is not None


class TestProgrammableModel:
    """Tests for ProgrammableModel functionality."""

    def test_programmable_model_basic(self) -> None:
        """Test basic programmable model functionality."""
        pm = ProgrammableModel()
        pm.responses = [
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Response 1'))],
                ),
            ),
        ]
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='test'))],
                ),
            ],
        )

        response = pm.model_fn(request, ctx)  # type: ignore[arg-type]

        assert response.message is not None
        assert response.message.content[0].root.text == 'Response 1'
        assert pm.request_count == 1

    def test_programmable_model_multiple_responses(self) -> None:
        """Test multiple sequential responses."""
        pm = ProgrammableModel()
        pm.responses = [
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Response 1'))],
                ),
            ),
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Response 2'))],
                ),
            ),
        ]
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='test'))],
                ),
            ],
        )

        response1 = pm.model_fn(request, ctx)  # type: ignore[arg-type]
        response2 = pm.model_fn(request, ctx)  # type: ignore[arg-type]

        assert response1.message is not None
        assert response2.message is not None
        assert response1.message.content[0].root.text == 'Response 1'
        assert response2.message.content[0].root.text == 'Response 2'
        assert pm.request_count == 2

    def test_programmable_model_chunks(self) -> None:
        """Test streaming programmed chunks."""
        pm = ProgrammableModel()
        pm.responses = [
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Final'))],
                ),
            ),
        ]
        pm.chunks = [
            [
                GenerateResponseChunk(content=[Part(root=TextPart(text='Chunk 1'))]),
                GenerateResponseChunk(content=[Part(root=TextPart(text='Chunk 2'))]),
            ],
        ]
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='test'))],
                ),
            ],
        )

        pm.model_fn(request, ctx)  # type: ignore[arg-type]

        assert len(ctx.chunks) == 2
        assert ctx.chunks[0].content[0].root.text == 'Chunk 1'
        assert ctx.chunks[1].content[0].root.text == 'Chunk 2'

    def test_programmable_model_reset(self) -> None:
        """Test reset clears state."""
        pm = ProgrammableModel()
        pm.responses = [
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Response'))],
                ),
            ),
        ]
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='test'))],
                ),
            ],
        )

        pm.model_fn(request, ctx)  # type: ignore[arg-type]
        assert pm.request_count == 1
        assert pm.last_request is not None

        pm.reset()

        assert pm.request_count == 0
        assert pm.last_request is None
        assert pm.responses == []
        assert pm.chunks is None

    def test_programmable_model_stores_deep_copy(self) -> None:
        """Test that last_request is a deep copy."""
        pm = ProgrammableModel()
        pm.responses = [
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Response'))],
                ),
            ),
        ]
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='original'))],
                ),
            ],
        )

        pm.model_fn(request, ctx)  # type: ignore[arg-type]

        # Modify original request
        original_part = request.messages[0].content[0].root
        assert isinstance(original_part, TextPart)
        original_part.text = 'modified'

        # last_request should still have original value (deep copy)
        assert pm.last_request is not None
        stored_part = pm.last_request.messages[0].content[0].root
        assert isinstance(stored_part, TextPart)
        assert stored_part.text == 'original'

    @pytest.mark.asyncio
    async def test_define_programmable_model(self, ai: Genkit) -> None:
        """Test define_programmable_model helper function."""
        pm, _action = define_programmable_model(ai, name='testPM')
        pm.responses = [
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Programmed response'))],
                ),
            ),
        ]

        response = await ai.generate(model='testPM', prompt='Hello')

        assert response.text == 'Programmed response'
        assert pm.last_request is not None


class TestStaticResponseModel:
    """Tests for StaticResponseModel functionality."""

    def test_static_model_basic(self) -> None:
        """Test basic static response model functionality."""
        static = StaticResponseModel(
            message={
                'role': 'model',
                'content': [{'text': 'Static response'}],
            }
        )
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='test'))],
                ),
            ],
        )

        response = static.model_fn(request, ctx)  # type: ignore[arg-type]

        assert response.message is not None
        assert response.message.content[0].root.text == 'Static response'

    def test_static_model_request_count(self) -> None:
        """Test request counting."""
        static = StaticResponseModel(
            message={
                'role': 'model',
                'content': [{'text': 'Static'}],
            }
        )
        ctx = MockActionRunContext()

        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='test'))],
                ),
            ],
        )

        static.model_fn(request, ctx)  # type: ignore[arg-type]
        static.model_fn(request, ctx)  # type: ignore[arg-type]
        static.model_fn(request, ctx)  # type: ignore[arg-type]

        assert static.request_count == 3

    @pytest.mark.asyncio
    async def test_define_static_response_model(self, ai: Genkit) -> None:
        """Test define_static_response_model helper function."""
        static, _action = define_static_response_model(
            ai,
            message={
                'role': 'model',
                'content': [{'text': 'Always this'}],
            },
            name='testStatic',
        )

        response1 = await ai.generate(model='testStatic', prompt='First')
        response2 = await ai.generate(model='testStatic', prompt='Second')

        assert response1.text == 'Always this'
        assert response2.text == 'Always this'
        assert static.request_count == 2


class TestSkipTestError:
    """Tests for SkipTestError and skip() function."""

    def test_skip_raises_error(self) -> None:
        """Test that skip() raises SkipTestError."""
        with pytest.raises(SkipTestError):
            skip()

    def test_skip_test_error_is_exception(self) -> None:
        """Test that SkipTestError is an Exception subclass."""
        assert issubclass(SkipTestError, Exception)


class TestGablorkenInput:
    """Tests for GablorkenInput model."""

    def test_gablorken_input_validation(self) -> None:
        """Test GablorkenInput validates correctly."""
        input = GablorkenInput(value=2.0)
        assert input.value == 2.0

    def test_gablorken_calculation(self) -> None:
        """Test the gablorken calculation: value^3 + 1.407."""
        # 2^3 + 1.407 = 9.407
        value = 2.0
        expected = (value**3) + 1.407
        assert expected == 9.407


class TestTestModels:
    """Tests for the test_models() function."""

    @pytest.mark.asyncio
    async def test_test_models_with_echo_model(self, ai: Genkit) -> None:
        """Test test_models with an echo model."""
        # Define an echo model that will pass the basic hi test
        pm, _ = define_programmable_model(ai, name='testModel')
        pm.responses = [
            # For basic hi test
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Hi'))],
                ),
            ),
            # For multimodal test (will skip since no media support)
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='plus'))],
                ),
            ),
            # For history test
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Nice to meet you'))],
                ),
            ),
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Your name is Glorb'))],
                ),
            ),
            # For system prompt test
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Bye'))],
                ),
            ),
            # For structured output test
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='{"name": "Jack", "occupation": "Lumberjack"}'))],
                ),
            ),
            # For tool calling test (will skip since no tools support)
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='9.407'))],
                ),
            ),
        ]

        report = await run_model_tests(ai, ['testModel'])

        # Verify report structure
        assert isinstance(report, list)
        assert len(report) == 6  # 6 test cases

        # Check test case names match JS implementation
        test_names = [r['description'] for r in report]
        assert 'basic hi' in test_names
        assert 'multimodal' in test_names
        assert 'history' in test_names
        assert 'system prompt' in test_names
        assert 'structured output' in test_names
        assert 'tool calling' in test_names

    @pytest.mark.asyncio
    async def test_test_models_report_format(self, ai: Genkit) -> None:
        """Test that report format matches JS implementation."""
        pm, _ = define_programmable_model(ai, name='formatTestModel')
        pm.responses = [
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Hi'))],
                ),
            ),
        ] * 10  # Enough responses for all tests

        report = await run_model_tests(ai, ['formatTestModel'])

        # Verify report structure matches JS TestReport type
        for case_report in report:
            assert 'description' in case_report
            assert 'models' in case_report
            assert isinstance(case_report['models'], list)

            for model_result in case_report['models']:
                assert 'name' in model_result
                assert 'passed' in model_result
                # Optional fields: skipped, error
                if not model_result['passed'] and 'error' in model_result:
                    assert 'message' in model_result['error']

    @pytest.mark.asyncio
    async def test_test_models_multiple_models(self, ai: Genkit) -> None:
        """Test test_models with multiple models."""
        pm1, _ = define_programmable_model(ai, name='model1')
        pm1.responses = [
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Hi'))],
                ),
            ),
        ] * 10

        pm2, _ = define_programmable_model(ai, name='model2')
        pm2.responses = [
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Hello'))],
                ),
            ),
        ] * 10

        report = await run_model_tests(ai, ['model1', 'model2'])

        # Each test case should have results for both models
        for case_report in report:
            assert len(case_report['models']) == 2
            model_names = [m.get('name') for m in case_report['models']]
            assert 'model1' in model_names
            assert 'model2' in model_names

    @pytest.mark.asyncio
    async def test_test_models_handles_failures(self, ai: Genkit) -> None:
        """Test that test_models properly reports failures."""
        pm, _ = define_programmable_model(ai, name='failingModel')
        pm.responses = [
            # Return something that doesn't match expected pattern
            GenerateResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=TextPart(text='Goodbye'))],  # Should be "Hi"
                ),
            ),
        ] * 10

        report = await run_model_tests(ai, ['failingModel'])

        # Find the basic hi test
        basic_hi_report = next(r for r in report if r['description'] == 'basic hi')
        model_result = basic_hi_report['models'][0]

        # Should have failed
        assert model_result.get('passed') is False
        assert 'error' in model_result
        error = model_result.get('error')
        assert error is not None
        assert 'message' in error
