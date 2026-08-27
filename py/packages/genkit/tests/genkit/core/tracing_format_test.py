# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""LoggingInstrumentor must not rewrite the process log format."""

from unittest.mock import MagicMock, patch

from genkit._core._tracing import init_provider


def test_init_provider_does_not_rewrite_log_format() -> None:
    """The shared TTY stays as the process configured it — no trace_id= wall."""
    instrumentor = MagicMock()
    with (
        patch('genkit._core._tracing.trace_api.get_tracer_provider', return_value=None),
        patch('genkit._core._tracing.trace_api.set_tracer_provider'),
        patch('genkit._core._tracing.LoggingInstrumentor', return_value=instrumentor),
    ):
        init_provider()

    instrumentor.instrument.assert_called_once_with(set_logging_format=False)
