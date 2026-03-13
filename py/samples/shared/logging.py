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
"""Common developer-experience setup for Genkit samples.

Provides a single ``setup_sample()`` call that configures:

- **Rich tracebacks** -- pretty, colorful exception rendering with local
  variables shown in context.
- **Colored structlog + stdlib logging** -- human-readable, color-coded
  log messages for *both* structlog loggers (used by sample code) and
  stdlib loggers (used by uvicorn, third-party libraries, etc.).

The integration uses ``structlog.stdlib.ProcessorFormatter`` so that
**every** log line -- regardless of origin -- flows through the same
colored ``ConsoleRenderer``.

Usage::

    from samples.shared.logging import setup_sample

    setup_sample()  # call once at the top of each sample main.py
"""

import logging
import os
import sys

import structlog
from rich.traceback import install as _install_rich_traceback


def _want_colors() -> bool:
    """Decide whether to emit ANSI color codes.

    Color is enabled unless explicitly suppressed via ``NO_COLOR=1``
    (see https://no-color.org).  We default to **True** rather than
    checking ``isatty()`` because ``genkit start`` pipes
    stdout/stderr through the dev-server, which makes ``isatty()``
    return ``False`` even though the output ultimately lands in a
    color-capable terminal or the Dev UI.
    """
    return os.environ.get('NO_COLOR', '') == ''


def setup_sample(log_level: int = logging.DEBUG) -> None:
    """One-stop developer-experience setup for Genkit samples.

    Installs Rich tracebacks and configures *both* structlog and
    Python's standard ``logging`` module for pretty, colored console
    output.  This ensures that log lines from uvicorn, third-party
    libraries, and Genkit internals all render with the same style.

    Call this once at the top of each sample's ``main.py`` before
    any logging calls.

    Args:
        log_level: Minimum log level to display.  Defaults to
            ``logging.DEBUG``.
    """
    # ── Rich tracebacks ─────────────────────────────────────────
    _install_rich_traceback(show_locals=True, width=120, extra_lines=3)

    # ── Shared processor chain ──────────────────────────────────
    # These processors run on EVERY log entry -- both structlog and
    # stdlib ("foreign") entries.  They must NOT include a final
    # renderer; the renderer lives in the ProcessorFormatter below.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt='iso'),
    ]

    # ── structlog configuration ─────────────────────────────────
    # Route structlog entries into stdlib logging so that a single
    # ProcessorFormatter handles final rendering for everything.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ── stdlib logging configuration ────────────────────────────
    # A single handler on the root logger with ProcessorFormatter
    # ensures uvicorn, third-party, and structlog entries all get
    # the same colored ConsoleRenderer treatment.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=_want_colors()),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
