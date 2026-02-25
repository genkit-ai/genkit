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

"""Shared types for the ``conform`` tool.

This module contains types that are used across multiple modules
(``display``, ``runner``, ``cli``, ``plugins``) to avoid circular
imports.  It has **no** internal dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


class Status(enum.Enum):
    """Status of a plugin conformance run."""

    PENDING = 'pending'
    RUNNING = 'running'
    PASSED = 'passed'
    FAILED = 'failed'
    SKIPPED = 'skipped'
    ERROR = 'error'


@runtime_checkable
class Runtime(Protocol):
    """Protocol defining a conformance test runtime.

    Each runtime (Python, JS, Go, Dart, Java, Rust) implements this
    protocol to tell the runner how to discover plugins, locate specs,
    and build the command to execute an entry point.

    A concrete implementation is :class:`conform.config.RuntimeConfig`.
    New runtimes can be added by either:

    * Adding a ``[tool.conform.runtimes.<name>]`` section to
      ``pyproject.toml`` (parsed into :class:`RuntimeConfig`), or
    * Creating a new class that satisfies this protocol.
    """

    @property
    def name(self) -> str:
        """Runtime identifier, e.g. ``python``, ``js``, ``go``."""
        ...

    @property
    def specs_dir(self) -> Path:
        """Directory containing conformance specs (per-plugin subdirs)."""
        ...

    @property
    def plugins_dir(self) -> Path:
        """Directory containing plugin source trees."""
        ...

    @property
    def entry_command(self) -> list[str]:
        """Command prefix to run an entry point.

        The entry-point file path is appended as the last argument.
        Example: ``["uv", "run", "--project", "/path/to/py", "--active"]``.
        """
        ...

    @property
    def cwd(self) -> Path | None:
        """Working directory for subprocess execution.

        ``None`` means use the repo root (default).  JS uses ``js/``
        so Node module resolution finds the pnpm workspace packages.
        """
        ...

    @property
    def entry_filename(self) -> str:
        """Entry-point filename (e.g. ``conformance_entry.py``)."""
        ...

    @property
    def model_marker(self) -> str:
        """Glob pattern to locate model provider source files.

        Used by ``check-plugin`` to discover which plugins are model
        providers.  Example: ``model_info.py`` for Python,
        ``models.ts`` for JS.
        """
        ...


@dataclass
class FailedTest:
    """A single failed test case — used for the end-of-run error summary."""

    test_name: str
    model: str
    error: str


@dataclass
class PluginResult:
    """Result of running conformance tests for a single plugin.

    When multiple runtimes are active, each (plugin, runtime) pair
    produces a separate ``PluginResult``.  The ``runtime`` field
    identifies which runtime produced the result.
    """

    plugin: str
    runtime: str = ''
    status: Status = Status.PENDING
    elapsed_s: float = 0.0
    stdout: str = ''
    stderr: str = ''
    return_code: int = -1
    error_message: str = ''
    missing_env_vars: list[str] = field(default_factory=list)
    tests_total: int = 0
    tests_supports: int = 0
    tests_custom: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    failed_tests: list[FailedTest] = field(default_factory=list)
