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

"""Native model conformance test runner.

Replaces ``genkit dev:test-model`` entirely — no genkit CLI dependency.

Two execution modes (same validator/reporting logic):

1. **In-process (Python runtime):** Imports the conformance entry point
   module, which creates a ``Genkit`` instance and registers model actions.
   Tests are run via ``ai.generate()`` so the full framework pipeline
   (format parsing, ``extract_json``, etc.) is exercised — matching the
   real user experience.

2. **Subprocess + async HTTP (JS, Go, other runtimes):** Starts the entry
   point as a subprocess (e.g. ``node entry.ts``), then communicates with
   the reflection server via ``ReflectionClient`` (httpx).

Canonical JS source (keep in sync):
    genkit-tools/cli/src/commands/dev-test-model.ts
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import yaml

from conform.config import ConformConfig, RuntimeConfig
from conform.display import console
from conform.executors.in_process_runner import InProcessRunner
from conform.plugins import entry_point, spec_file
from conform.reflection import ReflectionClient
from conform.util_test_cases import TEST_CASES
from conform.validators import ValidationError, get_validator

logger = logging.getLogger(__name__)

# How long to wait for the runtime file to appear (seconds).
_RUNTIME_FILE_TIMEOUT = 30.0
_RUNTIME_FILE_POLL = 0.5

# Cap for exponential backoff on test retries (seconds).
_MAX_BACKOFF = 60.0


@dataclass
class TestResult:
    """Result of a single test case."""

    name: str
    passed: bool = False
    error: str = ''
    elapsed_s: float = 0.0


@dataclass
class SuiteResult:
    """Result of testing a single model."""

    model: str
    tests: list[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        """Count of passed tests."""
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed(self) -> int:
        """Count of failed tests."""
        return sum(1 for t in self.tests if not t.passed)


@dataclass
class RunResult:
    """Aggregate result of all test suites."""

    suites: list[SuiteResult] = field(default_factory=list)

    @property
    def total_passed(self) -> int:
        """Total passed tests across all suites."""
        return sum(s.passed for s in self.suites)

    @property
    def total_failed(self) -> int:
        """Total failed tests across all suites."""
        return sum(s.failed for s in self.suites)


class ActionRunner(Protocol):
    """Protocol for running model actions.

    Three implementations:
    - ``InProcessRunner``: Python-only, calls actions via ``ai.generate()``.
    - ``ReflectionRunner``: Any runtime, talks to reflection server via HTTP.
    - ``NativeRunner``: Any runtime, communicates via JSONL-over-stdio.
    """

    async def run_action(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        stream: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run a model action and return (response, chunks)."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...


# InProcessRunner is in conform.executors.conformance_native
# (imported above) alongside the Go and JS executors.


class ReflectionRunner:
    """Run model actions via a subprocess reflection server.

    Starts the entry point subprocess, discovers the reflection server
    URL from the runtime file, and communicates via async HTTP.
    """

    def __init__(
        self,
        entry_cmd: list[str],
        cwd: str,
        required_keys: set[str],
        *,
        action_timeout: float = 120.0,
        health_timeout: float = 5.0,
        startup_timeout: float = 30.0,
    ) -> None:
        """Initialize with the entry command, working directory, and required keys."""
        self._entry_cmd = entry_cmd
        self._cwd = cwd
        self._required_keys = required_keys
        self._action_timeout = action_timeout
        self._health_timeout = health_timeout
        self._startup_timeout = startup_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._client: ReflectionClient | None = None

    async def start(self) -> None:
        """Start the subprocess and wait for the reflection server."""
        # Clean up stale runtime files.
        runtimes_dir = Path(self._cwd) / '.genkit' / 'runtimes'
        if runtimes_dir.exists():
            for f in runtimes_dir.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except OSError:
                        pass

        env = {**os.environ, 'GENKIT_ENV': 'dev'}
        console.print(f'[dim]Starting entry point: {" ".join(self._entry_cmd)}[/dim]')

        self._proc = await asyncio.create_subprocess_exec(
            *self._entry_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=env,
        )

        # Discover the reflection server URL.
        url = await _find_runtime_url(Path(self._cwd), timeout=self._startup_timeout)
        if not url:
            raise RuntimeError(f'Runtime file not found after {self._startup_timeout}s.')

        self._client = ReflectionClient(
            url,
            action_timeout=self._action_timeout,
            health_timeout=self._health_timeout,
        )

        if not await self._client.wait_for_health():
            raise RuntimeError(f'Reflection server not healthy after {self._startup_timeout}s.')

        if not await self._client.wait_for_actions(self._required_keys):
            console.print('[yellow]Warning:[/yellow] Not all model actions registered. Proceeding anyway.')

    async def run_action(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        stream: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run a model action via the reflection server."""
        if self._client is None:
            raise RuntimeError('ReflectionRunner not started. Call start() first.')
        return await self._client.run_action(key, input_data, stream=stream)

    async def close(self) -> None:
        """Stop the subprocess and close the HTTP client."""
        if self._client:
            await self._client.close()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()


class NativeRunner:
    r"""Run model actions via a native subprocess using JSONL-over-stdio.

    Starts the native executor once per plugin.  The executor initializes
    the plugin (``genkit.Init`` / ``genkit(...)``), then enters a read loop:

    1. Reads one JSON line from **stdin** (a ``GenerateRequest``).
    2. Calls ``generate()`` natively using the SDK.
    3. Writes one JSON line to **stdout** (a ``GenerateResponse``).

    Protocol (JSONL-over-stdio)::

        --> {"model": "googleai/gemini-2.5-flash", "messages": [...], ...}\\n
        <-- {"response": {...}, "chunks": [...], "error": null}\\n

    Advantages over ``ReflectionRunner``:
    - No HTTP server or port management.
    - Lower latency (no network overhead).
    - Simpler process lifecycle.

    The executor is built/compiled once at start, then handles all test
    cases for all models in that plugin.

    Thread safety: an ``asyncio.Lock`` serializes access to the
    subprocess pipes so concurrent test coroutines don't race on
    ``readline()``.
    """

    # 100 MB — large enough for base64-encoded images in JSON responses.
    _STREAM_LIMIT: int = 100 * 1024 * 1024

    def __init__(
        self,
        entry_cmd: list[str],
        cwd: str,
        *,
        action_timeout: float = 120.0,
    ) -> None:
        """Initialize with the command to start the native executor.

        Args:
            entry_cmd: Command to start the executor
                (e.g. ``["go", "run", "conformance_native.go"]``).
            cwd: Working directory for subprocess execution.
            action_timeout: Timeout per generate() call in seconds.
        """
        self._entry_cmd = entry_cmd
        self._cwd = cwd
        self._action_timeout = action_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        # Pre-resolve the NODE_PATH so the JS/TS executor can resolve
        # workspace packages from the cwd (e.g. js/) even though the
        # executor file lives outside the JS workspace tree.
        cwd_path = Path(self._cwd).resolve()
        self._node_path = str(cwd_path / 'node_modules')

    async def start(self) -> None:
        """Start the native executor subprocess."""
        env = {**os.environ}
        existing = env.get('NODE_PATH', '')
        env['NODE_PATH'] = f'{self._node_path}:{existing}' if existing else self._node_path
        console.print(f'[dim]Starting native executor: {" ".join(self._entry_cmd)}[/dim]')

        self._proc = await asyncio.create_subprocess_exec(
            *self._entry_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=env,
            limit=self._STREAM_LIMIT,
        )

        # Wait for the "ready" signal from the executor.
        # The executor prints a JSON line {"ready": true} when initialized.
        if self._proc.stdout is None:
            raise RuntimeError('Native executor stdout is not piped.')

        try:
            ready_line = await asyncio.wait_for(
                self._proc.stdout.readline(),
                timeout=60.0,  # Plugin init can be slow (Go compilation).
            )
        except asyncio.TimeoutError:
            await self._kill()
            raise RuntimeError(
                f'Native executor did not send ready signal within 60s.\nCommand: {" ".join(self._entry_cmd)}'
            ) from None

        if not ready_line:
            stderr_output = await self._drain_stderr()
            await self._kill()
            raise RuntimeError(f'Native executor exited before sending ready signal.\nstderr: {stderr_output}')

        try:
            ready_data = json.loads(ready_line)
        except json.JSONDecodeError:
            await self._kill()
            raise RuntimeError(f'Native executor sent invalid ready signal: {ready_line!r}') from None

        if not ready_data.get('ready'):
            await self._kill()
            raise RuntimeError(f'Native executor ready signal was not {{"ready": true}}: {ready_data}')

        console.print('[dim]Native executor ready.[/dim]')

    async def run_action(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        stream: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Send a generate request and read the response via stdio.

        Access to the subprocess pipes is serialized with an asyncio
        lock because the JSONL protocol is strictly request-response
        and concurrent ``readline()`` calls on the same stream race.
        """
        async with self._lock:
            return await self._run_action_locked(key, input_data, stream=stream)

    async def _run_action_locked(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        stream: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run a single action while holding the lock."""
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError('Native executor not started. Call start() first.')

        # Check if the process has already exited.
        if self._proc.returncode is not None:
            stderr_output = await self._drain_stderr()
            raise RuntimeError(f'Native executor has exited (rc={self._proc.returncode}).\nstderr: {stderr_output}')

        # Build the request payload.
        model_name = key.removeprefix('/model/')
        request = {
            'model': model_name,
            'input': input_data,
            'stream': stream,
        }

        # Send request as a single JSON line.
        line = json.dumps(request, separators=(',', ':')) + '\n'
        self._proc.stdin.write(line.encode('utf-8'))
        await self._proc.stdin.drain()

        # Read response as a single JSON line.
        try:
            response_line = await asyncio.wait_for(
                self._proc.stdout.readline(),
                timeout=self._action_timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f'Native executor timed out after {self._action_timeout}s for model {model_name}'
            ) from None

        if not response_line:
            stderr_output = await self._drain_stderr()
            raise RuntimeError(f'Native executor closed stdout unexpectedly.\nstderr: {stderr_output}')

        try:
            result = json.loads(response_line)
        except json.JSONDecodeError:
            raise RuntimeError(f'Native executor returned invalid JSON: {response_line[:500]!r}') from None

        # Check for executor-level error.
        error = result.get('error')
        if error:
            raise RuntimeError(f'Native executor error: {error}')

        response = result.get('response', {})
        chunks = result.get('chunks', [])
        return response, chunks

    async def _drain_stderr(self) -> str:
        """Read available stderr from the subprocess (non-blocking)."""
        if self._proc is None or self._proc.stderr is None:
            return ''
        try:
            stderr_bytes = await asyncio.wait_for(self._proc.stderr.read(8192), timeout=2.0)
            return stderr_bytes.decode('utf-8', errors='replace')
        except asyncio.TimeoutError:
            return ''

    async def _kill(self) -> None:
        """Force-kill the executor subprocess."""
        if self._proc and self._proc.returncode is None:
            self._proc.kill()
            await self._proc.wait()

    async def close(self) -> None:
        """Shut down the executor subprocess."""
        if self._proc and self._proc.returncode is None:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()


def _load_spec(spec_path: Path) -> list[dict[str, Any]]:
    """Load and parse a model-conformance.yaml file."""
    content = spec_path.read_text(encoding='utf-8')
    parsed = yaml.safe_load(content)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f'Unexpected spec format in {spec_path}')


class SpecTestCounts(NamedTuple):
    """Pre-calculated test counts from a spec file."""

    supports: int
    custom: int
    total: int


def count_spec_tests(
    spec_path: Path,
    default_supports: list[str] | None = None,
) -> SpecTestCounts:
    """Count the total number of tests in a spec file without running them.

    This allows the progress table to show ``passed/total`` from the start
    rather than incrementing the denominator as tests complete.

    Returns:
        A :class:`SpecTestCounts` with the breakdown of built-in
        (supports) tests, custom tests, and the total.
    """
    if default_supports is None:
        default_supports = [
            'tool-request',
            'structured-output',
            'multiturn',
            'system-role',
            'input-image-base64',
            'input-image-url',
            'streaming-multiturn',
            'streaming-tool-request',
            'streaming-structured-output',
        ]

    if not spec_path.exists():
        return SpecTestCounts(supports=0, custom=0, total=0)

    suites = _load_spec(spec_path)
    n_supports = 0
    n_custom = 0
    for suite in suites:
        caps = suite.get('supports') or ([] if suite.get('tests') else default_supports)
        # Count only capabilities that have a matching built-in test case.
        n_supports += sum(1 for cap in caps if cap in TEST_CASES)
        # Count custom tests.
        n_custom += len(suite.get('tests', []))
    return SpecTestCounts(supports=n_supports, custom=n_custom, total=n_supports + n_custom)


def _model_to_action_key(model: str) -> str:
    """Convert a model name to an action key."""
    return model if model.startswith('/') else f'/model/{model}'


async def _find_runtime_url(project_root: Path, *, timeout: float = _RUNTIME_FILE_TIMEOUT) -> str | None:
    """Poll for a runtime file and return the reflection server URL."""
    runtimes_dir = project_root / '.genkit' / 'runtimes'
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if runtimes_dir.exists():
            for f in sorted(runtimes_dir.iterdir(), reverse=True):
                if f.is_file():
                    try:
                        data = json.loads(f.read_text(encoding='utf-8'))
                        url = data.get('reflectionServerUrl')
                        if url:
                            logger.info('Found runtime at %s', url)
                            return url
                    except (json.JSONDecodeError, OSError):
                        pass
        await asyncio.sleep(_RUNTIME_FILE_POLL)

    return None


async def _run_single_test(
    runner: ActionRunner,
    model: str,
    test_case: dict[str, Any],
    *,
    max_retries: int = 0,
    retry_base_delay: float = 1.0,
) -> TestResult:
    """Execute a single test case with optional retry.

    When *max_retries* > 0, a failed test is retried with exponential
    backoff and full jitter (delay = random() × min(base × 2^k, 60)).
    The caller (``_run_suite``) switches to serial execution on retry
    to avoid hammering rate-limited APIs.
    """
    name = test_case.get('name', 'Unnamed Test')
    result = TestResult(name=name)
    start = time.monotonic()

    for attempt in range(1 + max_retries):
        try:
            action_key = _model_to_action_key(model)
            should_stream = bool(test_case.get('stream'))
            input_data = test_case['input']

            response, chunks = await runner.run_action(
                action_key,
                input_data,
                stream=should_stream,
            )

            if should_stream and not chunks:
                raise ValidationError('Streaming requested but no chunks received.')

            # Run all validators.
            for v_spec in test_case.get('validators', []):
                parts = v_spec.split(':', 1)
                v_name = parts[0]
                v_arg = parts[1] if len(parts) > 1 else None
                validator = get_validator(v_name)
                validator(response, v_arg, chunks)

            result.passed = True
            if attempt > 0:
                console.print(f'  [green]✅ Passed:[/green] {name} [dim](after {attempt} retry/retries)[/dim]')
            else:
                console.print(f'  [green]✅ Passed:[/green] {name}')
            break

        except ValidationError as exc:
            result.error = str(exc)
            if attempt < max_retries:
                delay = random.random() * min(retry_base_delay * (2**attempt), _MAX_BACKOFF)
                console.print(
                    f'  [yellow]⟳ Retry {attempt + 1}/{max_retries}:[/yellow] {name}'
                    f' — {exc} [dim](backoff {delay:.1f}s)[/dim]'
                )
                await asyncio.sleep(delay)
            else:
                console.print(f'  [red]❌ Failed:[/red] {name} — {exc}')
        except Exception as exc:
            # Walk the cause chain to surface the real error, not just
            # the GenkitError wrapper ("INTERNAL: Error while running action …").
            cause = exc.__cause__ if exc.__cause__ else exc
            error_msg = f'{exc}' if cause is exc else f'{exc} — caused by: {cause}'
            result.error = error_msg
            if attempt < max_retries:
                delay = random.random() * min(retry_base_delay * (2**attempt), _MAX_BACKOFF)
                console.print(
                    f'  [yellow]⟳ Retry {attempt + 1}/{max_retries}:[/yellow] {name}'
                    f' — {error_msg} [dim](backoff {delay:.1f}s)[/dim]'
                )
                await asyncio.sleep(delay)
            else:
                console.print(f'  [red]❌ Failed:[/red] {name} — {error_msg}')

    result.elapsed_s = time.monotonic() - start
    return result


# Type alias for the optional per-test progress callback.
OnTestDone = Callable[[TestResult], None] | None


async def _run_suite(
    runner: ActionRunner,
    suite: dict[str, Any],
    default_supports: list[str],
    on_test_done: OnTestDone = None,
    test_concurrency: int = 3,
    max_retries: int = 0,
    retry_base_delay: float = 1.0,
) -> SuiteResult:
    """Run all tests for a single model suite.

    Args:
        runner: The action runner to execute tests with.
        suite: A single model suite from the spec file.
        default_supports: Default capabilities to test.
        on_test_done: Optional per-test progress callback.
        test_concurrency: Maximum parallel requests per model.
            Defaults to 3.
        max_retries: Maximum retries per failed test (0 = no retries).
        retry_base_delay: Base delay in seconds for exponential backoff.
    """
    model = suite['model']
    supports = suite.get('supports') or ([] if suite.get('tests') else default_supports)

    console.print(f'\n[bold cyan]Testing model:[/bold cyan] {model}')

    result = SuiteResult(model=model)

    # Collect all test cases (built-in + custom).
    all_cases: list[dict[str, Any]] = []
    for capability in supports:
        test_case = TEST_CASES.get(capability)
        if test_case:
            all_cases.append(test_case)
        else:
            console.print(f'  [yellow]⚠ Unknown capability:[/yellow] {capability}')

    for custom in suite.get('tests', []):
        all_cases.append({
            'name': custom.get('name', 'Custom Test'),
            'input': custom['input'],
            'validators': custom.get('validators', []),
            'stream': custom.get('stream', False),
        })

    retry_kwargs: dict[str, Any] = {
        'max_retries': max_retries,
        'retry_base_delay': retry_base_delay,
    }

    if test_concurrency <= 1:
        # Sequential execution (default, safest for rate-limited APIs).
        for test_case in all_cases:
            tr = await _run_single_test(runner, model, test_case, **retry_kwargs)
            result.tests.append(tr)
            if on_test_done:
                on_test_done(tr)
    else:
        # Parallel execution bounded by a semaphore.
        sem = asyncio.Semaphore(test_concurrency)

        async def _bounded(tc: dict[str, Any]) -> TestResult:
            async with sem:
                return await _run_single_test(runner, model, tc, **retry_kwargs)

        tasks = [asyncio.ensure_future(_bounded(tc)) for tc in all_cases]
        for coro in asyncio.as_completed(tasks):
            tr = await coro
            result.tests.append(tr)
            if on_test_done:
                on_test_done(tr)

    # If any tests failed during parallel execution, re-run them
    # serially with retries to rule out flakes caused by rate limiting.
    if max_retries > 0 and test_concurrency > 1:
        failed_names = {tr.name for tr in result.tests if not tr.passed}
        failed_cases = [tc for tc in all_cases if tc.get('name', 'Unnamed Test') in failed_names]
        if failed_cases:
            console.print(
                f'  [yellow]⚠ {len(failed_cases)} test(s) failed — re-running serially'
                f' with retries to rule out flakes.[/yellow]'
            )
            for test_case in failed_cases:
                tc_name = test_case.get('name', 'Unnamed Test')
                tr = await _run_single_test(runner, model, test_case, **retry_kwargs)
                # Replace the original failed result.
                for i, old_tr in enumerate(result.tests):
                    if old_tr.name == tc_name and not old_tr.passed:
                        result.tests[i] = tr
                        break
                if on_test_done:
                    on_test_done(tr)

    return result


async def run_test_model(
    plugin: str,
    config: ConformConfig,
    *,
    default_supports: list[str] | None = None,
    on_test_done: OnTestDone = None,
    runner_type: str = 'auto',
) -> RunResult:
    """Run model conformance tests for a plugin.

    For the Python runtime, tests run in-process via ``ai.generate()`` so
    the full framework pipeline is exercised.  For other runtimes, the
    runner type determines how tests are executed:

    - ``auto`` (default): Python → in-process, others → native (if a
      native executor exists), else reflection.
    - ``native``: Use the JSONL-over-stdio native executor.  The entry
      point must be a ``conformance_native.{go,ts}`` file.
    - ``reflection``: Use the reflection server HTTP protocol.
    - ``in-process``: Force in-process execution (Python only).

    Args:
        plugin: Plugin name (e.g. ``google-genai``).
        config: Conform configuration.
        default_supports: Default capabilities to test if not specified
            in the suite.
        on_test_done: Optional callback invoked after each individual
            test completes, useful for incremental progress updates.
        runner_type: Which runner to use: ``auto``, ``native``,
            ``reflection``, or ``in-process``.

    Returns:
        Aggregate test results.
    """
    if default_supports is None:
        default_supports = [
            'tool-request',
            'structured-output',
            'multiturn',
            'system-role',
            'input-image-base64',
            'input-image-url',
            'streaming-multiturn',
            'streaming-tool-request',
            'streaming-structured-output',
        ]

    spec = spec_file(plugin, config)
    entry = entry_point(plugin, config)

    if not spec.exists():
        console.print(f'[red]Error:[/red] Spec not found: {spec}')
        return RunResult()
    if not entry.exists():
        console.print(f'[red]Error:[/red] Entry point not found: {entry}')
        return RunResult()

    # Load test suites from spec.
    suites = _load_spec(spec)
    if not suites:
        console.print('[yellow]No test suites found in spec.[/yellow]')
        return RunResult()

    # Compute required action keys for readiness check.
    required_keys = {_model_to_action_key(s['model']) for s in suites if s.get('model')}

    # Choose runner based on runtime and runner_type.
    runtime = config.runtime

    runner: ActionRunner
    resolved_runner = _resolve_runner_type(runner_type, runtime, plugin, config)

    # If native was selected but no executor exists, fall back to reflection.
    if resolved_runner == 'native' and not _find_native_entry(plugin, config):
        console.print(
            f'[yellow]⚠ No native executor for {plugin} ({runtime.name}) — falling back to reflection runner.[/yellow]'
        )
        resolved_runner = 'reflection'

    if resolved_runner == 'in-process':
        console.print('[dim]Using in-process runner (Python).[/dim]')
        runner = InProcessRunner(entry)

    elif resolved_runner == 'native':
        native_entry = _find_native_entry(plugin, config)
        # At this point native_entry is guaranteed to exist (checked above).
        entry_cmd = list(runtime.entry_command) + [str(native_entry), '--plugin', plugin]
        if not runtime.cwd:
            raise RuntimeError(
                f"Runtime '{runtime.name}' has no 'cwd' configured.  "
                f"Set 'cwd' in the [conform.runtimes.{runtime.name}] section."
            )
        cwd = str(runtime.cwd)
        console.print(f'[dim]Using native runner ({runtime.name}).[/dim]')
        runner = NativeRunner(
            entry_cmd,
            cwd,
            action_timeout=config.action_timeout_for(plugin),
        )
        await runner.start()

    elif resolved_runner == 'reflection':
        # Reflection runner (original behavior).
        entry_cmd = list(runtime.entry_command) + [str(entry)]
        if not runtime.cwd:
            raise RuntimeError(
                f"Runtime '{runtime.name}' has no 'cwd' configured.  "
                f"Set 'cwd' in the [conform.runtimes.{runtime.name}] section."
            )
        cwd = str(runtime.cwd)
        console.print(f'[dim]Using reflection runner ({runtime.name}).[/dim]')
        runner = ReflectionRunner(
            entry_cmd,
            cwd,
            required_keys,
            action_timeout=config.action_timeout_for(plugin),
            health_timeout=config.health_timeout,
            startup_timeout=config.startup_timeout,
        )
        await runner.start()

    else:
        raise RuntimeError(f'Unknown runner type: {resolved_runner!r}')

    run_result = RunResult()

    try:
        console.print(f'[bold green]All {len(required_keys)} model actions ready. Starting tests.[/bold green]')

        # Run test suites sequentially (each suite is one model).
        for suite in suites:
            if not suite.get('model'):
                console.print('[red]Error:[/red] Model name required in test suite.')
                continue
            sr = await _run_suite(
                runner,
                suite,
                default_supports,
                on_test_done,
                test_concurrency=config.test_concurrency_for(plugin),
                max_retries=config.max_retries,
                retry_base_delay=config.retry_base_delay,
            )
            run_result.suites.append(sr)

        # Summary.
        console.print()
        console.print('[dim]' + '—' * 50 + '[/dim]')
        passed = run_result.total_passed
        failed = run_result.total_failed
        if failed > 0:
            console.print(f'[bold]Tests Completed:[/bold] [green]{passed} Passed[/green], [red]{failed} Failed[/red]')
        else:
            console.print(f'[bold green]Tests Completed: {passed} Passed, 0 Failed[/bold green]')

    finally:
        await runner.close()

    return run_result


# Fallback native executor entry filenames per runtime (used when
# ``native_entry_filename`` is not set in the runtime config).
_NATIVE_ENTRY_FILENAMES: dict[str, str] = {
    'go': 'conformance_native.go',
    'js': 'conformance_native.ts',
    'python': 'conformance_native.py',
}

# Directory containing the native executor files, relative to this module.
_EXECUTORS_DIR = Path(__file__).resolve().parent / 'executors'


def _find_native_entry(
    plugin: str,
    config: ConformConfig,
) -> Path | None:
    """Find the shared native executor in the conform tool package.

    Looks for ``conformance_native.{go,ts}`` in the ``executors/``
    directory within the conform package (``py/tools/conform/
    src/conform/executors/``).  The executor is generic — it receives
    ``--plugin <name>`` as a CLI argument and uses a built-in
    plugin registry map to initialize the correct plugin.

    Uses ``runtime.native_entry_filename`` from the config (set via
    ``native-entry-filename`` in ``conform.toml``), falling back to
    the hardcoded ``_NATIVE_ENTRY_FILENAMES`` map.
    """
    runtime = config.runtime
    filename = runtime.native_entry_filename or _NATIVE_ENTRY_FILENAMES.get(runtime.name)
    if not filename:
        return None
    candidate = _EXECUTORS_DIR / filename
    return candidate if candidate.exists() else None


def _resolve_runner_type(
    runner_type: str,
    runtime: RuntimeConfig,
    plugin: str,
    config: ConformConfig,
) -> str:
    """Resolve the effective runner type.

    - ``in-process``: Always returns ``in-process``.
    - ``native``: Always returns ``native`` (caller handles fallback
      if no executor exists).
    - ``reflection``: Always returns ``reflection``.
    - ``auto``: Uses ``runtime.default_runner`` from config.
      Python defaults to ``in-process``, Go/JS default to ``native``.
      Falls back if the preferred runner isn't available.
    """
    if runner_type in ('in-process', 'native', 'reflection'):
        return runner_type

    # Auto: use the runtime's configured default runner.
    preferred = runtime.default_runner

    # If preferred is native, verify the executor exists.
    if preferred == 'native' and _find_native_entry(plugin, config):
        return 'native'

    # If preferred is in-process (Python), use it directly.
    if preferred == 'in-process':
        return 'in-process'

    # Fallback: try native if executor exists, else reflection.
    if _find_native_entry(plugin, config):
        return 'native'
    if runtime.name == 'python':
        return 'in-process'
    return 'reflection'
