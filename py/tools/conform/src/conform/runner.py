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

"""Async parallel test runner for ``conform check-model``.

Uses an ``asyncio.Queue`` with a fixed-size worker pool for concurrency
control.  Each worker pulls the next plugin from the queue, runs
``genkit dev:test-model``, and records the result.  The live Rich table
updates after every completion.

Worker pool model::

    ┌───────────┐     ┌──────────┐     ┌──────────────────────────────┐
    │  Queue    │ ──► │ Worker 1 │ ──► │ genkit dev:test-model ...    │
    │ (plugins) │ ──► │ Worker 2 │ ──► │ genkit dev:test-model ...    │
    │           │ ──► │ Worker N │ ──► │ genkit dev:test-model ...    │
    └───────────┘     └──────────┘     └──────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import re
import time

from rich.live import Live
from rich.panel import Panel

from conform.config import ConformConfig
from conform.display import (
    build_progress_table,
    build_summary_table,
    console,
    print_summary_footer,
    rust_error,
    rust_warning,
)
from conform.plugins import check_env, entry_point, spec_file
from conform.types import PluginResult, Status

# Maximum lines of output to show per failure in the error log.
_ERROR_LOG_LINES = 15

# Pattern to extract pass/fail counts from genkit's test summary line.
# Example: "Tests Completed: 25 Passed, 1 Failed"
_TESTS_COMPLETED_RE = re.compile(r'Tests Completed:\s*(\d+)\s*Passed,\s*(\d+)\s*Failed')


def _preflight(plugin: str, result: PluginResult, config: ConformConfig) -> bool:
    """Run pre-flight checks for a plugin.

    Returns ``True`` if the plugin is ready to run, ``False`` if it
    should be skipped (result is updated in-place).

    Note: ``Path.exists()`` is a stat syscall (sub-microsecond on local
    filesystems) and is safe to call from the async event loop without
    offloading to a thread.
    """
    spec = spec_file(plugin, config)
    if not spec.exists():
        result.status = Status.ERROR
        result.error_message = f'spec not found: {spec}'
        return False

    entry = entry_point(plugin, config)
    if not entry.exists():
        result.status = Status.ERROR
        result.error_message = f'entry point not found: {entry}'
        return False

    missing = check_env(plugin, config)
    if missing:
        result.status = Status.SKIPPED
        result.missing_env_vars = missing
        result.error_message = f'missing env: {", ".join(missing)}'
        return False

    return True


async def _run_subprocess(
    plugin: str,
    result: PluginResult,
    config: ConformConfig,
) -> PluginResult:
    """Spawn ``genkit dev:test-model`` for a single plugin."""
    spec = spec_file(plugin, config)
    entry = entry_point(plugin, config)
    runtime = config.runtime
    entry_cmd = list(runtime.entry_command) + [str(entry)]

    result.status = Status.RUNNING
    start = time.monotonic()

    try:
        proc = await asyncio.create_subprocess_exec(
            'genkit',
            'dev:test-model',
            '--from-file',
            str(spec),
            '--',
            *entry_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(runtime.cwd) if runtime.cwd else None,
        )

        stdout_bytes, stderr_bytes = await proc.communicate()
        result.elapsed_s = time.monotonic() - start
        result.stdout = stdout_bytes.decode(errors='replace')
        result.stderr = stderr_bytes.decode(errors='replace')
        result.return_code = proc.returncode if proc.returncode is not None else -1

        # Parse test counts from genkit's summary line.
        combined = result.stdout + result.stderr
        m = _TESTS_COMPLETED_RE.search(combined)
        if m:
            result.tests_passed = int(m.group(1))
            result.tests_failed = int(m.group(2))

        if result.return_code == 0:
            result.status = Status.PASSED
        else:
            result.status = Status.FAILED
            lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
            result.error_message = lines[-1] if lines else f'exit code {result.return_code}'

    except FileNotFoundError:
        result.elapsed_s = time.monotonic() - start
        result.status = Status.ERROR
        result.error_message = 'genkit CLI not found in PATH'
    except OSError as exc:
        result.elapsed_s = time.monotonic() - start
        result.status = Status.ERROR
        result.error_message = str(exc)

    return result


async def _worker(
    queue: asyncio.Queue[str],
    results: dict[str, PluginResult],
    config: ConformConfig,
    on_complete: asyncio.Event,
) -> None:
    """Worker coroutine — pulls plugins from the queue until empty.

    Sets *on_complete* after each plugin finishes so the live display
    can refresh.
    """
    while True:
        try:
            plugin = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        result = results[plugin]

        if _preflight(plugin, result, config):
            await _run_subprocess(plugin, result, config)

        on_complete.set()
        queue.task_done()


def _print_error_log(results: dict[str, PluginResult]) -> None:
    """Print a consolidated error log for all failures.

    Shows the last ``_ERROR_LOG_LINES`` lines of output per failure.
    """
    failures = {k: v for k, v in results.items() if v.status in (Status.FAILED, Status.ERROR)}
    if not failures:
        return

    console.print()
    console.rule('[bold red]Error Log[/bold red]', style='red')
    console.print()

    for plugin, result in failures.items():
        combined = (result.stdout + result.stderr).rstrip()
        if not combined:
            combined = result.error_message or '(no output)'

        lines = combined.splitlines()
        if len(lines) > _ERROR_LOG_LINES:
            content = (
                f'[dim]… ({len(lines) - _ERROR_LOG_LINES} lines omitted,'
                f' use -v for full output)[/dim]\n' + '\n'.join(lines[-_ERROR_LOG_LINES:])
            )
        else:
            content = '\n'.join(lines)

        console.print(
            Panel(
                content,
                title=f'[bold red]{plugin}[/bold red] — exit code {result.return_code}',
                border_style='red',
                expand=False,
                padding=(0, 1),
            )
        )
        console.print()


async def _worker_verbose(
    queue: asyncio.Queue[str],
    results: dict[str, PluginResult],
    config: ConformConfig,
) -> None:
    """Worker for verbose mode — prints progress as plain log lines."""
    while True:
        try:
            plugin = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        result = results[plugin]

        if not _preflight(plugin, result, config):
            _log_result(plugin, result)
            queue.task_done()
            continue

        console.print(f'[dim]  ▶ {plugin}[/dim] — starting …')
        await _run_subprocess(plugin, result, config)
        _log_result(plugin, result)
        queue.task_done()


def _log_result(plugin: str, result: PluginResult) -> None:
    """Print a single-line result in verbose mode."""
    elapsed = f'{result.elapsed_s:.1f}s' if result.elapsed_s else ''

    counts = ''
    if result.tests_passed or result.tests_failed:
        counts = f' ({result.tests_passed}/{result.tests_passed + result.tests_failed})'

    if result.status == Status.PASSED:
        console.print(f'  [green]✓[/green] {plugin} — [green]passed[/green]{counts} {elapsed}')
    elif result.status == Status.SKIPPED:
        console.print(
            f'  [yellow]⊘[/yellow] {plugin} — [yellow]skipped[/yellow] (missing: {", ".join(result.missing_env_vars)})',
        )
    elif result.status == Status.FAILED:
        console.print(f'  [red]✗[/red] {plugin} — [red]failed[/red]{counts} {elapsed}')
        # Show full output inline.
        combined = (result.stdout + result.stderr).rstrip()
        if combined:
            for line in combined.splitlines():
                console.print(f'    [dim]│[/dim] {line}')
    elif result.status == Status.ERROR:
        console.print(
            f'  [red]![/red] {plugin} — [red]error[/red]: {result.error_message}',
        )


async def run_all(
    plugins: list[str],
    config: ConformConfig,
    *,
    verbose: bool = False,
) -> dict[str, PluginResult]:
    """Run conformance tests for all *plugins* with a worker pool.

    Spawns ``config.concurrency`` workers that pull plugins from an
    ``asyncio.Queue``.  A live Rich table updates after every
    completion.

    Args:
        plugins: List of plugin names to test.
        config: Resolved tool configuration.
        verbose: If ``True``, use plain-text logging instead of
            the Rich live table.

    Returns:
        Ordered dict mapping plugin names to their results.
    """
    queue: asyncio.Queue[str] = asyncio.Queue()
    for plugin in plugins:
        queue.put_nowait(plugin)

    results: dict[str, PluginResult] = {p: PluginResult(plugin=p) for p in plugins}

    num_workers = min(config.concurrency, len(plugins))

    if verbose:
        # Plain-text streaming mode.
        workers = [
            asyncio.create_task(
                _worker_verbose(queue, results, config),
                name=f'worker-{i}',
            )
            for i in range(num_workers)
        ]
        await asyncio.gather(*workers)

        # Summary counts.
        passed = sum(1 for r in results.values() if r.status == Status.PASSED)
        failed = sum(1 for r in results.values() if r.status in (Status.FAILED, Status.ERROR))
        skipped = sum(1 for r in results.values() if r.status == Status.SKIPPED)
        console.print()
        console.print(
            f'[bold]Done:[/bold] {passed} passed, {failed} failed, {skipped} skipped',
        )
    else:
        on_complete = asyncio.Event()

        workers = [
            asyncio.create_task(
                _worker(queue, results, config, on_complete),
                name=f'worker-{i}',
            )
            for i in range(num_workers)
        ]

        # Live-update the table as results come in.
        with Live(
            build_progress_table(results),
            console=console,
            refresh_per_second=4,
            transient=True,
        ) as live:
            while not queue.empty() or any(not w.done() for w in workers):
                on_complete.clear()
                # Wait for either a completion event or a short timeout
                # (so the live display refreshes even during long runs).
                try:
                    await asyncio.wait_for(on_complete.wait(), timeout=0.5)
                except TimeoutError:
                    pass
                live.update(build_progress_table(results))

            # Wait for all workers to finish.
            await asyncio.gather(*workers)
            live.update(build_progress_table(results))

        # Print the final summary table.
        console.print()
        console.print(build_summary_table(results))
        print_summary_footer(results)

        # Print the error log (shows full output in rich panels).
        _print_error_log(results)

    # Rust-style error/warning summaries (both modes).
    for plugin, result in results.items():
        if result.status == Status.SKIPPED:
            rust_warning(
                'W0001',
                f'skipping `{plugin}`: missing environment variables',
                note=', '.join(result.missing_env_vars),
            )
        elif result.status == Status.ERROR:
            rust_error(
                'E0002',
                f'internal error running `{plugin}`',
                note=result.error_message,
                hint='check that `genkit` CLI is installed and on PATH',
            )
        elif result.status == Status.FAILED:
            rust_error(
                'E0003',
                f'conformance tests failed for `{plugin}`',
                note=result.error_message,
                hint=f'run with -v or test manually: conform check-model {plugin} -v',
            )

    return results
