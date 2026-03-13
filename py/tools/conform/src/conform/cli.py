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

"""CLI entry point for the ``conform`` tool.

Provides subcommands::

    conform [GLOBAL FLAGS] check-model [PLUGIN...] [-j N] [-v] [--runner TYPE]
        Run model conformance tests (all runtimes).

    conform [GLOBAL FLAGS] check-plugin
        Verify that every model plugin has conformance files.

    conform [GLOBAL FLAGS] list
        List available plugins, their runtimes, and env-var readiness.

Global flags (apply to all subcommands)::

    --config FILE               Path to conform.toml config file
    --runtime NAME[,NAME,...]   Runtimes to use (default: all configured)
    --specs-dir DIR             Override the specs directory
    --plugins-dir DIR           Override the plugins directory

When invoked with no subcommand, displays the help screen followed by
the plugin env-var readiness table.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from conform.checker import check_model_conformance
from conform.config import ConformConfig, load_all_runtime_names, load_config
from conform.display import (
    build_progress_table,
    build_summary_table,
    console,
    print_summary_footer,
    rust_error,
    rust_warning,
    stdout_console,
)
from conform.plugins import check_env, discover_plugins, entry_point, spec_file
from conform.runner import run_all
from conform.types import FailedTest, PluginResult, Status
from conform.util_test_model import TestResult, count_spec_tests, run_test_model


def _resolve_runtime_names(
    runtime_arg: str | None,
    config_path: Path | None = None,
) -> list[str]:
    """Resolve which runtimes to use.

    Args:
        runtime_arg: Comma-separated runtime name(s), or ``None`` for all
            configured.
        config_path: Explicit path to the config file.

    Returns:
        List of runtime names to operate on.
    """
    if runtime_arg:
        return [name.strip() for name in runtime_arg.split(',') if name.strip()]
    names = load_all_runtime_names(config_path=config_path)
    return names if names else ['python']


def _plugin_runtimes(
    plugin: str,
    config_path: Path | None = None,
) -> list[str]:
    """Return the runtime names that have an entry point for *plugin*."""
    available: list[str] = []
    for rt_name in load_all_runtime_names(config_path=config_path):
        rt_config = load_config(runtime_name=rt_name, config_path=config_path)
        ep = entry_point(plugin, rt_config)
        if ep.exists():
            available.append(rt_name)
    return available


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add flags shared by every subcommand to *p*.

    These are added directly to each subparser (not via a parent parser)
    to avoid the argparse bug where parent-parser defaults silently
    overwrite values parsed by the main parser.
    """
    p.add_argument(
        '--config',
        '-c',
        default=None,
        metavar='FILE',
        help='Path to conform.toml config file (default: auto-detected).',
    )
    p.add_argument(
        '--runtime',
        default=None,
        metavar='NAME[,NAME,...]',
        help=(
            'Runtime(s) to use, comma-separated (e.g. python, python,js). '
            'If not specified, runs all configured runtimes.'
        ),
    )
    p.add_argument(
        '--specs-dir',
        default=None,
        metavar='DIR',
        help='Override the specs directory (default: from runtime config).',
    )
    p.add_argument(
        '--plugins-dir',
        default=None,
        metavar='DIR',
        help='Override the plugins directory (default: from runtime config).',
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with rich-argparse formatting."""
    try:
        from rich_argparse import RichHelpFormatter

        RichHelpFormatter.styles['argparse.groups'] = 'yellow bold'
        RichHelpFormatter.styles['argparse.args'] = 'cyan'
        RichHelpFormatter.styles['argparse.metavar'] = 'dark_cyan'
        formatter_class: type[argparse.HelpFormatter] = RichHelpFormatter
    except ImportError:
        formatter_class = argparse.HelpFormatter

    parser = argparse.ArgumentParser(
        prog='conform',
        description=(
            'Model conformance test runner for Genkit plugins.\n\n'
            'Runs model conformance tests against multiple plugins and runtimes\n'
            'concurrently, collecting results and displaying a unified table.'
        ),
        formatter_class=formatter_class,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available subcommands')

    cm_parser = subparsers.add_parser(
        'check-model',
        help='Run model conformance tests.',
        description=(
            'Run model conformance tests against one or more plugins using the\n'
            'configured runner.  By default runs all plugins across all configured\n'
            'runtimes.  Results are displayed in a unified table.\n\n'
            'Use --runner cli to fall back to genkit dev:test-model subprocess.'
        ),
        formatter_class=formatter_class,
    )
    _add_common_args(cm_parser)
    cm_parser.add_argument(
        'plugins',
        nargs='*',
        default=[],
        metavar='PLUGIN',
        help='Plugin name(s) to test (default: all available).',
    )
    cm_parser.add_argument(
        '--concurrency',
        '-j',
        type=int,
        default=-1,
        metavar='N',
        help='Maximum concurrent plugins (overrides pyproject.toml).',
    )
    cm_parser.add_argument(
        '--test-concurrency',
        '-t',
        type=int,
        default=-1,
        metavar='N',
        help='Maximum concurrent tests per model spec (overrides config, default: 3).',
    )
    cm_parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Print full stdout/stderr for failed plugins.',
    )
    cm_parser.add_argument(
        '--max-retries',
        type=int,
        default=-1,
        metavar='N',
        help='Max retries per failed test with exponential backoff (overrides config, default: 2).',
    )
    cm_parser.add_argument(
        '--retry-base-delay',
        type=float,
        default=-1.0,
        metavar='SECS',
        help='Base delay in seconds for retry backoff (overrides config, default: 1.0).',
    )
    cm_parser.add_argument(
        '--runner',
        choices=['auto', 'native', 'reflection', 'in-process', 'cli'],
        default='auto',
        metavar='TYPE',
        help=(
            'Runner type: auto (default), native (JSONL-over-stdio), '
            'reflection (HTTP server), in-process (Python only), '
            'or cli (genkit dev:test-model subprocess).'
        ),
    )

    cp_parser = subparsers.add_parser(
        'check-plugin',
        help='Check that every model plugin has conformance files.',
        description=(
            'Verify that every model plugin (identified by model_info.py or\n'
            'listed in [tool.conform.additional-model-plugins]) has both a\n'
            'model-conformance.yaml spec and a conformance_entry.py file.'
        ),
        formatter_class=formatter_class,
    )
    _add_common_args(cp_parser)

    list_parser = subparsers.add_parser(
        'list',
        help='List available plugins and their env-var readiness.',
        description=(
            'Display a table of all plugins that have conformance specs,\n'
            'showing which runtimes and environment variables are available.'
        ),
        formatter_class=formatter_class,
    )
    _add_common_args(list_parser)

    return parser


async def _run_native_check(
    plugins: list[str],
    runtime_names: list[str],
    concurrency: int,
    test_concurrency: int,
    max_retries: int,
    retry_base_delay: float,
    specs_dir_override: Path | None,
    plugins_dir_override: Path | None,
    verbose: bool,
    config_path: Path | None = None,
    runner_type: str = 'auto',
) -> dict[str, PluginResult]:
    """Run native conformance tests for all (plugin, runtime) pairs.

    Creates a unified results dict keyed by ``plugin@runtime`` (or just
    ``plugin`` when only one runtime is active).  Uses an asyncio
    semaphore to bound concurrency.
    """
    single_rt = len(runtime_names) == 1
    results: dict[str, PluginResult] = {}
    tasks: list[tuple[str, str, str, ConformConfig]] = []

    for rt_name in runtime_names:
        rt_config = load_config(
            runtime_name=rt_name,
            concurrency_override=concurrency,
            test_concurrency_override=test_concurrency,
            max_retries_override=max_retries,
            retry_base_delay_override=retry_base_delay,
            config_path=config_path,
        )
        overrides: dict[str, Any] = {}
        if specs_dir_override:
            overrides['specs_dir'] = specs_dir_override
        if plugins_dir_override:
            overrides['plugins_dir'] = plugins_dir_override
        if overrides:
            new_rt = replace(rt_config.runtime, **overrides)
            rt_config = replace(rt_config, runtime=new_rt)

        available = discover_plugins(rt_config.runtime)

        if plugins:
            unknown = [p for p in plugins if p not in available]
            if unknown:
                for p in unknown:
                    rust_error(
                        'E0001',
                        f'unknown plugin `{p}` for runtime `{rt_name}`',
                        note=f'available: {", ".join(available)}',
                        hint='run `conform list` to see available plugins',
                    )
                return {}
            selected = plugins
        else:
            selected = available

        for plugin in selected:
            ep = entry_point(plugin, rt_config)
            if not ep.exists():
                continue
            key = plugin if single_rt else f'{plugin}@{rt_name}'
            result = PluginResult(plugin=plugin, runtime=rt_name)

            # Pre-flight: check env vars.
            missing = check_env(plugin, rt_config)
            if missing:
                result.status = Status.SKIPPED
                result.missing_env_vars = missing
                result.error_message = f'missing env: {", ".join(missing)}'
                results[key] = result
                continue

            # Pre-calculate total test count from the spec so the
            # progress table can show passed/total from the start.
            spec = spec_file(plugin, rt_config)
            counts = count_spec_tests(spec)
            result.tests_total = counts.total
            result.tests_supports = counts.supports
            result.tests_custom = counts.custom

            results[key] = result
            tasks.append((key, plugin, rt_name, rt_config))

    if not tasks and not results:
        console.print('[yellow]No plugins found to test.[/yellow]')
        return results

    effective_concurrency = tasks[0][3].concurrency if tasks else 8
    sem = asyncio.Semaphore(max(effective_concurrency, 1))
    on_complete = asyncio.Event()

    # Mutable reference to the Live object so the inner function can
    # print log lines above the live table via live.console.print().
    live_ref: list[Live | None] = [None]

    async def _run_one(key: str, plugin: str, rt_name: str, config: ConformConfig) -> None:
        async with sem:
            result = results[key]
            result.status = Status.RUNNING
            on_complete.set()  # Trigger refresh to show RUNNING status.
            start = time.monotonic()

            def _on_test_done(tr: TestResult) -> None:
                """Update counts after each individual test."""
                if tr.passed:
                    result.tests_passed += 1
                else:
                    result.tests_failed += 1
                result.elapsed_s = time.monotonic() - start
                on_complete.set()

            try:
                run_result = await run_test_model(
                    plugin,
                    config,
                    on_test_done=_on_test_done,
                    runner_type=runner_type,
                )
                result.elapsed_s = time.monotonic() - start
                result.tests_passed = run_result.total_passed
                result.tests_failed = run_result.total_failed
                if run_result.total_failed > 0:
                    result.status = Status.FAILED
                    result.error_message = f'{run_result.total_failed} test(s) failed'
                    # Collect per-test failure details for the error summary.
                    for suite in run_result.suites:
                        for tr in suite.tests:
                            if not tr.passed:
                                result.failed_tests.append(
                                    FailedTest(
                                        test_name=tr.name,
                                        model=suite.model,
                                        error=tr.error,
                                    )
                                )
                else:
                    result.status = Status.PASSED
            except Exception as exc:
                result.elapsed_s = time.monotonic() - start
                result.status = Status.ERROR
                result.error_message = str(exc)

            # Print a log line above the live table (or to console in
            # verbose mode).  Using live.console.print() ensures the
            # line renders above the pinned table.
            out = live_ref[0].console if live_ref[0] else console
            _log_native_result(plugin, rt_name, result, out)
            on_complete.set()  # Trigger refresh to show final status.

    coros = [_run_one(key, plugin, rt_name, cfg) for key, plugin, rt_name, cfg in tasks]

    if verbose:
        # Plain-text streaming mode (no live table).
        await asyncio.gather(*coros)

        # Summary counts.
        passed = sum(1 for r in results.values() if r.status == Status.PASSED)
        failed = sum(1 for r in results.values() if r.status in (Status.FAILED, Status.ERROR))
        skipped = sum(1 for r in results.values() if r.status == Status.SKIPPED)
        console.print()
        console.print(
            f'[bold]Done:[/bold] {passed} passed, {failed} failed, {skipped} skipped',
        )
    else:
        # Live Rich table pinned at the bottom, log lines scroll above.
        gather_task = asyncio.ensure_future(asyncio.gather(*coros))

        with Live(
            build_progress_table(results),
            console=console,
            refresh_per_second=4,
            transient=True,
        ) as live:
            live_ref[0] = live
            while not gather_task.done():
                on_complete.clear()
                try:
                    await asyncio.wait_for(on_complete.wait(), timeout=0.5)
                except TimeoutError:
                    pass
                live.update(build_progress_table(results))

            # Final refresh after all tasks complete.
            await gather_task
            live.update(build_progress_table(results))
            live_ref[0] = None

    return results


def _log_native_result(
    plugin: str,
    rt_name: str,
    result: PluginResult,
    out: Console,
) -> None:
    """Print a single-line completion log.

    When called from within a ``Live`` context, *out* should be
    ``live.console`` so the line renders above the pinned table.
    """
    label = f'{plugin} ({rt_name})'
    elapsed = f'{result.elapsed_s:.1f}s' if result.elapsed_s else ''

    counts = ''
    if result.tests_passed or result.tests_failed:
        counts = f' ({result.tests_passed}/{result.tests_passed + result.tests_failed})'

    if result.status == Status.PASSED:
        out.print(f'  [green]✓[/green] {label} — [green]passed[/green]{counts} {elapsed}')
    elif result.status == Status.SKIPPED:
        out.print(
            f'  [yellow]⊘[/yellow] {label} — [yellow]skipped[/yellow] (missing: {", ".join(result.missing_env_vars)})',
        )
    elif result.status == Status.FAILED:
        out.print(f'  [red]✗[/red] {label} — [red]failed[/red]{counts} {elapsed}')
    elif result.status == Status.ERROR:
        out.print(
            f'  [red]![/red] {label} — [red]error[/red]: {result.error_message}',
        )


def _print_error_summary(results: dict[str, PluginResult]) -> None:
    """Print a consolidated error log at the end of the run.

    Groups all individual test failures by plugin so the user doesn't
    have to scroll through the full log to find them.  Only prints
    when there are actual failures.
    """
    # Collect all failures across all plugins.
    has_failures = any(r.failed_tests for r in results.values())
    if not has_failures:
        return

    table = Table(
        title='🔍 Failed Tests',
        box=box.ROUNDED,
        title_style='bold red',
        border_style='dim',
        header_style='bold',
        expand=False,
        pad_edge=True,
        show_lines=True,
    )
    table.add_column('Plugin', style='bold', min_width=18)
    table.add_column('Runtime', style='cyan', min_width=8)
    table.add_column('Model', style='yellow', min_width=20)
    table.add_column('Test', min_width=24)
    table.add_column('Error', ratio=1, style='red')

    for _key, result in results.items():
        for ft in result.failed_tests:
            table.add_row(
                result.plugin,
                result.runtime,
                ft.model,
                ft.test_name,
                ft.error or '(no details)',
            )

    console.print()
    console.print(table)
    console.print()


def _cmd_check_model(
    args: argparse.Namespace,
    runtime_names: list[str],
    config_path: Path | None = None,
) -> int:
    """Handle ``conform check-model``.

    Uses the configured runner (default: auto).  Falls back to the
    genkit CLI subprocess when ``--runner cli`` is specified.
    """
    specs_dir_raw: str | None = getattr(args, 'specs_dir', None)
    plugins_dir_raw: str | None = getattr(args, 'plugins_dir', None)
    resolved_specs_dir: Path | None = Path(specs_dir_raw).resolve() if specs_dir_raw else None
    resolved_plugins_dir: Path | None = Path(plugins_dir_raw).resolve() if plugins_dir_raw else None
    concurrency_override: int = getattr(args, 'concurrency', -1)
    test_concurrency_override: int = getattr(args, 'test_concurrency', -1)
    max_retries_override: int = getattr(args, 'max_retries', -1)
    retry_base_delay_override: float = getattr(args, 'retry_base_delay', -1.0)
    verbose = getattr(args, 'verbose', False)
    runner_type = getattr(args, 'runner', 'auto')

    if runner_type == 'cli':
        return _cmd_check_model_cli(args, runtime_names, config_path=config_path)

    # Native runner path.
    results = asyncio.run(
        _run_native_check(
            plugins=args.plugins,
            runtime_names=runtime_names,
            concurrency=concurrency_override,
            test_concurrency=test_concurrency_override,
            max_retries=max_retries_override,
            retry_base_delay=retry_base_delay_override,
            specs_dir_override=resolved_specs_dir,
            plugins_dir_override=resolved_plugins_dir,
            verbose=verbose,
            config_path=config_path,
            runner_type=runner_type,
        )
    )

    if not results:
        return 1

    # Display unified results.
    console.print()
    console.print(build_summary_table(results))
    print_summary_footer(results)

    # Rust-style messages.
    for _key, result in results.items():
        if result.status == Status.SKIPPED:
            rust_warning(
                'W0001',
                f'skipping `{result.plugin}`: missing environment variables',
                note=', '.join(result.missing_env_vars),
            )
        elif result.status == Status.ERROR:
            rust_error(
                'E0002',
                f'internal error running `{result.plugin}`',
                note=result.error_message,
            )
        elif result.status == Status.FAILED:
            rust_error(
                'E0003',
                f'conformance tests failed for `{result.plugin}`',
                note=result.error_message,
                hint=f'run: conform check-model {result.plugin} -v',
            )

    # Consolidated error log — all individual test failures in one place.
    _print_error_summary(results)

    failed = sum(1 for r in results.values() if r.status in (Status.FAILED, Status.ERROR))
    return 1 if failed > 0 else 0


def _cmd_check_model_cli(
    args: argparse.Namespace,
    runtime_names: list[str],
    config_path: Path | None = None,
) -> int:
    """Handle ``conform check-model --runner cli`` (genkit CLI path)."""
    specs_dir_override: str | None = getattr(args, 'specs_dir', None)
    concurrency_override: int = getattr(args, 'concurrency', -1)
    total_failures = 0

    for rt_name in runtime_names:
        rt_config = load_config(
            runtime_name=rt_name,
            concurrency_override=concurrency_override,
            config_path=config_path,
        )
        if specs_dir_override:
            new_rt = replace(
                rt_config.runtime,
                specs_dir=Path(specs_dir_override).resolve(),
            )
            rt_config = replace(rt_config, runtime=new_rt)

        available = discover_plugins(rt_config.runtime)

        if args.plugins:
            unknown = [p for p in args.plugins if p not in available]
            if unknown:
                for p in unknown:
                    rust_error(
                        'E0001',
                        f'unknown plugin `{p}` for runtime `{rt_name}`',
                        note=f'available: {", ".join(available)}',
                        hint='run `conform list` to see available plugins',
                    )
                return 1
            plugins = args.plugins
        else:
            plugins = available

        if not plugins:
            continue

        console.print()
        console.print(
            f'[bold cyan]Running conformance tests (genkit CLI)[/bold cyan] '
            f'\u2014 [dim]{len(plugins)} plugin(s), '
            f'concurrency={rt_config.concurrency}, '
            f'runtime={rt_name}[/dim]'
        )
        console.print()

        results = asyncio.run(run_all(plugins, rt_config, verbose=args.verbose))

        failed = sum(1 for r in results.values() if r.status == Status.FAILED)
        errors = sum(1 for r in results.values() if r.status == Status.ERROR)
        if failed > 0 or errors > 0:
            total_failures += 1

    return 1 if total_failures > 0 else 0


def _cmd_check_plugin(config: ConformConfig) -> int:
    """Handle ``conform check-plugin``."""
    error_count = check_model_conformance(config)
    return 1 if error_count > 0 else 0


def _cmd_list(
    runtime_names: list[str],
    config_path: Path | None = None,
) -> int:
    """Handle ``conform list``.

    Shows all plugins across all active runtimes with a column indicating
    which runtimes have entry points.
    """
    # Collect the union of all plugins across runtimes.
    all_plugins: dict[str, ConformConfig] = {}
    for rt_name in runtime_names:
        rt_config = load_config(runtime_name=rt_name, config_path=config_path)
        for plugin in discover_plugins(rt_config.runtime):
            if plugin not in all_plugins:
                all_plugins[plugin] = rt_config

    if not all_plugins:
        stdout_console.print('[yellow]No plugins found.[/yellow]')
        return 0

    all_rt_names = load_all_runtime_names(config_path=config_path)

    table = Table(
        title='Available Plugins',
        box=box.ROUNDED,
        title_style='bold cyan',
        border_style='dim',
        header_style='bold',
        show_lines=False,
    )
    table.add_column('', width=2, justify='center')
    table.add_column('Plugin', style='bold', min_width=20)
    table.add_column('Runtimes', min_width=12)
    table.add_column('Environment Variables', ratio=1)

    for plugin in sorted(all_plugins):
        config = all_plugins[plugin]
        required = config.env.get(plugin, [])
        missing = check_env(plugin, config)

        # Readiness indicator.
        if not required:
            ready = Text('\u25cf', style='bold green')
            env_text = Text('(no credentials needed)', style='dim')
        elif not missing:
            ready = Text('\u25cf', style='bold green')
            parts: list[Text | str] = []
            for v in required:
                if parts:
                    parts.append('  ')
                parts.append(Text(v, style='blue'))
            env_text = Text.assemble(*parts)
        else:
            ready = Text('\u25cb', style='bold red')
            parts = []
            for v in required:
                if parts:
                    parts.append('  ')
                style = 'blue' if v not in missing else 'red'
                parts.append(Text(v, style=style))
            env_text = Text.assemble(*parts)

        # Runtime availability.
        runtimes = _plugin_runtimes(plugin)
        rt_parts: list[Text | str] = []
        for rt in all_rt_names:
            if rt_parts:
                rt_parts.append(' ')
            if rt in runtimes:
                rt_parts.append(Text(rt, style='green'))
            else:
                rt_parts.append(Text(rt, style='dim'))
        rt_text = Text.assemble(*rt_parts)

        table.add_row(ready, plugin, rt_text, env_text)

    stdout_console.print()
    stdout_console.print(table)
    stdout_console.print()
    stdout_console.print(
        '[bold]Legend:[/bold]  '
        '[green]\u25cf[/green] Ready  '
        '[red]\u25cb[/red] Missing env vars  '
        '[blue]VAR[/blue] Set  '
        '[red]VAR[/red] Not set  '
        '[green]runtime[/green] Available  '
        '[dim]runtime[/dim] Not available'
    )
    stdout_console.print()
    return 0


def _install_log_redaction() -> None:
    """Install a structlog processor that truncates data URIs in log output.

    Data URIs (e.g. base64-encoded images) can be 20,000+ characters and
    make debug logs unreadable during conformance tests with multimodal
    inputs.  This inserts the redaction processor before the final
    renderer in the structlog chain.
    """
    try:
        import structlog

        from conform.log_redact import redact_data_uris_processor

        cfg = structlog.get_config()
        processors = list(cfg.get('processors', []))
        # Insert before the last processor (usually the renderer).
        insert_pos = max(0, len(processors) - 1)
        processors.insert(insert_pos, redact_data_uris_processor)
        structlog.configure(processors=processors)
    except Exception:  # noqa: S110 - non-critical; logging still works, just verbose
        pass


def _extract_common_args(args: argparse.Namespace) -> tuple[list[str], Path | None]:
    """Extract the common flags from parsed *args*.

    Returns:
        ``(runtime_names, config_path)`` derived from ``--runtime``
        and ``--config``.
    """
    config_raw: str | None = getattr(args, 'config', None)
    config_path: Path | None = Path(config_raw).resolve() if config_raw else None
    runtime_arg: str | None = getattr(args, 'runtime', None)
    runtime_names = _resolve_runtime_names(runtime_arg, config_path=config_path)
    return runtime_names, config_path


def main() -> None:
    """Entry point for the ``conform`` CLI."""
    _install_log_redaction()

    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    runtime_names, config_path = _extract_common_args(args)

    if args.command == 'check-model':
        sys.exit(_cmd_check_model(args, runtime_names, config_path=config_path))
    elif args.command == 'check-plugin':
        exit_code = 0
        specs_dir: str | None = getattr(args, 'specs_dir', None)
        plugins_dir: str | None = getattr(args, 'plugins_dir', None)
        for rt_name in runtime_names:
            config = load_config(runtime_name=rt_name, config_path=config_path)
            overrides: dict[str, Any] = {}
            if specs_dir:
                overrides['specs_dir'] = Path(specs_dir).resolve()
            if plugins_dir:
                overrides['plugins_dir'] = Path(plugins_dir).resolve()
            if overrides:
                new_rt = replace(config.runtime, **overrides)
                config = replace(config, runtime=new_rt)
            if _cmd_check_plugin(config) != 0:
                exit_code = 1
        sys.exit(exit_code)
    elif args.command == 'list':
        sys.exit(_cmd_list(runtime_names, config_path=config_path))
    else:
        parser.print_help()
        sys.exit(0)
