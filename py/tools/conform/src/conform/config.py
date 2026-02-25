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

"""TOML configuration loader for the ``conform`` tool.

Reads a ``conform.toml`` (or ``pyproject.toml``) config file passed via
``--config``.  The config file contains all repository-specific settings:
concurrency, environment variables, and per-runtime paths.

Supports two TOML layouts:
- ``conform.toml``: top-level ``[conform]`` section.
- ``pyproject.toml``: nested under ``[tool.conform]``.

**Multi-runtime support:** Each runtime (Python, JS, Go, etc.) has its
own ``[conform.runtimes.<name>]`` section defining how to locate specs,
plugins, and the entry command to execute.  The ``--runtime`` CLI flag
selects which runtime config to use (default: ``python``).

Python 3.10 ships without ``tomllib``; the ``tomli`` backport is used
as a fallback.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Config file name used for auto-discovery.
_DEFAULT_CONFIG_NAME = 'conform.toml'


def _discover_config() -> Path | None:
    """Walk up from the current directory looking for ``conform.toml``.

    Returns the first ``conform.toml`` found, or ``None`` if the
    filesystem root is reached without finding one.
    """
    current = Path.cwd().resolve()
    while True:
        candidate = current / _DEFAULT_CONFIG_NAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            # Reached filesystem root.
            return None
        current = parent


if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        msg = "Python 3.10 requires the 'tomli' package: pip install tomli"
        raise SystemExit(msg) from exc

# Sentinel for "no CLI override".
_UNSET: int = -1


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for a single runtime (e.g. python, js, go).

    Attributes:
        name: Runtime identifier (e.g. ``python``, ``js``, ``go``).
        specs_dir: Absolute path to the directory containing conformance
            specs (one subdirectory per plugin with ``model-conformance.yaml``
            and a runtime-specific entry file).
        plugins_dir: Absolute path to the directory containing plugin
            source trees (used by ``check-plugin`` for model marker
            scanning).
        entry_command: Command prefix used to run a conformance entry
            point.  The entry-point path is appended as the last argument.
            Example: ``["uv", "run", "--project", "/path/to/py", "--active"]``.
        cwd: Working directory for subprocess execution.  Defaults to
            the repo root.  JS needs ``js/`` so Node module resolution
            finds the pnpm workspace packages.
        entry_filename: The conformance entry point filename.
            Defaults to ``conformance_entry.py`` for Python,
            ``conformance_entry.ts`` for JS, etc.
        model_marker: Glob pattern to identify model provider plugins.
            The ``check-plugin`` subcommand scans for files matching this
            pattern under ``plugins_dir/*/src/``. Defaults to
            ``model_info.py`` for Python.
    """

    name: str
    specs_dir: Path
    plugins_dir: Path
    entry_command: list[str]
    cwd: Path | None = None
    entry_filename: str = 'conformance_entry.py'
    native_entry_filename: str = ''
    default_runner: str = 'auto'
    model_marker: str = 'model_info.py'


@dataclass(frozen=True)
class ConformConfig:
    """Resolved configuration for the ``conform`` tool.

    Attributes:
        concurrency: Maximum number of plugins to test in parallel.
        test_concurrency: Maximum number of tests to run in parallel
            within a single model spec.  Defaults to 3.
        max_retries: Maximum number of retries for a failed test before
            giving up.  Defaults to 2.  Set to 0 to disable retries.
        retry_base_delay: Base delay in seconds for exponential backoff
            with full jitter.  The actual delay for attempt *k* is
            ``random() * min(base * 2**k, 60)``.
            Defaults to 1.0.
        action_timeout: Timeout in seconds for a single LLM action call
            (model generate request).  Defaults to 120.0.
        health_timeout: Timeout in seconds for a single health-check
            request to the reflection server.  Defaults to 5.0.
        startup_timeout: Maximum seconds to wait for the reflection
            server to become healthy after starting the entry point
            subprocess.  Defaults to 30.0.
        env: Mapping of plugin name → list of required environment variables.
        additional_model_plugins: Plugins that should have conformance specs
            but lack ``model_info.py`` (they use dynamic model registration).
        plugin_overrides: Per-plugin configuration overrides.  Keyed by
            plugin name, each value is a dict that may contain
            ``test-concurrency`` and ``action-timeout``.
        model_overrides: Per-model configuration overrides.  Keyed by
            model name (e.g. ``googleai/gemini-2.5-flash``), each value
            is a dict that may contain ``action-timeout``.
        runtime: The active runtime configuration.
    """

    concurrency: int = 8
    test_concurrency: int = 3
    max_retries: int = 2
    retry_base_delay: float = 1.0
    action_timeout: float = 120.0
    health_timeout: float = 5.0
    startup_timeout: float = 30.0
    env: dict[str, list[str]] = field(default_factory=dict)
    additional_model_plugins: list[str] = field(default_factory=list)
    plugin_overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    model_overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(
            name='python',
            specs_dir=Path(),
            plugins_dir=Path(),
            entry_command=[],
        )
    )
    runner: str = 'auto'

    def test_concurrency_for(self, plugin: str) -> int:
        """Return the effective test concurrency for *plugin*.

        Checks ``plugin_overrides`` first, then falls back to the
        global ``test_concurrency``.
        """
        overrides = self.plugin_overrides.get(plugin, {})
        tc = overrides.get('test-concurrency', self.test_concurrency)
        if isinstance(tc, int) and tc >= 1:
            return tc
        return self.test_concurrency

    def action_timeout_for(self, plugin: str, model: str = '') -> float:
        """Return the effective action timeout for *plugin* and *model*.

        Resolution order (most specific wins):
        1. ``model_overrides[model]['action-timeout']``
        2. ``plugin_overrides[plugin]['action-timeout']``
        3. Global ``action_timeout``
        """
        if model:
            mo = self.model_overrides.get(model, {})
            mt = mo.get('action-timeout')
            if isinstance(mt, (int, float)) and mt > 0:
                return float(mt)
        po = self.plugin_overrides.get(plugin, {})
        pt = po.get('action-timeout')
        if isinstance(pt, (int, float)) and pt > 0:
            return float(pt)
        return self.action_timeout


def _load_toml(path: Path) -> dict[str, object]:
    """Read and parse a TOML file."""
    with open(path, 'rb') as fh:
        return tomllib.load(fh)


def _resolve_path(raw: str, base_dir: Path) -> Path:
    """Resolve a path that may be relative to *base_dir*.

    Absolute paths are returned as-is.  Relative paths are resolved
    against *base_dir* (typically the directory containing the config
    file).
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


# Default entry filenames and model markers per runtime.
_RUNTIME_DEFAULTS: dict[str, dict[str, str]] = {
    'python': {
        'entry_filename': 'conformance_entry.py',
        'model_marker': 'model_info.py',
        'native_entry_filename': 'conformance_native.py',
        'default_runner': 'in-process',
    },
    'js': {
        'entry_filename': 'conformance_entry.ts',
        'model_marker': 'models.ts',
        'native_entry_filename': 'conformance_native.ts',
        'default_runner': 'native',
    },
    'go': {
        'entry_filename': 'conformance_entry.go',
        'model_marker': 'model_info.go',
        'native_entry_filename': 'conformance_native.go',
        'default_runner': 'native',
    },
}


def _parse_runtime(
    name: str,
    raw: dict[str, object],
    base_dir: Path,
) -> RuntimeConfig:
    """Parse a single ``[tool.conform.runtimes.<name>]`` section.

    *base_dir* is used to resolve relative paths (typically the
    directory containing the config file).
    """
    specs_dir_raw = raw.get('specs-dir', '')
    plugins_dir_raw = raw.get('plugins-dir', '')
    entry_cmd_raw = raw.get('entry-command', [])
    cwd_raw = raw.get('cwd', '')

    if not specs_dir_raw:
        raise SystemExit(
            f'error: [conform.runtimes.{name}] is missing required '
            f"key 'specs-dir'.  Set it in conform.toml or pass --specs-dir."
        )
    if not plugins_dir_raw:
        raise SystemExit(
            f'error: [conform.runtimes.{name}] is missing required '
            f"key 'plugins-dir'.  Set it in conform.toml or pass --plugins-dir."
        )

    specs_dir = _resolve_path(str(specs_dir_raw), base_dir)
    plugins_dir = _resolve_path(str(plugins_dir_raw), base_dir)
    cwd = _resolve_path(str(cwd_raw), base_dir) if cwd_raw else None

    entry_command: list[str] = []
    if isinstance(entry_cmd_raw, list):
        entry_command = [str(v) for v in entry_cmd_raw]

    # Runtime-specific defaults, overridable via TOML.
    defaults = _RUNTIME_DEFAULTS.get(name, _RUNTIME_DEFAULTS['python'])
    entry_filename = str(raw.get('entry-filename', defaults['entry_filename']))
    native_entry_filename = str(raw.get('native-entry-filename', defaults.get('native_entry_filename', '')))
    default_runner = str(raw.get('default-runner', defaults.get('default_runner', 'auto')))
    model_marker = str(raw.get('model-marker', defaults['model_marker']))

    return RuntimeConfig(
        name=name,
        specs_dir=specs_dir,
        plugins_dir=plugins_dir,
        entry_command=entry_command,
        cwd=cwd,
        entry_filename=entry_filename,
        native_entry_filename=native_entry_filename,
        default_runner=default_runner,
        model_marker=model_marker,
    )


def _extract_conform_section(data: dict[str, object]) -> dict[str, object]:
    """Extract the ``[conform]`` section from parsed TOML data.

    Supports two layouts:
    - ``conform.toml``: top-level ``[conform]`` key.
    - ``pyproject.toml``: nested under ``[tool.conform]``.
    """
    # Try top-level [conform] first (conform.toml).
    conform: object = data.get('conform')
    if isinstance(conform, dict):
        return {str(k): v for k, v in conform.items()}

    # Fall back to [tool.conform] (pyproject.toml).
    tool: object = data.get('tool')
    if isinstance(tool, dict):
        conform = tool.get('conform')  # type: ignore[call-overload]
        if isinstance(conform, dict):
            return {str(k): v for k, v in conform.items()}

    return {}


def load_config(
    *,
    concurrency_override: int = _UNSET,
    test_concurrency_override: int = _UNSET,
    max_retries_override: int = _UNSET,
    retry_base_delay_override: float = -1.0,
    runtime_name: str = 'python',
    config_path: Path | None = None,
) -> ConformConfig:
    """Load configuration from TOML, applying CLI overrides.

    Args:
        concurrency_override: If positive, overrides the TOML concurrency.
        test_concurrency_override: If positive, overrides the TOML
            test-concurrency (per-model parallel requests).
        max_retries_override: If non-negative, overrides the TOML
            max-retries.
        retry_base_delay_override: If non-negative, overrides the TOML
            retry-base-delay.
        runtime_name: Which runtime section to load (default: ``python``).
        config_path: Explicit path to a ``conform.toml`` or ``pyproject.toml``.
            When ``None``, auto-discovers ``conform.toml`` by walking up
            from the current working directory.

    Returns:
        Fully resolved :class:`ConformConfig`.
    """
    if config_path is None:
        config_path = _discover_config()

    if config_path is None or not config_path.is_file():
        raise SystemExit(
            f'error: conform.toml not found (searched from {Path.cwd()}).\n'
            f'Pass --config <path> or create a conform.toml in the specs directory.'
        )

    raw: dict[str, object] = _extract_conform_section(_load_toml(config_path))

    # Parse concurrency.
    toml_concurrency = raw.get('concurrency', 8)
    if not isinstance(toml_concurrency, int) or toml_concurrency < 1:
        toml_concurrency = 8
    concurrency = concurrency_override if concurrency_override > 0 else toml_concurrency

    # Parse test concurrency (per-model parallel requests).
    toml_test_concurrency = raw.get('test-concurrency', 3)
    if not isinstance(toml_test_concurrency, int) or toml_test_concurrency < 1:
        toml_test_concurrency = 3
    test_concurrency = test_concurrency_override if test_concurrency_override > 0 else toml_test_concurrency

    # Parse retry settings.
    toml_max_retries = raw.get('max-retries', 2)
    if not isinstance(toml_max_retries, int) or toml_max_retries < 0:
        toml_max_retries = 2
    max_retries = max_retries_override if max_retries_override >= 0 else toml_max_retries

    toml_retry_base_delay = raw.get('retry-base-delay', 1.0)
    if not isinstance(toml_retry_base_delay, (int, float)) or toml_retry_base_delay < 0:
        toml_retry_base_delay = 1.0
    retry_base_delay = retry_base_delay_override if retry_base_delay_override >= 0 else float(toml_retry_base_delay)

    # Parse timeout settings.
    toml_action_timeout = raw.get('action-timeout', 120.0)
    action_timeout = (
        float(toml_action_timeout)
        if isinstance(toml_action_timeout, (int, float)) and toml_action_timeout > 0
        else 120.0
    )

    toml_health_timeout = raw.get('health-timeout', 5.0)
    health_timeout = (
        float(toml_health_timeout) if isinstance(toml_health_timeout, (int, float)) and toml_health_timeout > 0 else 5.0
    )

    toml_startup_timeout = raw.get('startup-timeout', 30.0)
    startup_timeout = (
        float(toml_startup_timeout)
        if isinstance(toml_startup_timeout, (int, float)) and toml_startup_timeout > 0
        else 30.0
    )

    # Parse env vars.
    env_raw = raw.get('env', {})
    env: dict[str, list[str]] = {}
    if isinstance(env_raw, dict):
        for plugin, vars_list in env_raw.items():
            if isinstance(plugin, str) and isinstance(vars_list, list):
                env[plugin] = [v for v in vars_list if isinstance(v, str)]

    # Parse additional model plugins.
    amp_raw = raw.get('additional-model-plugins', [])
    additional_model_plugins: list[str] = []
    if isinstance(amp_raw, list):
        additional_model_plugins = [v for v in amp_raw if isinstance(v, str)]

    # Parse per-plugin overrides (e.g. test-concurrency, action-timeout).
    po_raw = raw.get('plugin-overrides', {})
    plugin_overrides: dict[str, dict[str, object]] = {}
    if isinstance(po_raw, dict):
        for plugin_name, overrides in po_raw.items():
            if isinstance(plugin_name, str) and isinstance(overrides, dict):
                plugin_overrides[plugin_name] = {str(k): v for k, v in overrides.items()}

    # Parse per-model overrides (e.g. action-timeout).
    mo_raw = raw.get('model-overrides', {})
    model_overrides: dict[str, dict[str, object]] = {}
    if isinstance(mo_raw, dict):
        for model_name, overrides in mo_raw.items():
            if isinstance(model_name, str) and isinstance(overrides, dict):
                model_overrides[model_name] = {str(k): v for k, v in overrides.items()}

    # Parse runtime config.
    base_dir = config_path.resolve().parent
    runtimes_raw = raw.get('runtimes', {})
    runtime: RuntimeConfig | None = None

    if isinstance(runtimes_raw, dict):
        rt_section: object = runtimes_raw.get(runtime_name)  # type: ignore[union-attr]
        if isinstance(rt_section, dict):
            # Narrow to dict[str, object] for _parse_runtime.
            rt_dict: dict[str, object] = {str(k): v for k, v in rt_section.items()}
            runtime = _parse_runtime(runtime_name, rt_dict, base_dir)

    if runtime is None:
        raise SystemExit(
            f'error: no [conform.runtimes.{runtime_name}] section found '
            f'in {config_path}.  Add it or pass --specs-dir / --plugins-dir.'
        )

    # Parse runner type.
    runner_raw = raw.get('runner', 'auto')
    runner = str(runner_raw) if isinstance(runner_raw, str) else 'auto'
    if runner not in ('auto', 'native', 'reflection', 'in-process'):
        runner = 'auto'

    return ConformConfig(
        concurrency=concurrency,
        test_concurrency=test_concurrency,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        action_timeout=action_timeout,
        health_timeout=health_timeout,
        startup_timeout=startup_timeout,
        env=env,
        additional_model_plugins=additional_model_plugins,
        plugin_overrides=plugin_overrides,
        model_overrides=model_overrides,
        runtime=runtime,
        runner=runner,
    )


def load_all_runtime_names(
    config_path: Path | None = None,
) -> list[str]:
    """Return the names of all configured runtimes.

    Reads ``[conform.runtimes.*]`` sections from the TOML config.
    Falls back to ``['python']`` if no runtimes are configured.
    """
    if config_path is None:
        config_path = _discover_config()

    if config_path is None or not config_path.is_file():
        return ['python']

    raw = _extract_conform_section(_load_toml(config_path))

    runtimes_raw: object = raw.get('runtimes')
    if not isinstance(runtimes_raw, dict):
        return ['python']

    names = [str(k) for k in runtimes_raw]
    return names if names else ['python']
