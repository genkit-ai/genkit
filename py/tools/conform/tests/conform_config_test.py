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

"""Tests for the ``conform.config`` module."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from conform.config import (
    ConformConfig,
    RuntimeConfig,
    _extract_conform_section,
    _parse_runtime,
    _resolve_path,
    load_all_runtime_names,
    load_config,
)

# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    """Tests for _resolve_path."""

    def test_absolute_path_unchanged(self, tmp_path: Path) -> None:
        """Absolute paths are returned as-is."""
        p = tmp_path / 'specs'
        assert _resolve_path(str(p), Path('/some/base')) == p

    def test_relative_path_resolved(self, tmp_path: Path) -> None:
        """Relative paths are resolved against the base directory."""
        result = _resolve_path('specs', tmp_path)
        assert result == (tmp_path / 'specs').resolve()

    def test_dotdot_path_resolved(self, tmp_path: Path) -> None:
        """Parent-relative paths are resolved correctly."""
        base = tmp_path / 'a' / 'b'
        base.mkdir(parents=True)
        result = _resolve_path('../../x', base)
        assert result == (tmp_path / 'x').resolve()


# ---------------------------------------------------------------------------
# _extract_conform_section
# ---------------------------------------------------------------------------


class TestExtractConformSection:
    """Tests for _extract_conform_section."""

    def test_top_level_conform(self) -> None:
        """Extract from top-level [conform] section."""
        data: dict[str, object] = {'conform': {'concurrency': 4}}
        assert _extract_conform_section(data) == {'concurrency': 4}

    def test_tool_conform(self) -> None:
        """Extract from [tool.conform] section."""
        data: dict[str, object] = {'tool': {'conform': {'concurrency': 4}}}
        assert _extract_conform_section(data) == {'concurrency': 4}

    def test_top_level_takes_precedence(self) -> None:
        """Top-level [conform] takes precedence over [tool.conform]."""
        data: dict[str, object] = {
            'conform': {'concurrency': 4},
            'tool': {'conform': {'concurrency': 8}},
        }
        assert _extract_conform_section(data) == {'concurrency': 4}

    def test_empty_data(self) -> None:
        """Return empty dict for empty input."""
        assert _extract_conform_section({}) == {}

    def test_no_conform_key(self) -> None:
        """Return empty dict when no conform key exists."""
        assert _extract_conform_section({'other': 'stuff'}) == {}


# ---------------------------------------------------------------------------
# _parse_runtime
# ---------------------------------------------------------------------------


class TestParseRuntime:
    """Tests for _parse_runtime."""

    def test_basic_parse(self, tmp_path: Path) -> None:
        """Parse a complete runtime section with all fields."""
        raw: dict[str, object] = {
            'specs-dir': 'specs',
            'plugins-dir': 'plugins',
            'entry-command': ['uv', 'run'],
            'cwd': '.',
        }
        rt = _parse_runtime('python', raw, tmp_path)
        assert rt.name == 'python'
        assert rt.specs_dir == (tmp_path / 'specs').resolve()
        assert rt.plugins_dir == (tmp_path / 'plugins').resolve()
        assert rt.entry_command == ['uv', 'run']
        assert rt.cwd == tmp_path.resolve()
        assert rt.entry_filename == 'conformance_entry.py'
        assert rt.model_marker == 'model_info.py'

    def test_js_defaults(self, tmp_path: Path) -> None:
        """JS runtime uses .ts entry filename and models.ts marker."""
        raw: dict[str, object] = {'specs-dir': '.', 'plugins-dir': 'plugins'}
        rt = _parse_runtime('js', raw, tmp_path)
        assert rt.entry_filename == 'conformance_entry.ts'
        assert rt.model_marker == 'models.ts'

    def test_go_defaults(self, tmp_path: Path) -> None:
        """Go runtime uses .go entry filename and model_info.go marker."""
        raw: dict[str, object] = {'specs-dir': '.', 'plugins-dir': 'plugins'}
        rt = _parse_runtime('go', raw, tmp_path)
        assert rt.entry_filename == 'conformance_entry.go'
        assert rt.model_marker == 'model_info.go'

    def test_custom_entry_filename(self, tmp_path: Path) -> None:
        """Custom entry-filename overrides the default."""
        raw: dict[str, object] = {
            'specs-dir': '.',
            'plugins-dir': 'plugins',
            'entry-filename': 'custom_entry.py',
        }
        rt = _parse_runtime('python', raw, tmp_path)
        assert rt.entry_filename == 'custom_entry.py'

    def test_missing_specs_dir_exits(self, tmp_path: Path) -> None:
        """Exit with error when specs-dir is missing."""
        raw: dict[str, object] = {'plugins-dir': 'p'}
        with pytest.raises(SystemExit, match='specs-dir'):
            _parse_runtime('python', raw, tmp_path)

    def test_missing_plugins_dir_exits(self, tmp_path: Path) -> None:
        """Exit with error when plugins-dir is missing."""
        raw: dict[str, object] = {'specs-dir': 's'}
        with pytest.raises(SystemExit, match='plugins-dir'):
            _parse_runtime('python', raw, tmp_path)

    def test_no_cwd_returns_none(self, tmp_path: Path) -> None:
        """Omitting cwd sets it to None."""
        raw: dict[str, object] = {'specs-dir': '.', 'plugins-dir': 'plugins'}
        rt = _parse_runtime('python', raw, tmp_path)
        assert rt.cwd is None


# ---------------------------------------------------------------------------
# ConformConfig
# ---------------------------------------------------------------------------


class TestConformConfig:
    """Tests for ConformConfig dataclass."""

    def test_defaults(self) -> None:
        """Default values match documented defaults."""
        cfg = ConformConfig()
        assert cfg.concurrency == 8
        assert cfg.test_concurrency == 3
        assert cfg.max_retries == 2
        assert cfg.retry_base_delay == 1.0
        assert cfg.env == {}
        assert cfg.additional_model_plugins == []
        assert cfg.plugin_overrides == {}

    def test_test_concurrency_for_global(self) -> None:
        """Global test_concurrency is used when no override exists."""
        cfg = ConformConfig(test_concurrency=5)
        assert cfg.test_concurrency_for('any-plugin') == 5

    def test_test_concurrency_for_override(self) -> None:
        """Per-plugin override takes precedence over global."""
        cfg = ConformConfig(
            test_concurrency=5,
            plugin_overrides={'slow-plugin': {'test-concurrency': 1}},
        )
        assert cfg.test_concurrency_for('slow-plugin') == 1
        assert cfg.test_concurrency_for('other-plugin') == 5

    def test_test_concurrency_for_invalid_override(self) -> None:
        """Invalid override type falls back to global."""
        cfg = ConformConfig(
            test_concurrency=5,
            plugin_overrides={'bad': {'test-concurrency': 'not-an-int'}},
        )
        assert cfg.test_concurrency_for('bad') == 5

    def test_test_concurrency_for_zero_override(self) -> None:
        """Zero override falls back to global."""
        cfg = ConformConfig(
            test_concurrency=5,
            plugin_overrides={'bad': {'test-concurrency': 0}},
        )
        assert cfg.test_concurrency_for('bad') == 5


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------


class TestRuntimeConfig:
    """Tests for RuntimeConfig dataclass."""

    def test_frozen(self) -> None:
        """RuntimeConfig is immutable (frozen dataclass)."""
        rt = RuntimeConfig(
            name='python',
            specs_dir=Path('.'),
            plugins_dir=Path('.'),
            entry_command=[],
        )
        with pytest.raises(AttributeError):
            rt.name = 'js'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_config (integration with tmp TOML files)
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for load_config with real TOML files."""

    def _write_toml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / 'conform.toml'
        p.write_text(dedent(content))
        return p

    def test_basic_load(self, tmp_path: Path) -> None:
        """Load all fields from a minimal conform.toml."""
        cfg_path = self._write_toml(
            tmp_path,
            """\
            [conform]
            concurrency = 4
            test-concurrency = 2
            max-retries = 1
            retry-base-delay = 0.5

            [conform.runtimes.python]
            specs-dir = "."
            plugins-dir = "plugins"
            entry-command = ["python"]
        """,
        )
        cfg = load_config(config_path=cfg_path)
        assert cfg.concurrency == 4
        assert cfg.test_concurrency == 2
        assert cfg.max_retries == 1
        assert cfg.retry_base_delay == 0.5

    def test_cli_overrides(self, tmp_path: Path) -> None:
        """CLI overrides take precedence over TOML values."""
        cfg_path = self._write_toml(
            tmp_path,
            """\
            [conform]
            concurrency = 4
            test-concurrency = 2
            max-retries = 1
            retry-base-delay = 0.5

            [conform.runtimes.python]
            specs-dir = "."
            plugins-dir = "plugins"
            entry-command = ["python"]
        """,
        )
        cfg = load_config(
            config_path=cfg_path,
            concurrency_override=16,
            test_concurrency_override=8,
            max_retries_override=0,
            retry_base_delay_override=2.0,
        )
        assert cfg.concurrency == 16
        assert cfg.test_concurrency == 8
        assert cfg.max_retries == 0
        assert cfg.retry_base_delay == 2.0

    def test_env_vars_parsed(self, tmp_path: Path) -> None:
        """Parse per-plugin env var requirements."""
        cfg_path = self._write_toml(
            tmp_path,
            """\
            [conform]
            [conform.env]
            google-genai = ["GEMINI_API_KEY"]
            ollama = []

            [conform.runtimes.python]
            specs-dir = "."
            plugins-dir = "plugins"
            entry-command = ["python"]
        """,
        )
        cfg = load_config(config_path=cfg_path)
        assert cfg.env == {'google-genai': ['GEMINI_API_KEY'], 'ollama': []}

    def test_additional_model_plugins(self, tmp_path: Path) -> None:
        """Parse additional-model-plugins list."""
        cfg_path = self._write_toml(
            tmp_path,
            """\
            [conform]
            additional-model-plugins = ["google-genai", "vertex-ai"]

            [conform.runtimes.python]
            specs-dir = "."
            plugins-dir = "plugins"
            entry-command = ["python"]
        """,
        )
        cfg = load_config(config_path=cfg_path)
        assert cfg.additional_model_plugins == ['google-genai', 'vertex-ai']

    def test_plugin_overrides(self, tmp_path: Path) -> None:
        """Parse per-plugin overrides from TOML."""
        cfg_path = self._write_toml(
            tmp_path,
            """\
            [conform]
            [conform.plugin-overrides.cloudflare]
            test-concurrency = 1

            [conform.runtimes.python]
            specs-dir = "."
            plugins-dir = "plugins"
            entry-command = ["python"]
        """,
        )
        cfg = load_config(config_path=cfg_path)
        assert cfg.test_concurrency_for('cloudflare') == 1

    def test_missing_config_exits(self) -> None:
        """Exit with error when config file does not exist."""
        with pytest.raises(SystemExit, match='conform.toml not found'):
            load_config(config_path=Path('/nonexistent/conform.toml'))

    def test_missing_runtime_exits(self, tmp_path: Path) -> None:
        """Exit with error when requested runtime is not configured."""
        cfg_path = self._write_toml(
            tmp_path,
            """\
            [conform]
            [conform.runtimes.python]
            specs-dir = "."
            plugins-dir = "plugins"
            entry-command = ["python"]
        """,
        )
        with pytest.raises(SystemExit, match='no.*runtimes.js'):
            load_config(config_path=cfg_path, runtime_name='js')

    def test_pyproject_toml_layout(self, tmp_path: Path) -> None:
        """Load config from [tool.conform] layout."""
        p = tmp_path / 'conform.toml'
        p.write_text(
            dedent("""\
            [tool.conform]
            concurrency = 6

            [tool.conform.runtimes.python]
            specs-dir = "."
            plugins-dir = "plugins"
            entry-command = ["python"]
        """)
        )
        cfg = load_config(config_path=p)
        assert cfg.concurrency == 6

    def test_invalid_concurrency_uses_default(self, tmp_path: Path) -> None:
        """Invalid concurrency falls back to default."""
        cfg_path = self._write_toml(
            tmp_path,
            """\
            [conform]
            concurrency = -1

            [conform.runtimes.python]
            specs-dir = "."
            plugins-dir = "plugins"
            entry-command = ["python"]
        """,
        )
        cfg = load_config(config_path=cfg_path)
        assert cfg.concurrency == 8

    def test_invalid_retry_uses_default(self, tmp_path: Path) -> None:
        """Invalid retry settings fall back to defaults."""
        cfg_path = self._write_toml(
            tmp_path,
            """\
            [conform]
            max-retries = -5
            retry-base-delay = -1.0

            [conform.runtimes.python]
            specs-dir = "."
            plugins-dir = "plugins"
            entry-command = ["python"]
        """,
        )
        cfg = load_config(config_path=cfg_path)
        assert cfg.max_retries == 2
        assert cfg.retry_base_delay == 1.0


# ---------------------------------------------------------------------------
# load_all_runtime_names
# ---------------------------------------------------------------------------


class TestLoadAllRuntimeNames:
    """Tests for load_all_runtime_names."""

    def test_returns_configured_runtimes(self, tmp_path: Path) -> None:
        """Return all runtime names from config."""
        p = tmp_path / 'conform.toml'
        p.write_text(
            dedent("""\
            [conform.runtimes.python]
            specs-dir = "."
            plugins-dir = "plugins"

            [conform.runtimes.js]
            specs-dir = "."
            plugins-dir = "plugins"
        """)
        )
        names = load_all_runtime_names(config_path=p)
        assert set(names) == {'python', 'js'}

    def test_missing_config_returns_python(self) -> None:
        """Fall back to ['python'] when config is missing."""
        names = load_all_runtime_names(config_path=Path('/nonexistent'))
        assert names == ['python']

    def test_no_runtimes_returns_python(self, tmp_path: Path) -> None:
        """Fall back to ['python'] when no runtimes are configured."""
        p = tmp_path / 'conform.toml'
        p.write_text('[conform]\n')
        names = load_all_runtime_names(config_path=p)
        assert names == ['python']
