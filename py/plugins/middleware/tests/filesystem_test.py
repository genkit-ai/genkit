# Copyright 2025 Google LLC
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

"""Tests for Filesystem middleware."""

import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import pytest

from genkit.plugins.middleware import Filesystem


def _make_call_state():
    """Return a fresh (cache, lock) pair matching what tools() creates per call."""
    return OrderedDict(), threading.Lock()

# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_filesystem_validates_root_dir() -> None:
    """Filesystem must reject an empty root_dir."""
    with pytest.raises(ValueError, match='root_dir'):
        Filesystem(root_dir='')


def test_filesystem_resolves_root() -> None:
    """root_dir is resolved to an absolute path on construction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        assert fs._root_abs == str(Path(tmpdir).resolve())


# ---------------------------------------------------------------------------
# _resolve_safe
# ---------------------------------------------------------------------------


def test_resolve_safe_allows_root() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        assert fs._resolve_safe('') == fs._root_abs


def test_resolve_safe_allows_child() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        child = Path(tmpdir) / 'sub' / 'file.txt'
        child.parent.mkdir(parents=True)
        assert fs._resolve_safe('sub/file.txt').endswith('sub/file.txt')


def test_resolve_safe_blocks_escape() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        with pytest.raises(ValueError, match='escapes'):
            fs._resolve_safe('../../../etc/passwd')


# ---------------------------------------------------------------------------
# _list_files
# ---------------------------------------------------------------------------


def test_list_files_returns_paths_relative_to_queried_dir() -> None:
    """list_files paths should be relative to the requested sub-dir, not root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sub = Path(tmpdir) / 'docs'
        sub.mkdir()
        (sub / 'api.md').write_text('hello')
        fs = Filesystem(root_dir=tmpdir)
        entries = fs._list_files('docs')
        names = [e['path'] for e in entries]
        # Must be 'api.md', not 'docs/api.md'
        assert 'api.md' in names
        assert 'docs/api.md' not in names


def test_list_files_root() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / 'a.txt').write_text('a')
        (Path(tmpdir) / 'b.txt').write_text('b')
        fs = Filesystem(root_dir=tmpdir)
        entries = fs._list_files()
        names = {e['path'] for e in entries}
        assert 'a.txt' in names
        assert 'b.txt' in names


# ---------------------------------------------------------------------------
# _read_file_impl (text files)
# ---------------------------------------------------------------------------


def test_read_file_queues_content() -> None:
    queued: list = []

    def enqueue_parts(parts) -> None:
        queued.extend(parts)

    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / 'hello.txt'
        f.write_text('hello world\n')
        fs = Filesystem(root_dir=tmpdir)
        cache, lock = _make_call_state()
        result = fs._read_file_impl('hello.txt', 0, 0, enqueue_parts, cache, lock)
        assert 'queued' in result.lower() or 'read' in result.lower()
        assert len(queued) == 1


def test_read_file_dedup_unchanged() -> None:
    """Identical read (same mtime/size/offset/limit) returns stub."""
    queued: list = []

    def enqueue_parts(parts) -> None:
        queued.extend(parts)

    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / 'hello.txt'
        f.write_text('hello world\n')
        fs = Filesystem(root_dir=tmpdir)
        cache, lock = _make_call_state()
        fs._read_file_impl('hello.txt', 0, 0, enqueue_parts, cache, lock)
        queued.clear()
        result = fs._read_file_impl('hello.txt', 0, 0, enqueue_parts, cache, lock)
        assert 'unchanged' in result.lower()
        assert len(queued) == 0


# ---------------------------------------------------------------------------
# _write_file_impl and _edit_file_impl
# ---------------------------------------------------------------------------


def test_write_file_requires_read_first() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / 'existing.txt'
        f.write_text('original\n')
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        cache, lock = _make_call_state()
        with pytest.raises(ValueError, match='read'):
            fs._write_file_impl('existing.txt', 'new content\n', cache, lock)


def test_write_new_file_succeeds() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        cache, lock = _make_call_state()
        result = fs._write_file_impl('new.txt', 'content\n', cache, lock)
        assert 'written' in result.lower()
        assert (Path(tmpdir) / 'new.txt').read_text() == 'content\n'


def test_edit_file_requires_read_first() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / 'edit_me.txt'
        f.write_text('hello world\n')
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        cache, lock = _make_call_state()
        with pytest.raises(ValueError, match='read'):
            fs._edit_file_impl('edit_me.txt', [{'old_string': 'hello', 'new_string': 'hi'}], cache, lock)


def test_edit_file_after_read() -> None:
    queued: list = []

    def enqueue_parts(parts) -> None:
        queued.extend(parts)

    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / 'edit_me.txt'
        f.write_text('hello world\n')
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        cache, lock = _make_call_state()
        fs._read_file_impl('edit_me.txt', 0, 0, enqueue_parts, cache, lock)
        result = fs._edit_file_impl('edit_me.txt', [{'old_string': 'hello', 'new_string': 'hi'}], cache, lock)
        assert 'edited' in result.lower()
        assert f.read_text() == 'hi world\n'


# ---------------------------------------------------------------------------
# tools() — dynamic tool registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_returns_read_and_list() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        tool_actions = fs.tools()
        names = {t.name for t in tool_actions}
        assert 'list_files' in names
        assert 'read_file' in names
        assert 'write_file' not in names


@pytest.mark.asyncio
async def test_tools_returns_write_when_allowed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        tool_actions = fs.tools()
        names = {t.name for t in tool_actions}
        assert 'write_file' in names
        assert 'edit_file' in names
