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

"""Filesystem middleware for Genkit.

Provides sandboxed file operations (list_files, read_file, write_file, edit_file)
confined to a root directory. Tracks file mtime/size to detect external modifications
and to suppress redundant reads. Error messages are queued as user-role messages via
the engine's ``enqueue_parts`` mechanism so the model can self-correct.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel as PydanticBaseModel, PrivateAttr, model_validator

from genkit._ai._tools import Interrupt, define_tool
from genkit._core._model import ModelResponse
from genkit._core._registry import Registry
from genkit._core._typing import (
    Media,
    MediaPart,
    Part,
    TextPart,
    ToolRequestPart,
)
from genkit.middleware import (
    BaseMiddleware,
    GenerateHookParams,
    MultipartToolResponse,
    ToolHookParams,
)

# ---------------------------------------------------------------------------
# Tool input schemas (module-level so Pydantic can resolve annotations)
# ---------------------------------------------------------------------------


class _ListFilesInput(PydanticBaseModel):
    """Input for list_files tool."""

    dir_path: str = ''
    recursive: bool = False


class _ReadFileInput(PydanticBaseModel):
    """Input for read_file tool."""

    file_path: str
    offset: int = 0
    limit: int = 0


class _WriteFileInput(PydanticBaseModel):
    """Input for write_file tool."""

    file_path: str
    content: str


class _EditSpec(PydanticBaseModel):
    """A single string-replacement edit."""

    old_string: str
    new_string: str
    replace_all: bool = False


class _EditFileInput(PydanticBaseModel):
    """Input for edit_file tool."""

    file_path: str
    edits: list[_EditSpec]


# File-size limits.
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB — absolute ceiling for reading
_MAX_READ_SLICE_BYTES = 256 * 1024         # 256 KB — max bytes returned per slice
_MAX_CACHE_ENTRIES = 200

# Stub returned when an unchanged file is re-requested at the same byte range.
_FILE_UNCHANGED_STUB = (
    'File unchanged since last read. '
    'The content from the earlier read_file result in this conversation is still current'
    ' — refer to that instead of re-reading.'
)


@dataclass
class _FileState:
    """Per-path read state used for change detection and redundant-read suppression."""

    mtime: float
    size: int
    offset: int  # 0 when the read covered the whole file
    limit: int   # 0 when the read covered the whole file


class Filesystem(BaseMiddleware):
    """Filesystem middleware with sandboxed file operations.

    Contributes ``list_files``, ``read_file``, and (if ``allow_write_access``) ``write_file``
    and ``edit_file`` tools via ``tools(enqueue_parts)``. All paths are restricted to
    ``root_dir``. Tool errors are queued as user messages via ``enqueue_parts`` so the
    model can observe and self-correct.

    A per-call file-state cache (mtime + size) is allocated inside ``tools()`` so that
    each ``generate()`` call has its own independent read/write tracking. Concurrent calls
    on the same ``Filesystem`` instance are fully isolated — write guards from one call
    cannot block another.
    """

    name: ClassVar[str] = 'filesystem'
    description: ClassVar[str | None] = 'Sandboxed filesystem operations'

    root_dir: str
    allow_write_access: bool = False
    tool_name_prefix: str = ''

    _root_abs: str = PrivateAttr(default='')
    # Cached short-name set used by ``wrap_tool`` to decide whether a tool
    # belongs to this middleware. Computed once after validation since
    # ``tool_name_prefix`` is immutable.
    _fs_tool_name_set: frozenset[str] = PrivateAttr(default_factory=frozenset)

    @model_validator(mode='after')
    def _validate_root(self) -> Filesystem:
        if not self.root_dir or not self.root_dir.strip():
            raise ValueError('Filesystem.root_dir must not be empty.')
        self._root_abs = str(Path(self.root_dir).resolve())
        names = {self._tool_name('list_files'), self._tool_name('read_file')}
        if self.allow_write_access:
            names |= {self._tool_name('write_file'), self._tool_name('edit_file')}
        self._fs_tool_name_set = frozenset(names)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tool_name(self, base: str) -> str:
        return f'{self.tool_name_prefix}{base}'

    def _resolve_safe(self, rel: str) -> str:
        """Resolve ``rel`` to an absolute path, raising ValueError if it escapes root.

        Comparison goes through ``os.path.normcase`` so the case-insensitive
        Windows filesystem doesn't punch a hole in the sandbox check.
        """
        # Strip both forward and back slashes — the latter matters on Windows
        # where users can paste backslash-prefixed relative paths.
        rel = rel.strip().lstrip('/').lstrip('\\')
        if not rel:
            rel = '.'
        candidate = os.path.realpath(os.path.join(self._root_abs, rel))
        c_norm = os.path.normcase(candidate)
        root_norm = os.path.normcase(self._root_abs)
        if c_norm != root_norm and not c_norm.startswith(root_norm + os.sep):
            raise ValueError(f'Path {rel!r} escapes the root directory.')
        return candidate

    @staticmethod
    def _get_cache(
        abs_path: str,
        cache: OrderedDict[str, _FileState],
        lock: threading.Lock,
    ) -> _FileState | None:
        with lock:
            return cache.get(abs_path)

    @staticmethod
    def _set_cache(
        abs_path: str,
        state: _FileState,
        cache: OrderedDict[str, _FileState],
        lock: threading.Lock,
    ) -> None:
        with lock:
            cache[abs_path] = state
            while len(cache) > _MAX_CACHE_ENTRIES:
                cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _list_files(self, dir_path: str = '', recursive: bool = False) -> list[dict[str, Any]]:
        """List files and directories under ``dir_path`` (relative to root).

        Paths in the returned entries are relative to ``dir_path``, not to root —
        matching the non-recursive case.
        """
        abs_dir = self._resolve_safe(dir_path)
        if not os.path.isdir(abs_dir):
            raise ValueError(f'Not a directory: {dir_path!r}')

        results: list[dict[str, Any]] = []
        if recursive:
            for root, dirs, files in os.walk(abs_dir):
                dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
                for name in sorted(files):
                    abs_path = os.path.join(root, name)
                    try:
                        stat = os.stat(abs_path)
                        rel = os.path.relpath(abs_path, abs_dir)
                        results.append({'path': rel, 'is_directory': False, 'size_bytes': stat.st_size})
                    except OSError:
                        continue
                for name in dirs:
                    rel = os.path.relpath(os.path.join(root, name), abs_dir)
                    results.append({'path': rel, 'is_directory': True, 'size_bytes': 0})
        else:
            for name in sorted(os.listdir(abs_dir)):
                abs_path = os.path.join(abs_dir, name)
                try:
                    stat = os.stat(abs_path)
                    is_dir = os.path.isdir(abs_path)
                    results.append({'path': name, 'is_directory': is_dir, 'size_bytes': 0 if is_dir else stat.st_size})
                except OSError:
                    continue

        return results

    def _read_file_impl(
        self,
        file_path: str,
        offset: int,
        limit: int,
        enqueue_parts: Callable[[list[Part]], None] | None,
        cache: OrderedDict[str, _FileState],
        lock: threading.Lock,
    ) -> str:
        """Read a file and enqueue its content as a user message.

        Returns a short acknowledgement string (not the file content itself) so
        the tool response stays small — actual content reaches the model via the
        enqueued user message.
        """
        abs_path = self._resolve_safe(file_path)
        if not os.path.isfile(abs_path):
            raise ValueError(f'File not found: {file_path!r}')

        stat = os.stat(abs_path)
        if stat.st_size > _MAX_FILE_SIZE_BYTES:
            raise ValueError(f'File too large ({stat.st_size:,} bytes; max {_MAX_FILE_SIZE_BYTES:,}).')

        # Dedup: if mtime, size, offset, and limit all match, skip re-read.
        cached = self._get_cache(abs_path, cache, lock)
        if (
            cached is not None
            and cached.mtime == stat.st_mtime
            and cached.size == stat.st_size
            and cached.offset == offset
            and cached.limit == limit
        ):
            return _FILE_UNCHANGED_STUB

        mime_type, _ = mimetypes.guess_type(abs_path)
        is_image = bool(mime_type and mime_type.startswith('image/'))

        if is_image:
            with open(abs_path, 'rb') as fh:
                raw = fh.read()
            if len(raw) > _MAX_READ_SLICE_BYTES:
                raise ValueError(f'Image too large ({len(raw):,} bytes; max {_MAX_READ_SLICE_BYTES:,}).')
            b64 = base64.b64encode(raw).decode('ascii')
            data_uri = f'data:{mime_type};base64,{b64}'
            if enqueue_parts:
                media_part = Part(root=MediaPart(media=Media(url=data_uri, content_type=mime_type)))
                enqueue_parts([media_part])
            new_state = _FileState(mtime=stat.st_mtime, size=stat.st_size, offset=offset, limit=limit)
            self._set_cache(abs_path, new_state, cache, lock)
            return f'Image {file_path} queued as media part.'

        # Text file
        with open(abs_path, encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()

        total = len(lines)
        start = max(0, offset - 1) if offset > 0 else 0
        end = total if limit == 0 else min(total, start + limit)
        sliced = ''.join(lines[start:end])

        if len(sliced.encode()) > _MAX_READ_SLICE_BYTES:
            raise ValueError(
                f'Slice too large ({len(sliced):,} chars). Use offset/limit to read smaller sections.'
            )

        if offset > 0 or limit > 0:
            wrapped = f'<read_file path="{file_path}" lines="{start + 1}-{end}">\n{sliced}\n</read_file>'
        else:
            wrapped = f'<read_file path="{file_path}" totalLines="{total}">\n{sliced}\n</read_file>'

        if enqueue_parts:
            enqueue_parts([Part(root=TextPart(text=wrapped))])
        new_state = _FileState(mtime=stat.st_mtime, size=stat.st_size, offset=offset, limit=limit)
        self._set_cache(abs_path, new_state, cache, lock)
        return f'File {file_path} read successfully. Content queued as user message.'

    def _write_file_impl(
        self,
        file_path: str,
        content: str,
        cache: OrderedDict[str, _FileState],
        lock: threading.Lock,
    ) -> str:
        abs_path = self._resolve_safe(file_path)
        exists = os.path.isfile(abs_path)

        if exists:
            cached = self._get_cache(abs_path, cache, lock)
            if cached is None:
                raise ValueError(f'File must be read before writing: {file_path!r}')
            stat = os.stat(abs_path)
            if cached.mtime != stat.st_mtime or cached.size != stat.st_size:
                raise ValueError(f'File externally modified since last read: {file_path!r}. Re-read before writing.')

        os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as fh:
            fh.write(content)

        stat = os.stat(abs_path)
        self._set_cache(abs_path, _FileState(mtime=stat.st_mtime, size=stat.st_size, offset=0, limit=0), cache, lock)
        return f'File {file_path} written successfully.'

    def _edit_file_impl(
        self,
        file_path: str,
        edits: list[dict[str, Any]],
        cache: OrderedDict[str, _FileState],
        lock: threading.Lock,
    ) -> str:
        abs_path = self._resolve_safe(file_path)
        if not os.path.isfile(abs_path):
            raise ValueError(f'File not found: {file_path!r}')

        cached = self._get_cache(abs_path, cache, lock)
        if cached is None:
            raise ValueError(f'File must be read before editing: {file_path!r}')
        stat = os.stat(abs_path)
        if cached.mtime != stat.st_mtime or cached.size != stat.st_size:
            raise ValueError(f'File externally modified since last read: {file_path!r}. Re-read before editing.')

        with open(abs_path, encoding='utf-8', errors='replace') as fh:
            content = fh.read()

        for spec in edits:
            old = spec.get('old_string', '')
            new = spec.get('new_string', '')
            replace_all = spec.get('replace_all', False)
            if not old:
                raise ValueError('old_string must be non-empty.')
            if old == new:
                raise ValueError('old_string and new_string must differ.')
            count = content.count(old)
            if count == 0:
                raise ValueError(f'old_string not found in file: {old!r}')
            if not replace_all and count > 1:
                raise ValueError(f'old_string matches {count} times but replace_all=False.')
            content = content.replace(old, new) if replace_all else content.replace(old, new, 1)

        with open(abs_path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        stat = os.stat(abs_path)
        self._set_cache(abs_path, _FileState(mtime=stat.st_mtime, size=stat.st_size, offset=0, limit=0), cache, lock)
        return f'File {file_path} edited successfully.'

    # ------------------------------------------------------------------
    # Middleware hooks
    # ------------------------------------------------------------------

    def tools(self, enqueue_parts: Callable[[list[Part]], None] | None = None) -> list[Any]:
        """Return call-scoped filesystem tool actions.

        A fresh file-state cache and lock are allocated here so each generate()
        call gets its own isolated read/write tracking — concurrent calls on the
        same ``Filesystem`` instance cannot interfere with each other's write guards.

        Tool closures capture ``enqueue_parts`` so they can queue file content and
        error messages as user messages for the next generate iteration.
        """
        # Per-call state: each generate() call gets its own cache so write guards
        # from one call cannot block a different concurrent call that read the same
        # file independently.
        _call_cache: OrderedDict[str, _FileState] = OrderedDict()
        _call_lock: threading.Lock = threading.Lock()

        scratch = Registry()

        async def list_files(input: _ListFilesInput) -> list[dict[str, Any]]:
            """List files and directories within the workspace.

            Returns entries with ``path`` (relative to the queried directory),
            ``is_directory``, and ``size_bytes``.
            """
            return self._list_files(input.dir_path, input.recursive)

        async def read_file(input: _ReadFileInput) -> str:
            """Read a file and queue its content as a user message.

            File content is delivered via an enqueued user message so the tool
            response stays small. Use ``offset`` (1-indexed first line) and
            ``limit`` (max lines) to read slices of large files.
            """
            return self._read_file_impl(
                input.file_path, input.offset, input.limit, enqueue_parts, _call_cache, _call_lock
            )

        t_list = define_tool(scratch, list_files, name=self._tool_name('list_files'))
        t_read = define_tool(scratch, read_file, name=self._tool_name('read_file'))
        tools_out = [t_list.action(), t_read.action()]

        if self.allow_write_access:
            async def write_file(input: _WriteFileInput) -> str:
                """Write content to a file (requires prior read for existing files).

                Creates parent directories if needed. Fails if the file was externally
                modified since the last ``read_file`` call.
                """
                return self._write_file_impl(input.file_path, input.content, _call_cache, _call_lock)

            async def edit_file(input: _EditFileInput) -> str:
                """Apply string-replacement edits to a file (requires prior read).

                Each edit must have ``old_string`` and ``new_string``. Set
                ``replace_all: true`` to replace every occurrence instead of just the first.
                """
                return self._edit_file_impl(
                    input.file_path,
                    [e.model_dump() for e in input.edits],
                    _call_cache,
                    _call_lock,
                )

            t_write = define_tool(scratch, write_file, name=self._tool_name('write_file'))
            t_edit = define_tool(scratch, edit_file, name=self._tool_name('edit_file'))
            tools_out += [t_write.action(), t_edit.action()]

        return tools_out

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        next_fn: Callable[[GenerateHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Pass through — the engine drains enqueued messages automatically."""
        return await next_fn(params)

    async def wrap_tool(
        self,
        params: ToolHookParams,
        next_fn: Callable[
            [ToolHookParams],
            Awaitable[tuple[MultipartToolResponse | None, ToolRequestPart | None]],
        ],
    ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
        """Catch filesystem tool errors and enqueue them as user messages.

        On success, returns the tool result as-is. On failure (excluding Interrupt),
        queues a brief error description and returns a minimal ``MultipartToolResponse``
        so the model receives acknowledgement of the failure and can retry.
        """
        if params.tool.name not in self._fs_tool_name_set:
            return await next_fn(params)

        try:
            return await next_fn(params)
        except Interrupt:
            raise
        except Exception as exc:
            error_msg = f'Tool "{params.tool.name}" failed: {exc}'
            if params.enqueue_parts:
                params.enqueue_parts([Part(root=TextPart(text=error_msg))])
            return (
                MultipartToolResponse(output='Tool call failed; see user message below for details.'),
                None,
            )
