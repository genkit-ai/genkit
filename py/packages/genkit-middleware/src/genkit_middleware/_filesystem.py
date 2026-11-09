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

Provides sandboxed file operations — ``list_files``, ``read_file``,
``write_file``, ``edit_file`` — confined to a configurable root directory.

Paths resolve against the root. An absolute path is accepted only when it
already points inside the root. Symlinks are resolved before the containment
check, so a link that points outside the root is refused.

``read_file`` queues file content as user messages so the tool response stays
small. Tool errors come back as the tool's own response, with the host root
path redacted, so the model can self-correct without a user turn landing
between the tool response and the next model turn.

Each ``generate()`` gets a fresh middleware instance with its own message
queue; ``wrap_generate`` drains queued messages into the request before the
next model call.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel as PydanticBaseModel, Field

from genkit import (
    Message,
    ModelResponse,
    ModelResponseChunk,
    MultipartToolResponse,
    Part,
    Role,
    Tool,
)
from genkit.middleware import (
    GenerateHookParams,
    Middleware,
    MiddlewareContext,
    ToolHookParams,
)
from genkit.plugin_api import GenkitConfig

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
    offset: int = Field(default=0, ge=0, description='1-based line to start from; 0 reads from the top.')
    limit: int = Field(default=0, ge=0, description='Number of lines to read; 0 reads to the end.')


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


_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB — absolute ceiling for reading
_MAX_READ_SLICE_BYTES = 256 * 1024  # 256 KB — max bytes returned per slice


def _attr_text(value: str) -> str:
    """Escape text placed in a double-quoted attribute of the ``<read_file>`` wrapper."""
    return (
        value
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace('\n', ' ')
        .replace('\r', ' ')
    )


class FilesystemConfig(GenkitConfig):
    """Sandbox root and write/tool naming options."""

    # The wire key is rootDirectory (matches JS and Go); Python keeps root_dir.
    # alias= would rename the kwarg for type checkers, so use the split aliases.
    root_dir: str = Field(
        validation_alias=AliasChoices('rootDirectory', 'root_dir'),
        serialization_alias='rootDirectory',
    )
    allow_write_access: bool = False
    tool_name_prefix: str = ''


class Filesystem(Middleware[FilesystemConfig]):
    """Filesystem middleware with sandboxed file operations.

    Contributes ``list_files``, ``read_file``, and optionally ``write_file``
    and ``edit_file``. Tool errors are returned as the tool response so the
    model can self-correct on the next turn.
    """

    name = 'filesystem'
    description = 'Sandboxed filesystem operations'

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        if not self.config.root_dir or not self.config.root_dir.strip():
            raise ValueError('Filesystem.root_dir must not be empty.')
        # One queue per generate() — the engine copies middleware per call.
        self._message_queue: list[Message] = []

    @property
    def _root_abs(self) -> str:
        return str(Path(self.config.root_dir).resolve())

    def _tool_name(self, base: str) -> str:
        return f'{self.config.tool_name_prefix}{base}'

    def _filesystem_tool_names(self) -> frozenset[str]:
        names = {self._tool_name('list_files'), self._tool_name('read_file')}
        if self.config.allow_write_access:
            names |= {self._tool_name('write_file'), self._tool_name('edit_file')}
        return frozenset(names)

    def _contains(self, abs_path: str) -> bool:
        """Whether a resolved absolute path is the root or under it."""
        c_norm = os.path.normcase(abs_path)
        root_norm = os.path.normcase(self._root_abs)
        return c_norm == root_norm or c_norm.startswith(root_norm.rstrip(os.sep) + os.sep)

    def _resolve_safe(self, path: str) -> str:
        """Resolve a model-supplied path to a real absolute path inside the root.

        A relative path resolves against the root. An absolute path is used
        as-is and must already be inside the root: ``/etc/hosts`` is refused,
        not rewritten to ``{root}/etc/hosts``. Raises ValueError otherwise.
        """
        path = path.strip()
        if '\x00' in path:
            raise ValueError(f'Path {path!r} contains a NUL byte.')
        candidate = os.path.realpath(os.path.join(self._root_abs, path or '.'))
        if not self._contains(candidate):
            if os.path.isabs(path):
                raise ValueError(f'Absolute path {path!r} escapes the root directory; use a path relative to the root.')
            raise ValueError(f'Path {path!r} escapes the root directory.')
        return candidate

    def _redact(self, text: str) -> str:
        """Rewrite host root paths in an error message relative to the root."""
        roots: set[str] = {self._root_abs, os.path.abspath(self.config.root_dir)}
        for root in sorted(roots, key=lambda r: len(r), reverse=True):
            if root == os.path.dirname(root):
                continue  # A filesystem root ('/') is no secret, and replacing it mangles every path.
            text = text.replace(root + os.sep, '').replace(root, '.')
        return text

    def _describe(self, abs_path: str) -> tuple[bool, int] | None:
        """Return ``(is_directory, size_bytes)`` for a listing entry.

        Uses ``lstat``, so a symlink is described by its target only when the
        target is inside the root. A symlink out of the root, or an entry that
        can't be stat'ed, is left out instead of leaking the target's type and
        size.
        """
        try:
            st = os.lstat(abs_path)
            if stat.S_ISLNK(st.st_mode):
                target = os.path.realpath(abs_path)
                if not self._contains(target):
                    return None
                st = os.stat(target)
        except OSError:
            return None
        is_dir = stat.S_ISDIR(st.st_mode)
        return is_dir, 0 if is_dir else st.st_size

    def _enqueue_parts(self, parts: list[Part]) -> None:
        """Append parts to the pending user message for the next model turn."""
        if self._message_queue and self._message_queue[-1].role == Role.USER:
            self._message_queue[-1].content.extend(parts)
        else:
            self._message_queue.append(Message(role=Role.USER, content=list(parts)))

    def _list_files(self, dir_path: str = '', recursive: bool = False) -> list[dict[str, Any]]:
        """List files and directories under ``dir_path`` (relative to root)."""
        abs_dir = self._resolve_safe(dir_path)
        if not os.path.isdir(abs_dir):
            raise ValueError(f'Not a directory: {dir_path!r}')

        results: list[dict[str, Any]] = []
        if recursive:
            # os.walk doesn't follow links: a symlinked directory is listed, never descended.
            for root, dirs, files in os.walk(abs_dir):
                dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
                for name in sorted(files):
                    abs_path = os.path.join(root, name)
                    info = self._describe(abs_path)
                    if info is None:
                        continue
                    is_dir, size = info
                    rel = os.path.relpath(abs_path, abs_dir)
                    results.append({'path': rel, 'is_directory': is_dir, 'size_bytes': size})
                for name in dirs:
                    abs_path = os.path.join(root, name)
                    if self._describe(abs_path) is None:
                        continue
                    rel = os.path.relpath(abs_path, abs_dir)
                    results.append({'path': rel, 'is_directory': True, 'size_bytes': 0})
        else:
            for name in sorted(os.listdir(abs_dir)):
                info = self._describe(os.path.join(abs_dir, name))
                if info is None:
                    continue
                is_dir, size = info
                results.append({'path': name, 'is_directory': is_dir, 'size_bytes': size})

        return results

    def _read_file_impl(self, file_path: str, offset: int, limit: int) -> str:
        """Read a file and enqueue its content as a user message."""
        if offset < 0 or limit < 0:
            raise ValueError(f'offset and limit must be >= 0 (got offset={offset}, limit={limit}).')
        abs_path = self._resolve_safe(file_path)
        if not os.path.isfile(abs_path):
            raise ValueError(f'File not found: {file_path!r}')

        stat_result = os.stat(abs_path)
        if stat_result.st_size > _MAX_FILE_SIZE_BYTES:
            raise ValueError(f'File too large ({stat_result.st_size:,} bytes; max {_MAX_FILE_SIZE_BYTES:,}).')

        mime_type, _ = mimetypes.guess_type(abs_path)
        is_image = bool(mime_type and mime_type.startswith('image/'))

        if is_image:
            with open(abs_path, 'rb') as fh:
                raw = fh.read()
            if len(raw) > _MAX_READ_SLICE_BYTES:
                raise ValueError(f'Image too large ({len(raw):,} bytes; max {_MAX_READ_SLICE_BYTES:,}).')
            b64 = base64.b64encode(raw).decode('ascii')
            data_uri = f'data:{mime_type};base64,{b64}'
            self._enqueue_parts([Part.from_media(data_uri, content_type=mime_type)])
            return f'Image {file_path} queued as media part.'

        with open(abs_path, encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()

        total = len(lines)
        start = max(0, offset - 1) if offset > 0 else 0
        end = total if limit == 0 else min(total, start + limit)
        sliced = ''.join(lines[start:end])

        if len(sliced.encode()) > _MAX_READ_SLICE_BYTES:
            raise ValueError(f'Slice too large ({len(sliced):,} chars). Use offset/limit to read smaller sections.')

        path_attr = _attr_text(file_path)
        if offset > 0 or limit > 0:
            wrapped = f'<read_file path="{path_attr}" lines="{start + 1}-{end}">\n{sliced}\n</read_file>'
        else:
            wrapped = f'<read_file path="{path_attr}" totalLines="{total}">\n{sliced}\n</read_file>'

        self._enqueue_parts([Part.from_text(wrapped)])
        return f'File {file_path} read successfully. Content queued as user message.'

    def _write_file_impl(self, file_path: str, content: str) -> str:
        abs_path = self._resolve_safe(file_path)
        os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        return f'File {file_path} written successfully.'

    def _edit_file_impl(self, file_path: str, edits: list[dict[str, Any]]) -> str:
        abs_path = self._resolve_safe(file_path)
        if not os.path.isfile(abs_path):
            raise ValueError(f'File not found: {file_path!r}')

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
        return f'File {file_path} edited successfully.'

    def tools(self, ctx: MiddlewareContext) -> list[Tool]:
        """Return filesystem tools for this generate() call."""

        async def list_files(input: _ListFilesInput) -> list[dict[str, Any]]:
            return await asyncio.to_thread(self._list_files, input.dir_path, input.recursive)

        async def read_file(input: _ReadFileInput) -> str:
            return await asyncio.to_thread(
                self._read_file_impl,
                input.file_path,
                input.offset,
                input.limit,
            )

        tools_out = [
            Tool(
                list_files,
                name=self._tool_name('list_files'),
                description='List files and directories under a path (optional recursive).',
            ),
            Tool(
                read_file,
                name=self._tool_name('read_file'),
                description='Read a text file, optionally from an offset/limit in lines.',
            ),
        ]

        if self.config.allow_write_access:

            async def write_file(input: _WriteFileInput) -> str:
                return await asyncio.to_thread(self._write_file_impl, input.file_path, input.content)

            async def edit_file(input: _EditFileInput) -> str:
                return await asyncio.to_thread(
                    self._edit_file_impl,
                    input.file_path,
                    [e.model_dump() for e in input.edits],
                )

            tools_out += [
                Tool(
                    write_file,
                    name=self._tool_name('write_file'),
                    description='Create or overwrite a text file with the given content.',
                ),
                Tool(
                    edit_file,
                    name=self._tool_name('edit_file'),
                    description='Apply search/replace edits to an existing text file.',
                ),
            ]

        return tools_out

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[GenerateHookParams, MiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Drain queued user messages into the request before the next model turn."""
        if not self._message_queue:
            return await next_fn(params, ctx)

        message_index = params.message_index
        if ctx.on_chunk:
            for msg in self._message_queue:
                ctx.send_chunk(ModelResponseChunk(role=msg.role, content=msg.content, index=message_index))
                message_index += 1

        new_options = params.options.model_copy()
        new_options.messages = [*params.options.messages, *self._message_queue]
        self._message_queue.clear()

        params = params.model_copy(
            update={
                'options': new_options,
                'message_index': message_index,
            }
        )
        return await next_fn(params, ctx)

    async def wrap_tool(
        self,
        params: ToolHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[ToolHookParams, MiddlewareContext], Awaitable[MultipartToolResponse]],
    ) -> MultipartToolResponse:
        """Return filesystem tool errors as the tool response, host paths redacted.

        The error stays in the tool turn, as in JS. Queueing it as a user
        message put a user turn between the tool response and the next model
        turn, and carried the absolute host root into the conversation.
        """
        if params.tool.name not in self._filesystem_tool_names():
            return await next_fn(params, ctx)

        try:
            return await next_fn(params, ctx)
        except Exception as exc:
            return MultipartToolResponse(output=f'Tool "{params.tool.name}" failed: {self._redact(str(exc))}')
