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

``read_file`` queues file content as user messages so the tool response stays
small. Tool errors are queued the same way so the model can self-correct on
the next turn.

Each ``generate()`` gets a fresh middleware instance with its own message
queue; ``wrap_generate`` drains queued messages into the request before the
next model call.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel as PydanticBaseModel

from genkit._ai._tools import Interrupt, define_tool
from genkit._core._model import Message, ModelResponse, ModelResponseChunk
from genkit._core._registry import Registry
from genkit._core._typing import (
    Media,
    MediaPart,
    Part,
    Role,
    TextPart,
)
from genkit.middleware import (
    BaseMiddleware,
    GenerateHookParams,
    GenerateMiddlewareContext,
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


_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB — absolute ceiling for reading
_MAX_READ_SLICE_BYTES = 256 * 1024  # 256 KB — max bytes returned per slice


class FilesystemConfig(PydanticBaseModel):
    """Sandbox root and write/tool naming options."""

    root_dir: str
    allow_write_access: bool = False
    tool_name_prefix: str = ''


class Filesystem(BaseMiddleware[FilesystemConfig]):
    """Filesystem middleware with sandboxed file operations.

    Contributes ``list_files``, ``read_file``, and optionally ``write_file``
    and ``edit_file``. Tool errors are queued as user messages so the model
    can self-correct on the next turn.
    """

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

    def _resolve_safe(self, rel: str) -> str:
        """Resolve ``rel`` to an absolute path, raising ValueError if it escapes root."""
        rel = rel.strip().lstrip('/').lstrip('\\')
        if not rel:
            rel = '.'
        candidate = os.path.realpath(os.path.join(self._root_abs, rel))
        c_norm = os.path.normcase(candidate)
        root_norm = os.path.normcase(self._root_abs)
        if c_norm != root_norm and not c_norm.startswith(root_norm + os.sep):
            raise ValueError(f'Path {rel!r} escapes the root directory.')
        return candidate

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

    def _read_file_impl(self, file_path: str, offset: int, limit: int) -> str:
        """Read a file and enqueue its content as a user message."""
        abs_path = self._resolve_safe(file_path)
        if not os.path.isfile(abs_path):
            raise ValueError(f'File not found: {file_path!r}')

        stat = os.stat(abs_path)
        if stat.st_size > _MAX_FILE_SIZE_BYTES:
            raise ValueError(f'File too large ({stat.st_size:,} bytes; max {_MAX_FILE_SIZE_BYTES:,}).')

        mime_type, _ = mimetypes.guess_type(abs_path)
        is_image = bool(mime_type and mime_type.startswith('image/'))

        if is_image:
            with open(abs_path, 'rb') as fh:
                raw = fh.read()
            if len(raw) > _MAX_READ_SLICE_BYTES:
                raise ValueError(f'Image too large ({len(raw):,} bytes; max {_MAX_READ_SLICE_BYTES:,}).')
            b64 = base64.b64encode(raw).decode('ascii')
            data_uri = f'data:{mime_type};base64,{b64}'
            self._enqueue_parts([Part(root=MediaPart(media=Media(url=data_uri, content_type=mime_type)))])
            return f'Image {file_path} queued as media part.'

        with open(abs_path, encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()

        total = len(lines)
        start = max(0, offset - 1) if offset > 0 else 0
        end = total if limit == 0 else min(total, start + limit)
        sliced = ''.join(lines[start:end])

        if len(sliced.encode()) > _MAX_READ_SLICE_BYTES:
            raise ValueError(f'Slice too large ({len(sliced):,} chars). Use offset/limit to read smaller sections.')

        if offset > 0 or limit > 0:
            wrapped = f'<read_file path="{file_path}" lines="{start + 1}-{end}">\n{sliced}\n</read_file>'
        else:
            wrapped = f'<read_file path="{file_path}" totalLines="{total}">\n{sliced}\n</read_file>'

        self._enqueue_parts([Part(root=TextPart(text=wrapped))])
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

    def tools(self, ctx: GenerateMiddlewareContext) -> list[Any]:
        """Return filesystem tool actions for this generate() call."""
        scratch = Registry()

        async def list_files(input: _ListFilesInput) -> list[dict[str, Any]]:
            return await asyncio.to_thread(self._list_files, input.dir_path, input.recursive)

        async def read_file(input: _ReadFileInput) -> str:
            return await asyncio.to_thread(
                self._read_file_impl,
                input.file_path,
                input.offset,
                input.limit,
            )

        t_list = define_tool(scratch, list_files, name=self._tool_name('list_files'))
        t_read = define_tool(scratch, read_file, name=self._tool_name('read_file'))
        tools_out = [t_list.action(), t_read.action()]

        if self.config.allow_write_access:

            async def write_file(input: _WriteFileInput) -> str:
                return await asyncio.to_thread(self._write_file_impl, input.file_path, input.content)

            async def edit_file(input: _EditFileInput) -> str:
                return await asyncio.to_thread(
                    self._edit_file_impl,
                    input.file_path,
                    [e.model_dump() for e in input.edits],
                )

            t_write = define_tool(scratch, write_file, name=self._tool_name('write_file'))
            t_edit = define_tool(scratch, edit_file, name=self._tool_name('edit_file'))
            tools_out += [t_write.action(), t_edit.action()]

        return tools_out

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Drain queued user messages into the request before the next model turn."""
        if not self._message_queue:
            return await next_fn(params, ctx)

        message_index = params.message_index
        if ctx.on_chunk:
            for msg in self._message_queue:
                ctx.send_chunk(ModelResponseChunk(role=msg.role, content=msg.content, index=message_index))
                message_index += 1

        new_request = params.request.model_copy()
        new_request.messages = [*params.request.messages, *self._message_queue]
        self._message_queue.clear()

        params = params.model_copy(
            update={
                'request': new_request,
                'message_index': message_index,
            }
        )
        return await next_fn(params, ctx)

    async def wrap_tool(
        self,
        params: ToolHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
    ) -> MultipartToolResponse:
        """Catch filesystem tool errors and enqueue them as user messages."""
        if params.tool.name not in self._filesystem_tool_names():
            return await next_fn(params, ctx)

        try:
            return await next_fn(params, ctx)
        except Interrupt:
            raise
        except Exception as exc:
            error_msg = f'Tool "{params.tool.name}" failed: {exc}'
            self._enqueue_parts([Part(root=TextPart(text=error_msg))])
            return MultipartToolResponse(output='Tool call failed; see user message below for details.')
