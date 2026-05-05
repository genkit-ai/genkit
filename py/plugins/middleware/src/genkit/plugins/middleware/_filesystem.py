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
confined to a root directory. Tracks file state to detect external modifications and
avoid redundant reads. Queues file content as user messages for model context.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, PrivateAttr

from genkit._ai._tools import Interrupt
from genkit._core._model import GenerateHookParams, ModelResponse, ToolHookParams
from genkit._core._typing import MessageData, Part, Role, TextPart, ToolRequest, ToolRequestPart, ToolResponse, ToolResponsePart
from genkit.middleware import BaseMiddleware


# Constants
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_READ_SLICE_BYTES = 256 * 1024  # 256 KB
MAX_CACHE_ENTRIES = 200


class EditSpec(dict):
    """Edit specification for edit_file tool."""
    old_string: str
    new_string: str
    replace_all: bool = False


class Filesystem(BaseMiddleware):
    """Filesystem middleware with sandboxed file operations.

    Provides list_files, read_file, write_file (if allow_write_access=True), and
    edit_file (if allow_write_access=True) tools. All paths are sandboxed to root_dir.
    File content is queued as user messages for model context instead of inline tool
    responses. Tracks file state to detect external modifications and avoid redundant reads.

    Note: Tool injection from wrap_generate requires API support for dynamic tool registration.
    Current implementation provides the logic but may need integration updates.
    """

    name: ClassVar[str] = 'middleware/filesystem'
    description: ClassVar[str | None] = 'Sandboxed filesystem operations'

    root_dir: str
    allow_write_access: bool = False
    tool_name_prefix: str = ''

    # Private state
    _file_cache: OrderedDict[str, tuple[float, int, int, int]] = PrivateAttr(default_factory=OrderedDict)
    _queue: list[MessageData] = PrivateAttr(default_factory=list)
    _queue_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _cache_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _tool_names: set[str] = PrivateAttr(default_factory=set)
    _root_abs: str = PrivateAttr(default='')

    def model_post_init(self, __context: Any) -> None:
        """Initialize absolute root path and tool names after pydantic validation."""
        super().model_post_init(__context)
        self._root_abs = str(Path(self.root_dir).resolve())
        # Build tool name set
        self._tool_names = {
            f'{self.tool_name_prefix}list_files',
            f'{self.tool_name_prefix}read_file',
        }
        if self.allow_write_access:
            self._tool_names.add(f'{self.tool_name_prefix}write_file')
            self._tool_names.add(f'{self.tool_name_prefix}edit_file')

    def _resolve_safe(self, rel: str) -> str:
        """Resolve relative path to absolute, raising ValueError if it escapes root."""
        rel = rel.strip().lstrip('/')
        if not rel:
            rel = '.'
        candidate = os.path.realpath(os.path.join(self._root_abs, rel))
        if not candidate.startswith(self._root_abs + os.sep) and candidate != self._root_abs:
            raise ValueError(f'Path {rel!r} escapes root directory')
        return candidate

    def _enqueue_user_text(self, text: str) -> None:
        """Queue a text user message."""
        part = Part(root=TextPart(text=text))
        msg = MessageData(role=Role.USER, content=[part])
        with self._queue_lock:
            self._queue.append(msg)

    def _enqueue_user_data_uri(self, data_uri: str, description: str) -> None:
        """Queue a user message with a data URI (for images)."""
        # In Genkit, media parts typically use MediaPart, but for simplicity
        # we can queue as text with the data URI embedded.
        # The actual representation depends on the model's handling.
        text = f'{description}\n{data_uri}'
        self._enqueue_user_text(text)

    def _get_file_state(self, abs_path: str) -> tuple[float, int] | None:
        """Get (mtime, size) from cache, or None if not cached."""
        with self._cache_lock:
            entry = self._file_cache.get(abs_path)
            if entry:
                return (entry[0], entry[1])
        return None

    def _set_file_state(self, abs_path: str, mtime: float, size: int, offset: int, limit: int) -> None:
        """Update file state in cache."""
        with self._cache_lock:
            self._file_cache[abs_path] = (mtime, size, offset, limit)
            # Evict oldest if cache exceeds max size
            while len(self._file_cache) > MAX_CACHE_ENTRIES:
                self._file_cache.popitem(last=False)

    def _list_files(self, dir_path: str = '', recursive: bool = False) -> list[dict[str, Any]]:
        """List files in directory (relative to root)."""
        abs_dir = self._resolve_safe(dir_path)
        if not os.path.isdir(abs_dir):
            raise ValueError(f'Not a directory: {dir_path!r}')

        results = []
        if recursive:
            for root, dirs, files in os.walk(abs_dir):
                # Filter hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for name in files:
                    abs_path = os.path.join(root, name)
                    try:
                        stat = os.stat(abs_path)
                        rel_path = os.path.relpath(abs_path, self._root_abs)
                        results.append({'path': rel_path, 'is_directory': False, 'size_bytes': stat.st_size})
                    except Exception:
                        continue
                for name in dirs:
                    abs_path = os.path.join(root, name)
                    rel_path = os.path.relpath(abs_path, self._root_abs)
                    results.append({'path': rel_path, 'is_directory': True, 'size_bytes': 0})
        else:
            for name in os.listdir(abs_dir):
                abs_path = os.path.join(abs_dir, name)
                try:
                    stat = os.stat(abs_path)
                    rel_path = os.path.relpath(abs_path, self._root_abs)
                    is_dir = os.path.isdir(abs_path)
                    results.append({'path': rel_path, 'is_directory': is_dir, 'size_bytes': 0 if is_dir else stat.st_size})
                except Exception:
                    continue

        return results

    def _read_file(self, file_path: str, offset: int = 0, limit: int = 0) -> str:
        """Read file content (with optional line slicing). Queue content as user message."""
        abs_path = self._resolve_safe(file_path)
        if not os.path.isfile(abs_path):
            raise ValueError(f'File not found: {file_path!r}')

        stat = os.stat(abs_path)
        if stat.st_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f'File too large ({stat.st_size} bytes, max {MAX_FILE_SIZE_BYTES})')

        # Check for dedup
        cached_state = self._get_file_state(abs_path)
        if cached_state:
            cached_mtime, cached_size = cached_state
            if cached_mtime == stat.st_mtime and cached_size == stat.st_size:
                # Check if offset/limit match
                with self._cache_lock:
                    entry = self._file_cache.get(abs_path)
                    if entry and entry[2] == offset and entry[3] == limit:
                        return 'File unchanged since last read. The content from the earlier read_file result in this conversation is still current — refer to that instead of re-reading.'

        # Detect mime type
        mime_type, _ = mimetypes.guess_type(abs_path)
        is_image = mime_type and mime_type.startswith('image/')

        if is_image:
            # Read as binary and encode as data URI
            with open(abs_path, 'rb') as f:
                data = f.read()
            b64_data = base64.b64encode(data).decode('ascii')
            data_uri = f'data:{mime_type};base64,{b64_data}'
            description = f'File {file_path} read successfully, see contents below.'
            self._enqueue_user_data_uri(data_uri, description)
            self._set_file_state(abs_path, stat.st_mtime, stat.st_size, offset, limit)
            return description

        # Text file
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        total_lines = len(lines)
        start_line = 0 if offset == 0 else offset - 1  # offset is 1-indexed
        end_line = total_lines if limit == 0 else start_line + limit

        if start_line < 0:
            start_line = 0
        if end_line > total_lines:
            end_line = total_lines

        sliced_lines = lines[start_line:end_line]
        content_text = ''.join(sliced_lines)

        if len(content_text) > MAX_READ_SLICE_BYTES:
            raise ValueError(
                f'File slice too large ({len(content_text)} bytes, max {MAX_READ_SLICE_BYTES}). '
                f'Use offset and limit parameters to read smaller sections.'
            )

        # Wrap content
        if offset > 0 or limit > 0:
            wrapped = f'<read_file path="{file_path}" lines="{start_line + 1}-{end_line}">\n{content_text}\n</read_file>'
        else:
            wrapped = f'<read_file path="{file_path}" totalLines="{total_lines}">\n{content_text}\n</read_file>'

        self._enqueue_user_text(wrapped)
        self._set_file_state(abs_path, stat.st_mtime, stat.st_size, offset, limit)
        return f'File {file_path} read successfully. Content queued as user message.'

    def _write_file(self, file_path: str, content: str) -> str:
        """Write file content (requires prior read unless new file)."""
        abs_path = self._resolve_safe(file_path)
        exists = os.path.isfile(abs_path)

        if exists:
            # Require prior read
            cached = self._get_file_state(abs_path)
            if not cached:
                raise ValueError(f'File must be read before writing: {file_path!r}')

            # Check if externally modified
            stat = os.stat(abs_path)
            if cached[0] != stat.st_mtime or cached[1] != stat.st_size:
                raise ValueError(f'File was modified externally since last read: {file_path!r}. Re-read before writing.')

        # Create parent directories if needed
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)

        stat = os.stat(abs_path)
        self._set_file_state(abs_path, stat.st_mtime, stat.st_size, 0, 0)
        return f'File {file_path} written successfully.'

    def _edit_file(self, file_path: str, edits: list[dict[str, Any]]) -> str:
        """Apply string replacement edits to file."""
        abs_path = self._resolve_safe(file_path)
        if not os.path.isfile(abs_path):
            raise ValueError(f'File not found: {file_path!r}')

        # Require prior read
        cached = self._get_file_state(abs_path)
        if not cached:
            raise ValueError(f'File must be read before editing: {file_path!r}')

        # Check if externally modified
        stat = os.stat(abs_path)
        if cached[0] != stat.st_mtime or cached[1] != stat.st_size:
            raise ValueError(f'File was modified externally since last read: {file_path!r}. Re-read before editing.')

        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Apply edits sequentially
        for edit_spec in edits:
            old_string = edit_spec.get('old_string', '')
            new_string = edit_spec.get('new_string', '')
            replace_all = edit_spec.get('replace_all', False)

            if not old_string:
                raise ValueError('old_string must be non-empty')
            if old_string == new_string:
                raise ValueError('old_string and new_string must differ')

            count = content.count(old_string)
            if count == 0:
                raise ValueError(f'old_string not found in file: {old_string!r}')
            if not replace_all and count > 1:
                raise ValueError(f'old_string matches {count} times, but replace_all=False')

            if replace_all:
                content = content.replace(old_string, new_string)
            else:
                content = content.replace(old_string, new_string, 1)

        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)

        stat = os.stat(abs_path)
        self._set_file_state(abs_path, stat.st_mtime, stat.st_size, 0, 0)
        return f'File {file_path} edited successfully.'

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        next_fn,
    ) -> ModelResponse:
        """Drain queued messages and prepend to request before calling next_fn."""
        with self._queue_lock:
            queued = self._queue[:]
            self._queue.clear()

        if queued:
            # Make a shallow copy and prepend queued messages
            params.request = params.request.model_copy(deep=False)
            params.request.messages = list(params.request.messages) + queued

        # TODO: Inject filesystem tools into params.request or params.options.
        # Current middleware API does not provide a clean mechanism for dynamically
        # registering tools from wrap_generate. Once tool injection is supported:
        # - Define list_files, read_file, write_file, edit_file tool schemas
        # - Attach tool implementations (self._list_files, self._read_file, etc.)
        # - Inject into params.request.tools or params.options.tools

        return await next_fn(params)

    async def wrap_tool(
        self,
        params: ToolHookParams,
        next_fn,
    ):
        """Intercept filesystem tool calls and convert errors to queued user messages."""
        if params.tool.name not in self._tool_names:
            return await next_fn(params)

        try:
            result = await next_fn(params)
            return result
        except Interrupt:
            raise
        except Exception as e:
            error_msg = f'Tool "{params.tool.name}" failed: {e}'
            self._enqueue_user_text(error_msg)
            # Return a minimal tool response so the model knows the call was attempted
            return (
                ToolResponsePart(
                    tool_response=ToolResponse(
                        name=params.tool_request_part.tool_request.name,
                        output='Tool call failed; see user message for details.',
                    )
                ),
                None,
            )
