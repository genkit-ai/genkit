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

"""Python in-process conformance executor.

Runs model actions in-process via ``ai.generate()`` by importing the
plugin's conformance entry point and calling its ``Genkit`` instance
directly.  This exercises the full framework pipeline (format parsing,
``extract_json``, output processing) — matching the real user experience.

Protocol (in-process — no subprocess):

1.  Imports the conformance entry point module.
2.  Extracts the ``ai`` (``Genkit``) instance from module scope.
3.  For each test, calls ``ai.generate()`` or ``ai.generate_stream()``.
4.  Returns serialized response and streaming chunks.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any, cast

from rich.console import Console

from genkit.codec import dump_dict

console = Console(stderr=True)


class InProcessRunner:
    """Run model actions in-process via ``ai.generate()``.

    The entry point (e.g. ``conformance_entry.py``) creates a ``Genkit``
    instance at module level, registering all model actions.  We import
    it, grab the ``ai`` instance, and call ``ai.generate()`` so the full
    framework pipeline (format parsing, ``extract_json``, output
    processing) is exercised — matching the real user experience.
    """

    def __init__(self, entry_path: Path) -> None:
        """Initialize with the path to the conformance entry point."""
        self._entry_path = entry_path
        self._ai: Any = None  # noqa: ANN401

    async def _load(self) -> None:
        """Import the entry point module and extract the Genkit instance."""
        if self._ai is not None:
            return

        # Import the module by file path without executing __main__ block.
        spec = importlib.util.spec_from_file_location(
            '_conform_entry',
            str(self._entry_path),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Cannot load entry point: {self._entry_path}')

        module = importlib.util.module_from_spec(spec)

        # Ensure the entry point's directory is on sys.path so relative
        # imports work (e.g. if the entry point imports sibling modules).
        entry_dir = str(self._entry_path.parent)
        if entry_dir not in sys.path:
            sys.path.insert(0, entry_dir)

        spec.loader.exec_module(module)

        # Find the Genkit instance — by convention it's named `ai`.
        ai = getattr(module, 'ai', None)
        if ai is None:
            raise RuntimeError(
                f'Entry point {self._entry_path} does not expose an `ai` (Genkit) instance at module level.'
            )
        self._ai = ai
        console.print('[dim]Loaded entry point in-process.[/dim]')

    async def run_action(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        stream: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run a model action via ``ai.generate()`` in-process.

        Converts the raw test input (GenerateRequest-shaped dict) into
        ``ai.generate()`` keyword arguments so the full framework
        pipeline is exercised, including output format handling
        (``extract_json`` for JSON output) and streaming chunk wrapping.
        """
        try:
            return await self._run_action_impl(key, input_data, stream=stream)
        except Exception:
            # Match the native subprocess executor's error contract:
            # wrap exceptions in RuntimeError with the full traceback so
            # that _run_single_test can display a clean error message
            # instead of receiving a raw GenkitError whose __str__
            # embeds a multi-line traceback in the message itself.
            raise RuntimeError(f'In-process executor error: {traceback.format_exc()}') from None

    async def _run_action_impl(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        stream: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Inner implementation of run_action (no exception wrapping)."""
        await self._load()

        # Extract model name from the action key.
        # e.g. "/model/googleai/gemini-2.5-flash" -> "googleai/gemini-2.5-flash"
        model_name = key.removeprefix('/model/')

        # Extract fields from the GenerateRequest-shaped input.
        messages = input_data.get('messages', [])
        output_config = input_data.get('output')
        tools_defs = input_data.get('tools')
        config = input_data.get('config')

        # Convert raw message dicts to Message objects.
        from genkit.core.typing import Message, OutputConfig, Part

        msg_objects = [Message.model_validate(m) for m in messages]

        # Separate system messages from user/model messages.
        # ai.generate() takes system as a separate argument.
        system_parts: list[Part] | None = None
        non_system_messages: list[Message] = []
        for msg in msg_objects:
            if msg.role == 'system':
                system_parts = msg.content
            else:
                non_system_messages.append(msg)

        # Build output config if present.
        output_obj: OutputConfig | None = None
        if output_config:
            output_obj = OutputConfig.model_validate(output_config)

        # For tool definitions: the test cases contain inline tool
        # definitions (with inputSchema), but ai.generate() resolves
        # tools by name from the registry.  We register ephemeral tools
        # in the registry so generate() can resolve them.
        tool_names: list[str] | None = None
        if tools_defs:
            tool_names = []
            for tdef in tools_defs:
                t_name = tdef['name']
                tool_names.append(t_name)
                # Register a no-op tool so the framework can resolve its
                # definition.  The conformance tests only check that the
                # model *requests* the tool — they never execute it.
                _register_ephemeral_tool(self._ai, t_name, tdef)

        chunks: list[dict[str, Any]] = []

        if stream:
            # Use generate_stream for streaming tests.
            stream_iter, response_future = self._ai.generate_stream(
                model=model_name,
                system=system_parts,
                messages=non_system_messages,
                tools=tool_names,
                config=config,
                output=output_obj,
                return_tool_requests=True,
            )
            # Consume the stream to collect chunks.
            async for chunk in stream_iter:
                chunks.append(cast(dict[str, Any], dump_dict(chunk)))
            result = await response_future
        else:
            result = await self._ai.generate(
                model=model_name,
                system=system_parts,
                messages=non_system_messages,
                tools=tool_names,
                config=config,
                output=output_obj,
                return_tool_requests=True,
            )

        response = cast(dict[str, Any], dump_dict(result))
        return response, chunks

    async def close(self) -> None:
        """No resources to clean up for in-process runner."""


def _register_ephemeral_tool(
    ai: Any,  # noqa: ANN401
    name: str,
    tool_def: dict[str, Any],
) -> None:
    """Register a no-op tool so generate() can resolve its definition.

    Conformance tests provide inline tool definitions.  ``ai.generate()``
    resolves tools by name from the registry, so we register lightweight
    stubs whose only purpose is to carry the correct ``input_schema``
    and ``description`` so ``to_tool_definition()`` can build the right
    ``ToolDefinition`` for the model.
    """
    from genkit.core.action.types import ActionKind

    registry = ai.registry

    # Only register if not already present.
    with registry._lock:  # pyright: ignore[reportPrivateUsage]
        tool_entries = registry._entries.get(cast(ActionKind, ActionKind.TOOL), {})  # pyright: ignore[reportPrivateUsage]
        if name in tool_entries:
            return

    description = tool_def.get('description', '')
    input_schema = tool_def.get('inputSchema')

    async def noop_tool(input_data: Any, ctx: Any) -> dict[str, str]:  # noqa: ANN401
        """No-op tool stub for conformance testing."""
        return {'result': 'noop'}

    action = registry.register_action(
        kind=cast(ActionKind, ActionKind.TOOL),
        name=name,
        fn=noop_tool,
        description=description,
        metadata={'tool': {'type': 'tool'}},
    )
    # Override input schema from the test definition so the model
    # receives the correct JSON Schema for this tool.
    if input_schema:
        action._input_schema = input_schema  # pyright: ignore[reportPrivateUsage]
