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

"""Skills middleware for Genkit.

Scans skill directories for SKILL.md files and injects a system prompt describing
available skills. Provides a ``use_skill`` tool for loading full skill instructions
into the conversation context on demand.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import Field

from genkit._ai._model import Message
from genkit._ai._tools import define_tool
from genkit._core._model import ModelRequest, ModelResponse
from genkit._core._registry import Registry
from genkit._core._typing import Part, Role, TextPart
from genkit.middleware import BaseMiddleware, GenerateHookParams

# Marker placed in TextPart.metadata so later iterations can find and replace
# the skills block without duplicating it.
_SKILLS_MARKER = 'skills-instructions'

_MISSING_DESCRIPTION = 'No description provided.'


class Skills(BaseMiddleware):
    """Skills middleware that exposes SKILL.md files as loadable instructions.

    Scans directories for subdirectories containing SKILL.md files. Each skill is
    exposed via a system prompt that lists available skills and their descriptions.
    A ``use_skill`` tool is contributed per generate call so the model can load the
    full SKILL.md content on demand.

    Skills are scanned once per generate() call (inside ``tools()``) so filesystem
    changes between calls are always picked up.
    """

    name: ClassVar[str] = 'skills'
    description: ClassVar[str | None] = 'Provides access to skill library for specialized instructions'

    skill_paths: list[str] = Field(default_factory=lambda: ['skills'])

    # --- skill scanning ---

    def _scan_skills(self) -> dict[str, dict[str, str]]:
        """Scan skill directories and return ``{skill_name: {path, description}}``."""
        skills: dict[str, dict[str, str]] = {}
        for path_str in self.skill_paths:
            path = Path(path_str).resolve()
            if not path.is_dir():
                continue
            for subdir in sorted(path.iterdir()):
                if not subdir.is_dir() or subdir.name.startswith('.'):
                    continue
                skill_file = subdir / 'SKILL.md'
                if not skill_file.is_file():
                    continue
                name, description = self._parse_skill_file(skill_file)
                if not name:
                    name = subdir.name
                skills[name] = {
                    'path': str(skill_file),
                    'description': description or '',
                }
        return skills

    def _parse_skill_file(self, path: Path) -> tuple[str, str]:
        """Return ``(name, description)`` from SKILL.md YAML frontmatter."""
        try:
            content = path.read_text(encoding='utf-8').lstrip('\ufeff')
        except Exception:
            return '', ''
        if not content.startswith('---\n'):
            return '', ''
        end_idx = content.find('\n---', 4)
        if end_idx == -1:
            return '', ''
        try:
            data = yaml.safe_load(content[4:end_idx])
            if not isinstance(data, dict):
                return '', ''
            return data.get('name', ''), data.get('description', '')
        except Exception:
            return '', ''

    # --- prompt injection ---

    def _build_skills_prompt(self, skills: dict[str, dict[str, str]]) -> str:
        """Build the skills system prompt, omitting skills with no description."""
        if not skills:
            return ''
        lines = [
            '<skills>',
            'You have access to a library of skills that serve as specialized instructions/personas.',
            'Strongly prefer to use them when working on anything related to them.',
            'Only use them once to load the context.',
            'Here are the available skills:',
        ]
        for skill_name in sorted(skills.keys()):
            desc = skills[skill_name]['description']
            if desc and desc != _MISSING_DESCRIPTION:
                lines.append(f' - {skill_name} - {desc}')
            else:
                lines.append(f' - {skill_name}')
        lines.append('</skills>')
        return '\n'.join(lines)

    def _inject_skills_prompt(self, request: ModelRequest, prompt_text: str) -> ModelRequest:
        """Inject or refresh skills prompt in the system message."""
        messages = list(request.messages)
        system_idx: int | None = None
        for i, msg in enumerate(messages):
            if msg.role == Role.SYSTEM:
                system_idx = i
                break

        marker_meta: dict[str, Any] = {_SKILLS_MARKER: True}
        new_part = Part(root=TextPart(text=prompt_text, metadata=marker_meta))

        if system_idx is not None:
            msg = messages[system_idx]
            # Replace existing skills part or append.
            new_content = []
            replaced = False
            for part in msg.content:
                meta = part.root.metadata if isinstance(part.root, TextPart) else None
                if isinstance(meta, dict) and meta.get(_SKILLS_MARKER):
                    new_content.append(new_part)
                    replaced = True
                else:
                    new_content.append(part)
            if not replaced:
                new_content.append(new_part)
            messages[system_idx] = Message(role=Role.SYSTEM, content=new_content)
        else:
            messages.insert(0, Message(role=Role.SYSTEM, content=[new_part]))

        return request.model_copy(update={'messages': messages})

    # --- middleware hooks ---

    def tools(self, enqueue_parts: Callable[[list[Part]], None] | None = None) -> list[Any]:
        """Return a ``use_skill`` action scoped to the current generate call.

        Skills are scanned fresh inside ``use_skill`` so SKILL.md files added or
        modified mid-conversation are picked up — matches ``wrap_generate``,
        which re-scans each turn to refresh the system prompt.
        """
        # Initial scan only decides whether to expose the tool at all; the
        # closure rescans on every invocation so the model sees up-to-date
        # skills even within a single generate() call.
        if not self._scan_skills():
            return []

        scratch = Registry()

        async def use_skill(skill_name: str) -> str:
            """Load the full instructions for a named skill.

            Args:
                skill_name: The name of the skill to load (as listed in the system prompt).

            Returns:
                The full SKILL.md content for that skill.
            """
            skills = self._scan_skills()
            info = skills.get(skill_name)
            if info is None:
                available = ', '.join(sorted(skills.keys()))
                return f'Unknown skill "{skill_name}". Available skills: {available}'
            try:
                skill_path = Path(info['path'])
                return skill_path.read_text(encoding='utf-8')  # noqa: ASYNC240
            except Exception as exc:
                return f'Failed to read skill "{skill_name}": {exc}'

        t = define_tool(scratch, use_skill, name='use_skill')
        return [t.action()]

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        next_fn: Callable[[GenerateHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject the skills system prompt before each generate iteration."""
        skills = self._scan_skills()
        if skills:
            prompt_text = self._build_skills_prompt(skills)
            if prompt_text:
                new_req = self._inject_skills_prompt(params.request, prompt_text)
                params = params.model_copy(update={'request': new_req})
        return await next_fn(params)
