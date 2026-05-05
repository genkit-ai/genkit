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
available skills. Provides a use_skill tool for loading skill instructions into
the conversation context.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import Field, PrivateAttr

from genkit._core._model import GenerateHookParams, ModelResponse
from genkit.middleware import BaseMiddleware


class Skills(BaseMiddleware):
    """Skills middleware that exposes SKILL.md files as loadable instructions.

    Scans directories for subdirectories containing SKILL.md files. Each skill is
    exposed via a system prompt that lists available skills and their descriptions.
    The use_skill tool (when available) allows models to load full skill content.

    Skills are scanned once at initialization and cached for reuse across generate calls.
    """

    name: ClassVar[str] = 'middleware/skills'
    description: ClassVar[str | None] = 'Provides access to skill library for specialized instructions'

    skill_paths: list[str] = Field(default_factory=lambda: ['skills'])

    # Private cached state
    _skills_info: dict[str, dict[str, str]] | None = PrivateAttr(default=None)

    def _scan_skills(self) -> dict[str, dict[str, str]]:
        """Scan skill directories and return {skill_name: {path, description}}."""
        skills = {}

        for path_str in self.skill_paths:
            path = Path(path_str).resolve()
            if not path.is_dir():
                continue

            # List immediate subdirectories (non-hidden)
            for subdir in path.iterdir():
                if not subdir.is_dir() or subdir.name.startswith('.'):
                    continue

                skill_file = subdir / 'SKILL.md'
                if not skill_file.is_file():
                    continue

                # Parse frontmatter and extract description
                name, description = self._parse_skill_file(skill_file)
                if not name:
                    name = subdir.name

                skills[name] = {
                    'path': str(skill_file),
                    'description': description or 'No description provided.',
                }

        return skills

    def _parse_skill_file(self, path: Path) -> tuple[str, str]:
        """Parse SKILL.md frontmatter and return (name, description)."""
        try:
            content = path.read_text(encoding='utf-8')
        except Exception:
            return '', ''

        # Strip optional BOM
        content = content.lstrip('\ufeff')

        # Check for YAML frontmatter
        if not content.startswith('---\n'):
            return '', ''

        # Find end of frontmatter
        end_idx = content.find('\n---', 4)
        if end_idx == -1:
            return '', ''

        frontmatter = content[4:end_idx]
        try:
            data = yaml.safe_load(frontmatter)
            if not isinstance(data, dict):
                return '', ''
            return data.get('name', ''), data.get('description', '')
        except Exception:
            return '', ''

    def _get_skills_info(self) -> dict[str, dict[str, str]]:
        """Get or initialize cached skills info."""
        if self._skills_info is None:
            self._skills_info = self._scan_skills()
        return self._skills_info

    def _build_skills_prompt(self, info: dict[str, dict[str, str]]) -> str:
        """Build the skills system prompt from scanned info."""
        if not info:
            return ''

        lines = [
            '<skills>',
            'You have access to a library of skills that serve as specialized instructions/personas.',
            'Strongly prefer to use them when working on anything related to them.',
            'Only use them once to load the context.',
            'Here are the available skills:',
        ]

        for name in sorted(info.keys()):
            desc = info[name]['description']
            lines.append(f' - {name} - {desc}')

        lines.append('</skills>')
        return '\n'.join(lines)

    def _inject_skills_prompt(self, request: Any, prompt_text: str) -> Any:
        """Inject skills prompt into request as a system message part with marker metadata.

        Returns a shallow copy of the request with the prompt injected.
        """
        # Import here to avoid circular imports
        from genkit._core._typing import MessageData, Part, Role, TextPart

        # Make a shallow copy of the request
        request = request.model_copy(deep=False)

        # Find or create system message
        messages = list(request.messages)
        system_msg = None
        system_idx = None

        for i, msg in enumerate(messages):
            if msg.role == Role.SYSTEM:
                system_msg = msg
                system_idx = i
                break

        # Check if we already have the skills prompt
        marker_metadata = {'skills-instructions': True}

        if system_msg:
            # Check if identical prompt already exists
            for part in system_msg.content:
                if isinstance(part.root, TextPart):
                    if (
                        part.root.metadata
                        and part.root.metadata.get('skills-instructions')
                        and part.root.text == prompt_text
                    ):
                        # Already present, no change needed
                        return request

            # Add or update the skills prompt
            new_content = []
            found = False
            for part in system_msg.content:
                if isinstance(part.root, TextPart) and part.root.metadata and part.root.metadata.get('skills-instructions'):
                    # Replace existing skills prompt
                    new_content.append(Part(root=TextPart(text=prompt_text, metadata=marker_metadata)))
                    found = True
                else:
                    new_content.append(part)

            if not found:
                # Append new skills prompt
                new_content.append(Part(root=TextPart(text=prompt_text, metadata=marker_metadata)))

            # Update the system message
            system_msg = MessageData(role=Role.SYSTEM, content=new_content)
            messages[system_idx] = system_msg
        else:
            # Create a new system message
            part = Part(root=TextPart(text=prompt_text, metadata=marker_metadata))
            system_msg = MessageData(role=Role.SYSTEM, content=[part])
            messages.insert(0, system_msg)

        request.messages = messages
        return request

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        next_fn,
    ) -> ModelResponse:
        """Inject skills prompt and use_skill tool (if possible) into the request."""
        info = self._get_skills_info()
        if not info:
            return await next_fn(params)

        # Build and inject skills prompt
        prompt_text = self._build_skills_prompt(info)
        if prompt_text:
            params.request = self._inject_skills_prompt(params.request, prompt_text)

        # TODO: Inject use_skill tool — requires wrap_generate tool injection API.
        # The use_skill tool should accept a skill_name string and return the full
        # SKILL.md content. Current middleware API does not provide a clean way to
        # inject tools into params.request.tools or params.options.tools from wrap_generate.
        # Once the API supports tool injection, implement:
        # - Tool definition: name='use_skill', input schema for skill_name
        # - Tool implementation: read and return skills_info[skill_name]['path'] content
        # - Inject into params.request or params.options

        return await next_fn(params)
