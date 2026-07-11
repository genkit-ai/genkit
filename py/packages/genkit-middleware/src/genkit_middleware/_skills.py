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

"""Skills middleware for Genkit."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel as PydanticBaseModel, Field

from genkit._ai._model import Message
from genkit._ai._tools import define_tool
from genkit._core._model import ModelRequest, ModelResponse
from genkit._core._registry import Registry
from genkit._core._typing import Part, Role, TextPart
from genkit.middleware import BaseMiddleware, GenerateHookParams, GenerateMiddlewareContext

_SKILLS_MARKER = 'skills-instructions'
_MISSING_DESCRIPTION = 'No description provided.'


class _UseSkillInput(PydanticBaseModel):
    """Input for the ``use_skill`` tool."""

    skill_name: str = Field(description='The name of the skill to load (as listed in the system prompt).')


class SkillsConfig(PydanticBaseModel):
    """Directories to scan for skill folders containing ``SKILL.md``."""

    skill_paths: list[str] = Field(default_factory=lambda: ['skills'])


class Skills(BaseMiddleware[SkillsConfig]):
    """Skills middleware that exposes ``SKILL.md`` files as loadable instructions."""

    def _scan_skills(self) -> dict[str, dict[str, str]]:
        skills: dict[str, dict[str, str]] = {}
        for path_str in self.config.skill_paths:
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
        try:
            content = path.read_text(encoding='utf-8').lstrip('\ufeff')
        except Exception:
            return '', ''
        if content.startswith('---\r\n'):
            start_idx = 5
            end_marker = '\r\n---'
        elif content.startswith('---\n'):
            start_idx = 4
            end_marker = '\n---'
        else:
            return '', ''
        end_idx = content.find(end_marker, start_idx)
        if end_idx == -1:
            return '', ''
        try:
            data = yaml.safe_load(content[start_idx:end_idx])
            if not isinstance(data, dict):
                return '', ''
            return data.get('name', ''), data.get('description', '')
        except Exception:
            return '', ''

    def _build_skills_prompt(self, skills: dict[str, dict[str, str]]) -> str:
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

        new_request = request.model_copy()
        new_request.messages = messages
        return new_request

    def tools(self, ctx: GenerateMiddlewareContext) -> list[Any]:
        if not self._scan_skills():
            return []

        scratch = Registry()

        async def use_skill(input: _UseSkillInput) -> str:
            skill_name = input.skill_name
            skills = await asyncio.to_thread(self._scan_skills)
            info = skills.get(skill_name)
            if info is None:
                available = ', '.join(sorted(skills.keys()))
                return f'Unknown skill "{skill_name}". Available skills: {available}'
            try:
                skill_path = Path(info['path'])
                return await asyncio.to_thread(skill_path.read_text, encoding='utf-8')
            except Exception as exc:
                return f'Failed to read skill "{skill_name}": {exc}'

        t = define_tool(scratch, use_skill, name='use_skill')
        return [t.action()]

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        skills = await asyncio.to_thread(self._scan_skills)
        if skills:
            prompt_text = self._build_skills_prompt(skills)
            if prompt_text:
                params = params.model_copy()
                params.request = self._inject_skills_prompt(params.request, prompt_text)
        return await next_fn(params, ctx)
