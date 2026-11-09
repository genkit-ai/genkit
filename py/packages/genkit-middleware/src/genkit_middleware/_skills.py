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

Each entry in ``skill_paths`` is a skill root. Every direct subdirectory that
holds a regular ``SKILL.md`` is a skill. The scan and ``use_skill`` stay inside
the skill root: a subdirectory or ``SKILL.md`` whose real path leaves the root
is skipped, and a ``SKILL.md`` that is a symlink, is not a regular file, or is
over 1 MiB is never read.
"""

from __future__ import annotations

import asyncio
import errno
import os
import re
import stat
import unicodedata
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel as PydanticBaseModel, Field

from genkit import Message, ModelResponse, Part, Role, Tool, get_logger
from genkit.middleware import GenerateHookParams, Middleware, MiddlewareContext
from genkit.model import GenerateActionOptions
from genkit.plugin_api import GenkitConfig

logger = get_logger(__name__)

_SKILLS_MARKER = 'skills-instructions'
_MISSING_DESCRIPTION = 'No description provided.'
_SKILL_FILE_NAME = 'SKILL.md'

# Process resource limit, same as Go: an oversized SKILL.md is skipped, never truncated.
_SKILL_MAX_BYTES = 1 << 20
# Agent Skills spec bounds. A longer frontmatter name falls back to the
# directory name; a longer description is clipped in the catalog only.
_SKILL_NAME_MAX_CHARS = 64
_SKILL_DESCRIPTION_MAX_CHARS = 1024

# A '<' that opens a tag, e.g. '</skills>' or '<system>'.
_TAG_OPEN = re.compile(r'<(?=/?[A-Za-z])')

_O_NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)
_O_NONBLOCK = getattr(os, 'O_NONBLOCK', 0)


class _UseSkillInput(PydanticBaseModel):
    """Input for the ``use_skill`` tool."""

    skill_name: str = Field(description='The name of the skill to load (as listed in the system prompt).')


class SkillsConfig(GenkitConfig):
    """Directories to scan for skill folders containing ``SKILL.md``."""

    skill_paths: list[str] = Field(default_factory=lambda: ['skills'])


def _single_line(value: str) -> str:
    """Drop control and format characters (NUL, bidi overrides) and collapse whitespace."""
    kept = ''.join(' ' if unicodedata.category(c) in ('Cc', 'Cf', 'Zl', 'Zp') else c for c in value)
    return ' '.join(kept.split())


def _frontmatter_text(value: object) -> str:
    """Coerce a YAML scalar to text: ``name: 404`` is ``'404'``. Mappings and lists are ignored."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return str(value)
    return ''


def _catalog_text(value: str) -> str:
    """Escape tag openers so skill metadata can't close ``<skills>`` or open a new block."""
    return _TAG_OPEN.sub('&lt;', value)


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + '...'


def _is_within(path: Path, root: Path) -> bool:
    """Whether ``path``'s real path is ``root`` or under it. ``root`` must already be resolved."""
    try:
        return path.resolve().is_relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False


def _read_skill_file(path: Path, root: Path) -> str:
    """Read a ``SKILL.md`` that is a regular, in-root, non-symlink file within the size limit.

    Type and size are checked on ``lstat`` before the open, so a symlink or a
    named pipe is never opened. ``O_NOFOLLOW`` and the ``fstat`` identity check
    catch a swap between the check and the open.
    """
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f'{_SKILL_FILE_NAME} is not a regular file.')
    if st.st_size > _SKILL_MAX_BYTES:
        raise ValueError(f'{_SKILL_FILE_NAME} is {st.st_size:,} bytes, over the {_SKILL_MAX_BYTES:,} byte limit.')
    if not _is_within(path, root):
        raise ValueError(f'{_SKILL_FILE_NAME} is outside its skill root.')
    fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK)
    with os.fdopen(fd, 'rb') as fh:
        opened = os.fstat(fh.fileno())
        if (opened.st_dev, opened.st_ino) != (st.st_dev, st.st_ino) or not stat.S_ISREG(opened.st_mode):
            raise OSError(errno.ESTALE, f'{_SKILL_FILE_NAME} changed while it was being opened.')
        data = fh.read(_SKILL_MAX_BYTES + 1)
    if len(data) > _SKILL_MAX_BYTES:
        raise ValueError(f'{_SKILL_FILE_NAME} is over the {_SKILL_MAX_BYTES:,} byte limit.')
    return data.decode('utf-8')


class Skills(Middleware[SkillsConfig]):
    """Skills middleware that exposes ``SKILL.md`` files as loadable instructions.

    A skill's name is its frontmatter ``name``, or its directory name when that
    is missing, empty, or over 64 characters. When two skills share a name the
    first one scanned wins: earlier ``skill_paths`` entries beat later ones,
    and directories within a root are scanned in sorted order.
    """

    name = 'skills'
    description = 'Provides access to skill library for specialized instructions'

    def _scan_skills(self) -> dict[str, dict[str, str]]:
        skills: dict[str, dict[str, str]] = {}
        for path_str in self.config.skill_paths:
            try:
                root = Path(path_str).resolve()
                if not root.is_dir():
                    continue
                subdirs = sorted(root.iterdir())
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning('skills path could not be read, skipping', path=path_str, error=str(exc))
                continue
            for subdir in subdirs:
                if subdir.name.startswith('.'):
                    continue
                skill_file = subdir / _SKILL_FILE_NAME
                try:
                    if not subdir.is_dir():
                        continue
                    if not _is_within(subdir, root):
                        logger.warning('skill directory resolves outside its skill root, skipping', path=str(subdir))
                        continue
                    if not os.path.lexists(skill_file):
                        continue
                    content = _read_skill_file(skill_file, root)
                except (OSError, RuntimeError, ValueError) as exc:
                    logger.warning('skill could not be loaded, skipping', path=str(subdir), error=str(exc))
                    continue
                fm_name, fm_description = self._parse_frontmatter(content)
                name = _single_line(fm_name)
                if not name or len(name) > _SKILL_NAME_MAX_CHARS:
                    name = _single_line(subdir.name)
                if not name:
                    continue
                if name in skills:
                    logger.warning(
                        'skill name is taken by an earlier skill, skipping',
                        skill=name,
                        path=str(skill_file),
                        kept=skills[name]['path'],
                    )
                    continue
                skills[name] = {
                    'path': str(skill_file),
                    'root': str(root),
                    'description': _single_line(fm_description),
                }
        return skills

    def _parse_frontmatter(self, content: str) -> tuple[str, str]:
        content = content.lstrip('\ufeff')
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
        except Exception:
            return '', ''
        if not isinstance(data, dict):
            return '', ''
        return _frontmatter_text(data.get('name')), _frontmatter_text(data.get('description'))

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
            name_text = _catalog_text(skill_name)
            if desc and desc != _MISSING_DESCRIPTION:
                lines.append(f' - {name_text} - {_catalog_text(_clip(desc, _SKILL_DESCRIPTION_MAX_CHARS))}')
            else:
                lines.append(f' - {name_text}')
        lines.append('</skills>')
        return '\n'.join(lines)

    def _inject_skills_prompt(self, options: GenerateActionOptions, prompt_text: str) -> GenerateActionOptions:
        messages = list(options.messages)
        system_idx: int | None = None
        for i, msg in enumerate(messages):
            if msg.role == Role.SYSTEM:
                system_idx = i
                break

        marker_meta: dict[str, Any] = {_SKILLS_MARKER: True}
        new_part = Part.from_text(prompt_text, metadata=marker_meta)

        if system_idx is not None:
            msg = messages[system_idx]
            new_content = []
            replaced = False
            for part in msg.content:
                meta = part.metadata if part.text is not None else None
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

        new_options = options.model_copy()
        new_options.messages = messages
        return new_options

    def tools(self, ctx: MiddlewareContext) -> list[Tool]:
        if not self._scan_skills():
            return []

        async def use_skill(input: _UseSkillInput) -> str:
            skill_name = input.skill_name
            skills = await asyncio.to_thread(self._scan_skills)
            # The catalog shows names with tag openers escaped; accept either form.
            info = skills.get(skill_name) or skills.get(skill_name.replace('&lt;', '<'))
            if info is None:
                available = ', '.join(sorted(skills.keys()))
                return f'Unknown skill "{skill_name}". Available skills: {available}'
            try:
                return await asyncio.to_thread(_read_skill_file, Path(info['path']), Path(info['root']))
            except (OSError, ValueError) as exc:
                # Log the host path; keep it out of the model-visible result.
                logger.warning('skill could not be read', skill=skill_name, path=info['path'], error=str(exc))
                return f'Failed to read skill "{skill_name}".'

        return [Tool(use_skill, description='Load a skill by name.')]

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: MiddlewareContext,
        next_fn: Callable[[GenerateHookParams, MiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        skills = await asyncio.to_thread(self._scan_skills)
        if skills:
            prompt_text = self._build_skills_prompt(skills)
            if prompt_text:
                params = params.model_copy()
                params.options = self._inject_skills_prompt(params.options, prompt_text)
        return await next_fn(params, ctx)
