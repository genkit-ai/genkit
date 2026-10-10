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

"""Tests for Skills middleware."""

import tempfile
from pathlib import Path

import pytest
from genkit_middleware import Skills

from genkit import Message, ModelResponse, Part
from genkit._core._model import GenerateActionOptions
from genkit._core._typing import Role
from genkit.middleware import GenerateHookParams, GenerateMiddlewareContext


def _make_params() -> GenerateHookParams:
    return GenerateHookParams(
        options=GenerateActionOptions(messages=[]),
        iteration=0,
    )


def _write_skill(path: Path, content: str = 'Skill instructions.') -> None:
    path.mkdir(parents=True)
    (path / 'SKILL.md').write_text(content, encoding='utf-8')


@pytest.mark.parametrize('paths', [['.agents/skills'], ['skills'], ['.agents/skills', 'skills']])
def test_skills_default_paths(tmp_path, monkeypatch, paths) -> None:
    monkeypatch.chdir(tmp_path)
    for index, path in enumerate(paths):
        _write_skill(tmp_path / path / f'skill-{index}')

    scanned = Skills()._scan_skills()

    assert set(scanned) == {f'skill-{index}' for index in range(len(paths))}


def test_skills_default_paths_later_directory_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_skill(tmp_path / '.agents/skills/shared', 'Agent instructions.')
    _write_skill(tmp_path / 'skills/shared', 'Project instructions.')

    scanned = Skills()._scan_skills()

    assert set(scanned) == {'shared'}
    assert Path(scanned['shared']['path']) == tmp_path / 'skills/shared/SKILL.md'


@pytest.mark.parametrize('paths', [['custom'], []])
def test_skills_explicit_paths_replace_defaults(tmp_path, monkeypatch, paths) -> None:
    monkeypatch.chdir(tmp_path)
    _write_skill(tmp_path / '.agents/skills/agent')
    _write_skill(tmp_path / 'skills/project')
    _write_skill(tmp_path / 'custom/explicit')

    scanned = Skills(skill_paths=paths)._scan_skills()

    assert set(scanned) == ({'explicit'} if paths else set())


@pytest.mark.asyncio
@pytest.mark.parametrize('has_system_message', [False, True])
async def test_skills_catalog_names_activation_tool(tmp_path, ctx, has_system_message) -> None:
    _write_skill(tmp_path / 'poetry')
    skills = Skills(skill_paths=[str(tmp_path)])
    params = _make_params()
    if has_system_message:
        params.options.messages.append(Message(role=Role.SYSTEM, content=[Part.from_text('System instructions.')]))
    params.options.messages.append(Message(role=Role.USER, content=[Part.from_text('Write a poem.')]))
    original = params.model_copy(deep=True)
    tool = skills.tools(ctx)[0]

    async def next_fn(updated, ctx):
        system = updated.options.messages[0]
        assert system.role == Role.SYSTEM
        catalog = system.content[-1]
        assert catalog.text is not None
        assert f"Call the {tool.name} tool with a skill's name to load its instructions." in catalog.text
        assert catalog.metadata == {'skills-instructions': True, 'skillsActivationTool': tool.name}
        assert ' - poetry\n' in catalog.text
        if has_system_message:
            assert system.content[0] == original.options.messages[0].content[0]
        assert updated.options.messages[1:] == original.options.messages[-1:]
        return ModelResponse(message=None)

    await skills.wrap_generate(params, ctx, next_fn)
    assert params == original


@pytest.mark.parametrize(
    'owner_metadata',
    [
        {},
        {'skillsActivationTool': 'use_skill'},
        {'skillsActivationTool': None},
        {'skillsActivationTool': False},
        {'skillsActivationTool': 123},
        {'skillsActivationTool': []},
        {'skillsActivationTool': {}},
    ],
    ids=['legacy', 'same-owner', 'null', 'boolean', 'number', 'list', 'object'],
)
def test_skills_refreshes_owned_and_legacy_catalogs(owner_metadata) -> None:
    skills = Skills()
    options = GenerateActionOptions(
        messages=[
            Message(
                role=Role.SYSTEM,
                content=[
                    Part.from_text('System instructions.'),
                    Part.from_text('Old catalog.', metadata={'skills-instructions': True, **owner_metadata}),
                ],
            ),
        ],
    )
    original = options.model_copy(deep=True)

    updated = skills._inject_skills_prompt(options, 'Updated catalog.')
    updated = skills._inject_skills_prompt(updated, 'Latest catalog.')

    assert options == original
    assert len(updated.messages[0].content) == 2
    assert updated.messages[0].content[0] == original.messages[0].content[0]
    assert updated.messages[0].content[1].text == 'Latest catalog.'
    assert updated.messages[0].content[1].metadata == {
        'skills-instructions': True,
        'skillsActivationTool': 'use_skill',
    }


@pytest.mark.parametrize('owner', ['', 'other_use_skill'])
def test_skills_preserves_foreign_catalogs(owner) -> None:
    skills = Skills()
    options = GenerateActionOptions(
        messages=[
            Message(
                role=Role.SYSTEM,
                content=[
                    Part.from_text('System instructions.'),
                    Part.from_text(
                        'Foreign catalog.', metadata={'skills-instructions': True, 'skillsActivationTool': owner}
                    ),
                ],
            ),
        ],
    )
    original = options.model_copy(deep=True)

    updated = skills._inject_skills_prompt(options, 'New catalog.')
    updated = skills._inject_skills_prompt(updated, 'Updated catalog.')

    assert options == original
    assert len(updated.messages[0].content) == 3
    assert updated.messages[0].content[:2] == original.messages[0].content
    assert updated.messages[0].content[2].text == 'Updated catalog.'
    assert updated.messages[0].content[2].metadata == {
        'skills-instructions': True,
        'skillsActivationTool': 'use_skill',
    }


@pytest.mark.asyncio
async def test_skills_unknown_name_lists_available_skills(tmp_path, ctx) -> None:
    _write_skill(tmp_path / 'zebra')
    _write_skill(tmp_path / 'poetry', 'Write a poem.')
    skills = Skills(skill_paths=[str(tmp_path)])
    tool = skills.tools(ctx)[0]

    unknown = await tool.action().run(input={'skill_name': 'missing'})
    assert unknown.response.output == 'Unknown skill "missing". Available skills: poetry, zebra'

    known = await tool.action().run(input={'skill_name': 'poetry'})
    assert known.response.output == 'Write a poem.'


@pytest.mark.asyncio
async def test_skills_no_paths(ctx: GenerateMiddlewareContext) -> None:
    """Test that middleware works with no skill paths."""
    skills = Skills(skill_paths=[])

    async def next_fn(params, ctx):
        return ModelResponse(message=None)

    result = await skills.wrap_generate(_make_params(), ctx, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_skills_nonexistent_path(ctx: GenerateMiddlewareContext) -> None:
    """Test that nonexistent paths are silently skipped."""
    skills = Skills(skill_paths=['/nonexistent/path'])

    async def next_fn(params, ctx):
        return ModelResponse(message=None)

    result = await skills.wrap_generate(_make_params(), ctx, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_skills_scan_with_skill(ctx: GenerateMiddlewareContext) -> None:
    """Test that skills are scanned and injected into system message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / 'test-skill'
        skill_dir.mkdir()
        skill_file = skill_dir / 'SKILL.md'
        skill_file.write_text("""---
name: test-skill
description: A test skill
---
You are a test assistant.
""")

        skills = Skills(skill_paths=[tmpdir])

        async def next_fn(params, ctx):
            # Check that skills prompt was injected
            assert len(params.options.messages) > 0
            return ModelResponse(message=None)

        result = await skills.wrap_generate(_make_params(), ctx, next_fn)
        assert result is not None


@pytest.mark.asyncio
async def test_skills_parse_frontmatter() -> None:
    """Test that YAML frontmatter is parsed correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / 'python-expert'
        skill_dir.mkdir()
        skill_file = skill_dir / 'SKILL.md'
        skill_file.write_text("""---
name: python-expert
description: Expert Python programming assistance
---
You are an expert Python programmer.
""")

        skills = Skills(skill_paths=[tmpdir])
        info = skills._scan_skills()

        assert 'python-expert' in info
        assert info['python-expert']['description'] == 'Expert Python programming assistance'


def test_skills_parse_frontmatter_crlf() -> None:
    """Frontmatter with CRLF line endings parses like LF (Windows-checked-out files)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / 'win-skill'
        skill_dir.mkdir()
        skill_file = skill_dir / 'SKILL.md'
        skill_file.write_bytes(b'---\r\nname: win-skill\r\ndescription: Windows line endings\r\n---\r\nBody.\r\n')

        skills = Skills(skill_paths=[tmpdir])
        info = skills._scan_skills()

        assert 'win-skill' in info
        assert info['win-skill']['description'] == 'Windows line endings'


def test_skills_parse_no_frontmatter() -> None:
    """Test that files without frontmatter use directory name; description is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / 'test-skill'
        skill_dir.mkdir()
        skill_file = skill_dir / 'SKILL.md'
        skill_file.write_text('You are a test assistant.')

        skills = Skills(skill_paths=[tmpdir])
        info = skills._scan_skills()

        assert 'test-skill' in info
        # No frontmatter → empty description (displayed without placeholder in the prompt)
        assert info['test-skill']['description'] == ''


def test_skills_placeholder_description_not_shown_in_prompt() -> None:
    """Frontmatter that uses the placeholder sentence lists the skill name only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / 'bare-skill'
        skill_dir.mkdir()
        skill_file = skill_dir / 'SKILL.md'
        skill_file.write_text("""---
name: bare-skill
description: No description provided.
---
Skill body.
""")

        skills = Skills(skill_paths=[tmpdir])
        scanned = skills._scan_skills()
        prompt = skills._build_skills_prompt(scanned)

        assert ' - bare-skill\n' in prompt
        assert 'No description provided' not in prompt


def test_use_skill_description_is_nonempty(ctx: GenerateMiddlewareContext) -> None:
    """The skill tool the model sees has a non-empty description."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / 'test-skill'
        skill_dir.mkdir()
        (skill_dir / 'SKILL.md').write_text('You are a test assistant.')

        handles = Skills(skill_paths=[tmpdir]).tools(ctx)
        assert handles
        assert handles[0].description
