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

from genkit._core._model import GenerateActionOptions, GenerateHookParams, ModelRequest, ModelResponse
from genkit.plugins.middleware import Skills


@pytest.mark.asyncio
async def test_skills_no_paths() -> None:
    """Test that middleware works with no skill paths."""
    skills = Skills(skill_paths=[])

    async def next_fn(params):
        return ModelResponse(message=None)

    request = ModelRequest(messages=[])
    options = GenerateActionOptions(messages=[])
    params = GenerateHookParams(options=options, request=request, iteration=0)

    result = await skills.wrap_generate(params, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_skills_nonexistent_path() -> None:
    """Test that nonexistent paths are silently skipped."""
    skills = Skills(skill_paths=['/nonexistent/path'])

    async def next_fn(params):
        return ModelResponse(message=None)

    request = ModelRequest(messages=[])
    options = GenerateActionOptions(messages=[])
    params = GenerateHookParams(options=options, request=request, iteration=0)

    result = await skills.wrap_generate(params, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_skills_scan_with_skill() -> None:
    """Test that skills are scanned and injected into system message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / 'test-skill'
        skill_dir.mkdir()
        skill_file = skill_dir / 'SKILL.md'
        skill_file.write_text('''---
name: test-skill
description: A test skill
---
You are a test assistant.
''')

        skills = Skills(skill_paths=[tmpdir])

        async def next_fn(params):
            # Check that skills prompt was injected
            assert len(params.request.messages) > 0
            return ModelResponse(message=None)

        request = ModelRequest(messages=[])
        options = GenerateActionOptions(messages=[])
        params = GenerateHookParams(options=options, request=request, iteration=0)

        result = await skills.wrap_generate(params, next_fn)
        assert result is not None


@pytest.mark.asyncio
async def test_skills_parse_frontmatter() -> None:
    """Test that YAML frontmatter is parsed correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / 'python-expert'
        skill_dir.mkdir()
        skill_file = skill_dir / 'SKILL.md'
        skill_file.write_text('''---
name: python-expert
description: Expert Python programming assistance
---
You are an expert Python programmer.
''')

        skills = Skills(skill_paths=[tmpdir])
        info = skills._scan_skills()

        assert 'python-expert' in info
        assert info['python-expert']['description'] == 'Expert Python programming assistance'


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
        skill_file.write_text('''---
name: bare-skill
description: No description provided.
---
Skill body.
''')

        skills = Skills(skill_paths=[tmpdir])
        scanned = skills._scan_skills()
        prompt = skills._build_skills_prompt(scanned)

        assert ' - bare-skill\n' in prompt
        assert 'No description provided' not in prompt
