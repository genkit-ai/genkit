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
async def test_skills_no_paths():
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
async def test_skills_nonexistent_path():
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
async def test_skills_scan_with_skill():
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
async def test_skills_parse_frontmatter():
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
        info = skills._get_skills_info()
        
        assert 'python-expert' in info
        assert info['python-expert']['description'] == 'Expert Python programming assistance'


def test_skills_parse_no_frontmatter():
    """Test that files without frontmatter use directory name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / 'test-skill'
        skill_dir.mkdir()
        skill_file = skill_dir / 'SKILL.md'
        skill_file.write_text('You are a test assistant.')
        
        skills = Skills(skill_paths=[tmpdir])
        info = skills._get_skills_info()
        
        assert 'test-skill' in info
        assert info['test-skill']['description'] == 'No description provided.'
