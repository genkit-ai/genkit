#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Dotprompt file loading: frontmatter output settings and per-file isolation."""

import tempfile
from pathlib import Path

import pytest

from genkit import Genkit
from genkit._ai._prompt import strip_frontmatter
from genkit.testing import define_scripted_model

_SCHEMA_PROMPT = """---
model: scriptedModel
output:
  schema:
    title: string
  constrained: {constrained}
  contentType: application/json
---
Suggest a dish.
"""


@pytest.mark.asyncio
@pytest.mark.parametrize('constrained', [False, True])
async def test_prompt_file_honors_output_constrained(constrained: bool) -> None:
    """`constrained: false` in frontmatter turns native JSON mode off; the schema stays."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        text = _SCHEMA_PROMPT.format(constrained=str(constrained).lower())
        (Path(tmp_dir) / 'dish.prompt').write_text(text)
        ai = Genkit(prompt_dir=tmp_dir, model='scriptedModel')
        define_scripted_model(ai)

        opts = await ai.prompt('dish').render()

        assert opts.output is not None
        assert opts.output.constrained is constrained
        assert opts.output.content_type == 'application/json'
        assert opts.output.json_schema is not None


@pytest.mark.asyncio
async def test_prompt_file_schema_without_constrained_defaults_to_native_json() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / 'dish.prompt').write_text(
            '---\nmodel: scriptedModel\noutput:\n  schema:\n    title: string\n---\nSuggest a dish.\n'
        )
        ai = Genkit(prompt_dir=tmp_dir, model='scriptedModel')
        define_scripted_model(ai)

        opts = await ai.prompt('dish').render()

        assert opts.output is not None
        assert opts.output.constrained is True


@pytest.mark.asyncio
async def test_one_bad_prompt_file_does_not_drop_its_siblings(capfd: pytest.CaptureFixture[str]) -> None:
    """A YAML typo in one file is logged with its path; the rest of the folder loads."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        (root / 'a_broken.prompt').write_text('---\ndescription: Note: always greet\n  bad: [\n---\nHi\n')
        (root / 'b_greet.prompt').write_text('---\nmodel: scriptedModel\n---\nHello\n')
        (root / 'sub').mkdir()
        (root / 'sub' / 'menu.prompt').write_text('---\nmodel: scriptedModel\n---\nMenu\n')

        ai = Genkit(prompt_dir=tmp_dir, model='scriptedModel')
        define_scripted_model(ai)

        assert (await ai.prompt('b_greet').render()).messages
        assert (await ai.prompt('sub/menu').render()).messages
        out, err = capfd.readouterr()
        assert 'a_broken.prompt' in out + err


@pytest.mark.asyncio
async def test_prompt_variant_keeps_every_dot_after_the_name() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / 'scorer.v2.1.prompt').write_text('---\nmodel: scriptedModel\n---\nScore it.\n')
        ai = Genkit(prompt_dir=tmp_dir, model='scriptedModel')
        define_scripted_model(ai)

        rendered = await ai.prompt('scorer', variant='v2.1').render()

        assert rendered.messages


def test_strip_frontmatter_keeps_markdown_dividers_in_the_body() -> None:
    assert strip_frontmatter('---\nname: x\n---\nIntro\n\n---\n\nMore') == 'Intro\n\n---\n\nMore'
    assert strip_frontmatter('Intro\n\n---\n\nMore') == 'Intro\n\n---\n\nMore'
    assert strip_frontmatter('---- not a delimiter ---\nbody') == '---- not a delimiter ---\nbody'
