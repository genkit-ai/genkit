#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Dotprompt file loading: frontmatter output settings and per-file isolation."""

import tempfile
from pathlib import Path

import pytest

from genkit import Genkit
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
