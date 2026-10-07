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

"""A typo in a prompt name, a prompt directory, or a .prompt file fails at startup."""

import os
from pathlib import Path
from typing import Any

import pytest

from genkit import Genkit, GenkitError
from genkit._ai._model import text_from_message
from genkit._ai._testing import define_echo_model
from genkit._core._error import RuntimeErrorReason
from genkit._core._model import GenerateActionOptions


def _genkit(prompt_dir: Path | None = None) -> Genkit:
    ai = Genkit(model='echoModel', prompt_dir=prompt_dir)
    define_echo_model(ai)
    return ai


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _rendered_text(rendered: GenerateActionOptions) -> str:
    return ''.join(text_from_message(m) for m in rendered.messages)


def _assert_prompt_not_found(exc: GenkitError, text: str) -> None:
    assert exc.status == 'NOT_FOUND'
    assert exc.details['reason'] == RuntimeErrorReason.ACTION_NOT_FOUND.value
    assert text in exc.original_message


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    prompts = tmp_path / 'prompts'
    _write(prompts / 'recipe.prompt', 'Cook {{food}}.')
    _write(prompts / 'recipe.robot.prompt', 'BEEP. Cook {{food}}.')
    _write(prompts / 'sub' / 'recipe.prompt', 'Sub-cook {{food}}.')
    return prompts


def test_prompt_lookup_unknown_name_raises_not_found(prompts_dir: Path) -> None:
    """`ai.prompt('typo')` raises `GenkitError` `NOT_FOUND` immediately, before any await."""
    ai = _genkit(prompts_dir)

    with pytest.raises(GenkitError) as exc_info:
        ai.prompt('recipy')

    _assert_prompt_not_found(exc_info.value, 'Prompt recipy not found')


def test_prompt_lookup_unknown_variant_raises_not_found(prompts_dir: Path) -> None:
    """`ai.prompt('recipe', variant='nope')` raises `NOT_FOUND` naming the variant."""
    ai = _genkit(prompts_dir)

    with pytest.raises(GenkitError) as exc_info:
        ai.prompt('recipe', variant='nope')

    _assert_prompt_not_found(exc_info.value, 'Prompt recipe (variant nope) not found')


@pytest.mark.asyncio
async def test_prompt_lookup_file_prompt_returns_handle(prompts_dir: Path) -> None:
    """`ai.prompt('recipe')` for a loaded `recipe.prompt` returns a callable prompt."""
    ai = _genkit(prompts_dir)

    rendered = await ai.prompt('recipe').render({'food': 'soup'})

    assert _rendered_text(rendered) == 'Cook soup.'


@pytest.mark.asyncio
async def test_prompt_lookup_file_variant_returns_handle(prompts_dir: Path) -> None:
    """`ai.prompt('recipe', variant='robot')` for a loaded `recipe.robot.prompt` returns that variant."""
    ai = _genkit(prompts_dir)

    rendered = await ai.prompt('recipe', variant='robot').render({'food': 'soup'})

    assert _rendered_text(rendered) == 'BEEP. Cook soup.'


@pytest.mark.asyncio
async def test_prompt_lookup_subdirectory_prompt_returns_handle(prompts_dir: Path) -> None:
    """A prompt in `prompts/sub/recipe.prompt` is found as `ai.prompt('sub/recipe')`."""
    ai = _genkit(prompts_dir)

    rendered = await ai.prompt('sub/recipe').render({'food': 'soup'})

    assert _rendered_text(rendered) == 'Sub-cook soup.'


@pytest.mark.asyncio
async def test_prompt_lookup_defined_prompt_returns_handle() -> None:
    """`ai.prompt('joke')` after `ai.define_prompt(name='joke', ...)` returns that prompt."""
    ai = _genkit()
    ai.define_prompt(name='joke', prompt='Tell a joke about {{topic}}.')

    rendered = await ai.prompt('joke').render({'topic': 'cats'})

    assert _rendered_text(rendered) == 'Tell a joke about cats.'


def test_prompt_lookup_before_define_raises_not_found() -> None:
    """`ai.prompt('joke')` before `define_prompt(name='joke')` raises `NOT_FOUND`."""
    ai = Genkit()

    with pytest.raises(GenkitError) as exc_info:
        ai.prompt('joke')

    _assert_prompt_not_found(exc_info.value, 'Prompt joke not found')


@pytest.mark.asyncio
async def test_prompt_lookup_frontmatter_name_is_not_the_lookup_name(tmp_path: Path) -> None:
    """A file `named.prompt` with `name: goodbye` is found as `named`, and `ai.prompt('goodbye')` raises."""
    prompts = tmp_path / 'prompts'
    _write(prompts / 'named.prompt', '---\nname: goodbye\n---\nHi {{who}}.')
    ai = _genkit(prompts)

    rendered = await ai.prompt('named').render({'who': 'Ana'})
    assert _rendered_text(rendered) == 'Hi Ana.'

    with pytest.raises(GenkitError) as exc_info:
        ai.prompt('goodbye')
    _assert_prompt_not_found(exc_info.value, 'Prompt goodbye not found')


def test_genkit_missing_prompt_dir_raises(tmp_path: Path) -> None:
    """`Genkit(prompt_dir='missing')` raises `INVALID_ARGUMENT`, naming the resolved path."""
    missing = tmp_path / 'promtps'

    with pytest.raises(GenkitError) as exc_info:
        Genkit(prompt_dir=missing)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.original_message == f'Prompt directory not found: {missing.resolve()}'


def test_genkit_prompt_dir_pointing_at_file_raises(prompts_dir: Path) -> None:
    """`Genkit(prompt_dir='recipe.prompt')` raises `INVALID_ARGUMENT` because the path isn't a directory."""
    file_path = prompts_dir / 'recipe.prompt'

    with pytest.raises(GenkitError) as exc_info:
        Genkit(prompt_dir=file_path)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.original_message == f'Prompt path is not a directory: {file_path.resolve()}'


def test_genkit_without_prompt_dir_and_no_prompts_folder_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Genkit()` in a directory with no `./prompts` starts normally."""
    monkeypatch.chdir(tmp_path)

    ai = Genkit()

    with pytest.raises(GenkitError) as exc_info:
        ai.prompt('recipe')
    _assert_prompt_not_found(exc_info.value, 'Prompt recipe not found')


def test_malformed_prompt_file_fails_startup_naming_the_file(prompts_dir: Path) -> None:
    """A folder with good files and one broken frontmatter raises `INVALID_ARGUMENT` naming the broken file."""
    broken = _write(prompts_dir / 'broken.prompt', '---\nmodel: [unclosed\n---\nHi.')

    with pytest.raises(GenkitError) as exc_info:
        Genkit(prompt_dir=prompts_dir)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert str(broken.resolve()) in exc_info.value.original_message
    assert 'Malformed frontmatter' in exc_info.value.original_message


def test_malformed_prompt_file_in_subfolder_fails_startup_naming_the_file(prompts_dir: Path) -> None:
    """A broken `.prompt` file nested in a subfolder still stops `Genkit(prompt_dir=...)`, naming that file."""
    broken = _write(prompts_dir / 'sub' / 'deeper' / 'broken.prompt', '---\nmodel: gemini\nHi, no closing line.')

    with pytest.raises(GenkitError) as exc_info:
        Genkit(prompt_dir=prompts_dir)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert str(broken.resolve()) in exc_info.value.original_message
    assert 'Malformed frontmatter' in exc_info.value.original_message


def test_prompt_file_not_utf8_fails_startup_naming_file(prompts_dir: Path) -> None:
    """A `.prompt` file that isn't valid UTF-8 makes `Genkit(prompt_dir=...)` raise `INVALID_ARGUMENT` naming it."""
    bad = prompts_dir / 'latin1.prompt'
    bad.write_bytes('Café {{food}}.'.encode('latin-1'))

    with pytest.raises(GenkitError) as exc_info:
        Genkit(prompt_dir=prompts_dir)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert str(bad.resolve()) in exc_info.value.original_message


def test_prompt_file_unreadable_fails_startup_naming_file(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `.prompt` file the process can't open makes `Genkit(prompt_dir=...)` raise `INVALID_ARGUMENT` naming it."""
    locked = _write(prompts_dir / 'locked.prompt', 'Hi.').resolve()
    real_open = Path.open

    def open_or_deny(self: Path, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if self.resolve() == locked:
            raise PermissionError(13, 'Permission denied', str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_or_deny)

    with pytest.raises(GenkitError) as exc_info:
        Genkit(prompt_dir=prompts_dir)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert str(locked) in exc_info.value.original_message


def test_prompt_subfolder_unlistable_fails_startup_naming_folder(
    prompts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nested folder the process can't list makes `Genkit(prompt_dir=...)` raise `INVALID_ARGUMENT` naming it."""
    locked = (prompts_dir / 'sub' / 'locked').resolve()
    _write(locked / 'hidden.prompt', 'Hi.')
    real_scandir = os.scandir

    def scandir_or_deny(path: Any) -> Any:  # noqa: ANN401
        if Path(path).resolve() == locked:
            raise PermissionError(13, 'Permission denied', str(path))
        return real_scandir(path)

    monkeypatch.setattr(os, 'scandir', scandir_or_deny)

    with pytest.raises(GenkitError) as exc_info:
        Genkit(prompt_dir=prompts_dir)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert str(locked) in exc_info.value.original_message


def test_prompt_partial_file_not_utf8_fails_startup_naming_file(prompts_dir: Path) -> None:
    """A partial `_header.prompt` that isn't valid UTF-8 makes `Genkit(prompt_dir=...)` raise, naming it."""
    bad = prompts_dir / '_header.prompt'
    bad.write_bytes('Café.'.encode('latin-1'))

    with pytest.raises(GenkitError) as exc_info:
        Genkit(prompt_dir=prompts_dir)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert str(bad.resolve()) in exc_info.value.original_message


@pytest.mark.asyncio
async def test_prompt_folder_without_broken_files_loads_every_prompt(tmp_path: Path) -> None:
    """With no broken files, every prompt in every nested folder is loaded and callable."""
    prompts = tmp_path / 'prompts'
    names = ['a', 'b', 'one/a', 'one/two/a', 'one/two/three/z']
    for name in names:
        _write(prompts / f'{name}.prompt', f'---\nconfig:\n  temperature: 0.1\n---\nI am {name}.')

    ai = _genkit(prompts)

    for name in names:
        rendered = await ai.prompt(name).render()
        assert _rendered_text(rendered) == f'I am {name}.'
