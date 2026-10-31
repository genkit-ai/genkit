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

"""Every export rendered by py/docs/index.md has a docstring griffe can see.

mkdocstrings skips members without a docstring (``show_if_no_docstring`` is
off by default), so an undocumented export silently drops out of the API
reference. Griffe reads source statically: classes and functions use their
docstring, module-level assignments (constants, type aliases, TypeVars) use a
string literal on the statement right after the assignment. This test follows
re-exports through the source the same way.
"""

from __future__ import annotations

import ast
import importlib
import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

import genkit

_SRC_ROOT = Path(genkit.__file__).resolve().parent.parent
_DOCS_INDEX = Path(__file__).resolve().parents[4] / 'docs' / 'index.md'


def _documented_modules() -> list[str]:
    """Module targets of the ``::: genkit...`` directives in docs/index.md."""
    text = _DOCS_INDEX.read_text(encoding='utf-8')
    return re.findall(r'^:::\s+(genkit(?:\.\w+)*)\s*$', text, flags=re.MULTILINE)


def _module_file(module: str) -> Path | None:
    """Source file for a genkit module, or None if it lives outside genkit."""
    if module != 'genkit' and not module.startswith('genkit.'):
        return None
    base = _SRC_ROOT.joinpath(*module.split('.'))
    for candidate in (base / '__init__.py', base.with_suffix('.py')):
        if candidate.is_file():
            return candidate
    return None


@cache
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _sibling_runs(body: list[ast.stmt]) -> Iterator[list[ast.stmt]]:
    """Yield each statement list at module level, including if/try branches."""
    yield body
    for stmt in body:
        if isinstance(stmt, ast.If):
            yield from _sibling_runs(stmt.body)
            yield from _sibling_runs(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            yield from _sibling_runs(stmt.body)
            for handler in stmt.handlers:
                yield from _sibling_runs(handler.body)
            yield from _sibling_runs(stmt.orelse)
            yield from _sibling_runs(stmt.finalbody)


def _resolve_from(module: str, node: ast.ImportFrom, is_package: bool) -> str:
    if node.level == 0:
        return node.module or ''
    parts = module.split('.')
    if not is_package:
        parts = parts[:-1]
    parts = parts[: len(parts) - (node.level - 1)]
    return '.'.join([*parts, node.module] if node.module else parts)


def _assigned_names(stmt: ast.stmt) -> list[str]:
    if isinstance(stmt, ast.Assign):
        return [t.id for t in stmt.targets if isinstance(t, ast.Name)]
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return [stmt.target.id]
    return []


def _is_string_expr(stmt: ast.stmt | None) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
        and bool(stmt.value.value.strip())
    )


def _has_docstring(module: str, name: str, seen: frozenset[tuple[str, str]] = frozenset()) -> bool | None:
    """Whether ``module.name`` resolves to a documented definition.

    Returns None when the definition lives outside genkit (e.g. a stdlib
    re-export); griffe renders those from the external package, not our source.
    """
    if (module, name) in seen:
        raise AssertionError(f'import cycle resolving {module}.{name}')
    seen = seen | {(module, name)}
    path = _module_file(module)
    if path is None:
        return None
    is_package = path.name == '__init__.py'

    # Last binding wins, matching runtime and griffe.
    result: tuple[int, bool | None] | None = None

    def record(lineno: int, value: bool | None) -> None:
        nonlocal result
        if result is None or lineno >= result[0]:
            result = (lineno, value)

    for run in _sibling_runs(_parse(path).body):
        for i, stmt in enumerate(run):
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name:
                record(stmt.lineno, bool(ast.get_docstring(stmt)))
            elif name in _assigned_names(stmt):
                nxt = run[i + 1] if i + 1 < len(run) else None
                record(stmt.lineno, _is_string_expr(nxt))
            elif isinstance(stmt, ast.ImportFrom):
                source = _resolve_from(module, stmt, is_package)
                for alias in stmt.names:
                    if (alias.asname or alias.name) != name:
                        continue
                    submodule = f'{source}.{alias.name}'
                    if _module_file(submodule) is not None and not _binds(source, alias.name):
                        record(stmt.lineno, bool(ast.get_docstring(_parse(_module_file(submodule)))))  # type: ignore[arg-type]
                    else:
                        record(stmt.lineno, _has_docstring(source, alias.name, seen))
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.asname == name:
                        sub_path = _module_file(alias.name)
                        record(stmt.lineno, None if sub_path is None else bool(ast.get_docstring(_parse(sub_path))))

    if result is None:
        raise AssertionError(f'{module}.{name}: no binding found in {path}')
    return result[1]


def _binds(module: str, name: str) -> bool:
    """Whether ``module`` binds ``name`` itself (vs. ``name`` being a submodule)."""
    path = _module_file(module)
    if path is None:
        return True
    for run in _sibling_runs(_parse(path).body):
        for stmt in run:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name:
                return True
            if name in _assigned_names(stmt):
                return True
            if isinstance(stmt, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == name for a in stmt.names):
                return True
    return False


def test_docs_index_targets_modules() -> None:
    """The docs index lists module targets; an empty list means the regex rotted."""
    assert _documented_modules(), f'no ::: genkit directives found in {_DOCS_INDEX}'


@pytest.mark.parametrize('module', _documented_modules())
def test_every_export_has_docstring(module: str) -> None:
    """Each name in ``__all__`` of a documented module resolves to a docstring."""
    exports: list[str] = list(importlib.import_module(module).__all__)
    missing = [name for name in exports if _has_docstring(module, name) is False]
    assert not missing, (
        f'{module} exports without a docstring (mkdocstrings will skip them): {missing}. '
        'For constants, type aliases and TypeVars, put a string literal on the line after the assignment.'
    )
