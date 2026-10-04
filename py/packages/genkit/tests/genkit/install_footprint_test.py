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

"""`uv add genkit` installs only what the SDK imports."""

import re
import subprocess  # noqa: S404
import sys
import textwrap
from importlib.metadata import requires
from pathlib import Path

# These may still be installed in the dev environment through other packages,
# so each test hides them from the interpreter to prove genkit never reaches for them.
_UNUSED_MODULES = ['rich', 'psutil', 'PIL', 'sse_starlette', 'asgiref']
_UNUSED_DISTRIBUTIONS = {'rich', 'psutil', 'pillow', 'sse-starlette', 'asgiref'}

_BLOCKER = textwrap.dedent(
    """
    import sys

    BLOCKED = {blocked!r}

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if any(name == b or name.startswith(b + '.') for b in BLOCKED):
                raise ModuleNotFoundError(f'blocked for test: {{name}}', name=name)
            return None

    sys.meta_path.insert(0, _Blocker())
    """
)


def _run_with_unused_modules_blocked(body: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    code = _BLOCKER.format(blocked=_UNUSED_MODULES) + textwrap.dedent(body)
    return subprocess.run(  # noqa: S603
        [sys.executable, '-c', code],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _requirement_names(distribution: str) -> set[str]:
    return {re.split(r'[\s<>=!~;\[]', req, maxsplit=1)[0].lower() for req in requires(distribution) or []}


def test_import_genkit_works_without_unused_libraries(tmp_path: Path) -> None:
    """`import genkit` and `Genkit()` succeed with rich, psutil, PIL, sse_starlette, and asgiref missing."""
    result = _run_with_unused_modules_blocked(
        """
        import os
        os.environ.pop('GENKIT_ENV', None)

        from genkit import Genkit

        Genkit()
        print('ok')
        """,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'ok'


def test_genkit_dev_server_starts_without_unused_libraries(tmp_path: Path) -> None:
    """In dev mode with those libraries missing, `Genkit()` starts the Dev UI server and `/api/__health` answers 200."""
    result = _run_with_unused_modules_blocked(
        """
        import os
        import socket

        import httpx

        from genkit import Genkit
        from genkit._core._reflection import ServerSpec

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]

        os.environ['GENKIT_ENV'] = 'dev'
        ai = Genkit(reflection_server_spec=ServerSpec(scheme='http', host='127.0.0.1', port=port))
        assert ai._reflection_ready.wait(timeout=10), 'dev server never became ready'
        print(httpx.get(f'http://127.0.0.1:{port}/api/__health', timeout=5.0).status_code, flush=True)
        os._exit(0)
        """,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '200'


def test_genkit_base_dependencies_exclude_unused_libraries() -> None:
    """The requirements `uv add genkit` installs list none of rich, psutil, pillow, sse-starlette, or asgiref."""
    assert _requirement_names('genkit').isdisjoint(_UNUSED_DISTRIBUTIONS)
