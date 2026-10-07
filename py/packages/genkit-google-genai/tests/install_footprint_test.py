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

"""`uv add genkit-google-genai` doesn't install the Vertex training SDK."""

import re
import subprocess  # noqa: S404
import sys
import textwrap
from importlib.metadata import requires
from pathlib import Path

# The Vertex training SDK can still be installed in the dev environment through
# other packages, so the test hides it to prove the plugin never reaches for it.
_BLOCKED_MODULES = ['google.cloud.aiplatform', 'vertexai']

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


def test_import_genkit_google_genai_works_without_aiplatform(tmp_path: Path) -> None:
    """With the Vertex SDK missing, `import genkit_google_genai`, `GoogleAI()`, and `VertexAI(...)` all succeed."""
    body = """
        import os
        os.environ.pop('GENKIT_ENV', None)
        os.environ['GEMINI_API_KEY'] = 'test-key'

        from genkit_google_genai import GoogleAI, VertexAI

        GoogleAI()
        VertexAI(project='p', location='us-central1')
        print('ok')
        """
    result = subprocess.run(  # noqa: S603
        [sys.executable, '-c', _BLOCKER.format(blocked=_BLOCKED_MODULES) + textwrap.dedent(body)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'ok'


def test_genkit_google_genai_base_dependencies_exclude_aiplatform() -> None:
    """The requirements `uv add genkit-google-genai` installs don't list google-cloud-aiplatform."""
    names = {re.split(r'[\s<>=!~;\[]', req, maxsplit=1)[0].lower() for req in requires('genkit-google-genai') or []}
    assert 'google-genai' in names, names
    assert 'google-cloud-aiplatform' not in names
