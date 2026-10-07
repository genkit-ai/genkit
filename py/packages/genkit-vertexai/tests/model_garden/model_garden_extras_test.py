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

"""What `genkit-vertexai` does with and without its `[anthropic]` / `[openai]` extras.

A missing extra is simulated in a fresh interpreter whose imports of the
blocked packages fail the same way they do when the package isn't installed.
"""

import importlib.metadata
import json
import os
import subprocess  # noqa: S404
import sys
import textwrap
from typing import Any

import pytest
from genkit_vertexai.model_garden import ModelGarden
from genkit_vertexai.model_garden._model_info import SUPPORTED_OPENAI_COMPAT_MODELS
from packaging.requirements import Requirement

ANTHROPIC_EXTRA = ('anthropic', 'genkit_anthropic')
OPENAI_EXTRA = ('openai', 'genkit_openai')

CLAUDE_MESSAGE = "Model Garden Claude models need the anthropic extra: uv add 'genkit-vertexai[anthropic]'"
OPENAI_COMPAT_MESSAGE = (
    'Model Garden Llama, Mistral, and other OpenAI-compatible models need the openai extra: '
    "uv add 'genkit-vertexai[openai]'"
)

_BLOCK_IMPORTS = """
import importlib.abc
import sys

_blocked = {name for name in sys.argv[1].split(',') if name}


class _NotInstalled(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in _blocked:
            raise ModuleNotFoundError(f'No module named {name!r}', name=name)
        return None


sys.meta_path.insert(0, _NotInstalled())
"""

_GENERATE = """
import asyncio
import json

from genkit import Genkit
from genkit_vertexai.model_garden import ModelGarden


async def main():
    ai = Genkit(plugins=[ModelGarden({plugin_args})])
    try:
        response = await ai.generate(model={model!r}, prompt='hi')
    except Exception as e:
        print(json.dumps({{
            'error': type(e).__name__,
            'status': getattr(e, 'status', None),
            'message': getattr(e, 'original_message', str(e)),
        }}))
    else:
        print(json.dumps({{'error': None, 'text': response.text}}))


asyncio.run(main())
"""


def _run_python(code: str, *, blocked: tuple[str, ...] = ()) -> tuple[dict[str, Any], str]:
    """Runs `code` in a fresh interpreter where `blocked` packages aren't importable.

    Returns the JSON object the script printed last, plus its stderr.
    """
    env = {k: v for k, v in os.environ.items() if k not in {'GCLOUD_PROJECT', 'GOOGLE_CLOUD_PROJECT'}}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, '-c', _BLOCK_IMPORTS + textwrap.dedent(code), ','.join(blocked)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


def _generate_without(
    model: str, *, blocked: tuple[str, ...], plugin_args: str = "project_id='my-project'"
) -> dict[str, Any]:
    result, _ = _run_python(_GENERATE.format(model=model, plugin_args=plugin_args), blocked=blocked)
    return result


def test_import_genkit_vertexai_does_not_load_publisher_sdks() -> None:
    """`import genkit_vertexai` and `ModelGarden()` leave the Anthropic and OpenAI SDKs and plugins unloaded."""
    result, _ = _run_python("""
        import json, sys
        import genkit_vertexai
        from genkit_vertexai.model_garden import ModelGarden
        ModelGarden(project_id='my-project')
        publishers = ('anthropic', 'genkit_anthropic', 'openai', 'genkit_openai')
        print(json.dumps({'loaded': sorted(m for m in publishers if m in sys.modules)}))
    """)

    assert result['loaded'] == []


@pytest.mark.parametrize(
    ('model', 'blocked', 'plugin_args', 'message'),
    [
        pytest.param(
            'modelgarden/anthropic/claude-sonnet-4@20250514',
            ANTHROPIC_EXTRA + OPENAI_EXTRA,
            "project_id='my-project'",
            CLAUDE_MESSAGE,
            id='claude-no-extras',
        ),
        pytest.param(
            'modelgarden/anthropic/claude-sonnet-4@20250514',
            ANTHROPIC_EXTRA,
            "project_id='my-project'",
            CLAUDE_MESSAGE,
            id='claude-openai-extra-only',
        ),
        pytest.param(
            'modelgarden/anthropic/claude-sonnet-4@20250514',
            ANTHROPIC_EXTRA,
            '',
            CLAUDE_MESSAGE,
            id='claude-no-project-reports-extra-first',
        ),
        pytest.param(
            'modelgarden/meta/llama-3.1-405b-instruct-maas',
            ANTHROPIC_EXTRA + OPENAI_EXTRA,
            "project_id='my-project'",
            OPENAI_COMPAT_MESSAGE,
            id='llama-no-extras',
        ),
        pytest.param(
            'modelgarden/mistralai/mistral-small-2503',
            OPENAI_EXTRA,
            "project_id='my-project'",
            OPENAI_COMPAT_MESSAGE,
            id='mistral-uncataloged-anthropic-extra-only',
        ),
    ],
)
def test_generate_model_garden_model_without_its_extra_raises_install_command(
    model: str, blocked: tuple[str, ...], plugin_args: str, message: str
) -> None:
    """Generating with a Model Garden model whose extra is missing raises FAILED_PRECONDITION naming `uv add`."""
    result = _generate_without(model, blocked=blocked, plugin_args=plugin_args)

    assert result['error'] == 'GenkitError'
    assert result['status'] == 'FAILED_PRECONDITION'
    assert result['message'] == message


def test_model_garden_list_actions_without_openai_extra_returns_empty() -> None:
    """Without `[openai]`, the model list the Dev UI reads is empty and nothing is logged as a traceback."""
    result, stderr = _run_python(
        """
        import asyncio, json
        from genkit_vertexai.model_garden import ModelGarden
        actions = asyncio.run(ModelGarden(project_id='my-project').list_actions())
        print(json.dumps({'names': [a.name for a in actions]}))
        """,
        blocked=OPENAI_EXTRA,
    )

    assert result['names'] == []
    assert 'Traceback' not in stderr


@pytest.mark.asyncio
async def test_model_garden_list_actions_with_openai_extra_lists_llama_models() -> None:
    """With `[openai]`, the Dev UI lists every built-in OpenAI-compatible model, Llama included."""
    actions = await ModelGarden(project_id='my-project').list_actions()

    names = [a.name for a in actions]
    assert names == [f'modelgarden/{model}' for model in SUPPORTED_OPENAI_COMPAT_MODELS]
    assert 'modelgarden/meta/llama-3.1-405b-instruct-maas' in names


def test_genkit_vertexai_base_dependencies_exclude_unused_sdks() -> None:
    """`uv add genkit-vertexai` skips both publisher SDKs and unused Google SDKs; each extra adds its plugin."""
    requirements = [Requirement(r) for r in importlib.metadata.requires('genkit-vertexai') or []]
    base = {r.name for r in requirements if r.marker is None or 'extra' not in str(r.marker)}
    by_extra = {
        extra: {r.name for r in requirements if r.marker is not None and r.marker.evaluate({'extra': extra})} - base
        for extra in ('anthropic', 'openai')
    }

    assert base.isdisjoint({
        'anthropic',
        'openai',
        'genkit-anthropic',
        'genkit-openai',
        'google-cloud-aiplatform',
        'google-cloud-bigquery',
        'google-genai',
        'structlog',
        'strenum',
    })
    assert {'genkit', 'google-auth'} <= base
    assert by_extra == {'anthropic': {'genkit-anthropic'}, 'openai': {'genkit-openai'}}


@pytest.mark.parametrize(
    'model',
    [
        pytest.param('modelgarden/anthropic/claude-sonnet-4@20250514', id='claude'),
        pytest.param('modelgarden/meta/llama-3.1-405b-instruct-maas', id='llama'),
    ],
)
def test_generate_model_garden_broken_sdk_dependency_surfaces_real_import_error(model: str) -> None:
    """An installed SDK missing its own dependency (`distro`) raises that import error, not the extra hint."""
    result = _generate_without(model, blocked=('distro',))

    assert result['message'] not in {CLAUDE_MESSAGE, OPENAI_COMPAT_MESSAGE}
    assert 'distro' in result['message']


def test_model_garden_list_actions_broken_openai_dependency_raises() -> None:
    """`list_actions()` raises when `openai` is installed but can't import, instead of listing nothing."""
    result, _ = _run_python(
        """
        import asyncio, json
        from genkit_vertexai.model_garden import ModelGarden
        try:
            asyncio.run(ModelGarden(project_id='my-project').list_actions())
        except ModuleNotFoundError as e:
            print(json.dumps({'error': type(e).__name__, 'name': e.name}))
        else:
            print(json.dumps({'error': None}))
        """,
        blocked=('distro',),
    )

    assert result == {'error': 'ModuleNotFoundError', 'name': 'distro'}
