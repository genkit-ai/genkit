# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What `uv add genkit-vertexai` installs."""

from __future__ import annotations

import subprocess  # noqa: S404 — fresh interpreter; the test process may already have imported these SDKs
import sys
from importlib.metadata import requires

from packaging.requirements import Requirement

_DROPPED = (
    'google-cloud-aiplatform',
    'google-cloud-bigquery',
    'google-genai',
    'structlog',
    'strenum',
)


def test_import_genkit_vertexai_works_without_aiplatform() -> None:
    """import genkit_vertexai and ModelGarden() succeed without the Vertex training SDK or BigQuery."""
    script = """
import sys

class _Block:
    def find_spec(self, fullname, path, target=None):
        blocked = ('google.cloud.aiplatform', 'vertexai', 'google.cloud.bigquery')
        for name in blocked:
            if fullname == name or fullname.startswith(name + '.'):
                raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, _Block())
import genkit_vertexai
from genkit_vertexai.model_garden import ModelGarden
genkit_vertexai.ModelGarden(project_id='p', location='us-central1')
ModelGarden(project_id='p', location='us-central1')
"""
    completed = subprocess.run(  # noqa: S603 — interpreter and script are fixed; nothing comes from outside the test
        [sys.executable, '-c', script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_genkit_vertexai_requirements_exclude_unused_google_sdks() -> None:
    """genkit-vertexai's installed requirements don't list libraries the package never imports."""
    listed = [Requirement(raw) for raw in requires('genkit-vertexai') or []]
    names = {item.name for item in listed}
    assert names.isdisjoint(_DROPPED)
    auth = next(item for item in listed if item.name == 'google-auth')
    assert 'requests' in auth.extras
