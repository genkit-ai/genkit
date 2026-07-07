#!/usr/bin/env python3
# ruff: noqa
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Generates deprecated genkit-plugin-* tombstone wheels into dist/ for PyPI publishing."""

import os
import shutil
import subprocess
import sys
import tempfile


def get_version() -> str:
    ref = os.environ.get('GITHUB_REF_NAME', '')
    if ref.startswith('py/v'):
        return ref[4:]
    core_toml = os.path.join(os.path.dirname(__file__), '..', 'packages', 'genkit', 'pyproject.toml')
    if os.path.exists(core_toml):
        with open(core_toml) as f:
            for line in f:
                if line.startswith('version = '):
                    return line.split('"')[1]
    # A wrong version silently pins every tombstone to a nonexistent release and
    # breaks installs for everyone upgrading, so refuse to guess.
    sys.exit(
        "publish_tombstones: could not resolve the release version. Expected a 'py/v*' "
        'GITHUB_REF_NAME tag or a \'version = "..."\' line in packages/genkit/pyproject.toml.'
    )


PLUGINS = [
    {
        'old_dist': 'genkit-plugin-anthropic',
        'new_dist': 'genkit-anthropic',
        'old_import': 'anthropic',
        'new_import': 'genkit_anthropic',
    },
    {
        'old_dist': 'genkit-plugin-compat-oai',
        'new_dist': 'genkit-openai',
        'old_import': 'compat_oai',
        'new_import': 'genkit_openai',
    },
    {
        'old_dist': 'genkit-plugin-django',
        'new_dist': 'genkit-django',
        'old_import': 'django',
        'new_import': 'genkit_django',
    },
    {
        'old_dist': 'genkit-plugin-evaluators',
        'new_dist': 'genkit-evaluators',
        'old_import': 'evaluators',
        'new_import': 'genkit_evaluators',
    },
    {
        'old_dist': 'genkit-plugin-fastapi',
        'new_dist': 'genkit-fastapi',
        'old_import': 'fastapi',
        'new_import': 'genkit_fastapi',
    },
    {
        'old_dist': 'genkit-plugin-flask',
        'new_dist': 'genkit-flask',
        'old_import': 'flask',
        'new_import': 'genkit_flask',
    },
    {
        'old_dist': 'genkit-plugin-google-cloud',
        'new_dist': 'genkit-googlecloud',
        'old_import': 'google_cloud',
        'new_import': 'genkit_googlecloud',
    },
    {
        'old_dist': 'genkit-plugin-google-genai',
        'new_dist': 'genkit-googleai',
        'old_import': 'google_genai',
        'new_import': 'genkit_googleai',
    },
    {
        'old_dist': 'genkit-plugin-middleware',
        'new_dist': 'genkit-middleware',
        'old_import': 'middleware',
        'new_import': 'genkit_middleware',
    },
    {
        'old_dist': 'genkit-plugin-ollama',
        'new_dist': 'genkit-ollama',
        'old_import': 'ollama',
        'new_import': 'genkit_ollama',
    },
    {
        'old_dist': 'genkit-plugin-vertex-ai',
        'new_dist': 'genkit-vertexai',
        'old_import': 'vertex_ai',
        'new_import': 'genkit_vertexai',
    },
]

PYPROJECT = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{old_dist}"
version = "{version}"
description = "Deprecated: This package has been renamed to {new_dist}."
readme = "README.md"
requires-python = ">=3.10"
license = {{ text = "Apache-2.0" }}
dependencies = ["{new_dist}=={version}"]

[tool.hatch.build.targets.wheel]
packages = ["src/genkit"]
"""

README = """# Deprecated Package: {old_dist}

**IMPORTANT**: This package has been renamed to **[{new_dist}](https://pypi.org/project/{new_dist}/)**.

### Migration

1. Update your dependencies:
   ```bash
   uv remove {old_dist}
   uv add {new_dist}
   ```
2. Update your imports:
   ```python
   # Old
   from genkit.plugins import {old_import}

   # New
   import {new_import}
   ```

Importing from `genkit.plugins.{old_import}` raises an `ImportError`.
"""

INIT = """raise ImportError(
    "The '{old_dist}' package has been renamed to '{new_dist}'. "
    "Please update your dependencies to '{new_dist}' and swap your imports "
    "from 'genkit.plugins.{old_import}' to '{new_import}'."
)
"""


def main() -> None:
    version = get_version()
    dist_dir = os.path.abspath(
        sys.argv[2]
        if len(sys.argv) > 2 and sys.argv[1] == '--dist-dir'
        else (sys.argv[1] if len(sys.argv) > 1 else 'dist/')
    )
    os.makedirs(dist_dir, exist_ok=True)

    for p in PLUGINS:
        old_dist, new_dist, old_import, new_import = p['old_dist'], p['new_dist'], p['old_import'], p['new_import']
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f'{tmpdir}/pyproject.toml', 'w') as f:
                f.write(PYPROJECT.format(old_dist=old_dist, new_dist=new_dist, version=version))
            with open(f'{tmpdir}/README.md', 'w') as f:
                f.write(
                    README.format(old_dist=old_dist, new_dist=new_dist, old_import=old_import, new_import=new_import)
                )

            mod_dir = f'{tmpdir}/src/genkit/plugins/{old_import}'
            os.makedirs(mod_dir, exist_ok=True)
            with open(f'{mod_dir}/__init__.py', 'w') as f:
                f.write(INIT.format(old_dist=old_dist, new_dist=new_dist, old_import=old_import, new_import=new_import))

            subprocess.run(['uv', '--no-config', 'build', '--wheel', '--out-dir', 'out'], cwd=tmpdir, check=True)
            whl = [w for w in os.listdir(f'{tmpdir}/out') if w.endswith('.whl')][0]
            shutil.copy(f'{tmpdir}/out/{whl}', f'{dist_dir}/{whl}')
            print(f'Built tombstone: {whl}')


if __name__ == '__main__':
    main()
