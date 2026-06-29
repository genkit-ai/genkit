#!/usr/bin/env python3
# ruff: noqa
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

"""Script to dynamically generate and publish the 11 deprecated genkit-plugin-*.

tombstone packages to PyPI at version 0.8.0.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

# Mapping of deprecated plugin names to their new package names and import namespaces
PLUGINS_MAPPING = [
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
        'new_dist': 'genkit-google-cloud',
        'old_import': 'google_cloud',
        'new_import': 'genkit_google_cloud',
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
        'new_dist': 'genkit-vertex-ai',
        'old_import': 'vertex_ai',
        'new_import': 'genkit_vertexai',
    },
]

PYPROJECT_TEMPLATE = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{old_dist}"
version = "0.8.0"
description = "Deprecated: This package has been renamed to {new_dist}."
readme = "README.md"
requires-python = ">=3.10"
license = {{ text = "Apache-2.0" }}
dependencies = [
    "{new_dist}==0.8.0"
]

[tool.hatch.build.targets.wheel]
packages = ["src/genkit"]
"""

README_TEMPLATE = """# Deprecated Package: {old_dist}

**IMPORTANT**: This package has been renamed to **[{new_dist}](https://pypi.org/project/{new_dist}/)** as part of the Genkit Python SDK reorganization.

### Migration

1. Update your dependencies:
   ```bash
   pip uninstall {old_dist}
   pip install {new_dist}
   ```
2. Update your import statements in code:
   ```python
   # Old
   from genkit.plugins import {old_import}

   # New
   import {new_import}
   ```

This package `{old_dist}` is no longer maintained and will not receive further updates. Importing from this package will raise an `ImportError`.
"""

INIT_TEMPLATE = """# Copyright 2026 Google LLC
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

raise ImportError(
    "The '{old_dist}' package has been renamed to '{new_dist}'. "
    "Please update your requirements to '{new_dist}' and your imports to 'import {new_import}'."
)
"""


def build_and_publish(mapping, dist_dir, publish=False) -> None:
    old_dist = mapping['old_dist']
    new_dist = mapping['new_dist']
    old_import = mapping['old_import']
    new_import = mapping['new_import']

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Write pyproject.toml
        pyproject_content = PYPROJECT_TEMPLATE.format(old_dist=old_dist, new_dist=new_dist)
        with open(os.path.join(tmpdir, 'pyproject.toml'), 'w') as f:
            f.write(pyproject_content)

        # 2. Write README.md
        readme_content = README_TEMPLATE.format(
            old_dist=old_dist,
            new_dist=new_dist,
            old_import=old_import,
            new_import=new_import,
        )
        with open(os.path.join(tmpdir, 'README.md'), 'w') as f:
            f.write(readme_content)

        # 3. Write src/genkit/plugins/{old_import}/__init__.py
        shim_dir = os.path.join(tmpdir, 'src', 'genkit', 'plugins', old_import)
        os.makedirs(shim_dir, exist_ok=True)

        init_content = INIT_TEMPLATE.format(old_dist=old_dist, new_dist=new_dist, new_import=new_import)
        with open(os.path.join(shim_dir, '__init__.py'), 'w') as f:
            f.write(init_content)

        # 4. Build package using uv build with public PyPI registry and no workspace config
        build_out_dir = os.path.join(tmpdir, 'dist')
        subprocess.run(
            [
                'uv',
                '--no-config',
                'build',
                '--wheel',
                '--out-dir',
                build_out_dir,
            ],
            cwd=tmpdir,
            check=True,
        )

        wheel_files = [f for f in os.listdir(build_out_dir) if f.endswith('.whl')]
        if not wheel_files:
            raise FileNotFoundError(f'No wheel file found after building {old_dist}')

        wheel_name = wheel_files[0]
        shutil.copy(os.path.join(build_out_dir, wheel_name), os.path.join(dist_dir, wheel_name))
        wheel_file = os.path.join(dist_dir, wheel_name)

        # 5. Publish package if requested
        if publish:
            # Using uv publish or twine depending on environment setup
            subprocess.run(
                ['uv', 'run', 'twine', 'upload', wheel_file],
                check=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description='Build and publish Genkit Python tombstone packages.')
    parser.add_argument(
        '--publish',
        action='store_true',
        help="Actually publish the built wheels to PyPI using 'uv publish'.",
    )
    parser.add_argument(
        '--dist-dir',
        default='./dist-tombstones',
        help='Directory to save the built wheel artifacts.',
    )
    args = parser.parse_args()

    dist_dir = os.path.abspath(args.dist_dir)
    os.makedirs(dist_dir, exist_ok=True)

    for mapping in PLUGINS_MAPPING:
        build_and_publish(mapping, dist_dir, publish=args.publish)


if __name__ == '__main__':
    main()
