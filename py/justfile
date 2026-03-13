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
# Genkit Python SDK — run `just py` to see all available commands.
#
# Usage: just py <command>
#
# Examples:
#   just py lint           # Lint and type-check all Python code
#   just py test           # Run all Python tests
#   just py sample         # Interactive sample runner
#   just py fmt            # Format Python code

set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

# source_directory() returns the directory of *this* justfile, even when
# invoked as a submodule via `mod py` from the root justfile.

py_dir := source_directory()
top_dir := parent_directory(py_dir)

# Default: show available commands.
default:
    @just --list --unsorted

# --- Development -------------------------------------------------------

# Lint and type-check all Python code (ruff, ty, pyrefly, pyright).
lint:
    uv run --directory "{{ py_dir }}" ruff check --fix --preview --unsafe-fixes .
    uv run --directory "{{ py_dir }}" ruff format --preview .
    uv run --directory "{{ py_dir }}" ty check .
    uv run --directory "{{ py_dir }}" pyrefly check .
    uv run --directory "{{ py_dir }}" pyright packages/

# Format Python code with ruff.
fmt:
    uv run --directory "{{ py_dir }}" ruff format --preview .
    uv run --directory "{{ py_dir }}" ruff check --fix --preview --unsafe-fixes .

# Run all Python tests.
test:
    "{{ py_dir }}/bin/run_python_tests"

# Watch mode for tests (re-run on file changes).
test-watch:
    "{{ py_dir }}/bin/watch_python_tests"

# Run tests with nox (multi-version).
test-nox:
    "{{ py_dir }}/bin/run_python_tests_with_nox"

# Run tests with tox (multi-version).
test-tox:
    "{{ py_dir }}/bin/run_python_tests_with_tox"

# Run security checks (bandit, pip-audit).
security:
    "{{ py_dir }}/bin/run_python_security_checks"

# Sync workspace and verify all packages install.
sync:
    uv sync --directory "{{ py_dir }}"
    uv pip check --directory "{{ py_dir }}"

# Check workspace consistency (versions, naming, deps).
check:
    "{{ py_dir }}/bin/check_consistency"

# Check lockfile is up to date.
check-lock:
    uv lock --check --directory "{{ py_dir }}"

# Clean build artifacts and caches.
clean:
    "{{ py_dir }}/bin/cleanup"

# --- Samples -----------------------------------------------------------

# Run a sample (interactive picker or by name).
sample NAME="":
    "{{ py_dir }}/bin/run_sample" {{ NAME }}

# Test flows in a sample.
test-sample NAME="":
    "{{ py_dir }}/bin/test_sample_flows" {{ NAME }}

# --- Code Generation ---------------------------------------------------

# Regenerate typing.py from JSON schema.
generate-schema:
    "{{ py_dir }}/bin/generate_schema_typing"

# --- Release -----------------------------------------------------------

# Bump version in all packages.
bump-version VERSION:
    "{{ py_dir }}/bin/bump_version" {{ VERSION }}

# Pre-release validation suite.
release-check:
    "{{ py_dir }}/bin/release_check"

# Build wheel/sdist packages.
build:
    "{{ py_dir }}/bin/build_dists"

# Create a GitHub release.
create-release:
    "{{ py_dir }}/bin/create_release"

# Publish to PyPI.
publish:
    "{{ py_dir }}/bin/publish_pypi.sh"

# Check version consistency across packages.
check-versions:
    "{{ py_dir }}/bin/check_versions"

# Validate release documentation.
validate-docs:
    "{{ py_dir }}/bin/validate_release_docs"

# --- Model Conformance -------------------------------------------------

# Run model conformance tests in parallel (see `conform check-model --help`).
test-conformance *ARGS:
    uv run --directory "{{ py_dir }}" --active conform check-model {{ ARGS }}

# Check that every model plugin has conformance files.
check-conformance:
    uv run --directory "{{ py_dir }}" --active conform check-plugin

# List available conformance plugins and env-var readiness.
list-conformance:
    uv run --directory "{{ py_dir }}" --active conform list
