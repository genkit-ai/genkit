#!/usr/bin/env python3
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the server module."""

import json
import os
import pathlib
import tempfile

import pytest

from genkit._ai._runtime import RuntimeManager
from genkit._core._reflection import ServerSpec


def test_server_spec() -> None:
    """Test the ServerSpec class.

    Verifies that the ServerSpec class correctly generates URLs and
    handles different schemes, hosts, and ports.
    """
    assert ServerSpec(scheme='http', host='localhost', port=3100).url == 'http://localhost:3100'

    # Test with different schemes and hosts
    assert ServerSpec(scheme='https', host='example.com', port=8080).url == 'https://example.com:8080'

    # Test with default values
    spec = ServerSpec(port=5000)
    assert spec.scheme == 'http'
    assert spec.host == 'localhost'
    assert spec.url == 'http://localhost:5000'


def test_runtime_manager() -> None:
    """Test the RuntimeManager class.

    Verifies that the RuntimeManager class correctly creates and
    manages runtime metadata files, including cleanup on exit.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        spec = ServerSpec(port=3100)

        # Test runtime file creation using context manager
        runtime_path = None
        with RuntimeManager(spec=spec, runtime_dir=temp_dir) as rm:
            runtime_path = rm.write_runtime_file()
            assert runtime_path.exists()

            # Verify file content
            content = json.loads(runtime_path.read_text(encoding='utf-8'))
            assert isinstance(content, dict)
            assert 'pid' in content
            assert content['reflectionServerUrl'] == 'http://localhost:3100'
            assert 'timestamp' in content

        # Verify cleanup on exit
        assert runtime_path is not None
        assert not runtime_path.exists()

        # Test directory creation
        new_dir = os.path.join(temp_dir, 'new_dir')
        with RuntimeManager(spec=spec, runtime_dir=new_dir) as rm:
            runtime_path = rm.write_runtime_file()
            assert pathlib.Path(new_dir).exists()
            assert runtime_path.exists()


def test_find_project_root_matches_cli_markers(tmp_path: pathlib.Path) -> None:
    """Runtime files land where the CLI's ``findProjectRoot`` looks."""
    from genkit._ai._runtime import find_project_root

    project = tmp_path / 'repo'
    sub = project / 'services' / 'api'
    sub.mkdir(parents=True)
    (project / 'pyproject.toml').write_text('')

    assert find_project_root(sub) == project
    (sub.parent / 'go.mod').write_text('')
    assert find_project_root(sub) == sub.parent


def test_find_project_root_falls_back_to_start(tmp_path: pathlib.Path) -> None:
    from genkit._ai._runtime import find_project_root

    bare = tmp_path / 'no' / 'markers'
    bare.mkdir(parents=True)
    root = find_project_root(bare)
    # A marker above tmp_path (e.g. a pyproject in a parent) is a legitimate hit.
    assert root == bare or bare.is_relative_to(root)


def test_runtime_manager_defaults_to_project_root(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / 'repo'
    sub = project / 'src' / 'app'
    sub.mkdir(parents=True)
    (project / 'requirements.txt').write_text('')
    monkeypatch.chdir(sub)

    with RuntimeManager(spec=ServerSpec(port=3100)) as rm:
        path = rm.write_runtime_file()
        assert path.parent == project / '.genkit' / 'runtimes'
    assert not path.exists()


def test_runtime_manager_removes_empty_runtime_dirs(tmp_path: pathlib.Path) -> None:
    """Exiting leaves no empty `.genkit/runtimes` behind in the project."""
    runtime_dir = tmp_path / '.genkit' / 'runtimes'
    with RuntimeManager(spec=ServerSpec(port=3100), runtime_dir=runtime_dir):
        assert runtime_dir.exists()
    assert not (tmp_path / '.genkit').exists()

    with RuntimeManager(spec=ServerSpec(port=3100), runtime_dir=runtime_dir, lazy_write=True):
        assert runtime_dir.exists()
    assert not (tmp_path / '.genkit').exists()


def test_runtime_manager_keeps_dirs_with_other_files(tmp_path: pathlib.Path) -> None:
    runtime_dir = tmp_path / '.genkit' / 'runtimes'
    runtime_dir.mkdir(parents=True)
    (tmp_path / '.genkit' / 'traces.json').write_text('{}')
    with RuntimeManager(spec=ServerSpec(port=3100), runtime_dir=runtime_dir):
        pass
    assert not runtime_dir.exists()
    assert (tmp_path / '.genkit' / 'traces.json').exists()
