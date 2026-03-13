#!/usr/bin/env python3
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the server module."""

import json
import os
import pathlib
import tempfile

from genkit.ai._runtime import RuntimeManager
from genkit.ai._server import ServerSpec


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
