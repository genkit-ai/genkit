# Copyright 2025 Google LLC
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

"""Smoke tests for package structure."""

from genkit_google_cloud import package_name as google_cloud_package_name
from genkit_google_genai import package_name as google_genai_package_name
from genkit_mcp import (
    McpClient,
    McpClientError,
    McpConnectionClosedError,
    McpConnectionFailedError,
    McpProtocolError,
    McpStdioServerConfig,
    create_mcp_client,
    define_mcp_client,
)
from genkit_ollama import package_name as ollama_package_name
from genkit_vertexai import package_name as vertex_ai_package_name


def test_package_names() -> None:
    """A test that ensure that the package imports work correctly.

    This test verifies that the package imports work correctly from the
    end-user perspective.
    """
    assert google_cloud_package_name() == 'genkit_google_cloud'
    assert google_genai_package_name() == 'genkit_google_genai'
    assert ollama_package_name() == 'genkit_ollama'
    assert vertex_ai_package_name() == 'genkit_vertexai'


def test_mcp_exports() -> None:
    """A test that ensures the MCP entry points import from the installed package.

    This test verifies that the package imports work correctly from the
    end-user perspective.
    """
    assert McpClient.__module__ == 'genkit_mcp._client'
    assert McpStdioServerConfig.__module__ == 'genkit_mcp._config'
    assert create_mcp_client.__module__ == 'genkit_mcp._client'
    assert define_mcp_client.__module__ == 'genkit_mcp._client'
    assert issubclass(McpConnectionClosedError, McpClientError)
    assert issubclass(McpConnectionFailedError, McpClientError)
    assert issubclass(McpProtocolError, McpClientError)
    assert not issubclass(McpClientError, RuntimeError)
