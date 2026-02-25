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


"""Simple MCP client example."""

import asyncio

from genkit.ai import Genkit
from genkit.plugins.mcp import McpServerConfig, create_mcp_client

try:
    from genkit.plugins.google_genai import GoogleAI
except ImportError:
    GoogleAI = None  # type: ignore[assignment] - optional dependency


# Simple client example connecting to 'everything' server using npx
async def main() -> None:
    """Run the simple MCP client."""
    # Define the client plugin
    everything_client = create_mcp_client(
        name='everything', config=McpServerConfig(command='npx', args=['-y', '@modelcontextprotocol/server-everything'])
    )

    plugins = [everything_client]
    if GoogleAI:
        plugins.append(GoogleAI())

    Genkit(plugins=plugins)

    await everything_client.connect()

    print('Connected! Listing tools...')  # noqa: T201 - example script

    tools = await everything_client.list_tools()
    for t in tools:
        print(f'- {t.name}: {t.description}')  # noqa: T201 - example script

    await everything_client.close()


if __name__ == '__main__':
    asyncio.run(main())
