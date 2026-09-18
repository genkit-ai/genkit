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

"""An MCP server's tools, handed to a model through generate()."""

import os
import sys
from pathlib import Path

from genkit_google_genai import GoogleAI
from genkit_mcp import McpStdioServerConfig, define_mcp_client

from genkit import Genkit


def google_ai() -> GoogleAI:
    """Build the Google AI plugin, or stop with a readable reason."""
    if not os.environ.get('GEMINI_API_KEY'):
        raise SystemExit(
            'GEMINI_API_KEY is not set.\n'
            'Get a key from https://aistudio.google.com/app/apikey, then:\n'
            '  export GEMINI_API_KEY=your-api-key'
        )
    return GoogleAI()


# Any command works here: `npx`, `uvx`, a binary on PATH. This one is the
# sample's own virtualenv, which is where the server's dependencies live. The
# fields match a `mcpServers` entry from Claude Desktop or Cursor.
BOOKSHOP = McpStdioServerConfig(
    command=sys.executable,
    args=[str(Path(__file__).with_name('bookshop_server.py'))],
)

ai = Genkit(plugins=[google_ai()], model=GoogleAI.gemini_model('gemini-flash-latest'))

# No process starts here. The server is launched the first time its tools are listed.
client = define_mcp_client(ai, 'bookshop', BOOKSHOP)


async def main() -> None:
    # This is what starts the server. Every tool is named after 'bookshop', the
    # name this client was registered under, not after what the server calls
    # itself.
    print('Tools:')
    for tool in await client.get_active_tools():
        print(f'  {tool.name}: {tool.description}')

    every_tool = await ai.generate(
        prompt='I want a fantasy novel. What do you have, and can I walk out with a copy today?',
        tools=[f'{client.name}:tool/*'],
    )
    print(f'All tools: {every_tool.text}')

    one_tool = await ai.generate(
        prompt='Am I too late to come in on Sunday?',
        tools=[f'{client.name}:tool/{client.tool_name("opening_hours")}'],
    )
    print(f'One tool: {one_tool.text}')


# The client is deliberately not closed here. Under `genkit start` this process
# outlives main(), and a closed client serves no tools to the Dev UI.
if __name__ == '__main__':
    ai.run_main(main())
