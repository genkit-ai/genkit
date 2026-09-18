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

import sys
from pathlib import Path

from genkit_google_genai import GoogleAI
from genkit_mcp import McpStdioServerConfig, define_mcp_client

from genkit import Genkit

# Any command works here: `npx`, `uvx`, a binary on PATH. This one is the
# sample's own virtualenv, which is where the server's dependencies live. The
# fields match a `mcpServers` entry from Claude Desktop or Cursor.
BOOKSHOP = McpStdioServerConfig(
    command=sys.executable,
    args=[str(Path(__file__).with_name('bookshop_server.py'))],
)

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))

# No process starts here. The server is launched the first time its tools are listed.
client = define_mcp_client(ai, 'bookshop', BOOKSHOP)


@ai.flow()
async def browse_shelves(topic: str) -> str:
    """Shop for a book on a topic with every bookshop tool on offer."""
    res = await ai.generate(
        prompt=f'I want a {topic} novel. What do you have, and can I walk out with a copy today?',
        tools=[f'{client.name}:tool/*'],
    )
    return res.text


@ai.flow()
async def ask_opening_hours(day: str) -> str:
    """Ask about a day's opening hours with just the one tool that answers it."""
    res = await ai.generate(
        prompt=f'Am I too late to come in on {day}?',
        tools=[f'{client.name}:tool/{client.tool_name("opening_hours")}'],
    )
    return res.text


async def main() -> None:
    # This is what starts the server. Every tool is named after 'bookshop', the
    # name this client was registered under, not after what the server calls
    # itself.
    print('Tools:')
    for tool in await client.get_active_tools():
        print(f'  {tool.name}: {tool.description}')

    print(f'All tools: {await browse_shelves("fantasy")}')
    print(f'One tool: {await ask_opening_hours("Sunday")}')


# The client stays open: under `genkit start` the flows run from the Dev UI
# after main() returns, and a closed client cannot be reopened.
if __name__ == '__main__':
    ai.run_main(main())
