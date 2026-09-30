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

"""A bookshop MCP server, small enough to read in one sitting.

Nothing here knows about Genkit. It is an ordinary stdio MCP server, the same
shape as the ones you would install from npm or PyPI, and `main.py` launches it
as a child process.
"""

from mcp.server.fastmcp import FastMCP

# The name a client sees during initialization. It is not what Genkit names the
# tools after, which is why it reads nothing like the prefix on them.
# A server logs to stderr, which is the sample's own terminal; INFO would print
# a line per request over the sample's output.
mcp = FastMCP('Corner Bookshop', log_level='WARNING')

CATALOGUE = [
    {'title': 'The Left Hand of Darkness', 'author': 'Ursula K. Le Guin', 'topic': 'science fiction'},
    {'title': 'A Wizard of Earthsea', 'author': 'Ursula K. Le Guin', 'topic': 'fantasy'},
    {'title': 'Piranesi', 'author': 'Susanna Clarke', 'topic': 'fantasy'},
    {'title': 'The Making of the Atomic Bomb', 'author': 'Richard Rhodes', 'topic': 'history'},
    {'title': 'Longitude', 'author': 'Dava Sobel', 'topic': 'history'},
]

SHELF = {
    'The Left Hand of Darkness': 4,
    'A Wizard of Earthsea': 0,
    'Piranesi': 2,
    'The Making of the Atomic Bomb': 1,
    'Longitude': 7,
}

HOURS = {
    'saturday': '9am to 6pm',
    'sunday': 'closed',
}


@mcp.tool()
def search_books(topic: str, limit: int = 3) -> list[dict[str, str]]:
    """Find books in the shop's catalogue on a topic, such as fantasy or history."""
    matches = [book for book in CATALOGUE if topic.lower() in book['topic']]
    return matches[:limit]


@mcp.tool()
def check_stock(title: str) -> dict[str, object]:
    """Report how many copies of a title are on the shelf right now."""
    copies = SHELF.get(title)
    if copies is None:
        return {'title': title, 'stocked': False, 'copies': 0}
    return {'title': title, 'stocked': copies > 0, 'copies': copies}


@mcp.tool()
def opening_hours(day: str) -> str:
    """Report the shop's opening hours for a day of the week."""
    return HOURS.get(day.lower(), '10am to 7pm')


if __name__ == '__main__':
    mcp.run()
