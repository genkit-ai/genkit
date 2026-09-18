# MCP hello

An MCP server's tools, handed to a model through `generate()`.

`bookshop_server.py` is an ordinary stdio MCP server with three tools:
`search_books`, `check_stock` and `opening_hours`. The sample launches it
itself, so there is nothing to install or download first.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

In Dev UI:

```bash
genkit start -- uv run src/main.py
```

Then open [http://localhost:4000](http://localhost:4000) and run
`browse_shelves` with a topic such as `fantasy`, or `ask_opening_hours` with a
day such as `Sunday`.

## What it shows

`define_mcp_client(ai, 'bookshop', ...)` registers the server under the name
`bookshop`. Nothing runs yet: the child process starts the first time something
asks for the tools.

Ask the client what it found, rather than reading the server's source:

```python
for tool in await client.get_active_tools():
    print(tool.name, tool.description)
```

Then select them the way you select any tool. `browse_shelves` binds every
tool and `ask_opening_hours` binds one:

```python
tools=[f'{client.name}:tool/*']                              # every tool
tools=[f'{client.name}:tool/{client.tool_name("opening_hours")}']  # one
```

The name after `tool/` is `bookshop_<tool>`, where `bookshop` is the name you
registered the server under. Pass `tool_prefix=` to `define_mcp_client` to
choose a different one. The separator is an underscore because Gemini and
OpenAI reject a `/` in a function declaration name.

The server calls itself `Corner Bookshop`, which no model would accept in a
tool name. That name is never used for naming, which is why `tool_name()` can
answer without connecting.

`check_stock` returns a dict, so it arrives as structured output. Text,
images, audio and resource links all map across too.

The sample never calls `client.close()`. Under `genkit start` the flows run
from the Dev UI after `main()` returns, and they need the client open. Closing
is final: after `close()`, listing tools, calling them and `restart()` all
raise `McpConnectionClosedError`. Call `close()` when you are done with a
server before your process is. Call `restart()` on a live client to reconnect
after the server changes, such as when you edit its tools. To use a closed
server again, call `define_mcp_client` for it again.

## Tests

The tests drive the same wiring with a fake model, so they need no API key and
no network, and they launch the real server:

```bash
uv run pytest samples/mcp-hello --no-cov   # from py/
```
