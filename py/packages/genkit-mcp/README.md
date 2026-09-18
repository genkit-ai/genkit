# Genkit MCP Client

Connects Genkit to a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio and
exposes that server's tools as Genkit `tool.v2` actions through a dynamic action provider.

Tracking issue: <https://github.com/genkit-ai/genkit/issues/6194>

## Usage

```python
from genkit import Genkit
from genkit_mcp import McpStdioServerConfig, define_mcp_client

ai = Genkit()

client = define_mcp_client(
    ai,
    name='everything',
    server=McpStdioServerConfig(
        command='npx',
        args=['-y', '@modelcontextprotocol/server-everything'],
    ),
)

# What the server offers, under the names a model is shown.
for tool in await client.get_active_tools():
    print(tool.name, tool.description)

# Every tool of this server, or just one.
result = await ai.generate(prompt='Echo "hello".', tools=['everything:tool/*'])
one = f'{client.name}:tool/{client.tool_name("echo")}'
result = await ai.generate(prompt='Echo "hello".', tools=[one])
```

## Building a client without registering it

`create_mcp_client` builds the same client and registers nothing:

```python
from genkit_mcp import McpStdioServerConfig, create_mcp_client

client = create_mcp_client(
    name='everything',
    server=McpStdioServerConfig(
        command='npx',
        args=['-y', '@modelcontextprotocol/server-everything'],
    ),
)
tools = await client.get_active_tools()
```

It takes no `Genkit` instance, and no selector reaches this server: the actions `get_active_tools()`
returns are unregistered, for a dynamic action provider of your own to serve. Naming, connecting,
`restart()` and `close()` behave as they do for a defined client. `get_active_tools()` lists from the
server on every call, registered or not, so what it returns is what the server serves now.

## Tool names

Every tool is named `<prefix>_<tool>`, so the tool above is `everything_echo`. The prefix is the
provider name you registered the server under, `everything` here, and `tool_prefix=` overrides it.
The separator is an underscore because Gemini and OpenAI reject a `/` in a function declaration
name, and the prefix is checked against the same rules when the client is defined, so a name a model
would refuse fails there rather than on your first `generate`.

The prefix is deliberately not the name the server advertises for itself, which is arbitrary
server-controlled text: `@modelcontextprotocol/server-everything` above calls itself
`mcp-servers/everything`, and that slash would be a 400 from both providers.

## Connecting and disconnecting

Connecting is lazy. The child process starts the first time something lists the tools, once per
event loop, and the Dev UI's reflection server has a loop of its own.

`await client.restart()` reconnects, for a server whose configuration or state changed underneath
you. `await client.close()` disconnects and stops the child process for good; anything that asks for
a tool afterwards fails with `McpConnectionClosedError`. A connection that died on its own is
replaced on next use, so one bad server start is not permanent.

Nothing needs closing at exit. Call `close()` when you are done with a server before your process
is, not as cleanup on the way out: a process running under `genkit start` outlives your `main()`,
and a closed client shows no tools in the Dev UI.
