# Genkit MCP Client

Connects Genkit to a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio and
exposes that server's tools as Genkit `tool.v2` actions through a dynamic action provider.

Tracking issue: <https://github.com/genkit-ai/genkit/issues/6194>

## Usage

```python
from genkit import Genkit
from genkit_mcp import define_mcp_client

ai = Genkit()

client = define_mcp_client(
    ai,
    'everything',
    command='npx',
    args=['-y', '@modelcontextprotocol/server-everything'],
)

# What the server offers, under the names a model is shown.
for tool in await client.get_active_tools():
    print(tool.name, tool.description)

# Every tool of this server, or just one.
result = await ai.generate(prompt='Echo "hello".', tools=['everything:tool/*'])
one = f'{client.name}:tool/{client.tool_name("echo")}'
result = await ai.generate(prompt='Echo "hello".', tools=[one])
```

## Tool names

Every tool is named `<prefix>_<tool>`, so the tool above is `everything_echo`. The prefix is the
provider name you registered the server under, `everything` here. The separator is an underscore
because Gemini and OpenAI reject a `/` in a function declaration name. The name is checked against
the same rules when the client is defined, so a name a model would refuse fails there rather than
on your first `generate`. A tool name the server lists with such a character, such as `look.up`, has it rewritten to an underscore with a warning. Two
tools that then share a name, such as `look.up` and `look/up`, fail the listing. Length is not
checked at definition, because the tool half of the name is unknown until the server lists it. A
`<prefix>_<tool>` longer than 64 characters is warned about when the tools are listed and passed
through, so the provider's 400 arrives on `generate`.

The prefix is deliberately not the name the server advertises for itself, which is arbitrary
server-controlled text: `@modelcontextprotocol/server-everything` above calls itself
`mcp-servers/everything`, and that slash would be a 400 from both providers.

## Connecting and disconnecting

Connecting is lazy. The child process starts the first time something lists the tools, once per
event loop, and the Dev UI's reflection server has a loop of its own. So under `genkit start`, a
flow run from the Dev UI and your `main()` talk to two separate server processes. A server that
keeps state, such as a session or an open file, keeps one copy of it for each.

`await client.restart()` reconnects a live client, for a server whose configuration or state
changed underneath you. `await client.close()` disconnects and stops the child process: a
closed client cannot be restarted, and anything that asks for a tool afterwards fails with a
`GenkitError` whose status is `FAILED_PRECONDITION`. To use that server again, define a new
client. A server that cannot start, or whose process dies, fails with status `UNAVAILABLE`. A
connection that died on its own is replaced on next use, so one bad server start is not
permanent.

Nothing needs closing at exit. Call `close()` when you are done with a server before your process
is, not as cleanup on the way out: a process running under `genkit start` outlives your `main()`,
and a closed client's tools can no longer be selected by a flow. Whether the Dev UI lists a
provider's tools at all depends on core registry support, closed or not.

## Several servers behind one name

`define_mcp_host` registers one provider over any number of servers, so a single selector reaches
all of them:

```python
from genkit import Genkit
from genkit_mcp import define_mcp_host

ai = Genkit()

host = define_mcp_host(
    ai,
    'hosted',
    mcp_servers={
        'everything': {'command': 'npx', 'args': ['-y', '@modelcontextprotocol/server-everything']},
        'git': {'command': 'uvx', 'args': ['mcp-server-git']},
    },
)

result = await ai.generate(prompt='Echo "hello".', tools=['hosted:tool/*'])
one = f'{host.name}:tool/{host.tool_name("everything", "echo")}'
result = await ai.generate(prompt='Echo "hello".', tools=[one])
```

`mcp_servers` has the shape of the `mcpServers` block that Claude Desktop and Cursor read, so an
existing one can be passed through: `mcp_servers=json.load(f)['mcpServers']`. Only stdio entries are
accepted, and an entry with any key besides `command`, `args`, `env`, `cwd` and `type` fails.

Each server keeps its own namespace, so the tools above are `everything_echo` and `git_git_log`. The
key a server has in `mcp_servers` is its prefix. The host's own name is only the provider name, so
it has to fit a selector but no model ever reads it.

A server that cannot start, or whose process dies, fails with status `UNAVAILABLE`. The host logs
it and skips it, so the other servers stay selectable. Any other failure fails the whole listing:
a server that breaks the protocol, a server that answers the listing with an error, and two tools
that share a name. Only the server you ask for is affected by `await host.connect(...)`,
`await host.disconnect(...)` and `await host.reconnect(...)`. `await host.close()` disconnects every
server the host holds, and like `McpClient.close()` it is final: afterwards the host fails with
status `FAILED_PRECONDITION`, so define a new host to use those servers again.

One provider means one listing. A tool call that fails on one server drops the listing for all of
them, so the next `generate` lists every server again.
