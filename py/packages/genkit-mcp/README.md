# Genkit MCP Client

Connects Genkit to a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio and
exposes that server's tools as Genkit `tool.v2` actions through a dynamic action provider.

Tracking issue: <https://github.com/genkit-ai/genkit/issues/6194>

## Usage

The example uses Gemini through `genkit-google-genai` and needs `GEMINI_API_KEY` set. Any model plugin works.

```python
import asyncio

from genkit import Genkit
from genkit_google_genai import GoogleAI
from genkit_mcp import define_mcp_client

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))

client = define_mcp_client(
    ai,
    'everything',
    command='npx',
    args=['-y', '@modelcontextprotocol/server-everything'],
)


async def main() -> None:
    try:
        # What the server offers, under the names a model is shown.
        tools = await client.get_active_tools()
        for tool in tools:
            print(tool.name, tool.description)

        # Every tool of this server, or just one.
        result = await ai.generate(prompt='Echo "hello".', tools=['everything:tool/*'])
        one = f'{client.name}:tool/{client.tool_name("echo")}'
        result = await ai.generate(prompt='Echo "hello".', tools=[one])

        # The listing itself is a list of Genkit tools, so it can be handed to a
        # generate call or called directly.
        result = await ai.generate(prompt='Echo "hello".', tools=tools)
        print(result.text)
        echoed = (await tools[0]({'message': 'hello'})).output
        print(echoed)
    finally:
        await client.close()


asyncio.run(main())
```

## Tool names

Every tool is named `<prefix>_<tool>`, so the tool above is `everything_echo`. The prefix is the
provider name you registered the server under, `everything` here. The separator is an underscore
because Gemini and OpenAI reject a `/` in a function declaration name. A prefix holding a character
they reject fails when the client is defined, rather than on your first `generate`. A tool name the
server lists with such a character, such as `look.up`, has it rewritten to an underscore with a
warning. Two tools that then share a name, such as `look.up` and `look/up`, are both left out with a
warning, since either could answer a call meant for the other. Length is not checked at definition,
because the tool half of the name is unknown until the server lists it. A `<prefix>_<tool>` longer
than 64 characters is warned about when the tools are listed and passed through, so the provider's
400 arrives on `generate`. Nor are names checked across servers: a
prefix `fake` with a tool `await_stall` and a prefix `fake_await` with a tool `stall` both make
`fake_await_stall`, and `generate` refuses to bind both. Each prefix names one provider, so a second
client under a name already registered fails when it is defined, unless the client holding the name
is closed.

The prefix is deliberately not the name the server advertises for itself, which is arbitrary
server-controlled text: `@modelcontextprotocol/server-everything` above calls itself
`mcp-servers/everything`, and that slash would be a 400 from both providers.

## Connecting and disconnecting

Connecting is lazy. The child process starts the first time something lists the tools, once per
event loop, and the Dev UI's reflection server has a loop of its own. So under `genkit start`, a
flow run from the Dev UI and your `main()` talk to two separate server processes. A server that
keeps state, such as a session or an open file, keeps one copy of it for each, and servers that bind
a port, take a lock file or own a browser profile can clash.

`await client.restart()` reconnects a live client, for a server whose configuration or state changed
underneath you. A request still in flight when it restarts, on any event loop, fails with status
`UNAVAILABLE` and can be retried. `await client.close()` disconnects and stops the child process: a
closed client cannot be restarted. To use that server again, define a new client. Afterwards
`generate` with one of its tools, and `get_active_tools()`, fail with a `GenkitError` whose status
is `FAILED_PRECONDITION`. A server that cannot start, or whose process dies, fails with status
`UNAVAILABLE`, and a request the server does not answer within `request_timeout_millis` fails with
`DEADLINE_EXCEEDED`. A connection that died on its own is replaced on next use, so one bad server start is
not permanent. Resolving a tool by its action key, as the Dev UI does to run one, reports neither
status: core finds no action for a closed client or a server that cannot start.

Nothing needs closing at exit. Call `close()` when you are done with a server before your process
is, not as cleanup on the way out: a process running under `genkit start` outlives your `main()`,
and a closed client's tools can no longer be selected by a flow. The Dev UI lists each registered
client's tools, and none for a closed one or one whose server cannot start.

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

`mcp_servers` takes an `mcpServers` block of stdio servers, as Claude Desktop and Cursor write it, so
such a block can be loaded from its file: `mcp_servers=json.load(f)['mcpServers']`. An entry that is
not a stdio entry, such as one with a `url`, `type: 'sse'` or `disabled: true`, fails the host with
an error naming that server. So does an entry with any key besides `command`, `args`, `env`, `cwd`
and `type`. `await host.connect(name, entry)` takes one entry too, so a server from the same file
can be added later: `await host.connect('git', config['mcpServers']['git'])`.

An entry's `env` often holds secrets, and a traceback that renders frame locals, as Genkit's console
logging does, can print them. Once a host is defined, the plugin keeps `env` out of its own frames.
It cannot do that for your frames: in the `connect` call above, `config` stays a local of the
calling function, so a failed connect logged with its traceback prints the whole block. The same
goes for a `mcp_servers` mapping that fails validation when the host is defined. The child does not
inherit this process's environment, so the secrets have to be in `env`. To keep them out of your
frames, build the entry in a helper and pass its result straight on, as in
`await host.connect('git', load_server('git'))`, so no frame that can still fail holds it.

Each server keeps its own namespace, so the tools above are `everything_echo` and `git_git_log`. The
key a server has in `mcp_servers` is its prefix. The host's own name is only the provider name, so
it has to fit a selector but no model ever reads it.

A server that cannot start, or whose process dies, fails with status `UNAVAILABLE`, and one that
does not answer a request in time fails with `DEADLINE_EXCEEDED`. The host logs either and skips it,
so the other servers stay selectable. A server disconnected or replaced while the host was listing
it is left out of that listing. Any other failure fails the whole listing:
a server that breaks the protocol, a server that answers the listing with an error, and two tools
that share a name. Only the server you ask for is affected by `await host.connect(...)`,
`await host.disconnect(...)` and `await host.reconnect(...)`. `connect` and `reconnect` start the
server and list it before they return, so a server that cannot start fails right there. A server
that `connect` cannot list is stopped, and the host keeps what it held before, including any server
of that name. `await host.close()` disconnects every
server the host holds, and like `McpClient.close()` it is final: afterwards the host fails with
status `FAILED_PRECONDITION`, so define a new host to use those servers again.

One provider means one listing. A tool call that fails on one server drops the listing for all of
them, so the next `generate` lists every server again.
