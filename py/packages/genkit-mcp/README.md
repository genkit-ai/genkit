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
`UNAVAILABLE`. A connection that died on its own is replaced on next use, so one bad server start is
not permanent. Resolving a tool by its action key, as the Dev UI does to run one, reports neither
status: core finds no action for a closed client or a server that cannot start.

Nothing needs closing at exit. Call `close()` when you are done with a server before your process
is, not as cleanup on the way out: a process running under `genkit start` outlives your `main()`,
and a closed client's tools can no longer be selected by a flow. The Dev UI lists each registered
client's tools, and none for a closed one or one whose server cannot start.
