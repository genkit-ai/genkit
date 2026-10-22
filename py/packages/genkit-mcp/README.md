# Genkit MCP Client

Connects Genkit to a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio and
exposes that server's tools as Genkit `tool.v2` actions through a dynamic action provider.

Under development: the package is not published yet and exports no public API. The entry point below
is the target shape, landing with the client implementation.

Tracking issue: <https://github.com/genkit-ai/genkit/issues/6194>

## Intended usage

The example uses Gemini through `genkit-google-genai` and needs `GEMINI_API_KEY` set. Any model plugin works.

```python
import asyncio

from genkit import Genkit
from genkit_google_genai import GoogleAI
from genkit_mcp import define_mcp_client

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))

client = define_mcp_client(
    ai,
    'everything',
    command='npx',
    args=['-y', '@modelcontextprotocol/server-everything'],
)


async def main() -> None:
    try:
        result = await ai.generate(prompt='Echo "hello".', tools=['everything:tool/*'])
        print(result.text)
    finally:
        await client.close()


asyncio.run(main())
```

## One server process per event loop

The client holds one connection, and so one server process, per event loop. While the Dev UI runs,
its reflection server has its own loop, so a second process starts for the same server. The two
processes do not share state, and servers that bind a port, take a lock file or own a browser profile
can clash.
