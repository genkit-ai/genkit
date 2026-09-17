# Genkit MCP Plugin

Status: **stub**. Nothing here is implemented yet. Every function raises `NotImplementedError`.

Connects Genkit to a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio and
exposes that server's tools as Genkit `tool.v2` actions through a dynamic action provider.

See [PLAN.md](./PLAN.md) for the implementation plan, the decisions already taken, and the four core
behaviours that constrain them. Start there.

Tracking issue: <https://github.com/genkit-ai/genkit/issues/6194>

## Intended usage

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

result = await ai.generate(prompt='Echo "hello".', tools=['everything:tool/*'])
await client.close()
```
