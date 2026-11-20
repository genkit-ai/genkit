# Genkit MCP

See [Genkit MCP documentation](https://genkit.dev/docs/model-context-protocol/).

> **Building with a coding agent? Install the Genkit JS skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-js
> ```
>
> It teaches your agent the current Genkit JS APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

This plugin provides integration between Genkit and the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). MCP is an open standard allowing developers to build "servers" which provide tools, resources, and prompts to clients. Genkit MCP allows Genkit developers to:
- Consume MCP tools, prompts, and resources as a client using `defineMcpHost` or `defineMcpClient`.
- Provide Genkit tools and prompts as an MCP server using `createMcpServer`.

## Installation

To get started, you'll need Genkit and the MCP plugin:

```bash
npm i genkit @genkit-ai/mcp
```

## MCP Host

To connect to one or more MCP servers, use the `defineMcpHost` function. It returns a `GenkitMcpHost` instance that manages connections to the configured MCP servers, and registers it with your Genkit instance under `name` so its tools can be referenced by name.

```ts
import { googleAI } from '@genkit-ai/google-genai';
import { defineMcpHost } from '@genkit-ai/mcp';
import { genkit } from 'genkit';

const ai = genkit({
  plugins: [googleAI()],
});

const mcpHost = defineMcpHost(ai, {
  name: 'myMcpClients', // Also the name used to reference the host's tools
  mcpServers: {
    // Each key (e.g., 'fs', 'memory') becomes a namespace for the server's tools.
    fs: {
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem', process.cwd()],
    },
    memory: {
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-memory'],
    },
  },
});

(async () => {
  // Provide MCP tools to the model of your choice.
  const { text } = await ai.generate({
    model: googleAI.model('gemini-flash-latest'),
    prompt: `Analyze all files in ${process.cwd()}.`,
    // All tools from all servers. 'myMcpClients:tool/fs/*' would select only
    // the filesystem tools, 'myMcpClients:tool/fs/read_file' a single one.
    tools: ['myMcpClients:tool/*'],
    // Alternatively, pass the tool and resource actions directly:
    // tools: await mcpHost.getActiveTools(ai),
    // resources: await mcpHost.getActiveResources(ai),
  });

  console.log(text);

  await mcpHost.close();
})();
```

### `defineMcpHost()` Options

-   **`name`**: (required, string) A name for the MCP host. It is advertised to MCP servers as the client name and is used to reference the host's tools and resources (e.g. `myMcpClients:tool/*`).
-   **`version`**: (optional, string) The version of the MCP host. Defaults to "1.0.0".
-   **`rawToolResponses`**: (optional, boolean) When `true`, tool responses are returned in their raw MCP format; otherwise, they are processed for Genkit compatibility. Defaults to `false`.
-   **`cacheTtlMillis`**: (optional, number) How long the list of available actions is cached. Defaults to 3000. Negative values disable caching. The cache is invalidated whenever a connection changes.
-   **`mcpServers`**: (optional, object) An object where each key is a client-side name (namespace) for an MCP server, and the value is the configuration for that server. Servers can also be added later with `mcpHost.connect(name, config)`.

    Each server configuration object can include:
    -   **`disabled`**: (optional, boolean) If `true`, this server connection will not be attempted. Defaults to `false`.
    -   One of the following server connection configurations:
        -   Parameters for launching a local server process using the stdio MCP transport.
            -   **`command`**: (required, string) Shell command path for launching the MCP server (e.g., `npx`, `python`).
            -   **`args`**: (optional, string[]) Array of string arguments to pass to the command.
            -   **`env`**: (optional, Record<string, string>) Key-value object of environment variables.
        -   **`url`**: (string) The URL of a remote server to connect to using the Streamable HTTP MCP transport.
        -   **`transport`**: An existing MCP transport object for connecting to the server.


## MCP Client (Single Server)

For scenarios where you only need to connect to a single MCP server, or prefer to manage client instances individually, you can use `defineMcpClient`.

```ts
import { googleAI } from '@genkit-ai/google-genai';
import { defineMcpClient } from '@genkit-ai/mcp';
import { genkit } from 'genkit';

const ai = genkit({
  plugins: [googleAI()],
});

const myFsClient = defineMcpClient(ai, {
  name: 'myFileSystemClient', // Also the name used to reference its tools
  mcpServer: {
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-filesystem', process.cwd()],
  },
  // rawToolResponses: true, // Optional: get raw MCP responses
});

(async () => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-flash-latest'),
    prompt: 'List files in ' + process.cwd(),
    tools: ['myFileSystemClient:tool/*'], // or: await myFsClient.getActiveTools(ai)
  });
  console.log(text);

  await myFsClient.disable();
})();
```

### `defineMcpClient()` Options

The `defineMcpClient` function takes an `McpClientOptionsWithCache` object:
-   **`name`**: (required, string) A unique name for this client instance. It is used to reference the client's tools and resources (e.g. `myFileSystemClient:tool/*`).
-   **`mcpServer`**: (required, object) The server configuration, with the same options as each entry of `mcpServers` in `defineMcpHost()`.
-   **`serverName`**: (optional, string) Namespace for the server's tools. Defaults to the name the server advertises.
-   **`version`**: (optional, string) Version for this client instance. Defaults to "1.0.0".
-   **`rawToolResponses`**: (optional, boolean) Same as for `defineMcpHost()`.
-   **`cacheTtlMillis`**: (optional, number) Same as for `defineMcpHost()`.

### Unregistered clients

`defineMcpHost` and `defineMcpClient` register a dynamic action provider with your Genkit instance, so call them once per name, typically at startup. If you need a short-lived connection that should not be registered (for example, a client per request that authenticates as the calling user), construct `GenkitMcpClient` directly and pass its tools to `generate`:

```ts
import { GenkitMcpClient } from '@genkit-ai/mcp';

const client = new GenkitMcpClient({
  name: 'perRequestClient',
  mcpServer: {
    url: 'https://example.com/mcp',
    requestInit: { headers: { Authorization: `Bearer ${userToken}` } },
  },
});
try {
  const { text } = await ai.generate({
    prompt: 'What is on my calendar today?',
    tools: await client.getActiveTools(ai),
  });
} finally {
  await client.disable();
}
```

### Using MCP Actions (Tools, Prompts)

Both `GenkitMcpHost` (via `getActiveTools()`) and `GenkitMcpClient` (via `getActiveTools()`) discover available tools from their connected and enabled MCP server(s). These tools are standard Genkit `ToolAction` instances and can be provided to Genkit models.

MCP prompts can be fetched using `mcpHost.getPrompt(ai, serverName, promptName)` or `mcpClient.getPrompt(ai, promptName)`. These return an `ExecutablePrompt`.

All MCP actions (tools, prompts, resources) are namespaced.
- For `defineMcpHost`, the namespace is the key you provide for that server in the `mcpServers` configuration (e.g., `localFs/read_file`).
- For `defineMcpClient`, the namespace is the `serverName` option, or the name the server advertises if not set (e.g., `<serverName>/read_file`).

### Tool Responses

MCP tools return a `content` array as opposed to a structured response like most Genkit tools. The Genkit MCP plugin attempts to parse and coerce returned content:

1. If the content is text and valid JSON, it is parsed and returned as a JSON object.
2. If the content is text but not valid JSON, the raw text is returned.
3. If the content contains a single non-text part (e.g., an image), that part is returned directly.
4. If the content contains multiple or mixed parts (e.g., text and an image), the full content response array is returned.

## MCP Server

You can also expose all of the tools and prompts from a Genkit instance as an MCP server using the `createMcpServer` function.

```ts
import { googleAI } from '@genkit-ai/google-genai';
import { createMcpServer } from '@genkit-ai/mcp';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { genkit, z } from 'genkit/beta';

const ai = genkit({
  plugins: [googleAI()],
});

ai.defineTool(
  {
    name: 'add',
    description: 'add two numbers together',
    inputSchema: z.object({ a: z.number(), b: z.number() }),
    outputSchema: z.number(),
  },
  async ({ a, b }) => {
    return a + b;
  }
);

ai.definePrompt(
  {
    name: 'happy',
    description: 'everybody together now',
    input: {
      schema: z.object({
        action: z.string().default('clap your hands').optional(),
      }),
    },
  },
  `If you're happy and you know it, {{action}}.`
);

ai.defineResource(
  {
    name: 'my resouces',
    uri: 'my://resource',
  },
  async () => {
    return {
      content: [
        {
          text: 'my resource',
        },
      ],
    };
  }
);

ai.defineResource(
  {
    name: 'file',
    template: 'file://{path}',
  },
  async ({ uri }) => {
    return {
      content: [
        {
          text: `file contents for ${uri}`,
        },
      ],
    };
  }
);

// Use createMcpServer
const server = createMcpServer(ai, {
  name: 'example_server',
  version: '0.0.1',
});
// Setup (async) then starts with stdio transport by default
server.setup().then(async () => {
  await server.start();
  const transport = new StdioServerTransport();
  await server!.server?.connect(transport);
});
```

The `createMcpServer` function returns a `GenkitMcpServer` instance. The `start()` method on this instance will start an MCP server (using the stdio transport by default) that exposes all registered Genkit tools and prompts. To start the server with a different MCP transport, you can pass the transport instance to the `start()` method (e.g., `server.start(customMcpTransport)`).

### `createMcpServer()` Options
- **`name`**: (required, string) The name you want to give your server for MCP inspection.
- **`version`**: (optional, string) The version your server will advertise to clients. Defaults to "1.0.0".

### Known Limitations

- MCP prompts are only able to take string parameters, so inputs to schemas must be objects with only string property values.
- MCP prompts only support `user` and `model` messages. `system` messages are not supported.
- MCP prompts only support a single "type" within a message so you can't mix media and text in the same message.

### Testing your MCP server

You can test your MCP server using the official inspector. For example, if your server code compiled into `dist/index.js`, you could run:

    npx @modelcontextprotocol/inspector dist/index.js

Once you start the inspector, you can list prompts and actions and test them out manually.
