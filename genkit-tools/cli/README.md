# Genkit CLI

The package contains the CLI for Genkit, an open source framework with rich local tooling to help app developers build, test, deploy, and monitor AI-powered features for their apps with confidence. Genkit is built by Firebase, Google's app development platform that is trusted by millions of businesses around the world.

> **Building with a coding agent? Install the Genkit JS skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-js
> ```
>
> It teaches your agent the current Genkit JS APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

Review the [documentation](https://genkit.dev/docs/get-started) for details and samples.

To install the CLI:

```bash
npm install -g genkit-cli
```

Available commands (run `genkit help <command>` for the options of each command):

- `start [options] [-- <command...>]`

  run a command in Genkit dev mode and start the Developer UI

- `start:flutter [options]`

  run a Flutter app in Genkit dev mode

- `flow:run [options] <flowName> [data] [-- <command...>]`

  run a flow using provided data as input

- `flow:batch-run [options] <flowName> <inputFileName> [-- <command...>]`

  batch run a flow using provided set of data from a file as input

- `eval:extract-data [options] <flowName>`

  extract evaluation data for a given flow from the trace store

- `eval:run [options] <dataset> [-- <command...>]`

  evaluate provided dataset against configured evaluators

- `eval:flow [options] <flowName> [data] [-- <command...>]`

  evaluate a flow against configured evaluators using provided data as input

- `trace:list [options]`

  list traces

- `trace:get [options] <traceId>`

  get a trace by id

- `log:list [options]`

  list logs, in reverse chronological order

- `docs:list [language]`, `docs:search <query> [language]`, `docs:read <filePath>`

  list, search, and read Genkit documentation

- `init:ai-tools [options]`

  initialize AI tools in a workspace with context about Genkit (experimental)

- `mcp [options]`

  run the Genkit MCP stdio server (experimental)

- `dev:test-model [options] [modelOrCmd] [args...]`

  test a model against the Genkit model specification

- `config`

  set development environment configuration

- `help [command]`

## MCP project root

When an MCP client launches `genkit mcp`, pass the absolute path to your Genkit
project if the client's working directory may be elsewhere:

```json
{
  "mcpServers": {
    "genkit": {
      "command": "genkit",
      "args": ["mcp", "--project-root", "/absolute/path/to/your/project"]
    }
  }
}
```

Without `--project-root`, the CLI searches upward from its working directory
for a project and otherwise uses that working directory. Trace files are stored
under `<projectRoot>/.genkit/traces` (or the equivalent Windows path). Launching
from a filesystem root can therefore attempt to write to `/.genkit/traces` on
Unix or `C:\.genkit\traces` on Windows. `GENKIT_HOME` does not configure this
path. Use `--project-root` or launch the MCP server from the project directory.
