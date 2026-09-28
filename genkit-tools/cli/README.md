# Genkit CLI

The package contains the CLI for Genkit, an open source framework with rich local tooling to help app developers build, test, deploy, and monitor AI-powered features for their apps with confidence. Genkit is built by Firebase, Google's app development platform that is trusted by millions of businesses around the world.

Review the [documentation](https://genkit.dev/docs/get-started) for details and samples.

To install the CLI:

```bash
npm install -g genkit-cli
```

Available commands:

- `init [options]`

  initialize a project directory with Genkit

- `start [options]`

  run the app in dev mode and start a Developer UI

- `flow:run [options] <flowName> [data]`

  run a flow using provided data as input

- `flow:batchRun [options] <flowName> <inputFileName>`

  batch run a flow using provided set of data from a file as input

- `flow:resume <flowName> <flowId> <data>`

  resume an interrupted flow (experimental)

- `eval:extractData [options] <flowName>`

  extract evaludation data for a given flow from the trace store

- `eval:run [options] <dataset>`

  evaluate provided dataset against configured evaluators

- `eval:flow [options] <flowName> [data]`

  evaluate a flow against configured evaluators using provided data as input

- `config`

  set development environment configuration

- `mcp [options]`

  run the experimental MCP server over stdio

- `help`

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
