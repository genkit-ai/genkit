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

- `help`
