# Genkit Ollama Plugin (Community)

> **Community Plugin** — This plugin is community-maintained and is not an
> official Google or Ollama product. It is provided on an "as-is" basis.
>
> **Preview** — This plugin is in preview and may have API changes in future releases.

This Genkit plugin provides a set of tools and utilities for working with Ollama.

## Setup environment

```bash
uv venv
source .venv/bin/activate
```

## Serving Ollama Locally

### Ollama Service Installation

#### MacOS  (brew)

```bash
brew install ollama
```

#### Debian/Ubuntu (apt installer)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Other installation options may be found [here](https://ollama.com/download)

### Start serving of Ollama locally

```bash
ollama serve
```

Ollama is served at `http://127.0.0.1:11434` by default.

### Installing Required Model

Once ollama service is serving - pull the required model version:

```bash
ollama pull <model-version>:<tag>
```

### Check installed models

Installed models can be reviewed with following command:

```bash
ollama list
```

## Disclaimer

This is a **community-maintained** plugin and is not officially supported by
Google or Ollama. Ollama is open-source software under the
[MIT License](https://github.com/ollama/ollama/blob/main/LICENSE). You are
responsible for ensuring your usage complies with the licenses of the models
you download and run.

- **Model Licensing** — Individual models pulled via Ollama have their own
  licenses. Review model cards before use in production.
- **Local Execution** — Ollama runs models locally on your hardware. No data
  is sent to external APIs unless you configure a remote Ollama server.

## License

Apache-2.0
