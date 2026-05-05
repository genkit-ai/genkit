![Genkit logo](docs/resources/genkit-logo-dark.png#gh-dark-mode-only 'Genkit')
![Genkit logo](docs/resources/genkit-logo.png#gh-light-mode-only 'Genkit')

[Genkit](https://genkit.dev) is an open-source framework for building full-stack AI-powered applications, built and used in production by Google's Firebase. It provides SDKs for multiple programming languages with varying levels of stability:

- **JavaScript/TypeScript**: Production-ready with full feature support
- **Go**: Production-ready with full feature support
- **Python (Alpha)**: Early development with core functionality

It offers a unified interface for integrating AI models from providers like [Google](https://genkit.dev/docs/plugins/google-genai), [OpenAI](https://genkit.dev/docs/plugins/openai), [Anthropic](https://thefireco.github.io/genkit-plugins/docs/plugins/genkitx-anthropic), [Ollama](https://genkit.dev/docs/plugins/ollama/), and more. Rapidly build and deploy production-ready chatbots, automations, and recommendation systems using streamlined APIs for multimodal content, structured outputs, tool calling, and agentic workflows.

Get started with just a few lines of code:

```ts
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({ plugins: [googleAI()] });

const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: 'Why is Firebase awesome?'
});
```

## Explore & build with Genkit

Play with AI sample apps, with visualizations of the Genkit code that powers
them, at no cost to you.

[Explore Genkit by Example](https://examples.genkit.dev)

## Key capabilities

<table>
  <tr>
    <td><strong>Broad AI model support</strong></td>
    <td>Use a unified interface to integrate with hundreds of models from providers like <a href="https://genkit.dev/docs/plugins/google-genai">Google</a>, <a href="https://genkit.dev/docs/plugins/openai">
    OpenAI</a>, <a href="https://thefireco.github.io/genkit-plugins/docs/plugins/genkitx-anthropic">
    Anthropic</a>, <a href="https://genkit.dev/docs/plugins/ollama">Ollama</a>, and more. Explore, compare, and use the best models for your needs.</td>
  </tr>
  <tr>
    <td><strong>Simplified AI development</strong></td>
    <td>Use streamlined APIs to build AI features with <a href="https://genkit.dev/docs/models#structured-output">
    structured output</a>, <a href="https://genkit.dev/docs/tool-calling">agentic tool calling</a>, <a href="https://genkit.dev/docs/rag">context-aware generation</a>, <a href="https://genkit.dev/docs/models#multimodal">multi-modal input/output</a>, and more. Genkit handles the complexity of AI development, so you can build and iterate faster.</td>
  </tr>
  <tr>
    <td><strong>Web and mobile ready</strong></td>
    <td>Integrate seamlessly with frameworks and platforms including Next.js, React, Angular, iOS, Android, using purpose-built <a href="https://genkit.dev/docs/firebase">client SDKs</a> and helpers.</td>
  </tr>
  <tr>
    <td><strong>Cross-language support</strong></td>
    <td>Build with the language that best fits your project. Genkit provides SDKs for JavaScript/TypeScript, Go, and Python (Alpha) with consistent APIs and capabilities across all supported languages.</td>
  </tr>
  <tr>
    <td><strong>Deploy anywhere</strong></td>
    <td>Deploy AI logic to any environment that supports your chosen programming language, such as <a href="https://genkit.dev/docs/firebase">Cloud Functions for Firebase</a>,
    <a href="https://genkit.dev/docs/cloud-run">Google Cloud Run</a>, or <a href="https://genkit.dev/docs/deploy-node">third-party platforms</a>,
    with or without Google services.</td>
  </tr>
  <tr>
    <td><strong>Developer tools</strong></td>
    <td>Accelerate AI development with a purpose-built, local <a href="https://genkit.dev/docs/devtools">CLI and Developer UI</a>. Test prompts and
    flows against individual inputs or datasets, compare outputs from different models, debug with detailed execution traces, and use immediate visual feedback to iterate rapidly on prompts.</td>
  </tr>
  <tr>
    <td><strong>Production monitoring</strong></td>
    <td>Ship AI features with confidence using comprehensive production monitoring. Track model performance, and request volumes, latency, and error rates in a <a href="https://genkit.dev/docs/observability/getting-started"> purpose-built dashboard</a>. Identify issues quickly with detailed observability metrics, and ensure your AI features meet quality and performance targets in real-world usage.</td>
  </tr>
</table>

## How does it work?

Genkit simplifies AI integration with an open-source SDK and unified APIs that
work across various model providers and programming languages. It abstracts away complexity so you can focus on delivering great user experiences.

Some key features offered by Genkit include:

* [Text and image generation](https://genkit.dev/docs/models)
* [Type-safe, structured data generation](https://genkit.dev/docs/models#structured-output)
* [Tool calling](https://genkit.dev/docs/tool-calling)
* [Prompt templating](https://genkit.dev/docs/dotprompt)
* [Persisted chat interfaces](https://genkit.dev/docs/chat)
* [AI workflows](https://genkit.dev/docs/flows)
* [AI-powered data retrieval (RAG)](https://genkit.dev/docs/rag)

Genkit is designed for server-side deployment in multiple language environments, and also provides seamless client-side integration through dedicated helpers and [client SDKs](https://genkit.dev/docs/firebase).

## Implementation path

<table>
<tr>
  <td><span>1</span></td>
  <td>Choose your language and model provider</td>
  <td>Select the Genkit SDK for your preferred language (JavaScript/TypeScript, Go, or Python (Alpha)). Choose a model provider like <a href="https://genkit.dev/docs/plugins/google-genai">Google Gemini</a> or Anthropic, and get an API key. Some providers, like <a href="https://genkit.dev/docs/plugins/vertex-ai">Vertex AI</a>, may rely on a different means of authentication.</td>
</tr>
<tr>
  <td><span>2</span></td>
  <td>Install the SDK and initialize</td>
  <td>Install the Genkit SDK, model-provider package of your choice, and the Genkit CLI. Import the Genkit and provider packages and initialize Genkit with the provider API key.</td>
</tr>
<tr>
  <td><span>3</span></td>
  <td>Write and test AI features</td>
  <td>Use the Genkit SDK to build AI features for your use case, from basic text generation to complex multi-step workflows and agents. Use the CLI and Developer UI to help you rapidly test and iterate.</td>
</tr>
<tr>
  <td><span>4</span></td>
  <td>Deploy and monitor</td>
  <td>Deploy your AI features to Firebase, Google Cloud Run, or any environment that supports your chosen programming language. Integrate them into your app, and monitor them in production in the Firebase console.</td>
</tr>
</table>

## Get started

- [JavaScript/TypeScript quickstart](https://genkit.dev/docs/get-started)
- [Go quickstart](https://genkit.dev/go/docs/get-started-go)
- [Python quickstart](https://genkit.dev/python/docs/get-started/) (Alpha)

## Development tools

Genkit provides a CLI and a local UI to streamline your AI development workflow.

### CLI

The Genkit CLI includes commands for running and evaluating your Genkit functions (flows) and collecting telemetry and logs.

- **Install:** `npm install -g genkit-cli`
- **Run a command, wrapped with telemetry, a interactive developer UI, etc:** `genkit start -- <command to run your code>`

### Developer UI

The Genkit developer UI is a local interface for testing, debugging, and iterating on your AI application.

Key features:

- **Run:** Execute and experiment with Genkit flows, prompts, queries, and more in dedicated playgrounds.
- **Inspect:** Analyze detailed traces of past executions, including step-by-step breakdowns of complex flows.
- **Evaluate:** Review the results of evaluations run against your flows, including performance metrics and links to relevant traces.

<img src="docs/resources/readme-ui-traces-screenshot.png" width="700" alt="Screenshot of Genkit Developer UI showing traces">

## Connect with us

- [**Join us on Discord**](https://discord.gg/qXt5zzQKpc) – Get help, share
ideas, and chat with other developers.
- [**Contribute on GitHub**](https://github.com/genkit-ai/genkit/issues) – Report
bugs, suggest features, or explore the source code.
- [**Contribute to Documentation and Samples**](https://github.com/genkit-ai/) – Report
issues in Genkit's [documentation](https://github.com/genkit-ai/docsite), or contribute to the [samples](https://github.com/genkit-ai/samples).

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/firebase/genkit)

## Contributing

Contributions to Genkit are welcome and highly appreciated! See our [Contribution Guide](CONTRIBUTING.md) to get started.

## Authors

Genkit is built by [Firebase](https://firebase.google.com/) with contributions from the [Open Source Community](https://github.com/genkit-ai/genkit/graphs/contributors).
