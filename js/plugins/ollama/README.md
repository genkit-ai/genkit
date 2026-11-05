# Ollama plugin for Genkit

See [official documentation](https://genkit.dev/docs/integrations/ollama/) for more.

> **Building with a coding agent? Install the Genkit JS skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-js
> ```
>
> It teaches your agent the current Genkit JS APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

## Installing the plugin

```bash
npm i --save genkitx-ollama
```

## Using the plugin

```ts
import { genkit } from 'genkit';
import { ollama } from 'genkitx-ollama';

const ai = genkit({
  plugins: [
    ollama({
      models: [{ name: 'gemma' }],
      serverAddress: 'http://127.0.0.1:11434', // default local address
    }),
  ],
});

async function main() {
  const { text } = await ai.generate({
    prompt: 'hi Gemini!',
    model: 'ollama/gemma',
  });
  console.log(text);
}

main();
```

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [Genkit documentation](https://genkit.dev/docs/plugins/ollama/).

License: Apache 2.0
