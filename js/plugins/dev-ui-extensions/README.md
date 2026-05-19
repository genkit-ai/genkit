# Genkit Dev UI Extensions Plugin

This plugin provides extensions for the Genkit Dev UI.

## Features

- **Enable Thinking**: Automatically enables "thinking" for Gemini models when `GENKIT_ENV=dev`.
- **Trace Decorator**: Adds a "hello-world" badge to spans in the Trace Viewer.

## Usage

```typescript
import { genkit } from 'genkit';
import { devUiExtensions } from '@genkit-ai/dev-ui-extensions';

const ai = genkit({
  plugins: [devUiExtensions],
});
```
