# Anthropic Plugin Sample

This test app demonstrates usage of the Genkit Anthropic plugin against both the stable and beta runners, organized by feature.

## Directory Structure

```
src/
  stable/
    basic.ts          - Basic stable API examples (hello, streaming)
    text-plain.ts     - Text/plain error handling demonstration
    webp.ts           - WEBP image handling demonstration
    pdf.ts            - PDF document processing examples
    vision.ts         - Image/vision analysis examples
    attention-first-page.pdf - Sample PDF file for testing
    sample-image.png  - Sample image file for vision demo
  beta/
    basic.ts          - Basic beta API examples
```

## Setup

1. From the repo root run `pnpm install` followed by `pnpm run setup` to link workspace dependencies.
2. In this directory, optionally run `pnpm install` if you want a local `node_modules/`.
3. Export an Anthropic API key (or add it to a `.env` file) before running any samples:

   ```bash
   export ANTHROPIC_API_KEY=your-key
   ```

## Available scripts

### Basic Examples
- `pnpm run build` – Compile the TypeScript sources into `lib/`.
- `pnpm run start:stable` – Run the compiled stable basic sample.
- `pnpm run start:beta` – Run the compiled beta basic sample.
- `pnpm run dev:stable` – Start the Genkit Dev UI over `src/stable/basic.ts` with live reload.
- `pnpm run dev:beta` – Start the Genkit Dev UI over `src/beta/basic.ts` with live reload.

### Feature-Specific Examples
- `pnpm run dev:stable:text-plain` – Start Dev UI for text/plain error handling demo.
- `pnpm run dev:stable:webp` – Start Dev UI for WEBP image handling demo.
- `pnpm run dev:stable:pdf` – Start Dev UI for PDF document processing demo.
- `pnpm run dev:stable:vision` – Start Dev UI for image/vision analysis demo.
- `pnpm run dev:stable:prompt-caching` – Start Dev UI for prompt caching demo.

## Flows

Each source file defines flows that can be invoked from the Dev UI or the Genkit CLI:

### Basic Examples
- `anthropic-stable-hello` – Simple greeting using stable API
- `anthropic-stable-stream` – Streaming response example
- `anthropic-beta-hello` – Simple greeting using beta API
- `anthropic-beta-stream` – Streaming response with beta API
- `anthropic-beta-opus41` – Test Opus 4.1 model with beta API

### Text/Plain Handling
- `stable-text-plain-error` – Demonstrates the helpful error when using text/plain as media
- `stable-text-plain-correct` – Shows the correct way to send text content

### WEBP Image Handling
- `stable-webp-matching` – WEBP image with matching contentType
- `stable-webp-mismatched` – WEBP image with mismatched contentType (demonstrates the fix)

### PDF Document Processing
- `stable-pdf-base64` – Process a PDF from a local file using base64 encoding
- `stable-pdf-url` – Process a PDF from a publicly accessible URL
- `stable-pdf-analysis` – Analyze a PDF document for key topics, concepts, and visual elements

### Vision/Image Analysis
- `stable-vision-url` – Analyze an image from a public URL
- `stable-vision-base64` – Analyze an image from a local file (base64 encoded)
- `stable-vision-conversation` – Multi-turn conversation about an image

### Prompt Caching
- `stable-prompt-caching-system` – Caches a long system prompt.
- `stable-prompt-caching-user` – Applies caching to user-provided text input.
- `stable-prompt-caching-image` – Caches prompts that combine text and an image.
- `stable-prompt-caching-pdf` – Caches prompts including PDF file content.
- `stable-prompt-caching-with-tool-call` – Demonstrates caching with tool-assisted prompts.

Example: `genkit flow:run anthropic-stable-hello`
