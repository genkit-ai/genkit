# TwelveLabs plugin for Genkit

This Genkit plugin adds [TwelveLabs](https://twelvelabs.io) video AI:

- **Pegasus** video-understanding models, exposed as Genkit **models** — describe,
  summarize, or answer questions about a video.
- **Marengo** multimodal embedding models, exposed as Genkit **embedders** —
  produce 512-dim text embeddings that share a space with TwelveLabs video
  embeddings (great for video retrieval / RAG).

The plugin is opt-in: nothing is registered unless you list `models` and/or
`embedders`.

## Installing the plugin

```bash
npm i --save genkitx-twelvelabs
```

## Using the plugin

```ts
import { genkit } from 'genkit';
import { twelvelabs } from 'genkitx-twelvelabs';

const ai = genkit({
  plugins: [
    twelvelabs({
      // apiKey defaults to process.env.TWELVELABS_API_KEY
      models: [{ name: 'pegasus1.5' }],
      embedders: [{ name: 'marengo3.0', dimensions: 512 }],
    }),
  ],
});

async function main() {
  // Pegasus: describe a video. The video is a media part with a public URL;
  // TwelveLabs fetches it server-side, so no prior indexing is required.
  const { text } = await ai.generate({
    model: 'twelvelabs/pegasus1.5',
    messages: [
      {
        role: 'user',
        content: [
          { text: 'Describe this video in one sentence.' },
          {
            media: {
              url: 'https://example.com/video.mp4',
              contentType: 'video/mp4',
            },
          },
        ],
      },
    ],
  });
  console.log(text);

  // Marengo: embed text into the multimodal space.
  const embedding = await ai.embed({
    embedder: 'twelvelabs/marengo3.0',
    content: 'a cat playing piano',
  });
  console.log(embedding[0].embedding.length); // 512
}

main();
```

Get a free API key at [twelvelabs.io](https://twelvelabs.io) — there's a
generous free tier.

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit)
repo. Please file issues and pull requests against that repo.

License: Apache 2.0
