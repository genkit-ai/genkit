# MiniMax plugin for Genkit

This plugin adds the MiniMax music generation models to Genkit.

## Installation

```bash
npm i @genkit-ai/minimax
```

## Configuration

The plugin reads the API key from the `MINIMAX_API_KEY` environment variable, or
it can be passed explicitly:

```ts
import { minimax } from '@genkit-ai/minimax';
import { genkit } from 'genkit';

const ai = genkit({
  plugins: [minimax({ apiKey: process.env.MINIMAX_API_KEY })],
});
```

### Regions

Requests go to the international endpoint by default. Set `region: 'cn'` to use
the mainland China endpoint instead:

```ts
const ai = genkit({
  plugins: [minimax({ region: 'cn' })],
});
```

| Region     | Endpoint                                     |
| ---------- | -------------------------------------------- |
| `global`   | `https://api.minimax.io/v1/music_generation` |
| `cn`       | `https://api.minimaxi.com/v1/music_generation` |

`baseUrl` overrides the endpoint derived from the region, and the region can also
be selected per request through the model config.

## Music generation

The music models turn a prompt, and optionally lyrics, into an audio track. The
generated audio is returned as a media part.

```ts
const { message } = await ai.generate({
  model: minimax.model('music-3.0'),
  prompt: 'An upbeat acoustic folk song about the sunrise',
  config: {
    lyrics: '[Verse]\nMorning light\n[Chorus]\nHere comes the sun',
  },
});

const audio = message?.content.find((part) => part.media)?.media;
```

Available models are `music-3.0`, `music-2.6`, `music-3.0-free` and
`music-2.6-free`.

### Instrumental tracks

Set `is_instrumental` to generate a vocal-free track, which makes `lyrics`
unnecessary. Alternatively, `lyrics_optimizer` lets the service derive the lyrics
from the prompt.

```ts
const { message } = await ai.generate({
  model: minimax.model('music-3.0'),
  prompt: 'A calm lo-fi study beat',
  config: { is_instrumental: true },
});
```

### Audio output

By default the audio bytes are returned as a hexadecimal string and inlined into
the media part as a `data:` URL. Set `output_format: 'url'` to receive a download
URL instead, which expires 24 hours after the request.

```ts
const { message } = await ai.generate({
  model: minimax.model('music-3.0'),
  prompt: 'A cinematic orchestral theme',
  config: {
    output_format: 'url',
    audio_setting: { sample_rate: 44100, bitrate: 256000, format: 'mp3' },
  },
});
```

`audio_setting.format` accepts `mp3`, `wav` and `pcm`, and determines the content
type reported on the media part.

### Watermarking

The `cn` endpoint accepts `aigc_watermark` to embed an AIGC watermark. Requests
that set it against another region are rejected before they are sent.

```ts
const { message } = await ai.generate({
  model: minimax.model('music-3.0'),
  prompt: 'A gentle piano lullaby',
  config: { region: 'cn', aigc_watermark: true },
});
```

### Streaming

The models always return a single response. The endpoint's streaming mode only
emits hexadecimal audio and is not exposed through this plugin, so requests that
set `stream` are rejected.

## Documentation

- International: https://platform.minimax.io/docs/api-reference/music-generation
- Mainland China: https://platform.minimaxi.com/docs/api-reference/music-generation

The sources for this package are in the main
[Genkit](https://github.com/genkit-ai/genkit) repository.

License: Apache 2.0
