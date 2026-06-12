# Gemini Text-to-Speech

This sample generates speech with Gemini TTS and writes a playable WAV file.

Gemini TTS responses are returned as Genkit media parts. The media data may be
raw PCM, usually:

```text
audio/L16;codec=pcm;rate=24000
```

Raw PCM is not a WAV or MP3 file. The sample decodes the returned media data URI
and wraps `audio/L16` bytes with a WAV header before writing
`generated-tts.wav`. This matches the JavaScript Gemini TTS samples, which decode
`media.url` and convert the returned PCM bytes with a `toWav` helper.

Run the sample with a Gemini API key:

```bash
export GEMINI_API_KEY='<your-api-key>'
cd go
go run ./samples/text-to-speech/gemini
```

Invoke the `text-to-speech-flow` from Dev UI. The default model is
`googleai/gemini-2.5-flash-preview-tts`, and the generated WAV path defaults to
`./generated-tts.wav`.

The dedicated `*-tts` models produce audio by default, so this sample configures
`speechConfig` only. Use `responseModalities: ["AUDIO"]` with conversational
Gemini models when you need to request audio as one of several possible output
modalities.
