/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import * as assert from 'assert';
import type { GenerateRequest } from 'genkit/model';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import {
  MiniMaxMusicConfigSchema,
  defineModel,
  isMiniMaxMusicModelName,
  listActions,
  model,
} from '../src/music.js';
import type {
  MiniMaxPluginOptions,
  MusicGenerationResponse,
} from '../src/types.js';

describe('MiniMax music generation', () => {
  const ORIGINAL_ENV = { ...process.env };
  let fetchStub: sinon.SinonStub;

  beforeEach(() => {
    process.env = { ...ORIGINAL_ENV };
    delete process.env.MINIMAX_API_KEY;
    fetchStub = sinon.stub(global, 'fetch');
  });

  afterEach(() => {
    sinon.restore();
    process.env = { ...ORIGINAL_ENV };
  });

  function mockFetchResponse(body: any, status = 200) {
    fetchStub.callsFake(() =>
      Promise.resolve(
        new Response(JSON.stringify(body), {
          status,
          statusText: status === 200 ? 'OK' : 'Error',
          headers: { 'Content-Type': 'application/json' },
        })
      )
    );
  }

  const defaultPluginOptions: MiniMaxPluginOptions = {
    apiKey: 'test-api-key-plugin',
  };

  const minimalRequest: GenerateRequest<typeof MiniMaxMusicConfigSchema> = {
    messages: [
      {
        role: 'user',
        content: [{ text: 'An upbeat acoustic folk song about the sunrise' }],
      },
    ],
  };

  // 0xff 0xd8 0xff encoded as base64 is '/9j/'.
  const HEX_AUDIO = 'ffd8ff';
  const HEX_AUDIO_BASE64 = '/9j/';

  const completedResponse: MusicGenerationResponse = {
    data: { status: 2, audio: HEX_AUDIO },
    base_resp: { status_code: 0, status_msg: 'success' },
    trace_id: 'trace-123',
  };

  function lastBody() {
    return JSON.parse(fetchStub.lastCall.args[1].body);
  }

  function lastUrl() {
    return fetchStub.lastCall.args[0];
  }

  function lastHeaders() {
    return fetchStub.lastCall.args[1].headers;
  }

  describe('endpoints', () => {
    it('calls the global endpoint by default', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      await action(minimalRequest, {} as any);

      sinon.assert.calledOnce(fetchStub);
      assert.strictEqual(
        lastUrl(),
        'https://api.minimax.io/v1/music_generation'
      );
      assert.strictEqual(fetchStub.lastCall.args[1].method, 'POST');
    });

    it('calls the cn endpoint when the plugin selects that region', async () => {
      const action = defineModel('music-3.0', {
        ...defaultPluginOptions,
        region: 'cn',
      });
      mockFetchResponse(completedResponse);

      await action(minimalRequest, {} as any);

      assert.strictEqual(
        lastUrl(),
        'https://api.minimaxi.com/v1/music_generation'
      );
    });

    it('lets the request config override the plugin region', async () => {
      const action = defineModel('music-3.0', {
        ...defaultPluginOptions,
        region: 'cn',
      });
      mockFetchResponse(completedResponse);

      await action(
        { ...minimalRequest, config: { region: 'global' } },
        {} as any
      );

      assert.strictEqual(
        lastUrl(),
        'https://api.minimax.io/v1/music_generation'
      );
    });

    it('lets baseUrl override the region endpoint', async () => {
      const action = defineModel('music-3.0', {
        ...defaultPluginOptions,
        baseUrl: 'https://proxy.example.com/',
      });
      mockFetchResponse(completedResponse);

      await action(minimalRequest, {} as any);

      assert.strictEqual(
        lastUrl(),
        'https://proxy.example.com/v1/music_generation'
      );
    });
  });

  describe('authorization', () => {
    it('sends the plugin API key as a bearer token with a JSON content type', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      await action(minimalRequest, {} as any);

      const headers = lastHeaders();
      assert.strictEqual(headers.Authorization, 'Bearer test-api-key-plugin');
      assert.strictEqual(headers['Content-Type'], 'application/json');
    });

    it('lets the request config override the API key', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      await action(
        { ...minimalRequest, config: { apiKey: 'test-api-key-request' } },
        {} as any
      );

      assert.strictEqual(
        lastHeaders().Authorization,
        'Bearer test-api-key-request'
      );
    });

    it('falls back to the environment variable', async () => {
      process.env.MINIMAX_API_KEY = 'test-api-key-env';
      const action = defineModel('music-3.0');
      mockFetchResponse(completedResponse);

      await action(minimalRequest, {} as any);

      assert.strictEqual(
        lastHeaders().Authorization,
        'Bearer test-api-key-env'
      );
    });

    it('throws when no API key is available', () => {
      assert.throws(() => defineModel('music-3.0'), /MINIMAX_API_KEY/);
    });

    it('defers the API key check when apiKey is false', async () => {
      const action = defineModel('music-3.0', { apiKey: false });
      mockFetchResponse(completedResponse);

      await assert.rejects(action(minimalRequest, {} as any), /apiKey: false/);
    });

    it('does not let custom headers overwrite the authorization header', async () => {
      const action = defineModel('music-3.0', {
        ...defaultPluginOptions,
        customHeaders: { Authorization: 'Bearer spoofed', 'X-Trace': 'on' },
      });
      mockFetchResponse(completedResponse);

      await action(minimalRequest, {} as any);

      const headers = lastHeaders();
      assert.strictEqual(headers.Authorization, 'Bearer test-api-key-plugin');
      assert.strictEqual(headers['X-Trace'], 'on');
    });
  });

  describe('request body', () => {
    it('sends the model and the prompt taken from the messages', async () => {
      const action = defineModel('music-2.6', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      await action(minimalRequest, {} as any);

      const body = lastBody();
      assert.strictEqual(body.model, 'music-2.6');
      assert.strictEqual(
        body.prompt,
        'An upbeat acoustic folk song about the sunrise'
      );
    });

    it('falls back to the prompt from the config', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      await action(
        {
          messages: [{ role: 'user', content: [] }],
          config: { prompt: 'A slow piano ballad' },
        },
        {} as any
      );

      assert.strictEqual(lastBody().prompt, 'A slow piano ballad');
    });

    it('forwards the generation fields', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      await action(
        {
          ...minimalRequest,
          config: {
            lyrics: '[Verse]\nMorning light\n[Chorus]\nHere comes the sun',
            output_format: 'hex',
            audio_setting: {
              sample_rate: 44100,
              bitrate: 256000,
              format: 'mp3',
            },
            lyrics_optimizer: true,
            is_instrumental: false,
          },
        },
        {} as any
      );

      const body = lastBody();
      assert.strictEqual(
        body.lyrics,
        '[Verse]\nMorning light\n[Chorus]\nHere comes the sun'
      );
      assert.strictEqual(body.output_format, 'hex');
      assert.deepStrictEqual(body.audio_setting, {
        sample_rate: 44100,
        bitrate: 256000,
        format: 'mp3',
      });
      assert.strictEqual(body.lyrics_optimizer, true);
      assert.strictEqual(body.is_instrumental, false);
    });

    it('never sends the connection fields to the API', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      await action(
        {
          ...minimalRequest,
          config: {
            apiKey: 'test-api-key-request',
            region: 'global',
            baseUrl: 'https://proxy.example.com',
            stream: false,
          },
        },
        {} as any
      );

      const body = lastBody();
      assert.ok(!('apiKey' in body));
      assert.ok(!('region' in body));
      assert.ok(!('baseUrl' in body));
      assert.ok(!('stream' in body));
    });

    it('rejects streaming output, which the model does not support', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      await assert.rejects(
        action({ ...minimalRequest, config: { stream: true } }, {} as any),
        /Streaming music generation is not supported/
      );
      sinon.assert.notCalled(fetchStub);
    });

    it('sends aigc_watermark to the cn endpoint', async () => {
      const action = defineModel('music-3.0', {
        ...defaultPluginOptions,
        region: 'cn',
      });
      mockFetchResponse(completedResponse);

      await action(
        { ...minimalRequest, config: { aigc_watermark: true } },
        {} as any
      );

      assert.strictEqual(lastBody().aigc_watermark, true);
    });

    it('rejects aigc_watermark outside the cn endpoint', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      await assert.rejects(
        action(
          { ...minimalRequest, config: { aigc_watermark: true } },
          {} as any
        ),
        /aigc_watermark.*only accepted by the `cn` region/
      );
      sinon.assert.notCalled(fetchStub);
    });

    it('forwards the abort signal', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);
      const abortSignal = AbortSignal.timeout(60_000);

      await action(minimalRequest, { abortSignal } as any);

      assert.strictEqual(fetchStub.lastCall.args[1].signal, abortSignal);
    });
  });

  describe('response parsing', () => {
    it('inlines hex audio as a data URL, defaulting to mp3', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      const response = await action(minimalRequest, {} as any);

      const media = response.message?.content[0].media;
      assert.strictEqual(
        media?.url,
        `data:audio/mpeg;base64,${HEX_AUDIO_BASE64}`
      );
      assert.strictEqual(media?.contentType, 'audio/mpeg');
      assert.strictEqual(response.finishReason, 'stop');
      assert.deepStrictEqual(response.custom, completedResponse);
    });

    it('uses the configured audio format for the content type', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      // 0x00 0xff encoded as base64 is 'AP8='.
      mockFetchResponse({
        ...completedResponse,
        data: { status: 2, audio: '00ff' },
      });

      const response = await action(
        {
          ...minimalRequest,
          config: { audio_setting: { format: 'wav' } },
        },
        {} as any
      );

      const media = response.message?.content[0].media;
      assert.strictEqual(media?.url, 'data:audio/wav;base64,AP8=');
      assert.strictEqual(media?.contentType, 'audio/wav');
    });

    it('passes a url output through unchanged', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      const audioUrl = 'https://files.example.com/song.mp3?expires=86400';
      mockFetchResponse({
        ...completedResponse,
        data: { status: 2, audio: audioUrl },
      });

      const response = await action(
        { ...minimalRequest, config: { output_format: 'url' } },
        {} as any
      );

      assert.strictEqual(response.message?.content[0].media?.url, audioUrl);
    });

    it('reports usage statistics', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse(completedResponse);

      const response = await action(minimalRequest, {} as any);

      assert.strictEqual(response.usage?.outputAudioFiles, 1);
      assert.strictEqual(
        response.usage?.inputCharacters,
        'An upbeat acoustic folk song about the sunrise'.length
      );
    });

    it('throws when the generation is still in progress', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse({
        ...completedResponse,
        data: { status: 1 },
      });

      await assert.rejects(
        action(minimalRequest, {} as any),
        /still in progress/
      );
    });

    it('throws when the response carries no audio', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse({ ...completedResponse, data: { status: 2 } });

      await assert.rejects(
        action(minimalRequest, {} as any),
        /did not contain any generated audio/
      );
    });

    it('throws when the hex audio is malformed', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse({
        ...completedResponse,
        data: { status: 2, audio: 'not-hex' },
      });

      await assert.rejects(
        action(minimalRequest, {} as any),
        /not a valid hexadecimal string/
      );
    });

    it('surfaces an API error reported in the response envelope', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse({
        base_resp: { status_code: 1004, status_msg: 'auth failed' },
      });

      await assert.rejects(
        action(minimalRequest, {} as any),
        /\[1004\] auth failed/
      );
    });

    it('surfaces an HTTP error', async () => {
      const action = defineModel('music-3.0', defaultPluginOptions);
      mockFetchResponse({ error: 'rate limited' }, 429);

      await assert.rejects(
        action(minimalRequest, {} as any),
        /Error fetching from https:\/\/api\.minimax\.io\/v1\/music_generation/
      );
    });
  });

  describe('model references', () => {
    it('namespaces known models', () => {
      assert.strictEqual(model('music-3.0').name, 'minimax/music-3.0');
      assert.strictEqual(model('minimax/music-2.6').name, 'minimax/music-2.6');
    });

    it('supports media output only', () => {
      assert.deepStrictEqual(model('music-3.0').info?.supports?.output, [
        'media',
      ]);
    });

    it('accepts unknown music model versions', () => {
      assert.strictEqual(model('music-9.9').name, 'minimax/music-9.9');
    });

    it('claims generation model names but not cover model names', () => {
      assert.ok(isMiniMaxMusicModelName('music-3.0'));
      assert.ok(isMiniMaxMusicModelName('music-2.6-free'));
      assert.ok(!isMiniMaxMusicModelName('music-cover'));
      assert.ok(!isMiniMaxMusicModelName('music-cover-free'));
      assert.ok(!isMiniMaxMusicModelName('speech-2.6'));
      assert.ok(!isMiniMaxMusicModelName(undefined));
    });

    it('lists the known generation models', () => {
      assert.deepStrictEqual(
        listActions().map((action) => action.name),
        [
          'minimax/music-3.0',
          'minimax/music-2.6',
          'minimax/music-3.0-free',
          'minimax/music-2.6-free',
        ]
      );
    });
  });
});
