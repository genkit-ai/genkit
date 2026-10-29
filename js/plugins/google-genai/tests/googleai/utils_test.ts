/**
 * Copyright 2025 Google LLC
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

import assert from 'node:assert';
import { afterEach, beforeEach, describe, it } from 'node:test';
import process from 'process';
import {
  MISSING_API_KEY_ERROR,
  appendToWavData,
  calculateApiKey,
  checkApiKey,
  convertObjectKeysToSnakeCase,
  getApiKeyFromEnvVar,
  stripLeadingId3Tags,
} from '../../src/googleai/utils.js';

/**
 * An ID3v2.3 tag shaped like the one seen live: header
 * `49 44 33 03 00 00 00 00 2f 63` (size 6,115, so 6,125 bytes in total), then
 * a GEOB frame. The frame body is filler here.
 */
function testId3Tag(): Buffer {
  const tag = Buffer.alloc(6125, 0x55);
  Buffer.from([
    0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x2f, 0x63,
  ]).copy(tag);
  tag.write('GEOB', 10, 'ascii');
  return tag;
}

describe('stripLeadingId3Tags', () => {
  it('removes a leading ID3 tag and keeps the audio after it', () => {
    const pcm = Buffer.alloc(5760, 0x07);
    const result = stripLeadingId3Tags(Buffer.concat([testId3Tag(), pcm]));
    assert.deepStrictEqual(result, pcm);
  });

  it('removes several tags in a row', () => {
    const pcm = Buffer.from([1, 2]);
    const result = stripLeadingId3Tags(
      Buffer.concat([testId3Tag(), testId3Tag(), pcm])
    );
    assert.deepStrictEqual(result, pcm);
  });

  it('returns audio without a tag unchanged', () => {
    const pcm = Buffer.from([0xf8, 0xff, 0xf9, 0xff]);
    assert.strictEqual(stripLeadingId3Tags(pcm), pcm);
  });

  it('keeps audio that only starts with the letters "ID3"', () => {
    // A size byte of 0x80 or more can't be a real tag header.
    const notATag = Buffer.from([
      0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 1, 2,
    ]);
    assert.strictEqual(stripLeadingId3Tags(notATag), notATag);
  });

  it('keeps the bytes if the tag would run past the end', () => {
    const truncated = testId3Tag().subarray(0, 100);
    assert.strictEqual(stripLeadingId3Tags(truncated), truncated);
  });
});

/**
 * Builds a WAV file shaped like the ones the TTS models return: a canonical
 * 44-byte header, the `data` chunk, then another chunk after the audio (the
 * server's WAVs end with extra bytes after `data`; their content is unknown,
 * so a generic chunk stands in for them).
 */
function testWav(audio: number[], trailing: number[] = [9, 9, 9, 9]): Buffer {
  const data = Buffer.from(audio);
  const extra = Buffer.from(trailing);
  const header = Buffer.alloc(44);
  header.write('RIFF', 0, 'ascii');
  header.write('WAVE', 8, 'ascii');
  header.write('fmt ', 12, 'ascii');
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(1, 22); // mono
  header.writeUInt32LE(24000, 24);
  header.writeUInt32LE(48000, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36, 'ascii');
  header.writeUInt32LE(data.length, 40);
  const extraHeader = Buffer.alloc(8);
  extraHeader.write('LIST', 0, 'ascii');
  extraHeader.writeUInt32LE(extra.length, 4);
  const wav = Buffer.concat([header, data, extraHeader, extra]);
  wav.writeUInt32LE(wav.length - 8, 4);
  return wav;
}

describe('appendToWavData', () => {
  it('inserts the audio at the end of the data chunk and fixes the sizes', () => {
    const result = appendToWavData(testWav([1, 2]), Buffer.from([3, 4, 5, 6]));
    assert.ok(result);
    assert.strictEqual(result.readUInt32LE(4), result.length - 8);
    assert.strictEqual(result.readUInt32LE(40), 6);
    assert.deepStrictEqual(
      result.subarray(44, 50),
      Buffer.from([1, 2, 3, 4, 5, 6])
    );
    // The chunk after `data` is kept, after the audio.
    assert.strictEqual(result.toString('ascii', 50, 54), 'LIST');
    assert.deepStrictEqual(result.subarray(58), Buffer.from([9, 9, 9, 9]));
  });

  it('returns undefined for data that is not a WAV file', () => {
    assert.strictEqual(
      appendToWavData(Buffer.from('not a wav file'), Buffer.from([1])),
      undefined
    );
  });

  it('returns undefined for a WAV without a data chunk', () => {
    const noData = testWav([1, 2]).subarray(0, 36);
    noData.writeUInt32LE(noData.length - 8, 4);
    assert.strictEqual(appendToWavData(noData, Buffer.from([1])), undefined);
  });
});

describe('API Key Utils', () => {
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    // Save original process.env
    originalEnv = { ...process.env };
  });

  afterEach(() => {
    // Restore original process.env
    process.env = originalEnv;
  });

  // Helper to clear specific env vars
  const clearEnvVars = () => {
    delete process.env.GEMINI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    delete process.env.GOOGLE_GENAI_API_KEY;
  };

  describe('getApiKeyFromEnvVar', () => {
    it('returns GEMINI_API_KEY if set (priority 1)', () => {
      clearEnvVars();
      process.env.GEMINI_API_KEY = 'gemini_key';
      process.env.GOOGLE_API_KEY = 'google_key';
      process.env.GOOGLE_GENAI_API_KEY = 'genai_key';
      assert.strictEqual(getApiKeyFromEnvVar(), 'gemini_key');
    });

    it('returns GOOGLE_API_KEY if GEMINI_API_KEY is not set (priority 2)', () => {
      clearEnvVars();
      process.env.GOOGLE_API_KEY = 'google_key';
      process.env.GOOGLE_GENAI_API_KEY = 'genai_key';
      assert.strictEqual(getApiKeyFromEnvVar(), 'google_key');
    });

    it('returns GOOGLE_GENAI_API_KEY if others are not set (priority 3)', () => {
      clearEnvVars();
      process.env.GOOGLE_GENAI_API_KEY = 'genai_key';
      assert.strictEqual(getApiKeyFromEnvVar(), 'genai_key');
    });

    it('returns undefined if no relevant env vars are set', () => {
      clearEnvVars();
      assert.strictEqual(getApiKeyFromEnvVar(), undefined);
    });
  });

  describe('checkApiKey', () => {
    beforeEach(() => {
      clearEnvVars();
    });

    it('returns apiKey1 if it is a non-empty string', () => {
      assert.strictEqual(checkApiKey('test_key'), 'test_key');
    });

    it('returns env var key if apiKey1 is undefined', () => {
      process.env.GOOGLE_API_KEY = 'env_key';
      assert.strictEqual(checkApiKey(undefined), 'env_key');
    });

    it('returns env var key if apiKey1 is an empty string', () => {
      process.env.GOOGLE_API_KEY = 'env_key';
      assert.strictEqual(checkApiKey(''), 'env_key');
    });

    it('returns undefined if apiKey1 is false', () => {
      process.env.GOOGLE_API_KEY = 'env_key'; // Should not be used
      assert.strictEqual(checkApiKey(false), undefined);
    });

    it('throws MISSING_API_KEY_ERROR if apiKey1 is undefined and no env var set', () => {
      assert.throws(() => checkApiKey(undefined), MISSING_API_KEY_ERROR);
    });

    it('throws MISSING_API_KEY_ERROR if apiKey1 is empty string and no env var set', () => {
      assert.throws(() => checkApiKey(''), MISSING_API_KEY_ERROR);
    });

    it('does not throw if apiKey1 is false, even with no env var', () => {
      assert.doesNotThrow(() => checkApiKey(false));
      assert.strictEqual(checkApiKey(false), undefined);
    });
  });

  describe('calculateApiKey', () => {
    beforeEach(() => {
      clearEnvVars();
    });

    it('returns apiKey2 if provided, ignoring apiKey1 and env', () => {
      process.env.GOOGLE_API_KEY = 'env_key';
      assert.strictEqual(calculateApiKey('api1_key', 'api2_key'), 'api2_key');
      assert.strictEqual(calculateApiKey(undefined, 'api2_key'), 'api2_key');
      assert.strictEqual(calculateApiKey(false, 'api2_key'), 'api2_key');
    });

    it('returns apiKey1 if apiKey2 is undefined', () => {
      assert.strictEqual(calculateApiKey('api1_key', undefined), 'api1_key');
    });

    it('returns env var key if apiKey1 and apiKey2 are undefined', () => {
      process.env.GOOGLE_API_KEY = 'env_key';
      assert.strictEqual(calculateApiKey(undefined, undefined), 'env_key');
    });

    it('returns undefined if apiKey1 is false and apiKey2 is undefined', () => {
      assert.strictEqual(calculateApiKey(false, undefined), undefined);
    });

    it('throws MISSING_API_KEY_ERROR if apiKey1 and apiKey2 are undefined and no env var', () => {
      assert.throws(
        () => calculateApiKey(undefined, undefined),
        MISSING_API_KEY_ERROR
      );
    });

    it('throws MISSING_API_KEY_ERROR if apiKey1 is empty, apiKey2 is undefined, and no env var', () => {
      assert.throws(
        () => calculateApiKey('', undefined),
        MISSING_API_KEY_ERROR
      );
    });

    it('returns env var if apiKey1 is empty and apiKey2 is undefined', () => {
      process.env.GOOGLE_API_KEY = 'env_key';
      assert.strictEqual(calculateApiKey('', undefined), 'env_key');
    });
  });

  describe('convertObjectKeysToSnakeCase', () => {
    it('converts basic camelCase to snake_case', () => {
      const input = {
        myProp: 1,
        anotherPropTest: 'string',
      };
      assert.deepStrictEqual(convertObjectKeysToSnakeCase(input), {
        my_prop: 1,
        another_prop_test: 'string',
      });
    });

    it('handles nested objects', () => {
      const input = {
        parentProp: {
          childPropA: true,
          childPropB: {
            deepProp: 123,
          },
        },
      };
      assert.deepStrictEqual(convertObjectKeysToSnakeCase(input), {
        parent_prop: {
          child_prop_a: true,
          child_prop_b: {
            deep_prop: 123,
          },
        },
      });
    });

    it('handles arrays of objects correctly instead of turning them into plain objects', () => {
      const input = {
        arrayProp: [{ innerProp: 1 }, { innerProp: 2 }, 'stringItem'],
      };
      assert.deepStrictEqual(convertObjectKeysToSnakeCase(input), {
        array_prop: [{ inner_prop: 1 }, { inner_prop: 2 }, 'stringItem'],
      });
    });

    it('passes through nulls and non-objects untouched', () => {
      assert.strictEqual(convertObjectKeysToSnakeCase(null), null);
      assert.strictEqual(convertObjectKeysToSnakeCase('string'), 'string');
      assert.strictEqual(convertObjectKeysToSnakeCase(123), 123);
    });
  });
});
