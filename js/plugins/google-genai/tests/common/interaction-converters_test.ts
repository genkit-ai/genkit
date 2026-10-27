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
import { GenkitError, MessageData, Part } from 'genkit';
import { ToolDefinition } from 'genkit/model';
import { describe, it } from 'node:test';
import {
  createInteractionChunkConverter,
  ensureToolIds,
  fromInteraction,
  fromInteractionContent,
  fromInteractionDelta,
  fromInteractionStep,
  fromInteractionSync,
  mergeStreamedAudio,
  removeId3FromAudio,
  toInteractionAudioResponseFormat,
  toInteractionConfigTool,
  toInteractionContent,
  toInteractionGenerationConfig,
  toInteractionGoogleSearch,
  toInteractionResponseFormat,
  toInteractionRole,
  toInteractionSafetySettings,
  toInteractionSpeechConfig,
  toInteractionSteps,
  toInteractionTool,
  toInteractionVideoResponseFormat,
} from '../../src/common/interaction-converters.js';
import {
  Content,
  GeminiInteraction,
  InteractionSseEvent,
  Step,
  StepDeltaData,
} from '../../src/common/interaction-types.js';

describe('Interaction Converters', () => {
  describe('toInteractionSafetySettings', () => {
    it('converts to lower-case types and thresholds', () => {
      // The server rejects upper-case values with a 400 listing the
      // lower-case ones (seen live on vertexAI, Oct 2026).
      assert.deepStrictEqual(
        toInteractionSafetySettings([
          {
            category: 'HARM_CATEGORY_HATE_SPEECH',
            threshold: 'BLOCK_LOW_AND_ABOVE',
          },
          { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
        ]),
        [
          { type: 'hate_speech', threshold: 'block_low_and_above' },
          { type: 'harassment', threshold: 'block_none' },
        ]
      );
    });

    it('lower-cases method', () => {
      // The server rejects 'SEVERITY' with a 400 (reported in review).
      assert.deepStrictEqual(
        toInteractionSafetySettings([
          {
            category: 'HARM_CATEGORY_HATE_SPEECH',
            threshold: 'BLOCK_ONLY_HIGH',
            method: 'SEVERITY',
          },
        ]),
        [
          {
            type: 'hate_speech',
            threshold: 'block_only_high',
            method: 'severity',
          },
        ]
      );
    });

    it('leaves out HARM_CATEGORY_UNSPECIFIED', () => {
      assert.strictEqual(
        toInteractionSafetySettings([
          {
            category: 'HARM_CATEGORY_UNSPECIFIED',
            threshold: 'BLOCK_ONLY_HIGH',
          },
        ]),
        undefined
      );
    });

    it('returns undefined when not set', () => {
      assert.strictEqual(toInteractionSafetySettings(undefined), undefined);
    });
  });

  describe('toInteractionAudioResponseFormat', () => {
    it('maps mimeType and sampleRate', () => {
      assert.deepStrictEqual(
        toInteractionAudioResponseFormat({
          mimeType: 'audio/l16',
          sampleRate: 16000,
        }),
        { type: 'audio', mime_type: 'audio/l16', sample_rate: 16000 }
      );
    });

    it('sends only the fields that are set', () => {
      assert.deepStrictEqual(
        toInteractionAudioResponseFormat({ mimeType: 'audio/wav' }),
        { type: 'audio', mime_type: 'audio/wav' }
      );
    });

    it('returns undefined when audioFormat is not set', () => {
      assert.strictEqual(
        toInteractionAudioResponseFormat(undefined),
        undefined
      );
    });
  });

  describe('toInteractionVideoResponseFormat', () => {
    it('maps aspectRatio and resolution', () => {
      assert.deepStrictEqual(
        toInteractionVideoResponseFormat({
          aspectRatio: '9:16',
          resolution: '4k',
        }),
        { type: 'video', aspect_ratio: '9:16', resolution: '4k' }
      );
    });

    it('sends only the fields that are set', () => {
      assert.deepStrictEqual(
        toInteractionVideoResponseFormat({ resolution: '360p' }),
        { type: 'video', resolution: '360p' }
      );
    });

    it('maps delivery', () => {
      assert.deepStrictEqual(
        toInteractionVideoResponseFormat({ delivery: 'uri' }),
        { type: 'video', delivery: 'uri' }
      );
    });

    it('returns undefined when videoFormat is not set', () => {
      assert.strictEqual(
        toInteractionVideoResponseFormat(undefined),
        undefined
      );
    });
  });

  describe('toInteractionResponseFormat', () => {
    it('returns undefined when nothing is set (server defaults apply)', () => {
      assert.strictEqual(toInteractionResponseFormat({}), undefined);
    });

    it('returns a single entry when there is one', () => {
      assert.deepStrictEqual(
        toInteractionResponseFormat({ modalities: ['AUDIO'] }),
        { type: 'audio' }
      );
    });

    it('makes one entry per modality, in order', () => {
      assert.deepStrictEqual(
        toInteractionResponseFormat({ modalities: ['TEXT', 'IMAGE'] }),
        [{ type: 'text' }, { type: 'image' }]
      );
    });

    it('puts imageConfig on the image entry', () => {
      assert.deepStrictEqual(
        toInteractionResponseFormat({
          modalities: ['TEXT', 'IMAGE'],
          imageConfig: { aspectRatio: '9:16', imageSize: '2K' },
        }),
        [
          { type: 'text' },
          { type: 'image', aspect_ratio: '9:16', image_size: '2K' },
        ]
      );
    });

    it('adds an image entry for imageConfig when no modalities are given', () => {
      assert.deepStrictEqual(
        toInteractionResponseFormat({ imageConfig: { aspectRatio: '1:1' } }),
        { type: 'image', aspect_ratio: '1:1' }
      );
    });

    it('puts audioFormat and videoFormat on their entries', () => {
      assert.deepStrictEqual(
        toInteractionResponseFormat({
          modalities: ['AUDIO'],
          audioFormat: { mimeType: 'audio/l16', sampleRate: 24000 },
        }),
        { type: 'audio', mime_type: 'audio/l16', sample_rate: 24000 }
      );
      assert.deepStrictEqual(
        toInteractionResponseFormat({
          videoFormat: { aspectRatio: '9:16', resolution: '1080p' },
        }),
        { type: 'video', aspect_ratio: '9:16', resolution: '1080p' }
      );
    });

    it('puts JSON output on the text entry, alongside other entries', () => {
      const schema = { type: 'object', properties: { a: { type: 'string' } } };
      assert.deepStrictEqual(
        toInteractionResponseFormat({
          modalities: ['TEXT', 'IMAGE'],
          json: { schema },
        }),
        [
          { type: 'text', mime_type: 'application/json', schema },
          { type: 'image' },
        ]
      );
      // JSON output alone (no modalities): one text entry.
      assert.deepStrictEqual(toInteractionResponseFormat({ json: {} }), {
        type: 'text',
        mime_type: 'application/json',
      });
    });

    it('keeps JSON output when audioFormat is set too', () => {
      // Used to be lost: the audio entry replaced the JSON text entry.
      assert.deepStrictEqual(
        toInteractionResponseFormat({
          audioFormat: { mimeType: 'audio/wav' },
          json: {},
        }),
        [
          { type: 'audio', mime_type: 'audio/wav' },
          { type: 'text', mime_type: 'application/json' },
        ]
      );
    });
  });

  describe('toInteractionGenerationConfig (video)', () => {
    it('keeps videoFormat out of generation_config and sends videoConfig as video_config', () => {
      const result = toInteractionGenerationConfig({
        videoFormat: { resolution: '1080p' },
        videoConfig: { task: 'extend' },
      });
      assert.deepStrictEqual(result, { video_config: { task: 'extend' } });
    });

    it('keeps imageConfig out of generation_config (sent as a response_format entry)', () => {
      const result = toInteractionGenerationConfig({
        imageConfig: { aspectRatio: '16:9' },
        temperature: 0.5,
      });
      assert.deepStrictEqual(result, { temperature: 0.5 });
    });
  });

  describe('computer use safety decisions', () => {
    // The plugin doesn't run computer use actions; users run the loop
    // themselves (returnToolRequests: true). The safety decision must reach
    // them, and their acknowledgement must reach the server. Shapes from the
    // computer use guide.
    const safetyDecision = {
      explanation: 'Must check check-box',
      decision: 'require_confirmation',
    };

    it('passes safety_decision through to the toolRequest input', () => {
      const result = fromInteractionSync({
        status: 'requires_action',
        steps: [
          {
            type: 'function_call',
            id: 'call_123',
            name: 'click',
            arguments: { x: 60, y: 100, safety_decision: safetyDecision },
          },
        ],
      });
      const toolRequest = result.message?.content.find(
        (p) => p.toolRequest
      )?.toolRequest;
      assert.deepStrictEqual(toolRequest, {
        name: 'click',
        ref: 'call_123',
        input: { x: 60, y: 100, safety_decision: safetyDecision },
      });
    });

    it('resends safety_decision and sends safety_acknowledgement back', () => {
      const steps = toInteractionSteps([
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'click',
                ref: 'call_123',
                input: { x: 60, y: 100, safety_decision: safetyDecision },
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'click',
                ref: 'call_123',
                output: {
                  url: 'https://example.com',
                  safety_acknowledgement: true,
                },
              },
            },
          ],
        },
      ]);
      assert.deepStrictEqual(steps, [
        {
          type: 'function_call',
          name: 'click',
          arguments: { x: 60, y: 100, safety_decision: safetyDecision },
          id: 'call_123',
        },
        {
          type: 'function_result',
          name: 'click',
          result: { url: 'https://example.com', safety_acknowledgement: true },
          call_id: 'call_123',
        },
      ]);
    });
  });

  describe('agentic video understanding', () => {
    // Shapes seen live on gemini-4-argon (Oct 2026).
    const processingCallStep: Step = {
      type: 'processing_call',
      id: 'call_598332',
      signature: 'sig_call',
    };
    const processingResultStep: Step = {
      type: 'processing_result',
      call_id: 'call_598332',
      signature: 'sig_result',
    };

    it('sends metadata.processing on a YouTube video', () => {
      assert.deepStrictEqual(
        toInteractionContent({
          media: { url: 'https://www.youtube.com/watch?v=abc123' },
          metadata: { processing: 'agentic' },
        }),
        {
          type: 'video',
          uri: 'https://www.youtube.com/watch?v=abc123',
          processing: 'agentic',
        }
      );
    });

    it('sends metadata.processing on inline and URI videos', () => {
      assert.deepStrictEqual(
        toInteractionContent({
          media: {
            url: 'data:video/mp4;base64,AAAA',
            contentType: 'video/mp4',
          },
          metadata: { processing: 'static' },
        }),
        {
          type: 'video',
          mime_type: 'video/mp4',
          data: 'AAAA',
          processing: 'static',
        }
      );
      assert.deepStrictEqual(
        toInteractionContent({
          media: {
            url: 'https://generativelanguage.googleapis.com/v1beta/files/abc123',
            contentType: 'video/mp4',
          },
          metadata: { processing: 'agentic' },
        }),
        {
          type: 'video',
          mime_type: 'video/mp4',
          uri: 'https://generativelanguage.googleapis.com/v1beta/files/abc123',
          processing: 'agentic',
        }
      );
    });

    it('sends no processing field when it is not set', () => {
      assert.deepStrictEqual(
        toInteractionContent({
          media: { url: 'https://www.youtube.com/watch?v=abc123' },
        }),
        { type: 'video', uri: 'https://www.youtube.com/watch?v=abc123' }
      );
    });

    it('ignores metadata.processing on media that is not video', () => {
      assert.deepStrictEqual(
        toInteractionContent({
          media: {
            url: 'data:image/png;base64,AAAA',
            contentType: 'image/png',
          },
          metadata: { processing: 'agentic' },
        }),
        { type: 'image', mime_type: 'image/png', data: 'AAAA' }
      );
    });

    it('returns processing steps as custom parts with their signatures', () => {
      assert.deepStrictEqual(fromInteractionStep(processingCallStep), [
        {
          custom: { processingCall: { id: 'call_598332' } },
          metadata: { thoughtSignature: 'sig_call' },
        },
      ]);
      assert.deepStrictEqual(fromInteractionStep(processingResultStep), [
        {
          custom: { processingResult: { callId: 'call_598332' } },
          metadata: { thoughtSignature: 'sig_result' },
        },
      ]);
    });

    it('resends processing steps unchanged in the history', () => {
      // Turn 1's response, converted to Genkit parts and sent back as history.
      const modelContent = [
        ...fromInteractionStep(processingCallStep),
        ...fromInteractionStep(processingResultStep),
        { text: 'The main moments are...' },
      ];
      const steps = toInteractionSteps([
        {
          role: 'user',
          content: [
            {
              media: { url: 'https://www.youtube.com/watch?v=abc123' },
              metadata: { processing: 'agentic' },
            },
            { text: 'What happens in this video?' },
          ],
        },
        { role: 'model', content: modelContent },
        { role: 'user', content: [{ text: 'What happens at the end?' }] },
      ]);
      assert.deepStrictEqual(steps, [
        {
          type: 'user_input',
          content: [
            {
              type: 'video',
              uri: 'https://www.youtube.com/watch?v=abc123',
              processing: 'agentic',
            },
            { type: 'text', text: 'What happens in this video?' },
          ],
        },
        processingCallStep,
        processingResultStep,
        {
          type: 'model_output',
          content: [{ type: 'text', text: 'The main moments are...' }],
        },
        {
          type: 'user_input',
          content: [{ type: 'text', text: 'What happens at the end?' }],
        },
      ]);
    });

    it('reports tool use tokens in usage.custom', () => {
      const result = fromInteractionSync({
        status: 'completed',
        steps: [
          processingCallStep,
          processingResultStep,
          { type: 'model_output', content: [{ type: 'text', text: 'done' }] },
        ],
        usage: {
          total_input_tokens: 416,
          total_output_tokens: 393,
          total_tool_use_tokens: 2054,
          total_thought_tokens: 2117,
          total_tokens: 4980,
        },
      });
      assert.deepStrictEqual(result.usage?.custom, { toolUseTokens: 2054 });
    });

    it('reports grounding tool counts in usage.custom', () => {
      // `grounding_tool_count` as seen live (googleAI, Google Search).
      const result = fromInteractionSync({
        status: 'completed',
        usage: {
          total_tokens: 1011,
          total_tool_use_tokens: 0,
          grounding_tool_count: [
            { type: 'google_search', count: 2 },
            { type: 'google_maps', count: 1 },
          ],
        },
      });
      assert.deepStrictEqual(result.usage?.custom, {
        googleSearchCount: 2,
        googleMapsCount: 1,
      });
    });

    it('leaves usage.custom unset without tool use tokens', () => {
      const result = fromInteractionSync({
        status: 'completed',
        usage: { total_input_tokens: 4, total_tokens: 4 },
      });
      assert.strictEqual(result.usage?.custom, undefined);
    });
  });

  describe('server-side built-in tool steps', () => {
    // Shapes seen live (Oct 2026): Google Maps on vertexAI gemini-3.8-flash
    // (no arguments, no signature) and URL context on googleAI
    // gemini-3.8-flash (with signatures).
    const mapsCall: Step = { type: 'google_maps_call', id: 'qpo3tzzp' };
    const place = {
      place_id: 'ChIJefTaTQZD04kRzuh2lpwlV-A',
      name: 'Queen Victoria Park - Google Maps',
      url: 'https://maps.google.com/maps?cid=16165430741939185870',
    };
    const mapsResult: Step = {
      type: 'google_maps_result',
      call_id: 'qpo3tzzp',
      result: [{ places: [place] }],
    };
    const changelog = 'https://ai.google.dev/gemini-api/docs/changelog';
    const urlCall: Step = {
      type: 'url_context_call',
      id: 'call_178104',
      signature: 'sig_url_call',
      arguments: { urls: [changelog] },
    };
    const urlResult: Step = {
      type: 'url_context_result',
      call_id: 'call_178104',
      signature: 'sig_url_result',
      result: [{ url: changelog, status: 'success' }],
      is_error: false,
    };
    // The other built-in tools, from the spec's examples.
    const otherSteps: Step[] = [
      { type: 'file_search_call', id: 'file_call_1', signature: 'sig_f1' },
      {
        type: 'file_search_result',
        call_id: 'file_call_1',
        signature: 'sig_f2',
      },
      {
        type: 'mcp_server_tool_call',
        id: 'mcp_call_1',
        name: 'calculate_tax',
        server_name: 'financial_mcp_server',
        arguments: { income: 120000, state: 'CA' },
      },
      {
        type: 'mcp_server_tool_result',
        call_id: 'mcp_call_1',
        name: 'calculate_tax',
        server_name: 'financial_mcp_server',
        result: { tax_due: 32400 },
      },
    ];
    const allSteps = [mapsCall, mapsResult, urlCall, urlResult, ...otherSteps];

    it('converts Google Maps steps into custom parts', () => {
      assert.deepStrictEqual(fromInteractionStep(mapsCall), [
        { custom: { googleMapsCall: { id: 'qpo3tzzp' } } },
      ]);
      assert.deepStrictEqual(fromInteractionStep(mapsResult), [
        {
          custom: {
            googleMapsResult: {
              callId: 'qpo3tzzp',
              result: [{ places: [place] }],
            },
          },
        },
      ]);
    });

    it('converts URL context steps into custom parts with their signatures', () => {
      assert.deepStrictEqual(fromInteractionStep(urlCall), [
        {
          custom: {
            urlContextCall: {
              id: 'call_178104',
              arguments: { urls: [changelog] },
            },
          },
          metadata: { thoughtSignature: 'sig_url_call' },
        },
      ]);
      assert.deepStrictEqual(fromInteractionStep(urlResult), [
        {
          custom: {
            urlContextResult: {
              callId: 'call_178104',
              result: [{ url: changelog, status: 'success' }],
              isError: false,
            },
          },
          metadata: { thoughtSignature: 'sig_url_result' },
        },
      ]);
    });

    it('never produces tool requests (nothing for Genkit to run)', () => {
      // Even with isPending, which turns client function calls into tool
      // requests: the server already ran these tools.
      const parts = allSteps.flatMap((step) => fromInteractionStep(step, true));
      assert.strictEqual(parts.length, allSteps.length);
      assert.ok(parts.every((p) => p.toolRequest === undefined));
      assert.ok(parts.every((p) => p.custom?.unknownStep === undefined));
    });

    it('sends every built-in tool step back unchanged in the history', () => {
      // The live bug: these steps (and their signatures) were dropped when
      // the history was resent, so the model lost the tool results.
      const history: MessageData[] = [
        { role: 'user', content: [{ text: 'Name two sights near me.' }] },
        {
          role: 'model',
          content: [
            ...allSteps.flatMap((step) => fromInteractionStep(step)),
            { text: 'Queen Victoria Park.' },
          ],
        },
        { role: 'user', content: [{ text: 'Which one is closer?' }] },
      ];
      assert.deepStrictEqual(toInteractionSteps(history), [
        {
          type: 'user_input',
          content: [{ type: 'text', text: 'Name two sights near me.' }],
        },
        ...allSteps,
        {
          type: 'model_output',
          content: [{ type: 'text', text: 'Queen Victoria Park.' }],
        },
        {
          type: 'user_input',
          content: [{ type: 'text', text: 'Which one is closer?' }],
        },
      ]);
    });

    it('keeps the custom parts in the final response', () => {
      const result = fromInteractionSync({
        status: 'completed',
        steps: [
          mapsCall,
          mapsResult,
          {
            type: 'model_output',
            content: [{ type: 'text', text: 'Queen Victoria Park.' }],
          },
        ],
      });
      const customKeys = (result.message?.content ?? []).flatMap((p) =>
        p.custom ? Object.keys(p.custom) : []
      );
      assert.deepStrictEqual(customKeys, [
        'googleMapsCall',
        'googleMapsResult',
      ]);
    });

    it('round-trips Google Search steps without arguments or result', () => {
      // Seen live on vertexAI (gemini-3.8-flash, url context turn): the
      // model also searched, and the platform sent these steps without
      // `arguments` / `result`. Resending the history used to throw.
      const searchCall: Step = {
        type: 'google_search_call',
        id: 'ssg39wjg',
        search_type: 'web_search',
      };
      const searchResult: Step = {
        type: 'google_search_result',
        call_id: 'ssg39wjg',
        is_error: false,
      };
      assert.deepStrictEqual(fromInteractionStep(searchCall), [
        {
          custom: {
            googleSearchCall: { id: 'ssg39wjg', searchType: 'web_search' },
          },
        },
      ]);
      assert.deepStrictEqual(fromInteractionStep(searchResult), [
        {
          custom: {
            googleSearchResult: { callId: 'ssg39wjg', isError: false },
          },
        },
      ]);

      const history: MessageData[] = [
        { role: 'user', content: [{ text: 'Read the changelog.' }] },
        {
          role: 'model',
          content: [
            ...fromInteractionStep(searchCall),
            ...fromInteractionStep(searchResult),
            { text: 'October 13, 2026.' },
          ],
        },
        { role: 'user', content: [{ text: 'And the entry before?' }] },
      ];
      const steps = toInteractionSteps(history);
      // Sent back as received: no empty arguments or result added.
      assert.deepStrictEqual(steps[1], {
        type: 'google_search_call',
        id: 'ssg39wjg',
        search_type: 'web_search',
        signature: undefined,
      });
      assert.deepStrictEqual(steps[2], {
        type: 'google_search_result',
        call_id: 'ssg39wjg',
        is_error: false,
        signature: undefined,
      });
    });

    it('round-trips built-in tool steps with only their ids', () => {
      // The platform can omit fields the spec marks as required; only the ids
      // are needed to round-trip.
      const bareSteps: Step[] = [
        { type: 'url_context_call', id: 'u1' },
        { type: 'url_context_result', call_id: 'u1' },
        { type: 'google_maps_call', id: 'm1' },
        { type: 'google_maps_result', call_id: 'm1' },
        { type: 'mcp_server_tool_call', id: 'p1' },
        { type: 'mcp_server_tool_result', call_id: 'p1' },
      ];
      const history: MessageData[] = [
        {
          role: 'model',
          content: bareSteps.flatMap((step) => fromInteractionStep(step)),
        },
      ];
      assert.deepStrictEqual(toInteractionSteps(history), bareSteps);
    });

    it('accepts an empty trailing model_output step', () => {
      // Seen live on vertexAI (Google Maps, store: true): the response ended
      // with `{ type: 'model_output' }` and no content, which used to throw
      // "Cannot read properties of undefined (reading 'map')".
      const result = fromInteractionSync({
        status: 'completed',
        steps: [
          { type: 'thought', signature: 'sig_thought' },
          {
            type: 'model_output',
            content: [{ type: 'text', text: 'Queen Victoria Park.' }],
          },
          mapsCall,
          mapsResult,
          { type: 'model_output' },
        ],
      });
      const content = result.message?.content ?? [];
      assert.strictEqual(content.length, 4);
      assert.strictEqual(content[1].text, 'Queen Victoria Park.');
      assert.ok(content[2].custom?.googleMapsCall);
      assert.ok(content[3].custom?.googleMapsResult);
    });
  });

  describe('removeId3FromAudio', () => {
    // An ID3 tag holding a C2PA manifest, at the start of a streamed
    // audio/l16 chunk, as seen live: 6,125-byte tag + 5,760 bytes of PCM.
    const id3Tag = () => {
      const tag = Buffer.alloc(6125, 0x55);
      Buffer.from([
        0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x2f, 0x63,
      ]).copy(tag);
      tag.write('GEOB', 10, 'ascii');
      return tag;
    };
    const part = (bytes: Buffer, contentType = 'audio/l16'): Part => ({
      media: {
        url: `data:${contentType};base64,${bytes.toString('base64')}`,
        contentType,
      },
    });
    const decode = (p: Part) => {
      const url = p.media?.url ?? '';
      return Buffer.from(url.substring(url.indexOf(',') + 1), 'base64');
    };

    it('removes the tag and keeps the audio after it', () => {
      const pcm = Buffer.alloc(5760, 0x07);
      const result = removeId3FromAudio(part(Buffer.concat([id3Tag(), pcm])));
      assert.strictEqual(result.media?.contentType, 'audio/l16');
      assert.deepStrictEqual(decode(result), pcm);
    });

    it('leaves chunks without a tag, and other parts, unchanged', () => {
      const pcm = part(Buffer.from([1, 2, 3, 4]));
      assert.strictEqual(removeId3FromAudio(pcm), pcm);
      const text: Part = { text: 'hi' };
      assert.strictEqual(removeId3FromAudio(text), text);
    });

    it('only changes headerless audio types', () => {
      const wav = part(
        Buffer.concat([id3Tag(), Buffer.from([1, 2])]),
        'audio/wav'
      );
      assert.strictEqual(removeId3FromAudio(wav), wav);
    });
  });

  describe('mergeStreamedAudio', () => {
    // Streamed TTS sends headerless chunks (`audio/l16` by default, as seen
    // live with gemini-3.8-flash-lite-tts).
    const audioPart = (bytes: number[], contentType = 'audio/l16'): Part => ({
      media: {
        url: `data:${contentType};base64,${Buffer.from(bytes).toString('base64')}`,
        contentType,
      },
    });
    const decode = (part: Part) => {
      const url = part.media?.url ?? '';
      return Buffer.from(url.substring(url.indexOf(',') + 1), 'base64');
    };

    it('joins audio/l16 chunks into one audio/l16 part', () => {
      const result = mergeStreamedAudio([
        audioPart([1, 2]),
        audioPart([3, 4]),
        audioPart([5, 6]),
      ]);

      assert.strictEqual(result.length, 1);
      assert.strictEqual(result[0].media?.contentType, 'audio/l16');
      assert.deepStrictEqual(
        decode(result[0]),
        Buffer.from([1, 2, 3, 4, 5, 6])
      );
    });

    it('keeps the requested headerless format', () => {
      for (const type of ['audio/mulaw', 'audio/alaw']) {
        const result = mergeStreamedAudio([
          audioPart([1], type),
          audioPart([2], type),
        ]);
        assert.strictEqual(result.length, 1);
        assert.strictEqual(result[0].media?.contentType, type);
        assert.deepStrictEqual(decode(result[0]), Buffer.from([1, 2]));
      }
    });

    it('keeps content type parameters such as rate', () => {
      const type = 'audio/L16;rate=16000';
      const result = mergeStreamedAudio([
        audioPart([1], type),
        audioPart([2], type),
      ]);
      assert.strictEqual(result[0].media?.contentType, type);
    });

    it('does not join chunks of different types', () => {
      const result = mergeStreamedAudio([
        audioPart([1], 'audio/l16'),
        audioPart([2], 'audio/mulaw'),
      ]);
      assert.strictEqual(result.length, 2);
    });

    // A streamed audio/wav request starts with a small complete WAV (header,
    // a little audio, then extra bytes after `data`) and continues with
    // audio/l16 chunks, as seen live.
    const wavPart = (audio: number[]): Part => {
      const data = Buffer.from(audio);
      const extra = Buffer.from([9, 9, 9, 9]);
      const header = Buffer.alloc(44);
      header.write('RIFF', 0, 'ascii');
      header.write('WAVE', 8, 'ascii');
      header.write('fmt ', 12, 'ascii');
      header.writeUInt32LE(16, 16);
      header.writeUInt16LE(1, 20);
      header.writeUInt16LE(1, 22);
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
      return {
        media: {
          url: `data:audio/wav;base64,${wav.toString('base64')}`,
          contentType: 'audio/wav',
        },
      };
    };

    it('puts the audio/l16 chunks of a streamed WAV into one WAV', () => {
      const result = mergeStreamedAudio([
        wavPart([1, 2]),
        audioPart([3, 4]),
        audioPart([5, 6]),
      ]);

      assert.strictEqual(result.length, 1);
      assert.strictEqual(result[0].media?.contentType, 'audio/wav');
      const wav = decode(result[0]);
      assert.strictEqual(wav.readUInt32LE(4), wav.length - 8);
      assert.strictEqual(wav.readUInt32LE(40), 6);
      assert.deepStrictEqual(
        wav.subarray(44, 50),
        Buffer.from([1, 2, 3, 4, 5, 6])
      );
      assert.strictEqual(wav.toString('ascii', 50, 54), 'LIST');
    });

    it('leaves a WAV without following chunks unchanged', () => {
      const wav = wavPart([1, 2]);
      assert.deepStrictEqual(mergeStreamedAudio([wav]), [wav]);
    });

    it('leaves media typed audio/wav that is not a WAV file unchanged', () => {
      const notWav = audioPart([1, 2], 'audio/wav');
      const pcm = audioPart([3, 4]);
      assert.deepStrictEqual(mergeStreamedAudio([notWav, pcm]), [notWav, pcm]);
    });

    it('leaves other parts alone and joins each run separately', () => {
      const text: Part = { text: 'hello' };
      const result = mergeStreamedAudio([
        audioPart([1, 2]),
        audioPart([3, 4]),
        text,
        audioPart([5, 6]),
      ]);

      assert.strictEqual(result.length, 3);
      assert.deepStrictEqual(decode(result[0]), Buffer.from([1, 2, 3, 4]));
      assert.deepStrictEqual(result[1], text);
      assert.deepStrictEqual(decode(result[2]), Buffer.from([5, 6]));
    });

    it('returns parts without audio unchanged', () => {
      const parts: Part[] = [{ text: 'a' }, { text: 'b' }];
      assert.deepStrictEqual(mergeStreamedAudio(parts), parts);
    });
  });

  describe('ensureToolIds', () => {
    it('should assign IDs to tool requests without refs', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            { toolRequest: { name: 'tool1', input: {} } },
            { toolRequest: { name: 'tool2', input: {} } },
          ],
        },
      ];
      const result = ensureToolIds(messages);
      const req1 = result[0].content[0].toolRequest!;
      const req2 = result[0].content[1].toolRequest!;
      assert.ok(req1.ref && req1.ref.startsWith('genkit-auto-id-'));
      assert.ok(req2.ref && req2.ref.startsWith('genkit-auto-id-'));
      assert.notStrictEqual(req1.ref, req2.ref);
    });

    it('should assign matching IDs to tool responses without refs based on order', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            { toolRequest: { name: 'tool1', input: {} } },
            { toolRequest: { name: 'tool2', input: {} } },
          ],
        },
        {
          role: 'tool',
          content: [
            { toolResponse: { name: 'tool1', output: {} } },
            { toolResponse: { name: 'tool2', output: {} } },
          ],
        },
      ];
      const result = ensureToolIds(messages);
      const req1 = result[0].content[0].toolRequest!;
      const req2 = result[0].content[1].toolRequest!;
      const res1 = result[1].content[0].toolResponse!;
      const res2 = result[1].content[1].toolResponse!;

      assert.ok(req1.ref);
      assert.strictEqual(req1.ref, res1.ref);
      assert.ok(req2.ref);
      assert.strictEqual(req2.ref, res2.ref);
    });

    it('should assign orphan ID to tool response if no matching request', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [{ toolResponse: { name: 'tool1', output: {} } }],
        },
      ];
      const result = ensureToolIds(messages);
      const res1 = result[0].content[0].toolResponse!;
      assert.ok(res1.ref && res1.ref.startsWith('genkit-orphan-id-'));
    });

    it('should preserve existing refs', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            { toolRequest: { name: 'tool1', input: {}, ref: 'existing-id' } },
          ],
        },
      ];
      const result = ensureToolIds(messages);
      const req1 = result[0].content[0].toolRequest!;
      assert.strictEqual(req1.ref, 'existing-id');
    });
  });

  describe('toInteractionRole', () => {
    it('should convert user role', () => {
      assert.strictEqual(toInteractionRole('user'), 'user');
    });
    it('should convert model role', () => {
      assert.strictEqual(toInteractionRole('model'), 'model');
    });
    it('should convert tool role to user', () => {
      assert.strictEqual(toInteractionRole('tool'), 'user');
    });
    it('should throw for system role', () => {
      assert.throws(
        () => toInteractionRole('system'),
        /System role should be handled as system_instruction/
      );
    });
  });

  describe('toInteractionTool', () => {
    it('should convert ToolDefinition to InteractionTool', () => {
      const tool: ToolDefinition = {
        name: 'myFunc',
        description: 'desc',
        inputSchema: {
          type: 'object',
          properties: { arg: { type: 'string' } },
        },
      };
      const result = toInteractionTool(tool);
      assert.deepStrictEqual(result, {
        type: 'function',
        name: 'myFunc',
        description: 'desc',
        parameters: {
          type: 'object',
          properties: { arg: { type: 'string' } },
        },
      });
    });
  });

  describe('toInteractionGenerationConfig', () => {
    it('should flatten thinkingConfig and map casing', () => {
      const config = {
        thinkingConfig: {
          thinkingLevel: 'HIGH',
          includeThoughts: true,
        },
      };
      const result = toInteractionGenerationConfig(config);
      assert.deepStrictEqual(result, {
        thinking_level: 'high',
        thinking_summaries: 'auto',
      });
    });

    it('should drop other thinkingConfig fields (no thinking_config) and report them', () => {
      const dropped: string[] = [];
      const result = toInteractionGenerationConfig(
        {
          thinkingConfig: {
            thinkingLevel: 'LOW',
            includeThoughts: false,
            thinkingBudget: 1024,
            unknownProp: 'test',
          },
        },
        (field) => dropped.push(field)
      );
      assert.deepStrictEqual(result, {
        thinking_level: 'low',
        thinking_summaries: 'none',
      });
      assert.deepStrictEqual(dropped, [
        'thinkingConfig.thinkingBudget',
        'thinkingConfig.unknownProp',
      ]);
    });

    it('should handle already snake_cased thinking_config', () => {
      const dropped: string[] = [];
      const result = toInteractionGenerationConfig(
        {
          thinking_config: {
            thinking_level: 'MEDIUM',
            include_thoughts: true,
            thinking_budget: 2048,
          },
        },
        (field) => dropped.push(field)
      );
      assert.deepStrictEqual(result, {
        thinking_level: 'medium',
        thinking_summaries: 'auto',
      });
      assert.deepStrictEqual(dropped, ['thinkingConfig.thinking_budget']);
    });

    it('should drop contextCache and report it', () => {
      const dropped: string[] = [];
      const result = toInteractionGenerationConfig(
        { contextCache: true, temperature: 0.5 },
        (field) => dropped.push(field)
      );
      assert.deepStrictEqual(result, { temperature: 0.5 });
      assert.deepStrictEqual(dropped, ['contextCache']);
    });

    it('should drop topK (no top_k in generation_config) and report it', () => {
      // vertexAI rejects `top_k` with a 400 ("Unknown parameter 'top_k' at
      // 'generation_config'", seen live Oct 2026); the Gemini API ignores it.
      // temperature and top_p are still sent (in the spec, deprecated).
      for (const config of [
        { topK: 40, temperature: 0.5, topP: 0.9 },
        { top_k: 40, temperature: 0.5, topP: 0.9 },
      ]) {
        const dropped: string[] = [];
        const result = toInteractionGenerationConfig(config, (field) =>
          dropped.push(field)
        );
        assert.deepStrictEqual(result, { temperature: 0.5, top_p: 0.9 });
        assert.deepStrictEqual(dropped, ['topK']);
      }
    });

    it('should forward unknown passthrough keys in snake_case', () => {
      const result = toInteractionGenerationConfig({ someNewField: 1 });
      assert.deepStrictEqual(result, { some_new_field: 1 });
    });
  });

  describe('toInteractionSpeechConfig', () => {
    it('maps a single prebuilt voice', () => {
      assert.deepStrictEqual(
        toInteractionSpeechConfig({
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
        }),
        [{ voice: 'Puck' }]
      );
    });

    it('maps multi-speaker voices', () => {
      assert.deepStrictEqual(
        toInteractionSpeechConfig({
          multiSpeakerVoiceConfig: {
            speakerVoiceConfigs: [
              {
                speaker: 'Joe',
                voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
              },
              {
                speaker: 'Jane',
                voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
              },
            ],
          },
        }),
        {
          speakers: [
            { voice: 'Kore', speaker: 'Joe' },
            { voice: 'Puck', speaker: 'Jane' },
          ],
        }
      );
    });

    it('applies languageCode to every entry', () => {
      assert.deepStrictEqual(
        toInteractionSpeechConfig({
          languageCode: 'fr-FR',
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
        }),
        [{ voice: 'Puck', language: 'fr-FR' }]
      );
    });

    it('returns undefined when no voice or language is set', () => {
      assert.strictEqual(toInteractionSpeechConfig(undefined), undefined);
      assert.strictEqual(toInteractionSpeechConfig({}), undefined);
      assert.strictEqual(
        toInteractionSpeechConfig({ voiceConfig: {} }),
        undefined
      );
    });

    it('is applied by toInteractionGenerationConfig', () => {
      assert.deepStrictEqual(
        toInteractionGenerationConfig({
          maxOutputTokens: 100,
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
          },
        }),
        {
          max_output_tokens: 100,
          speech_config: [{ voice: 'Puck' }],
        }
      );
    });
  });

  describe('speech_metadata annotations on text parts', () => {
    it('adds a speech_metadata annotation from metadata.speechMetadata', () => {
      assert.deepStrictEqual(
        toInteractionContent({
          text: "How's it going today Jane?",
          metadata: { speechMetadata: { speaker: 'Joe', style: 'cheerful' } },
        }),
        {
          type: 'text',
          text: "How's it going today Jane?",
          annotations: [
            { type: 'speech_metadata', speaker: 'Joe', style: 'cheerful' },
          ],
        }
      );
    });

    it('supports style alone (single-speaker)', () => {
      assert.deepStrictEqual(
        toInteractionContent({
          text: 'Have a wonderful day!',
          metadata: { speechMetadata: { style: 'cheerful and friendly' } },
        }),
        {
          type: 'text',
          text: 'Have a wonderful day!',
          annotations: [
            { type: 'speech_metadata', style: 'cheerful and friendly' },
          ],
        }
      );
    });

    it('adds no annotations for plain text or empty speechMetadata', () => {
      assert.deepStrictEqual(toInteractionContent({ text: 'Hi' }), {
        type: 'text',
        text: 'Hi',
      });
      assert.deepStrictEqual(
        toInteractionContent({ text: 'Hi', metadata: { speechMetadata: {} } }),
        { type: 'text', text: 'Hi' }
      );
    });
  });

  describe('toInteractionGoogleSearch', () => {
    it('should handle boolean true config', () => {
      const result = toInteractionGoogleSearch(true);
      assert.deepStrictEqual(result, { type: 'google_search' });
    });

    it('should handle array format in snake_case', () => {
      const result = toInteractionGoogleSearch({
        search_types: ['web_search', 'image_search'],
      });
      assert.deepStrictEqual(result, {
        type: 'google_search',
        search_types: ['web_search', 'image_search'],
      });
    });

    it('should handle object format in camelCase', () => {
      const result = toInteractionGoogleSearch({
        searchTypes: { webSearch: {}, enterpriseWebSearch: {} },
      });
      assert.deepStrictEqual(result, {
        type: 'google_search',
        search_types: ['web_search', 'enterprise_web_search'],
      });
    });

    it('should handle empty config object', () => {
      const result = toInteractionGoogleSearch({});
      assert.deepStrictEqual(result, { type: 'google_search' });
    });

    it('should pass through unrecognized fields from search types array', () => {
      const result = toInteractionGoogleSearch({
        searchTypes: ['web_search', 'my_custom_search'],
      });
      assert.deepStrictEqual(result, {
        type: 'google_search',
        search_types: ['web_search', 'my_custom_search'],
      });
    });

    it('should pass through unrecognized fields from search types object', () => {
      const result = toInteractionGoogleSearch({
        searchTypes: { webSearch: {}, customSearchPlugin: {} },
      });
      assert.deepStrictEqual(result, {
        type: 'google_search',
        search_types: ['web_search', 'custom_search_plugin'],
      });
    });

    it('should throw an error for invalid top-level config', () => {
      assert.throws(
        () => toInteractionGoogleSearch('invalid'),
        /Invalid configuration for googleSearch tool/
      );
    });

    it('should throw an error for invalid searchTypes format', () => {
      assert.throws(
        () => toInteractionGoogleSearch({ searchTypes: 'invalid' }),
        /Invalid searchTypes configuration/
      );
    });

    it('should throw an error for invalid search type in array', () => {
      assert.throws(
        () => toInteractionGoogleSearch({ searchTypes: [123] }),
        /Invalid search type/
      );
    });
  });

  describe('toInteractionConfigTool', () => {
    it('should throw an error if toolRaw is not an object', () => {
      assert.throws(
        () => toInteractionConfigTool('not-an-object'),
        /Invalid tool configuration/
      );
      assert.throws(
        () => toInteractionConfigTool(['not-an-object']),
        /Invalid tool configuration/
      );
      assert.throws(
        () => toInteractionConfigTool(null),
        /Invalid tool configuration/
      );
    });

    it('should handle built-in tool format with valid inputs', () => {
      const tool = { codeExecution: { someProp: 123 } };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'code_execution',
        some_prop: 123,
      });
    });

    it('should handle built-in tool format with boolean true', () => {
      const tool = { codeExecution: true };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'code_execution',
      });
    });

    it('should throw an error if built-in tool config is invalid type', () => {
      const tool = { codeExecution: 'invalid' };
      assert.throws(
        () => toInteractionConfigTool(tool),
        /Invalid configuration for codeExecution tool/
      );
    });

    it('should handle fileSearch configurations and validate array', () => {
      const tool = { fileSearch: { fileSearchStoreNames: ['store1'] } };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'file_search',
        file_search_store_names: ['store1'],
      });

      const invalidTool = { fileSearch: { fileSearchStoreNames: 'store1' } };
      assert.throws(
        () => toInteractionConfigTool(invalidTool),
        /fileSearchStoreNames must be an array of strings/
      );
    });

    it('should pass through arbitrary custom tools un-nested', () => {
      const tool = {
        type: 'function',
        name: 'myFunction',
        parameters: { type: 'object' },
      };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'function',
        name: 'myFunction',
        parameters: { type: 'object' },
      });
    });

    it('should pass through unrecognized fields from tool configurations', () => {
      const tool = {
        urlContext: {
          myUrlParam: 1,
        },
      };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'url_context',
        my_url_param: 1,
      });
    });

    it('should handle mcpServer tool configuration and normalize string array allowed_tools to objects', () => {
      const tool = {
        mcpServer: {
          name: 'my-server',
          url: 'https://mcp.example.com',
          allowedTools: ['search', 'read_doc'],
        },
      };
      const result = toInteractionConfigTool(tool);
      assert.deepStrictEqual(result, {
        type: 'mcp_server',
        name: 'my-server',
        url: 'https://mcp.example.com',
        allowed_tools: [{ tools: ['search', 'read_doc'] }],
      });
    });
  });

  describe('toInteractionContent', () => {
    it('should convert TextPart', () => {
      const part: Part = { text: 'Hello' };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, { type: 'text', text: 'Hello' });
    });

    it('should convert MediaPart (image data)', () => {
      const part: Part = {
        media: {
          url: 'data:image/png;base64,DATA',
          contentType: 'image/png',
        },
      };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, {
        type: 'image',
        data: 'DATA',
        mime_type: 'image/png',
      });
    });

    it('should convert MediaPart (image uri)', () => {
      const part: Part = {
        media: {
          url: 'gs://bucket/image.png',
          contentType: 'image/png',
        },
      };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, {
        type: 'image',
        uri: 'gs://bucket/image.png',
        mime_type: 'image/png',
      });
    });

    it('should convert MediaPart (audio)', () => {
      const part: Part = {
        media: {
          url: 'data:audio/mp3;base64,DATA',
          contentType: 'audio/mp3',
        },
      };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, {
        type: 'audio',
        data: 'DATA',
        mime_type: 'audio/mp3',
      });
    });

    it('should convert MediaPart (document)', () => {
      const part: Part = {
        media: {
          url: 'gs://bucket/doc.pdf',
          contentType: 'application/pdf',
        },
      };
      const result = toInteractionContent(part);
      assert.deepStrictEqual(result, {
        type: 'document',
        uri: 'gs://bucket/doc.pdf',
        mime_type: 'application/pdf',
      });
    });

    it('should convert a CSV MediaPart to a document', () => {
      const result = toInteractionContent({
        media: { url: 'data:text/csv;base64,DATA', contentType: 'text/csv' },
      });
      assert.deepStrictEqual(result, {
        type: 'document',
        data: 'DATA',
        mime_type: 'text/csv',
      });
    });

    it('should send a YouTube URL without contentType as video without mime_type', () => {
      const url = 'https://www.youtube.com/watch?v=abc123';
      assert.deepStrictEqual(toInteractionContent({ media: { url } }), {
        type: 'video',
        uri: url,
      });
      assert.deepStrictEqual(
        toInteractionContent({ media: { url: 'https://youtu.be/abc123' } }),
        { type: 'video', uri: 'https://youtu.be/abc123' }
      );
    });

    it('should send other URIs without a known contentType as documents without mime_type', () => {
      const url =
        'https://generativelanguage.googleapis.com/v1beta/files/abc123';
      assert.deepStrictEqual(toInteractionContent({ media: { url } }), {
        type: 'document',
        uri: url,
      });
    });

    it('should still throw for a data URL without a contentType', () => {
      assert.throws(
        () => toInteractionContent({ media: { url: 'data:;base64,DATA' } }),
        /Media part missing contentType/
      );
    });

    it('sends Word and Excel files as documents', () => {
      // User report: 1.43 threw "Unsupported media type" for these, although
      // the server accepts them.
      const docx =
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
      const xlsx =
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
      assert.deepStrictEqual(
        toInteractionContent({
          media: { url: `data:${docx};base64,DATA`, contentType: docx },
        }),
        { type: 'document', data: 'DATA', mime_type: docx }
      );
      assert.deepStrictEqual(
        toInteractionContent({
          media: { url: 'gs://bucket/sheet.xlsx', contentType: xlsx },
        }),
        { type: 'document', uri: 'gs://bucket/sheet.xlsx', mime_type: xlsx }
      );
    });

    it('leaves other content types for the server to accept or reject', () => {
      assert.deepStrictEqual(
        toInteractionContent({
          media: { url: 'gs://bucket/notes.txt', contentType: 'text/plain' },
        }),
        {
          type: 'document',
          uri: 'gs://bucket/notes.txt',
          mime_type: 'text/plain',
        }
      );
    });
  });

  describe('toInteractionSteps', () => {
    it('should convert ToolRequestPart to step', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'func',
                input: { a: 1 },
                ref: 'ref1',
              },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'function_call',
          name: 'func',
          arguments: { a: 1 },
          id: 'ref1',
        },
      ]);
    });

    it('should send a ToolRequestPart thoughtSignature back as the function_call signature', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              toolRequest: { name: 'func', input: { a: 1 }, ref: 'ref1' },
              metadata: { thoughtSignature: 'sig-call' },
            },
          ],
        },
      ];
      assert.deepStrictEqual(toInteractionSteps(messages), [
        {
          type: 'function_call',
          name: 'func',
          arguments: { a: 1 },
          id: 'ref1',
          signature: 'sig-call',
        },
      ]);
    });

    it('should convert ToolResponsePart to step', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'func',
                output: { result: 'ok' },
                ref: 'ref1',
              },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'function_result',
          name: 'func',
          result: { result: 'ok' },
          call_id: 'ref1',
        },
      ]);
    });

    it('should group text contents into model_output step', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [{ text: 'Thinking' }, { text: 'Done' }],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'model_output',
          content: [
            { type: 'text', text: 'Thinking' },
            { type: 'text', text: 'Done' },
          ],
        },
      ]);
    });

    it('should convert custom googleSearchCall to google_search_call step', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              custom: {
                googleSearchCall: {
                  id: 'gs1',
                  arguments: { queries: ['genkit'] },
                },
              },
              metadata: { thoughtSignature: 'sig' },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'google_search_call',
          id: 'gs1',
          arguments: { queries: ['genkit'] },
          signature: 'sig',
        },
      ]);
    });

    it('should convert custom googleSearchResult to google_search_result step', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [
            {
              custom: {
                googleSearchResult: {
                  callId: 'gs1',
                  result: { answer: 'framework' },
                },
              },
              metadata: { thoughtSignature: 'sig' },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'google_search_result',
          call_id: 'gs1',
          result: { answer: 'framework' },
          signature: 'sig',
        },
      ]);
    });

    it('should round-trip a google_search_result with an array result (tool loop history)', () => {
      // Shape returned live by gemini-3.6-flash. Resending it in the history
      // (second turn of a tool loop) used to fail Zod validation.
      const step: Step = {
        type: 'google_search_result',
        call_id: 'call_1',
        result: [{ search_suggestions: '<div>...</div>' }],
        signature: 'sig-result',
      };
      const parts = fromInteractionStep(step);
      assert.deepStrictEqual(
        toInteractionSteps([{ role: 'model', content: parts }]),
        [
          {
            type: 'google_search_result',
            call_id: 'call_1',
            result: [{ search_suggestions: '<div>...</div>' }],
            signature: 'sig-result',
          },
        ]
      );
    });

    it('should convert custom executableCode to code_execution_call step', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              custom: {
                executableCode: { code: 'print("hello")', language: 'PYTHON' },
              },
              metadata: { thoughtSignature: 'sig', callId: 'ce1' },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'code_execution_call',
          id: 'ce1',
          arguments: { code: 'print("hello")', language: 'PYTHON' },
          signature: 'sig',
        },
      ]);
    });

    it('should convert custom codeExecutionResult to code_execution_result step', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [
            {
              custom: {
                codeExecutionResult: { output: 'hello\n' },
              },
              metadata: { thoughtSignature: 'sig', callId: 'ce1' },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'code_execution_result',
          call_id: 'ce1',
          result: 'hello\n',
          signature: 'sig',
        },
      ]);
    });

    it('throws GenkitError when tool output contains non-text/image content', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'pdfTool',
                ref: 'call-1',
                output: undefined,
                content: [
                  {
                    media: {
                      url: 'data:application/pdf;base64,ABC',
                      contentType: 'application/pdf',
                    },
                  },
                ],
              },
            },
          ],
        },
      ];
      assert.throws(
        () => toInteractionSteps(messages),
        (err: any) => {
          assert.strictEqual(err.status, 'INVALID_ARGUMENT');
          assert.strictEqual(
            err.originalMessage,
            'Tool output for pdfTool may only contain text or image content.'
          );
          return true;
        }
      );
    });

    it('wraps plain array tool output in result object', () => {
      const messages: MessageData[] = [
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'listTool',
                ref: 'call-1',
                output: [1, 2, 3],
              },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'function_result',
          name: 'listTool',
          call_id: 'call-1',
          result: { result: [1, 2, 3] },
        },
      ]);
    });

    it('should convert empty reasoning part with thoughtSignature to thought step without summary', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              reasoning: '',
              metadata: {
                thoughtSignature: 'sig-123',
              },
              custom: {
                thought: {
                  type: 'thought',
                  summary: [],
                  signature: 'sig-123',
                },
              },
            },
            {
              toolRequest: {
                name: 'getWeather',
                ref: 'call-1',
                input: { location: 'London' },
              },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'thought',
          signature: 'sig-123',
        },
        {
          type: 'function_call',
          name: 'getWeather',
          id: 'call-1',
          arguments: { location: 'London' },
        },
      ]);
    });

    it('should convert non-empty reasoning part to thought step with summary and signature', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              reasoning: 'Looking up weather data',
              metadata: {
                thoughtSignature: 'sig-456',
              },
            },
            {
              text: 'The weather is sunny.',
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'thought',
          summary: [{ type: 'text', text: 'Looking up weather data' }],
          signature: 'sig-456',
        },
        {
          type: 'model_output',
          content: [{ type: 'text', text: 'The weather is sunny.' }],
        },
      ]);
    });

    it('should preserve rich summary from custom.thought when available', () => {
      const messages: MessageData[] = [
        {
          role: 'model',
          content: [
            {
              reasoning: 'Plan',
              custom: {
                thought: {
                  type: 'thought',
                  summary: [
                    {
                      type: 'text',
                      text: 'Plan',
                      annotations: [
                        {
                          type: 'url_citation',
                          title: 'Doc',
                          url: 'https://example.com',
                        },
                      ],
                    },
                  ],
                  signature: 'custom-sig-789',
                },
              },
            },
          ],
        },
      ];
      const result = toInteractionSteps(messages);
      assert.deepStrictEqual(result, [
        {
          type: 'thought',
          summary: [
            {
              type: 'text',
              text: 'Plan',
              annotations: [
                {
                  type: 'url_citation',
                  title: 'Doc',
                  url: 'https://example.com',
                },
              ],
            },
          ],
          signature: 'custom-sig-789',
        },
      ]);
    });
  });

  describe('fromInteractionContent', () => {
    it('should convert TextContent', () => {
      const content: Content = {
        type: 'text',
        text: 'Hello world',
        annotations: [
          {
            type: 'file_citation',
            start_index: 0,
            end_index: 5,
            source: 'source',
          },
        ],
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        text: 'Hello world',
        metadata: {
          annotations: [
            {
              type: 'file_citation',
              start_index: 0,
              end_index: 5,
              source: 'source',
            },
          ],
        },
      });
    });

    it('should convert ImageContent with data', () => {
      const content: Content = {
        type: 'image',
        data: 'BASE64DATA',
        mime_type: 'image/png',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        media: {
          url: 'data:image/png;base64,BASE64DATA',
          contentType: 'image/png',
        },
      });
    });

    it('should convert ImageContent with uri', () => {
      const content: Content = {
        type: 'image',
        uri: 'gs://bucket/image.png',
        mime_type: 'image/png',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        media: {
          url: 'gs://bucket/image.png',
          contentType: 'image/png',
        },
      });
    });

    it('should convert ImageContent with resolution', () => {
      const content: Content = {
        type: 'image',
        uri: 'gs://bucket/image.png',
        mime_type: 'image/png',
        resolution: 'high',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        media: {
          url: 'gs://bucket/image.png',
          contentType: 'image/png',
        },
        metadata: { resolution: 'high' },
      });
    });

    it('should convert ThoughtContent', () => {
      const content: Content = {
        type: 'thought',
        signature: 'SIG',
        summary: [{ type: 'text', text: 'Thinking...' }],
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        reasoning: 'Thinking...',
        metadata: {
          thoughtSignature: 'SIG',
        },
        custom: {
          thought: content,
        },
      });
    });

    it('should convert ThoughtContent with mixed summary', () => {
      const content: Content = {
        type: 'thought',
        signature: 'SIG',
        summary: [
          { type: 'text', text: 'Thinking about...' },
          { type: 'image', uri: 'gs://image.png' },
          { type: 'text', text: '...this image.' },
        ],
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        reasoning: 'Thinking about...\n[Image]\n...this image.',
        metadata: {
          thoughtSignature: 'SIG',
        },
        custom: {
          thought: content,
        },
      });
    });

    it('should convert FunctionCallContent', () => {
      const content: Content = {
        type: 'function_call',
        name: 'get_weather',
        arguments: { location: 'Paris' },
        id: 'call_123',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        toolRequest: {
          name: 'get_weather',
          input: { location: 'Paris' },
          ref: 'call_123',
        },
      });
    });

    it('should convert FunctionResultContent', () => {
      const content: Content = {
        type: 'function_result',
        name: 'get_weather',
        result: { temperature: 20 },
        call_id: 'call_123',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        toolResponse: {
          name: 'get_weather',
          output: { temperature: 20 },
          ref: 'call_123',
        },
      });
    });

    it('should convert FunctionResultContent with plain array result into tool output', () => {
      const content: Content = {
        type: 'function_result',
        name: 'list_items',
        result: [1, 2, 3] as any,
        call_id: 'call_123',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        toolResponse: {
          name: 'list_items',
          output: [1, 2, 3],
          ref: 'call_123',
        },
      });
    });

    it('should convert FunctionResultContent with multimodal Content array into tool content', () => {
      const content: Content = {
        type: 'function_result',
        name: 'multimodal_tool',
        result: [
          { type: 'text', text: 'description' },
          { type: 'image', uri: 'https://example.com/img.png' },
        ] as any,
        call_id: 'call_123',
      };
      const result = fromInteractionContent(content);
      assert.deepStrictEqual(result, {
        toolResponse: {
          name: 'multimodal_tool',
          content: [
            { text: 'description', metadata: { annotations: undefined } },
            {
              media: {
                url: 'https://example.com/img.png',
                contentType: undefined,
              },
            },
          ],
          ref: 'call_123',
        },
      });
    });
  });

  describe('fromInteractionStep', () => {
    it('should convert model_output step', () => {
      const step: any = {
        type: 'model_output',
        content: [{ type: 'text', text: 'Hello' }],
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        { text: 'Hello', metadata: { annotations: undefined } },
      ]);
    });

    it('should skip user_input step', () => {
      const step: any = {
        type: 'user_input',
        content: [{ type: 'text', text: 'Hello' }],
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, []);
    });

    it('should convert google_search_call step', () => {
      const step: any = {
        type: 'google_search_call',
        id: '123',
        arguments: { queries: ['foo'] },
        signature: 'sig',
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        {
          custom: {
            googleSearchCall: {
              id: '123',
              arguments: { queries: ['foo'] },
            },
          },
          metadata: { thoughtSignature: 'sig' },
        },
      ]);
    });

    it('should convert google_search_result step', () => {
      const step: any = {
        type: 'google_search_result',
        call_id: '123',
        result: { content: 'bar' },
        signature: 'sig',
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        {
          custom: {
            googleSearchResult: {
              callId: '123',
              result: { content: 'bar' },
            },
          },
          metadata: { thoughtSignature: 'sig' },
        },
      ]);
    });

    it('should convert code_execution_call step', () => {
      const step: any = {
        type: 'code_execution_call',
        id: '123',
        arguments: { code: 'print(1)', language: 'PYTHON' },
        signature: 'sig',
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        {
          custom: {
            executableCode: { code: 'print(1)', language: 'PYTHON' },
          },
          metadata: { callId: '123', thoughtSignature: 'sig' },
        },
      ]);
    });

    it('should convert code_execution_result step', () => {
      const step: any = {
        type: 'code_execution_result',
        call_id: '123',
        result: '1\n',
        signature: 'sig',
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        {
          custom: {
            codeExecutionResult: { output: '1\n', outcome: 'OUTCOME_OK' },
          },
          metadata: { callId: '123', thoughtSignature: 'sig' },
        },
      ]);
    });

    it('should convert thought step', () => {
      const step: Step = {
        type: 'thought',
        signature: '',
        summary: [
          { type: 'text', text: '**Protocol...**' },
          { type: 'text', text: ' **Evalua...**' },
        ],
      };
      const result = fromInteractionStep(step);
      assert.deepStrictEqual(result, [
        {
          reasoning: '**Protocol...**\n **Evalua...**',
          metadata: { thoughtSignature: '' },
          custom: { thought: step },
        },
      ]);
    });
  });

  describe('fromInteractionDelta', () => {
    it('should convert text delta', () => {
      const delta: StepDeltaData = { type: 'text', text: 'hello' };
      const result = fromInteractionDelta(delta);
      assert.deepStrictEqual(result, [{ text: 'hello' }]);
    });

    it('should convert image delta', () => {
      const delta: StepDeltaData = {
        type: 'image',
        data: 'XYZ',
        mime_type: 'image/png',
      };
      const result = fromInteractionDelta(delta);
      assert.deepStrictEqual(result, [
        {
          media: { url: 'data:image/png;base64,XYZ', contentType: 'image/png' },
        },
      ]);
    });

    it('returns no parts for tool step deltas (emitted when the step stops)', () => {
      const deltas: StepDeltaData[] = [
        { type: 'google_search_call', arguments: { queries: ['q'] } },
        { type: 'code_execution_result', result: '1\n' },
        { type: 'url_context_result', result: [] },
        { type: 'processing_call', signature: 'sig' },
      ];
      for (const delta of deltas) {
        assert.deepStrictEqual(fromInteractionDelta(delta), [], delta.type);
      }
    });

    it('should ignore arguments_delta', () => {
      const delta: StepDeltaData = {
        type: 'arguments_delta',
        arguments: '{"a":',
      };
      const result = fromInteractionDelta(delta);
      assert.deepStrictEqual(result, []);
    });

    it('should convert thought_summary delta to reasoning', () => {
      const delta: StepDeltaData = {
        type: 'thought_summary',
        content: { type: 'text', text: 'thinking process' },
      };
      const result = fromInteractionDelta(delta);
      assert.deepStrictEqual(result, [{ reasoning: 'thinking process' }]);
    });

    it('should convert a thought_summary image delta to an [Image] reasoning placeholder', () => {
      const delta: StepDeltaData = {
        type: 'thought_summary',
        content: { type: 'image', data: 'base64data', mime_type: 'image/jpeg' },
      };
      const result = fromInteractionDelta(delta);
      // The draft image is part of the thought, so it must not be streamed as
      // a media part (the final response only shows `[Image]` in reasoning).
      assert.deepStrictEqual(result, [{ reasoning: '[Image]' }]);
    });

    it('should convert other non-text thought_summary deltas to a type placeholder', () => {
      const audio: StepDeltaData = {
        type: 'thought_summary',
        content: { type: 'audio', data: 'base64data', mime_type: 'audio/wav' },
      };
      assert.deepStrictEqual(fromInteractionDelta(audio), [
        { reasoning: '[Audio]' },
      ]);

      const functionCall: StepDeltaData = {
        type: 'thought_summary',
        content: { type: 'function_call', name: 'myFunc', id: 'call_1' },
      };
      assert.deepStrictEqual(fromInteractionDelta(functionCall), [
        { reasoning: '[Function call]' },
      ]);
    });
  });

  describe('createInteractionChunkConverter', () => {
    // Event sequences seen live (Oct 2026); signatures and text shortened.
    // Every tool step streams the same way: step.start with the ids, one
    // delta with everything else, then step.stop.
    const start = (index: number, step: Step): InteractionSseEvent => ({
      event_type: 'step.start',
      index,
      step,
    });
    const delta = (index: number, d: StepDeltaData): InteractionSseEvent => ({
      event_type: 'step.delta',
      index,
      delta: d,
    });
    const stop = (index: number): InteractionSseEvent => ({
      event_type: 'step.stop',
      index,
    });
    const completed = (
      interaction: Partial<GeminiInteraction>
    ): InteractionSseEvent => ({
      event_type: 'interaction.completed',
      interaction,
    });
    const thoughtEvents = (index: number): InteractionSseEvent[] => [
      start(index, { type: 'thought' }),
      delta(index, { type: 'thought_signature', signature: 'sig_thought' }),
      stop(index),
    ];
    /** The chunk parts for each event, in order. */
    const chunksFor = (events: InteractionSseEvent[]): Part[][] => {
      const toChunkParts = createInteractionChunkConverter();
      return events.map((event) => toChunkParts(event));
    };

    it('streams a function call as a tool request when the interaction completes', () => {
      // vertexAI gemini-3.8-flash (tool-calling flow).
      const chunks = chunksFor([
        ...thoughtEvents(0),
        start(1, {
          id: 'call_7227796',
          type: 'function_call',
          name: 'getWeather',
          arguments: {},
        }),
        delta(1, {
          type: 'arguments_delta',
          arguments: '{"location":"Paris, France"}',
        }),
        {
          event_type: 'step.stop',
          index: 1,
          usage: { total_tokens: 210 },
        },
        completed({ id: '', status: 'requires_action' }),
      ]);
      assert.deepStrictEqual(chunks, [
        [],
        [{ metadata: { thoughtSignature: 'sig_thought' } }],
        [],
        [],
        [],
        [],
        [
          {
            toolRequest: {
              name: 'getWeather',
              ref: 'call_7227796',
              input: { location: 'Paris, France' },
            },
          },
        ],
      ]);
    });

    it('streams a long function call argument from a single arguments_delta', () => {
      // googleAI gemini-3.8-flash (streaming-tool-calling flow): a 300+ word
      // argument still came as one delta.
      const longText = 'In the misty heart of the Whispering Woods. '.repeat(
        40
      );
      const args = { longText, nestedData: { arrayItems: ['a'], id: 101 } };
      const chunks = chunksFor([
        ...thoughtEvents(0),
        start(1, {
          id: 'call_958433',
          type: 'function_call',
          name: 'getCharacterName',
          arguments: {},
        }),
        delta(1, { type: 'arguments_delta', arguments: JSON.stringify(args) }),
        stop(1),
        completed({ status: 'requires_action' }),
      ]);
      assert.deepStrictEqual(chunks[chunks.length - 1], [
        {
          toolRequest: {
            name: 'getCharacterName',
            ref: 'call_958433',
            input: args,
          },
        },
      ]);
    });

    it('does not stream a server-side function call as a tool request', () => {
      // Seen live with MCP on googleAI: the call and its result in the same
      // response, with status requires_action. Nothing for Genkit to run.
      const chunks = chunksFor([
        start(0, {
          id: 'call_1',
          type: 'function_call',
          name: 'complex_server:analyze_data',
          arguments: {},
        }),
        delta(0, { type: 'arguments_delta', arguments: '{"x":1}' }),
        stop(0),
        start(1, {
          type: 'function_result',
          call_id: 'call_1',
          name: 'complex_server:analyze_data',
          result: 'ok',
        }),
        stop(1),
        completed({ status: 'requires_action' }),
      ]);
      const parts = chunks.flat();
      assert.ok(parts.every((p) => p.toolRequest === undefined));
      assert.deepStrictEqual(
        parts.map((p) => Object.keys(p.custom ?? {})),
        [['serverFunctionCall'], ['serverFunctionResult']]
      );
    });

    it('streams Google Search steps with their ids when each step stops, matching the final response', () => {
      // googleAI gemini-3.6-flash (streaming-google-search flow).
      const queries = ['Nobel Prize in Physics 2026 winner'];
      const result = [{ search_suggestions: '<div>...</div>' }];
      const chunks = chunksFor([
        start(0, {
          id: 'call_1725402',
          signature: '',
          type: 'google_search_call',
          search_type: 'web_search',
        }),
        delta(0, {
          type: 'google_search_call',
          signature: 'sig_call',
          arguments: { queries },
        }),
        stop(0),
        start(1, {
          call_id: 'call_1725402',
          signature: '',
          type: 'google_search_result',
        }),
        delta(1, {
          type: 'google_search_result',
          signature: 'sig_result',
          result,
          is_error: false,
        }),
        stop(1),
        ...thoughtEvents(2),
        start(3, { type: 'model_output' }),
        delta(3, { type: 'text', text: 'The 2026 Nobel Prize ' }),
        delta(3, { type: 'text', text: 'in Physics...' }),
        stop(3),
        completed({ status: 'completed' }),
      ]);
      const final = fromInteractionSync({
        status: 'completed',
        steps: [
          {
            id: 'call_1725402',
            signature: 'sig_call',
            type: 'google_search_call',
            search_type: 'web_search',
            arguments: { queries },
          },
          {
            call_id: 'call_1725402',
            signature: 'sig_result',
            type: 'google_search_result',
            result,
            is_error: false,
          },
        ],
      });
      const [searchCall, searchResult] = final.message?.content ?? [];
      assert.deepStrictEqual(searchCall, {
        custom: {
          googleSearchCall: {
            id: 'call_1725402',
            arguments: { queries },
            searchType: 'web_search',
          },
        },
        metadata: { thoughtSignature: 'sig_call' },
      });
      // Nothing at the deltas; each step's part (as in the final response)
      // when it stops.
      assert.deepStrictEqual(chunks.slice(0, 6), [
        [],
        [],
        [searchCall],
        [],
        [],
        [searchResult],
      ]);
      // Text still streams as it arrives.
      assert.deepStrictEqual(chunks[10], [{ text: 'The 2026 Nobel Prize ' }]);
      assert.deepStrictEqual(chunks[11], [{ text: 'in Physics...' }]);
      assert.deepStrictEqual(chunks[12], []);
      assert.deepStrictEqual(chunks[13], []);
    });

    it('streams other built-in tool steps when they stop', () => {
      const chunks = chunksFor([
        start(0, { id: 'call_1', type: 'url_context_call', signature: '' }),
        delta(0, {
          type: 'url_context_call',
          signature: 'sig_url',
          arguments: { urls: ['https://example.com'] },
        }),
        stop(0),
        start(1, { id: 'proc_1', type: 'processing_call', signature: '' }),
        delta(1, { type: 'processing_call', signature: 'sig_proc' }),
        stop(1),
      ]);
      assert.deepStrictEqual(chunks, [
        [],
        [],
        [
          {
            custom: {
              urlContextCall: {
                id: 'call_1',
                arguments: { urls: ['https://example.com'] },
              },
            },
            metadata: { thoughtSignature: 'sig_url' },
          },
        ],
        [],
        [],
        [
          {
            custom: { processingCall: { id: 'proc_1' } },
            metadata: { thoughtSignature: 'sig_proc' },
          },
        ],
      ]);
    });

    it('does not change the events it reads', () => {
      // The final response is assembled from the same event objects.
      const startEvent = start(0, { id: 'c1', type: 'google_search_call' });
      chunksFor([
        startEvent,
        delta(0, { type: 'google_search_call', arguments: { queries: ['q'] } }),
        stop(0),
      ]);
      assert.deepStrictEqual(startEvent, {
        event_type: 'step.start',
        index: 0,
        step: { id: 'c1', type: 'google_search_call' },
      });
    });
  });

  describe('fromInteraction', () => {
    it('should convert cancelled interaction', () => {
      const interaction: GeminiInteraction = {
        id: '123',
        status: 'cancelled',
      };
      const result = fromInteraction(interaction);
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.output?.finishReason, 'aborted');
      assert.strictEqual(result.output?.finishMessage, 'Operation cancelled');
      assert.deepStrictEqual(result.output?.message?.content, [
        { text: 'Operation cancelled.' },
      ]);
    });

    it('should finish a failed operation with an error (no infinite polling)', () => {
      const result = fromInteraction({
        id: '123',
        status: 'failed',
        errors: [{ code: 'api_error', message: 'boom' }],
      });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.output, undefined);
      assert.ok(result.error?.message.includes('[api_error] boom'));
      assert.strictEqual(result.error?.code, 'api_error');
    });

    it('should finish a failed operation without errors[] with a generic error', () => {
      const result = fromInteraction({ id: '123', status: 'failed' });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.error?.message, 'Interaction failed');
    });

    it('should convert a content-blocked failed operation to finishReason blocked', () => {
      const result = fromInteraction({
        id: '123',
        status: 'failed',
        errors: [{ code: 'recitation', message: 'Recitation blocked.' }],
        usage: { total_input_tokens: 7, total_tokens: 7 },
      });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.error, undefined);
      assert.strictEqual(result.output?.finishReason, 'blocked');
      assert.strictEqual(result.output?.finishMessage, 'Recitation blocked.');
      assert.strictEqual(result.output?.usage?.inputTokens, 7);
    });

    it('should carry usage on a cancelled operation', () => {
      const result = fromInteraction({
        id: '123',
        status: 'cancelled',
        usage: { total_input_tokens: 3, total_tokens: 3 },
      });
      assert.strictEqual(result.output?.usage?.inputTokens, 3);
    });

    it('should finish a requires_action operation (collaborative planning) and surface the plan', () => {
      const result = fromInteraction({
        id: '123',
        status: 'requires_action',
        steps: [
          {
            type: 'model_output',
            content: [
              {
                type: 'text',
                text: "Here is the research plan I've prepared: 1. ...",
              },
            ],
          },
        ],
      });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.error, undefined);
      assert.strictEqual(result.output?.finishReason, 'stop');
      assert.deepStrictEqual(
        result.output?.message?.content.map((p) => p.text),
        ["Here is the research plan I've prepared: 1. ..."]
      );
      assert.strictEqual(
        result.output?.message?.metadata?.interactionStatus,
        'requires_action'
      );
      assert.strictEqual(
        result.output?.message?.metadata?.interactionId,
        '123'
      );
    });

    it('should surface pending client function calls as toolRequests on requires_action', () => {
      const result = fromInteraction({
        id: '123',
        status: 'requires_action',
        steps: [
          {
            type: 'function_call',
            id: 'call-1',
            name: 'lookup',
            arguments: { q: 'x' },
          },
        ],
      });
      assert.strictEqual(result.done, true);
      assert.deepStrictEqual(result.output?.message?.content, [
        { toolRequest: { name: 'lookup', ref: 'call-1', input: { q: 'x' } } },
      ]);
    });

    it('should keep the function_call signature and round-trip it in the next request', () => {
      // Shape observed live (gemini-flash-latest, store: false): the
      // function_call step carries its own signature.
      const result = fromInteraction({
        id: 'v1_abc123',
        status: 'requires_action',
        steps: [
          {
            type: 'function_call',
            id: 'call-1',
            name: 'lookup',
            arguments: { q: 'x' },
            signature: 'sig-call',
          },
        ],
      });
      const content = result.output?.message?.content ?? [];
      assert.deepStrictEqual(content, [
        {
          toolRequest: { name: 'lookup', ref: 'call-1', input: { q: 'x' } },
          metadata: { thoughtSignature: 'sig-call' },
        },
      ]);

      // Resending the history (stateless) includes the signature again.
      assert.deepStrictEqual(toInteractionSteps([{ role: 'model', content }]), [
        {
          type: 'function_call',
          name: 'lookup',
          arguments: { q: 'x' },
          id: 'call-1',
          signature: 'sig-call',
        },
      ]);
    });

    it('should keep the signature on completed (server-side) function calls', () => {
      const result = fromInteractionSync({
        status: 'completed',
        steps: [
          {
            type: 'function_call',
            id: 'call-1',
            name: 'lookup',
            arguments: { q: 'x' },
            signature: 'sig-call',
          },
          {
            type: 'function_result',
            call_id: 'call-1',
            name: 'lookup',
            result: { a: 1 },
          },
        ],
      });
      assert.deepStrictEqual(result.message?.content[0], {
        custom: {
          serverFunctionCall: {
            id: 'call-1',
            name: 'lookup',
            arguments: { q: 'x' },
          },
        },
        metadata: { thoughtSignature: 'sig-call' },
      });
    });

    it('should keep polling for in_progress', () => {
      const result = fromInteraction({ id: '123', status: 'in_progress' });
      assert.strictEqual(result.done, false);
    });

    it('should keep polling for queued (waiting for capacity)', () => {
      const result = fromInteraction({ id: '123', status: 'queued' });
      assert.strictEqual(result.done, false);
      assert.strictEqual(result.error, undefined);
    });

    it('should finish with an error for an unknown status (no infinite polling)', () => {
      const result = fromInteraction({
        id: '123',
        status: 'expired' as any,
      });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.output, undefined);
      assert.strictEqual(
        result.error?.message,
        'Unknown interaction status: expired'
      );
    });

    it('should convert an incomplete operation to finishReason length', () => {
      const result = fromInteraction({
        id: '123',
        status: 'incomplete',
        steps: [
          {
            type: 'model_output',
            content: [{ type: 'text', text: 'partial' }],
          },
        ],
      });
      assert.strictEqual(result.done, true);
      assert.strictEqual(result.output?.finishReason, 'length');
      assert.deepStrictEqual(
        result.output?.message?.content.map((p) => p.text),
        ['partial']
      );
    });
  });

  describe('fromInteractionSync', () => {
    it('returns finishReason blocked for a post-execution safety block', () => {
      const result = fromInteractionSync({
        id: 'int-1',
        status: 'failed',
        errors: [
          {
            code: 'safety',
            message: 'Response blocked due to safety violations.',
          },
        ],
      });
      assert.strictEqual(result.finishReason, 'blocked');
      assert.strictEqual(
        result.finishMessage,
        'Response blocked due to safety violations.'
      );
      assert.deepStrictEqual(result.message?.content, []);
      assert.strictEqual(result.message?.metadata?.interactionId, 'int-1');
    });

    it('returns finishReason blocked for every content-block code', () => {
      for (const code of [
        'safety',
        'recitation',
        'language',
        'prohibited_content',
        'spii',
        'blocklist',
        'image_safety',
        'image_prohibited_content',
        'image_recitation',
        'image_other',
        'content_blocked',
        'jailbreak',
        'model_armor',
      ]) {
        const result = fromInteractionSync({
          status: 'failed',
          errors: [{ code, message: 'blocked' }],
        });
        assert.strictEqual(result.finishReason, 'blocked', `code "${code}"`);
      }
    });

    it('throws a retryable ABORTED GenkitError for malformed_function_call', () => {
      const advice =
        'Model generated invalid JSON syntax and the output could not be parsed. Please retry the request.';
      assert.throws(
        () =>
          fromInteractionSync({
            status: 'failed',
            errors: [{ code: 'malformed_function_call', message: advice }],
          }),
        (err: any) => {
          assert.ok(err instanceof GenkitError);
          assert.strictEqual(err.status, 'ABORTED');
          assert.ok(err.message.includes('[malformed_function_call]'));
          assert.ok(err.message.includes(advice));
          return true;
        }
      );
    });

    it('throws ABORTED for unexpected_tool_call and no_image', () => {
      for (const code of ['unexpected_tool_call', 'no_image']) {
        assert.throws(
          () =>
            fromInteractionSync({
              status: 'failed',
              errors: [{ code, message: 'retry' }],
            }),
          (err: any) => err instanceof GenkitError && err.status === 'ABORTED',
          `code "${code}"`
        );
      }
    });

    it('throws a non-retryable UNKNOWN GenkitError for a failure without errors[]', () => {
      assert.throws(
        () => fromInteractionSync({ status: 'failed' }),
        (err: any) => {
          assert.ok(err instanceof GenkitError);
          assert.strictEqual(err.status, 'UNKNOWN');
          assert.ok(err.message.includes('Interaction failed'));
          return true;
        }
      );
    });

    it('throws a non-retryable UNKNOWN GenkitError for an unrecognized code', () => {
      assert.throws(
        () =>
          fromInteractionSync({
            status: 'failed',
            errors: [{ code: 'something_new', message: 'huh' }],
          }),
        (err: any) =>
          err instanceof GenkitError &&
          err.status === 'UNKNOWN' &&
          err.message.includes('[something_new] huh')
      );
    });

    it('returns finishReason length with partial content for an incomplete interaction', () => {
      const result = fromInteractionSync({
        id: 'int-1',
        status: 'incomplete',
        steps: [
          {
            type: 'model_output',
            content: [{ type: 'text', text: 'partial' }],
          },
        ],
      });
      assert.strictEqual(result.finishReason, 'length');
      assert.strictEqual(
        result.finishMessage,
        'Interaction incomplete (truncated output)'
      );
      assert.deepStrictEqual(
        result.message?.content.map((p) => p.text),
        ['partial']
      );
    });

    it('carries usage (including per-modality counts) on a blocked response', () => {
      const result = fromInteractionSync({
        status: 'failed',
        errors: [{ code: 'safety', message: 'blocked' }],
        usage: {
          total_input_tokens: 10,
          total_output_tokens: 0,
          total_tokens: 10,
          input_tokens_by_modality: [
            { modality: 'text', tokens: 8 },
            { modality: 'image', tokens: 2 },
          ],
        },
      });
      assert.strictEqual(result.finishReason, 'blocked');
      assert.deepStrictEqual(result.usage, {
        inputTokens: 10,
        outputTokens: 0,
        totalTokens: 10,
        cachedContentTokens: undefined,
        thoughtsTokens: undefined,
        inputCharacters: 8,
        inputImages: 2,
      });
    });

    it('maps video usage and returns the generated video (shape seen live)', () => {
      // gemini-omni-1.1-flash, text to video, store: false (no id).
      const result = fromInteractionSync({
        status: 'completed',
        model: 'gemini-omni-1.1-flash',
        steps: [
          { type: 'thought', signature: 'sig123' },
          {
            type: 'model_output',
            content: [{ type: 'video', mime_type: 'video/mp4', data: 'AAAA' }],
          },
        ],
        usage: {
          total_input_tokens: 16,
          total_output_tokens: 58936,
          total_thought_tokens: 383,
          total_tokens: 59335,
          input_tokens_by_modality: [{ modality: 'text', tokens: 16 }],
          output_tokens_by_modality: [{ modality: 'video', tokens: 57920 }],
        },
      });
      assert.strictEqual(result.finishReason, 'stop');
      assert.strictEqual(result.usage?.outputVideos, 57920);
      const video = result.message?.content.find((p) => p.media);
      assert.deepStrictEqual(video?.media, {
        url: 'data:video/mp4;base64,AAAA',
        contentType: 'video/mp4',
      });
    });

    it('maps video input usage', () => {
      const result = fromInteractionSync({
        status: 'completed',
        usage: {
          total_input_tokens: 30,
          input_tokens_by_modality: [{ modality: 'video', tokens: 30 }],
        },
      });
      assert.strictEqual(result.usage?.inputVideos, 30);
    });

    it('carries usage on a cancelled response', () => {
      const result = fromInteractionSync({
        status: 'cancelled',
        usage: { total_input_tokens: 4, total_tokens: 4 },
      });
      assert.strictEqual(result.finishReason, 'aborted');
      assert.strictEqual(result.usage?.inputTokens, 4);
    });

    it('carries usage even when the interaction has no steps', () => {
      const result = fromInteractionSync({
        status: 'completed',
        usage: { total_input_tokens: 5, total_tokens: 5 },
      });
      assert.strictEqual(result.usage?.inputTokens, 5);
    });

    it('still returns finishReason stop for a completed interaction', () => {
      const result = fromInteractionSync({
        status: 'completed',
        steps: [
          { type: 'model_output', content: [{ type: 'text', text: 'done' }] },
        ],
      });
      assert.strictEqual(result.finishReason, 'stop');
    });
  });
});
