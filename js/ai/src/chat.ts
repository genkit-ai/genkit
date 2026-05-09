/**
 * Copyright 2024 Google LLC
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

import type { StreamingCallback, z } from '@genkit-ai/core';
import { Channel } from '@genkit-ai/core/async';
import {
  ATTR_PREFIX,
  SPAN_TYPE_ATTR,
  runInNewSpan,
} from '@genkit-ai/core/tracing';
import {
  generate,
  type GenerateOptions,
  type GenerateResponse,
  type GenerateResponseChunk,
  type GenerateStreamOptions,
  type GenerateStreamResponse,
} from './generate.js';
import {
  GenerationCommonConfigSchema,
  type MessageData,
} from './model-types.js';
import { type Part } from './parts.js';
import {
  runWithSession,
  type BaseGenerateOptions,
  type Session,
  type SessionStore,
} from './session.js';

export const MAIN_THREAD = 'main';
export const SESSION_ID_ATTR = `${ATTR_PREFIX}:sessionId`;
export const THREAD_NAME_ATTR = `${ATTR_PREFIX}:threadName`;

export type ChatGenerateOptions<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> = GenerateOptions<O, CustomOptions>;

export interface PromptRenderOptions<I> {
  input?: I;
}

export type ChatOptions<I = undefined, S = any> = (
  | PromptRenderOptions<I>
  | BaseGenerateOptions
) & {
  store?: SessionStore<S>;
  sessionId?: string;
};

/**
 * Chat encapsulates a statful execution environment for chat.
 * Chat session executed within a session in this environment will have acesss to
 * session convesation history.
 *
 * ```ts
 * const ai = genkit({...});
 * const chat = ai.chat(); // create a Chat
 * let response = await chat.send('hi, my name is Genkit');
 * response = await chat.send('what is my name?'); // chat history aware conversation
 * ```
 */
export class Chat {
  private requestBase?: Promise<BaseGenerateOptions>;
  readonly sessionId: string;
  private _messages?: MessageData[];
  private threadName: string;

  constructor(
    readonly session: Session,
    requestBase: Promise<BaseGenerateOptions>,
    options: {
      id: string;
      thread: string;
      messages?: MessageData[];
    }
  ) {
    this.sessionId = options.id;
    this.threadName = options.thread;
    this.requestBase = requestBase?.then((rb) => {
      const requestBase = { ...rb };
      // this is handling dotprompt render case
      if (requestBase && requestBase['prompt']) {
        const basePrompt = requestBase['prompt'] as string | Part | Part[];
        let promptMessage: MessageData;
        if (typeof basePrompt === 'string') {
          promptMessage = {
            role: 'user',
            content: [{ text: basePrompt }],
          };
        } else if (Array.isArray(basePrompt)) {
          promptMessage = {
            role: 'user',
            content: basePrompt,
          };
        } else {
          promptMessage = {
            role: 'user',
            content: [basePrompt],
          };
        }
        requestBase.messages = [...(requestBase.messages ?? []), promptMessage];
      }
      if (hasPreamble(requestBase.messages)) {
        requestBase.messages = [
          // if request base contains a preamble, always put it first
          ...(getPreamble(requestBase.messages) ?? []),
          // strip out the preamble from history
          ...(stripPreamble(options.messages) ?? []),
          // add whatever non-preamble remains from request
          ...(stripPreamble(requestBase.messages) ?? []),
        ];
      } else {
        requestBase.messages = [
          ...(options.messages ?? []),
          ...(requestBase.messages ?? []),
        ];
      }
      this._messages = requestBase.messages;
      return requestBase;
    });
    this._messages = options.messages;
  }

  async send<
    O extends z.ZodTypeAny = z.ZodTypeAny,
    CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
  >(
    options: string | Part[] | ChatGenerateOptions<O, CustomOptions>
  ): Promise<GenerateResponse<z.infer<O>>> {
    return runWithSession(this.session.registry, this.session, () =>
      runInNewSpan(
        this.session.registry,
        {
          metadata: {
            name: 'send',
          },
          labels: {
            [SPAN_TYPE_ATTR]: 'helper',
            [SESSION_ID_ATTR]: this.session.id,
            [THREAD_NAME_ATTR]: this.threadName,
          },
        },
        async (metadata) => {
          const resolvedOptions = resolveSendOptions(options);
          let streamingCallback:
            | StreamingCallback<GenerateResponseChunk>
            | undefined = undefined;

          if (resolvedOptions.onChunk || resolvedOptions.streamingCallback) {
            streamingCallback =
              resolvedOptions.onChunk ?? resolvedOptions.streamingCallback;
          }
          const request: GenerateOptions = {
            ...(await this.requestBase),
            messages: this.messages,
            ...resolvedOptions,
          };
          metadata.input = resolvedOptions;
          const response = await generate(this.session.registry, {
            ...request,
            onChunk: streamingCallback,
          });
          this.requestBase = Promise.resolve({
            ...(await this.requestBase),
            // these things may get changed by tools calling within generate.
            tools: response?.request?.tools?.map((td) => td.name),
            toolChoice: response?.request?.toolChoice,
            config: response?.request?.config,
          });
          await this.updateMessages(response.messages);
          metadata.output = JSON.stringify(response);
          return response;
        }
      )
    );
  }

  sendStream<
    O extends z.ZodTypeAny = z.ZodTypeAny,
    CustomOptions extends z.ZodTypeAny = typeof GenerationCommonConfigSchema,
  >(
    options: string | Part[] | GenerateStreamOptions<O, CustomOptions>
  ): GenerateStreamResponse<z.infer<O>> {
    const channel = new Channel<GenerateResponseChunk>();
    const resolvedOptions = resolveSendOptions(options);

    const sent = this.send({
      ...resolvedOptions,
      onChunk: (chunk) => channel.send(chunk),
    });
    sent.then(
      () => channel.close(),
      (err) => channel.error(err)
    );

    return {
      response: sent,
      stream: channel,
    };
  }

  get messages(): MessageData[] {
    return this._messages ?? [];
  }

  private async updateMessages(messages: MessageData[]): Promise<void> {
    this._messages = messages;
    await this.session.updateMessages(this.threadName, messages);
  }
}

function hasPreamble(msgs?: MessageData[]) {
  return !!msgs?.find((m) => m.metadata?.preamble);
}

function getPreamble(msgs?: MessageData[]) {
  return msgs?.filter((m) => m.metadata?.preamble);
}

function stripPreamble(msgs?: MessageData[]) {
  return msgs?.filter((m) => !m.metadata?.preamble);
}

function resolveSendOptions<
  O extends z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny,
>(
  options: string | Part[] | ChatGenerateOptions<O, CustomOptions>
): ChatGenerateOptions<O, CustomOptions> {
  let resolvedOptions: ChatGenerateOptions<O, CustomOptions>;

  // string
  if (typeof options === 'string') {
    resolvedOptions = {
      prompt: options,
    } as ChatGenerateOptions<O, CustomOptions>;
  } else if (Array.isArray(options)) {
    // Part[]
    resolvedOptions = {
      prompt: options,
    } as ChatGenerateOptions<O, CustomOptions>;
  } else {
    resolvedOptions = options as ChatGenerateOptions<O, CustomOptions>;
  }
  return resolvedOptions;
}
