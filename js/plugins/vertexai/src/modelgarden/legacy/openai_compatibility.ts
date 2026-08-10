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

import { Message, z, type Genkit, type StreamingCallback } from 'genkit';
import {
  GenerationCommonConfigSchema,
  type CandidateData,
  type GenerateRequest,
  type GenerateResponseChunkData,
  type GenerateResponseData,
  type MessageData,
  type ModelAction,
  type ModelReference,
  type Part,
  type Role,
  type ToolDefinition,
  type ToolRequestPart,
} from 'genkit/model';
import type OpenAI from 'openai';
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionContentPart,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionRole,
  ChatCompletionTool,
  CompletionChoice,
} from 'openai/resources/index.mjs';

/**
 * See https://platform.openai.com/docs/api-reference/chat/create.
 * @deprecated
 */
export const OpenAIConfigSchema = GenerationCommonConfigSchema.extend({
  // TODO: topK is not supported and some of the other common config options
  // have different names in the above doc. Eg: max_completion_tokens.
  // Update to use the parameters in above doc.
  frequencyPenalty: z
    .number()
    .min(-2)
    .max(2)
    .describe(
      'Positive values penalize new tokens based on their ' +
        "existing frequency in the text so far, decreasing the model's " +
        'likelihood to repeat the same line verbatim.'
    )
    .optional(),
  logitBias: z
    .record(
      z.string().describe('Token string.'),
      z.number().min(-100).max(100).describe('Associated bias value.')
    )
    .describe(
      'Controls the likelihood of specified tokens appearing ' +
        'in the generated output. Map of tokens to an associated bias ' +
        'value from -100 (which will in most cases block that token ' +
        'from being generated) to 100 (exclusive selection of the ' +
        'token which makes it more likely to be generated). Moderate ' +
        'values like -1 and 1 will change the probability of a token ' +
        'being selected to a lesser degree.'
    )
    .optional(),
  logProbs: z
    .boolean()
    .describe(
      'Whether to return log probabilities of the output tokens or not.'
    )
    .optional(),
  presencePenalty: z
    .number()
    .min(-2)
    .max(2)
    .describe(
      'Positive values penalize new tokens based on whether ' +
        "they appear in the text so far, increasing the model's " +
        'likelihood to talk about new topics.'
    )
    .optional(),
  seed: z
    .number()
    .int()
    .describe(
      'If specified, the system will make a best effort to sample ' +
        'deterministically, such that repeated requests with the same seed ' +
        'and parameters should return the same result. Determinism is not ' +
        'guaranteed, and you should refer to the system_fingerprint response ' +
        'parameter to monitor changes in the backend.'
    )
    .optional(),
  topLogProbs: z
    .number()
    .int()
    .min(0)
    .max(20)
    .describe(
      'An integer specifying the number of most likely tokens to ' +
        'return at each token position, each with an associated log ' +
        'probability. logprobs must be set to true if this parameter is used.'
    )
    .optional(),
  user: z
    .string()
    .describe(
      'A unique identifier representing your end-user to monitor and detect abuse.'
    )
    .optional(),
});

/** @deprecated */
export function toOpenAIRole(role: Role): ChatCompletionRole {
  switch (role) {
    case 'user':
      return 'user';
    case 'model':
      return 'assistant';
    case 'system':
      return 'system';
    case 'tool':
      return 'tool';
    default:
      throw new Error(`role ${role} doesn't map to an OpenAI role.`);
  }
}

function toOpenAiTool(tool: ToolDefinition): ChatCompletionTool {
  return {
    type: 'function',
    function: {
      name: tool.name,
      parameters: tool.inputSchema || undefined,
    },
  };
}

/** @deprecated */
export function toOpenAiTextAndMedia(part: Part): ChatCompletionContentPart {
  if (part.text) {
    return {
      type: 'text',
      text: part.text,
    };
  } else if (part.media) {
    return {
      type: 'image_url',
      image_url: {
        url: part.media.url,
      },
    };
  }
  throw Error(
    `Unsupported genkit part fields encountered for current message role: ${JSON.stringify(part)}.`
  );
}

/** @deprecated */
export function toOpenAiMessages(
  messages: MessageData[]
): ChatCompletionMessageParam[] {
  const openAiMsgs: ChatCompletionMessageParam[] = [];
  for (const message of messages) {
    const msg = new Message(message);
    const role = toOpenAIRole(message.role);
    switch (role) {
      case 'user':
        openAiMsgs.push({
          role: role,
          content: msg.content.map((part) => toOpenAiTextAndMedia(part)),
        });
        break;
      case 'system':
        openAiMsgs.push({
          role: role,
          content: msg.text,
        });
        break;
      case 'assistant': {
        const toolCalls: ChatCompletionMessageToolCall[] = msg.content
          .filter(
            (
              part
            ): part is Part & {
              toolRequest: NonNullable<Part['toolRequest']>;
            } => Boolean(part.toolRequest)
          )
          .map((part) => ({
            id: part.toolRequest.ref ?? '',
            type: 'function',
            function: {
              name: part.toolRequest.name,
              arguments: JSON.stringify(part.toolRequest.input),
            },
          }));
        if (toolCalls.length > 0) {
          openAiMsgs.push({
            role: role,
            tool_calls: toolCalls,
          });
        } else {
          openAiMsgs.push({
            role: role,
            content: msg.text,
          });
        }
        break;
      }
      case 'tool': {
        const toolResponseParts = msg.toolResponseParts();
        toolResponseParts.map((part) => {
          openAiMsgs.push({
            role: role,
            tool_call_id: part.toolResponse.ref ?? '',
            content:
              typeof part.toolResponse.output === 'string'
                ? part.toolResponse.output
                : JSON.stringify(part.toolResponse.output),
          });
        });
        break;
      }
    }
  }
  return openAiMsgs;
}

const finishReasonMap: Record<
  CompletionChoice['finish_reason'] | 'tool_calls',
  CandidateData['finishReason']
> = {
  length: 'length',
  stop: 'stop',
  tool_calls: 'stop',
  content_filter: 'blocked',
};

/** @deprecated */
export function fromOpenAiToolCall(
  toolCall:
    | ChatCompletionMessageToolCall
    | ChatCompletionChunk.Choice.Delta.ToolCall
): ToolRequestPart {
  if (!toolCall.function) {
    throw Error(
      `Unexpected openAI chunk choice. tool_calls was provided but one or more tool_calls is missing.`
    );
  }
  const f = toolCall.function;
  return {
    toolRequest: {
      name: f.name!,
      ref: toolCall.id,
      input: f.arguments ? JSON.parse(f.arguments) : f.arguments,
    },
  };
}

/** @deprecated */
export function fromOpenAiChoice(
  choice: ChatCompletion.Choice,
  jsonMode = false
): CandidateData {
  const toolRequestParts = choice.message.tool_calls?.map(fromOpenAiToolCall);
  return {
    index: choice.index,
    finishReason: finishReasonMap[choice.finish_reason] || 'other',
    message: {
      role: 'model',
      content: toolRequestParts
        ? // Note: Not sure why I have to cast here exactly.
          // Otherwise it thinks toolRequest must be 'undefined' if provided
          (toolRequestParts as ToolRequestPart[])
        : [
            jsonMode
              ? { data: JSON.parse(choice.message.content!) }
              : { text: choice.message.content! },
          ],
    },
    custom: {},
  };
}

/** @deprecated */
export function fromOpenAiChunkChoice(
  choice: ChatCompletionChunk.Choice,
  jsonMode = false
): CandidateData {
  const toolRequestParts = choice.delta.tool_calls?.map(fromOpenAiToolCall);
  return {
    index: choice.index,
    finishReason: choice.finish_reason
      ? finishReasonMap[choice.finish_reason] || 'other'
      : 'unknown',
    message: {
      role: 'model',
      content: toolRequestParts
        ? (toolRequestParts as ToolRequestPart[])
        : [
            jsonMode
              ? { data: JSON.parse(choice.delta.content!) }
              : { text: choice.delta.content! },
          ],
    },
    custom: {},
  };
}

/** @deprecated */
export function toRequestBody(
  model: ModelReference<typeof OpenAIConfigSchema>,
  request: GenerateRequest<typeof OpenAIConfigSchema>
) {
  const openAiMessages = toOpenAiMessages(request.messages);
  const mappedModelName =
    request.config?.version || model.version || model.name;
  const body = {
    model: mappedModelName,
    messages: openAiMessages,
    temperature: request.config?.temperature,
    max_tokens: request.config?.maxOutputTokens,
    top_p: request.config?.topP,
    stop: request.config?.stopSequences,
    frequency_penalty: request.config?.frequencyPenalty,
    logit_bias: request.config?.logitBias,
    logprobs: request.config?.logProbs,
    presence_penalty: request.config?.presencePenalty,
    seed: request.config?.seed,
    top_logprobs: request.config?.topLogProbs,
    user: request.config?.user,
    tools: request.tools?.map(toOpenAiTool),
    n: request.candidates,
  } as ChatCompletionCreateParamsNonStreaming;
  const response_format = request.output?.format;
  if (response_format) {
    if (
      response_format === 'json' &&
      model.info?.supports?.output?.includes('json')
    ) {
      body.response_format = {
        type: 'json_object',
      };
    } else if (
      response_format === 'text' &&
      model.info?.supports?.output?.includes('text')
    ) {
      // this is default format, don't need to set it
      // body.response_format = {
      //   type: 'text',
      // };
    } else {
      throw new Error(`${response_format} format is not supported currently`);
    }
  }
  for (const key in body) {
    if (!body[key] || (Array.isArray(body[key]) && !body[key].length))
      delete body[key];
  }
  return body;
}

/** @deprecated */
export function openaiCompatibleModel<C extends typeof OpenAIConfigSchema>(
  ai: Genkit,
  model: ModelReference<any>,
  clientFactory: (request: GenerateRequest<C>) => Promise<OpenAI>
): ModelAction<C> {
  const modelId = model.name;
  if (!model) throw new Error(`Unsupported model: ${name}`);

  return ai.defineModel(
    {
      name: modelId,
      ...model.info,
      configSchema: model.configSchema,
    },
    async (
      request: GenerateRequest<C>,
      sendChunk?: StreamingCallback<GenerateResponseChunkData>
    ): Promise<GenerateResponseData> => {
      let response: ChatCompletion;
      const client = await clientFactory(request);
      const body = toRequestBody(model, request);
      if (sendChunk) {
        const stream = client.beta.chat.completions.stream({
          ...body,
          stream: true,
        });
        for await (const chunk of stream) {
          chunk.choices?.forEach((chunk) => {
            const c = fromOpenAiChunkChoice(chunk);
            sendChunk({
              index: c.index,
              content: c.message.content,
            });
          });
        }
        response = await stream.finalChatCompletion();
      } else {
        response = await client.chat.completions.create(body);
      }
      return {
        candidates: response.choices.map((c) =>
          fromOpenAiChoice(c, request.output?.format === 'json')
        ),
        usage: {
          inputTokens: response.usage?.prompt_tokens,
          outputTokens: response.usage?.completion_tokens,
          totalTokens: response.usage?.total_tokens,
        },
        custom: response,
      };
    }
  );
}
