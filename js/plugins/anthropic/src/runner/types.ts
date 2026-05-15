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

/**
 * Type contract that each Anthropic runner passes into the generic `BaseRunner`.
 *
 * The concrete runners (stable vs. beta SDKs) bind these slots to their SDK’s
 * concrete interfaces so the shared logic in `BaseRunner` can stay strongly typed
 * without knowing which SDK variant it is talking to.
 *
 * Properties are `unknown` by default, so every subclass must plug in the
 * correct Anthropic types to keep the generic plumbing sound.
 */
type RunnerTypes = {
  Message: unknown;
  Stream: AsyncIterable<unknown> & { finalMessage(): Promise<unknown> };
  StreamEvent: unknown;
  RequestBody: unknown;
  StreamingRequestBody: unknown;
  Tool: unknown;
  MessageParam: unknown;
  ContentBlockParam: unknown;
  ToolResponseContent: unknown;
};

type RunnerMessage<ApiTypes extends RunnerTypes> = ApiTypes['Message'];

/** Streaming handle that yields Anthropic events and exposes the final message. */
type RunnerStream<ApiTypes extends RunnerTypes> = ApiTypes['Stream'];

/** Discrete event emitted by the Anthropic stream (delta, block start, etc.). */
type RunnerStreamEvent<ApiTypes extends RunnerTypes> = ApiTypes['StreamEvent'];

/** Non-streaming request payload shape for create-message calls. */
type RunnerRequestBody<ApiTypes extends RunnerTypes> = ApiTypes['RequestBody'];
type RunnerStreamingRequestBody<ApiTypes extends RunnerTypes> =
  ApiTypes['StreamingRequestBody'];

/** Tool definition compatible with the target Anthropic SDK. */
type RunnerTool<ApiTypes extends RunnerTypes> = ApiTypes['Tool'];

/** Anthropic message param shape used when sending history to the API. */
type RunnerMessageParam<ApiTypes extends RunnerTypes> =
  ApiTypes['MessageParam'];

/** Content block that the runner sends to Anthropic for a single part. */
type RunnerContentBlockParam<ApiTypes extends RunnerTypes> =
  ApiTypes['ContentBlockParam'];

/** Tool response block that Anthropic expects when returning tool output. */
type RunnerToolResponseContent<ApiTypes extends RunnerTypes> =
  ApiTypes['ToolResponseContent'];

export {
  RunnerContentBlockParam,
  RunnerMessage,
  RunnerMessageParam,
  RunnerRequestBody,
  RunnerStream,
  RunnerStreamEvent,
  RunnerStreamingRequestBody,
  RunnerTool,
  RunnerToolResponseContent,
  RunnerTypes,
};
