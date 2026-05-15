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

import type { Action, z } from 'genkit';
import {
  runFlow as baseRunFlow,
  streamFlow as baseStreamFlow,
} from 'genkit/beta/client';

type Input<A extends Action> =
  A extends Action<infer I extends z.ZodTypeAny, any, any> ? z.infer<I> : never;
type Output<A extends Action> =
  A extends Action<any, infer O extends z.ZodTypeAny, any> ? z.infer<O> : never;
type Stream<A extends Action> =
  A extends Action<any, any, infer S extends z.ZodTypeAny> ? z.infer<S> : never;

export interface RequestData<T> {
  url: string;
  headers?: Record<string, string>;
  input?: T;
  streamId?: string;
  abortSignal?: AbortSignal;
}

export function runFlow<A extends Action = Action>(
  req: RequestData<Input<A>>
): Promise<Output<A>> {
  return baseRunFlow<Output<A>>(req);
}

export interface StreamResponse<A extends Action> {
  output: Promise<Output<A>>;
  stream: AsyncIterable<Stream<A>>;
  streamId: Promise<string | null>;
}

export function streamFlow<A extends Action = Action>(
  req: RequestData<Input<A>>
): StreamResponse<A> {
  const res = baseStreamFlow<Output<A>, Stream<A>>(req);
  return {
    output: res.output,
    stream: res.stream,
    streamId: res.streamId,
  };
}
