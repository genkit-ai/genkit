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

import { runInActionRuntimeContext } from './action.js';
import { getAsyncContext } from './async-context.js';
import { UserFacingError } from './error.js';

const contextAlsKey = 'core.auth.context';

/**
 * Action side channel data, like auth and other invocation context infromation provided by the invoker.
 */
export interface ActionContext {
  /** Information about the currently authenticated user if provided. */
  auth?: Record<string, any>;
  [additionalContext: string]: any;
}

/**
 * Execute the provided function in the runtime context. Call {@link getFlowContext()} anywhere
 * within the async call stack to retrieve the context. If context object is undefined, this function
 * is a no op passthrough, the function will be invoked as is.
 */
export function runWithContext<R>(
  context: ActionContext | undefined,
  fn: () => R
): R {
  if (context === undefined) {
    return fn();
  }
  return getAsyncContext().run(contextAlsKey, context, () =>
    runInActionRuntimeContext(fn)
  );
}

/**
 * Gets the runtime context of the current flow.
 */
export function getContext(): ActionContext | undefined {
  return getAsyncContext().getStore<ActionContext>(contextAlsKey);
}

/**
 * A universal type that request handling extensions (e.g. express, next) can map their request to.
 * This allows ContextProviders to build consistent interfacese on any web framework.
 * Headers must be lowercase to ensure portability.
 */
export interface RequestData<T = any> {
  method: 'GET' | 'PUT' | 'POST' | 'DELETE' | 'OPTIONS' | 'QUERY';
  headers: Record<string, string>;
  input: T;
}

/**
 * Middleware can read request data and add information to the context that will
 * be passed to the Action. If middleware throws an error, that error will fail
 * the request and the Action will not be invoked. Expected cases should return a
 * UserFacingError, which allows the request handler to know what data is safe to
 * return to end users.
 *
 * Middleware can provide validation in addition to parsing. For example, an auth
 * middleware can have policies for validating auth in addition to passing auth context
 * to the Action.
 */
export type ContextProvider<
  C extends ActionContext = ActionContext,
  T = any,
> = (request: RequestData<T>) => C | Promise<C>;

export interface ApiKeyContext extends ActionContext {
  auth: {
    apiKey: string | undefined;
  };
}

export function apiKey(
  policy: (context: ApiKeyContext) => void | Promise<void>
): ContextProvider<ApiKeyContext>;
export function apiKey(value?: string): ContextProvider<ApiKeyContext>;
export function apiKey(
  valueOrPolicy?: ((context: ApiKeyContext) => void | Promise<void>) | string
): ContextProvider<ApiKeyContext> {
  return async (request: RequestData): Promise<ApiKeyContext> => {
    const context: ApiKeyContext = {
      auth: { apiKey: request.headers['authorization'] },
    };
    if (typeof valueOrPolicy === 'string') {
      if (!context.auth?.apiKey) {
        console.error('THROWING UNAUTHENTICATED');
        throw new UserFacingError('UNAUTHENTICATED', 'Unauthenticated');
      }
      if (context.auth?.apiKey != valueOrPolicy) {
        console.error('Throwing PERMISSION_DENIED');
        throw new UserFacingError('PERMISSION_DENIED', 'Permission Denied');
      }
    } else if (typeof valueOrPolicy === 'function') {
      await valueOrPolicy(context);
    } else if (typeof valueOrPolicy !== 'undefined') {
      throw new Error(
        `Invalid type ${typeof valueOrPolicy} passed to apiKey()`
      );
    }
    return context;
  };
}
