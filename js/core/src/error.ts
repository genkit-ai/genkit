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

import type { Registry } from './registry.js';
import { httpStatusCode, type StatusName } from './statusTypes.js';

export type { StatusName };

export interface HttpErrorWireFormat {
  details?: unknown;
  message: string;
  status: StatusName;
}

/**
 * Metadata from the HTTP response that triggered this error.
 * This is not serialized into the wire format — it's only available
 * in-process for middleware (e.g. retry) to make informed decisions.
 */
export interface ErrorResponseMetadata {
  /** Provider-suggested retry delay in milliseconds (parsed from Retry-After header). */
  retryAfterMs?: number;
  /** Raw HTTP response headers, if available. */
  headers?: Record<string, string>;
}

/**
 * Base error class for Genkit errors.
 */
export class GenkitError extends Error {
  source?: string;
  status: StatusName;
  detail?: any;
  code: number;

  /**
   * Metadata from the HTTP response that triggered this error.
   * Not serialized into the wire format — only available in-process
   * for middleware (e.g. retry) to make informed decisions.
   */
  responseMetadata?: ErrorResponseMetadata;

  // For easy printing, we wrap the error with information like the source
  // and status, but that's redundant with JSON.
  originalMessage: string;

  constructor({
    status,
    message,
    detail,
    source,
    responseMetadata,
  }: {
    status: StatusName;
    message: string;
    detail?: any;
    source?: string;
    responseMetadata?: ErrorResponseMetadata;
  }) {
    super(`${source ? `${source}: ` : ''}${status}: ${message}`);
    this.originalMessage = message;
    this.code = httpStatusCode(status);
    this.status = status;
    this.detail = detail;
    this.responseMetadata = responseMetadata;
    this.name = 'GenkitError';
  }

  /**
   * Returns a JSON-serializable representation of this object.
   */
  public toJSON(): HttpErrorWireFormat {
    return {
      // This error type is used by 3P authors with the field "detail",
      // but the actual Callable protocol value is "details"
      ...(this.detail === undefined ? {} : { details: this.detail }),
      status: this.status,
      message: this.originalMessage,
    };
  }
}

export class UnstableApiError extends GenkitError {
  constructor(level: 'beta', message?: string) {
    super({
      status: 'FAILED_PRECONDITION',
      message: `${message ? message + ' ' : ''}This API requires '${level}' stability level.\n\nTo use this feature, initialize Genkit using \`import {genkit} from "genkit/${level}"\`.`,
    });
    this.name = 'UnstableApiError';
  }
}

/**
 * assertUnstable allows features to raise exceptions when using Genkit from *more* stable initialized instances.
 *
 * @param level The maximum stability channel allowed.
 * @param message An optional message describing which feature is not allowed.
 */
export function assertUnstable(
  registry: Registry,
  level: 'beta',
  message?: string
) {
  if (level === 'beta' && registry.apiStability === 'stable') {
    throw new UnstableApiError(level, message);
  }
}

/**
 * Creates a new class of Error for issues to be returned to users.
 * Using this error allows a web framework handler (e.g. express, next) to know it
 * is safe to return the message in a request. Other kinds of errors will
 * result in a generic 500 message to avoid the possibility of internal
 * exceptions being leaked to attackers.
 * In JSON requests, code will be an HTTP code and error will be a response body.
 * In streaming requests, { code, message } will be passed as the error message.
 */
export class UserFacingError extends GenkitError {
  constructor(status: StatusName, message: string, details?: any) {
    super({ status, detail: details, message });
    super.name = 'UserFacingError';
  }
}

export function getHttpStatus(e: any): number {
  if (e instanceof GenkitError) {
    return e.code;
  }
  return 500;
}

export function getCallableJSON(e: any): HttpErrorWireFormat {
  if (e instanceof GenkitError) {
    return e.toJSON();
  }
  return {
    message: 'Internal Error',
    status: 'INTERNAL',
  };
}

/**
 * Extracts error message from the given error object, or if input is not an error then just turn the error into a string.
 */
export function getErrorMessage(e: any): string {
  if (e instanceof Error) {
    return e.message;
  }
  return `${e}`;
}

/**
 * Extracts stack trace from the given error object, or if input is not an error then returns undefined.
 */
export function getErrorStack(e: any): string | undefined {
  if (e instanceof Error) {
    return e.stack;
  }
  return undefined;
}
