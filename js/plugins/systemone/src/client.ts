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

import { GenkitError, type StatusName } from 'genkit';
import { STATUS_CODES } from 'node:http';
import type { WireQuestion } from './protocol.js';
import { stringifyBody, type Media } from './state.js';

/**
 * Builds one model's request when the server's wire is not the native one. It
 * gets the model ID and the native body, which it may change, and returns a
 * path to append to the request path, which is joined with a slash when it
 * does not start with one, and the body to send.
 */
export type Route = (
  model: string,
  body: Record<string, unknown>
) => { path?: string; body: unknown };

/**
 * Extracts the native response from a server's envelope: it gets the response
 * body, parsed as JSON, and returns the native response. An error it throws is
 * reported under the plugin's name, with its status kept when it is a
 * `GenkitError` and as UNKNOWN otherwise: the server answered with a success
 * code, so neither the request nor the server is known to be at fault.
 */
export type Unwrap = (body: unknown) => unknown;

/**
 * How a server takes System One requests: the paths, and the hooks for a
 * server whose wire is not the native one.
 */
export interface Endpoint {
  /** Names the endpoint in errors. */
  name: string;
  /** The request path under the base URL. */
  path: string;
  /** The path of the model listing under the base URL, empty for none. */
  modelsPath: string;
  route?: Route;
  unwrap?: Unwrap;
}

/** The native request, before the endpoint's envelope. */
export interface DecideRequest {
  state: unknown;
  /**
   * Sent in the images, audio, and videos fields, which the servers place
   * before the state. Each field is sent only when it has media, since a
   * server that does not read a kind can refuse its field.
   */
  media: Media;
  questions: Record<string, WireQuestion>;
  /**
   * Merged into the top-level fields. It cannot replace a field the request
   * builds itself.
   */
  extra?: Record<string, unknown>;
}

/**
 * The top-level fields a request builds itself, which an extra field cannot
 * replace: a different model or questions would detach the request from the
 * model that was called and the questions that were compiled from the output
 * schema.
 */
export const RESERVED_FIELDS = [
  'model',
  'state',
  'questions',
  'images',
  'audio',
  'videos',
];

/** The native JSON body of a request. */
export function nativeBody(
  model: string,
  req: DecideRequest
): Record<string, unknown> {
  const body: Record<string, unknown> = { ...req.extra };
  for (const field of RESERVED_FIELDS) delete body[field];
  body.model = model;
  body.state = req.state;
  body.questions = req.questions;
  if (req.media.images.length > 0) body.images = req.media.images;
  if (req.media.audio.length > 0) body.audio = req.media.audio;
  if (req.media.videos.length > 0) body.videos = req.media.videos;
  return body;
}

/**
 * The native response. The provider, the ID, and the cost are gateway
 * additions, absent from a model vendor's own responses.
 */
export interface DecideResponse {
  model: string;
  provider?: string;
  id?: string;
  answers: Record<string, Record<string, unknown>>;
  usage?: { input_tokens?: number; output_tokens?: number; cost?: number };
}

/**
 * One entry of a model listing. A listing names a model by its name, or, as
 * OpenRouter's does, by its ID with a display name beside it.
 */
export interface ListedModel {
  id?: string;
  name?: string;
  description?: string;
  release_date?: string;
}

/**
 * Retry policy, as Genkit Go's System One client has it: two retries after
 * the first attempt, half a second doubling to five, a quarter of jitter, and
 * the server's Retry-After honored when it sends one, up to 30 seconds.
 */
const MAX_RETRIES = 2;
const RETRY_BASE_MS = 500;
const RETRY_CAP_MS = 5_000;
const RETRY_AFTER_CAP_MS = 30_000;
const MAX_BODY_BYTES = 8 << 20;

/** Bounds one attempt when the plugin supplies the transport. */
export const REQUEST_TIMEOUT_MS = 30_000;

/** Posts questions to one endpoint. */
export class Client {
  constructor(
    readonly options: {
      fetch: typeof fetch;
      /** The origin the endpoint's paths are joined to. */
      baseUrl: string;
      /**
       * Sent as a bearer token. A local server, such as Ollama, takes none,
       * and no authorization header is sent without one.
       */
      apiKey?: string;
      /** Sent on every request, after the authorization header. */
      headers?: Record<string, string>;
      endpoint: Endpoint;
      /** Bounds one attempt; unset leaves it unbounded. */
      timeoutMs?: number;
      /** Waits between attempts; replaced in tests. */
      sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
    }
  ) {}

  get endpoint(): Endpoint {
    return this.options.endpoint;
  }

  /**
   * Posts one request and decodes the answers. `apiKey` replaces the client's
   * key for this request.
   */
  async decide(
    model: string,
    req: DecideRequest,
    options: { signal?: AbortSignal; apiKey?: string } = {}
  ): Promise<DecideResponse> {
    const { endpoint, baseUrl } = this.options;
    const { signal } = options;
    let suffix = '';
    let body: unknown = nativeBody(model, req);
    if (endpoint.route) {
      const routed = hook(endpoint.name, 'route', () =>
        endpoint.route!(model, body as Record<string, unknown>)
      );
      if (
        !routed ||
        typeof routed !== 'object' ||
        isThenable(routed) ||
        routed.body === undefined ||
        routed.body === null
      ) {
        throw new GenkitError({
          status: 'INVALID_ARGUMENT',
          message: `${endpoint.name}: route must return {path, body} with a body, not a promise`,
        });
      }
      suffix = routed.path ?? '';
      body = routed.body;
      if (suffix !== '' && !suffix.startsWith('/')) suffix = `/${suffix}`;
    }
    let payload: string;
    try {
      payload = stringifyBody(body);
    } catch (e) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `${endpoint.name}: request is not JSON: ${(e as Error)?.message ?? e}`,
      });
    }
    const text = await this.send(
      'POST',
      baseUrl + endpoint.path + suffix,
      payload,
      signal,
      options.apiKey ?? this.options.apiKey
    );
    let raw: unknown;
    try {
      raw = JSON.parse(text);
    } catch (e) {
      throw new GenkitError({
        status: 'INTERNAL',
        message: `${endpoint.name}: response is not a System One answer: ${(e as Error).message}`,
      });
    }
    if (endpoint.unwrap) {
      const parsed = raw;
      raw = hook(endpoint.name, 'unwrap', () => endpoint.unwrap!(parsed));
      if (isThenable(raw)) {
        throw new GenkitError({
          status: 'UNKNOWN',
          message: `${endpoint.name}: unwrap must return the response, not a promise`,
        });
      }
    }
    const resp = raw as DecideResponse;
    if (!resp || typeof resp !== 'object' || Array.isArray(resp)) {
      throw new GenkitError({
        status: 'INTERNAL',
        message: `${endpoint.name}: response is not a System One answer`,
      });
    }
    if (
      !resp.answers ||
      typeof resp.answers !== 'object' ||
      Array.isArray(resp.answers)
    ) {
      throw new GenkitError({
        status: 'INTERNAL',
        message: `${endpoint.name}: response carries no answers`,
      });
    }
    return resp;
  }

  /**
   * Fetches the endpoint's model listing. It is sent once, with no retries:
   * the listing serves the Dev UI, and the caller falls back to the known
   * models when it fails. The listing's envelope is not pinned down by the
   * API reference, so a bare array and the usual wrapping keys are all
   * accepted.
   */
  async listModels(signal?: AbortSignal): Promise<ListedModel[]> {
    const { endpoint, baseUrl } = this.options;
    if (!endpoint.modelsPath) {
      throw new GenkitError({
        status: 'UNIMPLEMENTED',
        message: `${endpoint.name}: no model listing`,
      });
    }
    let text: string;
    try {
      ({ text } = await this.once(
        'GET',
        baseUrl + endpoint.modelsPath,
        undefined,
        signal,
        this.options.apiKey
      ));
    } catch (e) {
      throw e instanceof AttemptFailure ? e.error : e;
    }
    let raw: any;
    try {
      raw = JSON.parse(text);
    } catch (e) {
      throw new GenkitError({
        status: 'INTERNAL',
        message: `${endpoint.name}: model listing is not JSON: ${(e as Error).message}`,
      });
    }
    if (Array.isArray(raw)) return raw;
    for (const key of ['models', 'data']) {
      if (Array.isArray(raw?.[key])) return raw[key];
    }
    throw new GenkitError({
      status: 'INTERNAL',
      message: `${endpoint.name}: model listing has no models array`,
    });
  }

  /**
   * Sends one request with retries and returns the response body. A request
   * that fails to connect, times out, is rate limited, or hits a server error
   * other than 501 and 505 is retried; anything else is the caller's problem.
   */
  private async send(
    method: string,
    url: string,
    payload: string,
    signal: AbortSignal | undefined,
    apiKey: string | undefined
  ): Promise<string> {
    const sleep = this.options.sleep ?? abortableSleep;
    for (let attempt = 0; ; attempt++) {
      let failure: AttemptFailure;
      try {
        return (await this.once(method, url, payload, signal, apiKey)).text;
      } catch (e) {
        if (!(e instanceof AttemptFailure)) throw e;
        failure = e;
      }
      if (signal?.aborted) throw stopped(this.endpoint.name, signal, failure);
      if (attempt >= MAX_RETRIES || !failure.retry) throw failure.error;
      const delay =
        failure.retryAfterMs >= 0
          ? Math.min(failure.retryAfterMs, RETRY_AFTER_CAP_MS)
          : backoff(attempt);
      try {
        await sleep(delay, signal);
      } catch {
        throw stopped(this.endpoint.name, signal, failure);
      }
    }
  }

  /**
   * Sends one attempt. A failure is an {@link AttemptFailure} that says
   * whether it is worth a retry, which is a failure to reach the endpoint, a
   * timeout, a rate limit, or a server error, and the Retry-After the server
   * asked for.
   */
  private async once(
    method: string,
    url: string,
    payload: string | undefined,
    signal: AbortSignal | undefined,
    apiKey: string | undefined
  ): Promise<{ text: string }> {
    const { endpoint, headers, timeoutMs } = this.options;
    let requestHeaders: Headers;
    try {
      requestHeaders = buildHeaders(headers, apiKey, payload !== undefined);
    } catch (e) {
      // The plugin's own headers and key are checked when it is created, so
      // this is a request's key: the caller's to fix, and not worth a retry.
      throw new AttemptFailure(
        new GenkitError({
          status: 'INVALID_ARGUMENT',
          message: `${endpoint.name}: ${e instanceof HeaderError ? e.message : 'a header cannot be sent'}`,
        }),
        false,
        -1
      );
    }
    const attempt = anySignal(
      [signal, timeoutMs ? AbortSignal.timeout(timeoutMs) : undefined].filter(
        (s): s is AbortSignal => !!s
      )
    );
    let resp: Response;
    let text: string;
    try {
      resp = await this.options.fetch(url, {
        method,
        headers: requestHeaders,
        body: payload,
        signal: attempt.signal,
        // A redirect is refused rather than followed, so the key's handling
        // never depends on the transport's redirect rules.
        redirect: 'manual',
      });
      text = await readBody(resp);
    } catch (e) {
      if (signal?.aborted) throw stopped(endpoint.name, signal, e);
      if (e instanceof BodyTooLarge) {
        // Not worth a retry: the server would send it again.
        throw new AttemptFailure(
          new GenkitError({
            status: 'INTERNAL',
            message: `${endpoint.name}: ${e.message}`,
          }),
          false,
          -1
        );
      }
      throw new AttemptFailure(
        new GenkitError({
          status: 'UNAVAILABLE',
          message: `${endpoint.name}: ${(e as Error)?.message ?? String(e)}`,
        }),
        true,
        -1
      );
    } finally {
      attempt.release();
    }
    if (resp.ok) return { text };
    const code = resp.status;
    if (code >= 300 && code < 400) {
      throw new AttemptFailure(
        new GenkitError({
          status: 'FAILED_PRECONDITION',
          message: `${endpoint.name}: HTTP ${code}: the server redirects, and redirects are not followed; set the base URL to the final one`,
        }),
        false,
        -1
      );
    }
    const retry =
      code === 408 ||
      code === 429 ||
      (code >= 500 && code !== 501 && code !== 505);
    throw new AttemptFailure(
      httpError(endpoint.name, code, text),
      retry,
      parseRetryAfter(resp.headers.get('Retry-After'))
    );
  }
}

/**
 * A header HTTP cannot carry. Its message names the header only: the value,
 * which can be a key, is never quoted, nor the platform's error, which can
 * quote it.
 */
export class HeaderError extends Error {
  constructor(readonly header: string | undefined) {
    super(
      `${header ? `the ${header} header` : 'a header'} cannot be sent: its name or value is not valid HTTP`
    );
  }
}

/**
 * The headers of one request: a Headers object, so a header given in another
 * case is replaced rather than sent twice. It throws a {@link HeaderError}
 * for a name or a value HTTP cannot carry, such as a key with a newline.
 */
export function buildHeaders(
  headers: Record<string, string> | undefined,
  apiKey: string | undefined,
  hasBody: boolean
): Headers {
  const out = new Headers();
  const set = (name: string, value: string) => {
    try {
      // Headers accepts some control bytes, such as a vertical tab or DEL,
      // that the transport then refuses on every attempt, so a field value
      // is checked here too: no control byte but the horizontal tab.
      if (/[\x00-\x08\x0a-\x1f\x7f]/.test(value)) throw new Error();
      out.set(name, value);
    } catch {
      // A name that is not a valid token is not quoted either.
      throw new HeaderError(
        /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/.test(name) ? name : undefined
      );
    }
  };
  for (const [name, value] of Object.entries(headers ?? {})) set(name, value);
  if (apiKey) set('Authorization', `Bearer ${apiKey}`);
  set('Accept', 'application/json');
  if (hasBody) set('Content-Type', 'application/json');
  return out;
}

/** One failed attempt, and whether and when to retry it. */
class AttemptFailure extends Error {
  constructor(
    readonly error: GenkitError,
    readonly retry: boolean,
    /** Negative when the server sent no Retry-After; zero means at once. */
    readonly retryAfterMs: number
  ) {
    super(error.message);
  }
}

/**
 * The error for a request whose caller gave up: a cancellation, rather than
 * the retryable failure it interrupted, which is kept as text only, since its
 * status would make the call read as worth retrying.
 */
function stopped(
  name: string,
  signal: AbortSignal | undefined,
  last: unknown
): GenkitError {
  const lastMessage =
    last instanceof AttemptFailure
      ? last.error.originalMessage
      : ((last as Error)?.message ?? String(last));
  const reason = signal?.reason;
  const timedOut = reason instanceof Error && reason.name === 'TimeoutError';
  return new GenkitError({
    status: timedOut ? 'DEADLINE_EXCEEDED' : 'CANCELLED',
    message: `${name}: the request was ${timedOut ? 'timed out' : 'aborted'} by the caller (last attempt: ${lastMessage})`,
  });
}

/** A response body past {@link MAX_BODY_BYTES}. */
class BodyTooLarge extends Error {}

/**
 * A signal that aborts when any of the given ones does, with that one's
 * reason, as AbortSignal.any does, which Node has from 20.3 only. `release`
 * removes its listeners from the given signals once the attempt is over.
 */
function anySignal(signals: AbortSignal[]): {
  signal: AbortSignal | undefined;
  release: () => void;
} {
  if (signals.length === 0) return { signal: undefined, release: () => {} };
  const controller = new AbortController();
  const listeners: [AbortSignal, () => void][] = [];
  for (const signal of signals) {
    if (signal.aborted) {
      controller.abort(signal.reason);
      break;
    }
    const onAbort = () => controller.abort(signal.reason);
    signal.addEventListener('abort', onAbort, { once: true });
    listeners.push([signal, onAbort]);
  }
  return {
    signal: controller.signal,
    release: () => {
      for (const [signal, onAbort] of listeners) {
        signal.removeEventListener('abort', onAbort);
      }
    },
  };
}

function isThenable(value: unknown): boolean {
  return (
    !!value &&
    (typeof value === 'object' || typeof value === 'function') &&
    typeof (value as { then?: unknown }).then === 'function'
  );
}

/**
 * Runs a caller's hook. An error it throws is reported under the endpoint's
 * name, as UNKNOWN unless it is a GenkitError, whose status is kept, since
 * neither the request nor the server is known to be at fault. A GenkitError
 * from `route` goes out as it is, as Genkit Go returns a route's error.
 */
function hook<T>(name: string, which: 'route' | 'unwrap', run: () => T): T {
  try {
    return run();
  } catch (e) {
    if (e instanceof GenkitError) {
      if (which === 'route') throw e;
      throw new GenkitError({
        status: e.status,
        message: `${name}: ${e.originalMessage}`,
      });
    }
    throw new GenkitError({
      status: 'UNKNOWN',
      message: `${name}: ${(e as Error)?.message ?? String(e)}`,
    });
  }
}

async function readBody(resp: Response): Promise<string> {
  if (!resp.body) return '';
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > MAX_BODY_BYTES) {
      await reader.cancel();
      throw new BodyTooLarge(`response is larger than ${MAX_BODY_BYTES} bytes`);
    }
    chunks.push(value);
  }
  return Buffer.concat(chunks).toString('utf8');
}

function abortableSleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** The canonical HTTP mapping, as Genkit Go's status package has it. */
const HTTP_STATUS: Record<number, StatusName> = {
  400: 'INVALID_ARGUMENT',
  401: 'UNAUTHENTICATED',
  403: 'PERMISSION_DENIED',
  404: 'NOT_FOUND',
  409: 'ABORTED',
  429: 'RESOURCE_EXHAUSTED',
  499: 'CANCELLED',
  500: 'INTERNAL',
  501: 'UNIMPLEMENTED',
  503: 'UNAVAILABLE',
  504: 'DEADLINE_EXCEEDED',
};

/**
 * Maps an error response onto a Genkit status: the canonical HTTP mapping,
 * with the codes the endpoints give their own meaning handled first. The body
 * is searched for the message under the shapes the endpoints use.
 */
export function httpError(
  endpoint: string,
  code: number,
  body: string
): GenkitError {
  let status: StatusName;
  if (code === 402) {
    // Out of credit at a gateway: a quota, not a bad request.
    status = 'RESOURCE_EXHAUSTED';
  } else if (code === 408 || code === 504 || code === 524) {
    // 524 is Cloudflare's origin timeout.
    status = 'DEADLINE_EXCEEDED';
  } else if (code === 501 || code === 505) {
    // A server without the path or the protocol: a setup to fix, not a
    // failure to wait out.
    status = 'UNIMPLEMENTED';
  } else if (code > 500) {
    // Every other server-side failure reads as transient.
    status = 'UNAVAILABLE';
  } else {
    // A client error the mapping does not name, such as 422, is the
    // request's fault.
    status = HTTP_STATUS[code] ?? 'INVALID_ARGUMENT';
  }
  const message = errorMessage(body) || STATUS_CODES[code] || 'empty response';
  return new GenkitError({
    status,
    message: `${endpoint}: HTTP ${code}: ${message}`,
  });
}

/**
 * Reads the message out of an error body. The endpoints use four shapes:
 * `{error: {message}}` or `{error: "..."}`, a `{message}`, a Cloudflare
 * `{errors: [{code, message}]}` envelope, and a validation list, which
 * TypeSafe's API sends under `detail` as `{loc, msg}` records and OpenRouter
 * sends as the bare array of `{path, message}` records. A body in none of
 * these shapes is quoted as it came, up to 200 characters, and an empty one
 * gives "". Any message is cut at 500 characters.
 */
export function errorMessage(body: string): string {
  const message = bodyMessage(body);
  return message.length > MAX_ERROR_MESSAGE
    ? `${message.slice(0, MAX_ERROR_MESSAGE)}...`
    : message;
}

/** The longest server message an error quotes. */
const MAX_ERROR_MESSAGE = 500;

function bodyMessage(body: string): string {
  let parsed: any;
  try {
    parsed = JSON.parse(body);
  } catch {
    parsed = undefined;
  }
  const list = validationErrors(parsed);
  if (list) return list;
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    const { error, message, detail, errors } = parsed;
    if (typeof error?.message === 'string' && error.message) {
      return error.message;
    }
    if (typeof error === 'string' && error) return error;
    if (typeof message === 'string' && message) return message;
    const fromErrors = validationErrors(errors);
    if (fromErrors) return fromErrors;
    if (detail !== undefined && detail !== null) {
      const fromDetail = validationErrors(detail);
      if (fromDetail) return fromDetail;
      if (typeof detail === 'string' && detail) return detail;
      return JSON.stringify(detail);
    }
  }
  const text = body.trim();
  return text.length > 200 ? `${text.slice(0, 200)}...` : text;
}

/**
 * Joins a list of validation records into one line, each as its path and
 * message, or returns "" when the value is not such a list.
 */
function validationErrors(value: unknown): string {
  if (!Array.isArray(value) || value.length === 0) return '';
  const messages: string[] = [];
  for (const record of value) {
    if (!record || typeof record !== 'object') continue;
    let message: string = record.message || record.msg || '';
    if (typeof message !== 'string' || message === '') continue;
    const path = [
      ...(Array.isArray(record.path) ? record.path : []),
      ...(Array.isArray(record.loc) ? record.loc : []),
    ];
    if (path.length > 0) message = `${path.join('.')}: ${message}`;
    messages.push(message);
  }
  return messages.join('; ');
}

/** The wait before retry `attempt + 1`, with a quarter of jitter. */
export function backoff(attempt: number, random = Math.random): number {
  const delay = Math.min(RETRY_BASE_MS * 2 ** attempt, RETRY_CAP_MS);
  return delay - random() * 0.25 * delay;
}

/**
 * Reads a Retry-After header, given either as seconds or as an HTTP date. It
 * is negative when there is no usable header, and zero when the server asks
 * for an immediate retry.
 */
export function parseRetryAfter(
  header: string | null,
  now = Date.now()
): number {
  if (!header) return -1;
  if (/^\d+$/.test(header.trim())) {
    return Math.max(Number(header.trim()) * 1000, 0);
  }
  // An HTTP date starts with the day's name, as in the three forms RFC 9110
  // allows; Date.parse alone would read a number such as 1.5 as a past date.
  if (!/^[A-Za-z]{3}/.test(header.trim())) return -1;
  const at = Date.parse(header);
  if (!Number.isNaN(at)) return Math.max(at - now, 0);
  return -1;
}
