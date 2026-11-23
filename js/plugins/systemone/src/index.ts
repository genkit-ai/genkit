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

/**
 * `@genkit-ai/systemone` serves decision models, the models that speak the
 * System One protocol, and holds the question helpers a decision is declared
 * with. A decision model does not generate text. It evaluates a state against
 * typed questions and returns one typed answer per question, with calibrated
 * probabilities.
 *
 * The plugin serves such a model as one that speaks only constrained JSON,
 * which is the subset of the generate API it fits exactly. The questions are
 * the fields of the output schema, declared with {@link choice},
 * {@link score}, and {@link noul}, so a decision is one generate call with
 * nothing more than the model and the state:
 *
 * ```ts
 * import { choice, noul, systemOne } from '@genkit-ai/systemone';
 * import { genkit, z } from 'genkit';
 *
 * const Triage = z.object({
 *   department: choice({
 *     billing: 'Payments, invoicing, refunds',
 *     technical: 'Bugs, outages, integrations',
 *     other: 'None of the above',
 *   }).describe('Which team should handle this?'),
 *   isUrgent: noul().describe(
 *     'Does the ticket explicitly communicate time pressure?'
 *   ),
 * });
 *
 * const ai = genkit({ plugins: [systemOne.typeSafe()] }); // TYPESAFE_API_KEY
 *
 * const { output } = await ai.generate({
 *   model: 'typesafe/jev-1.13.0',
 *   prompt: ticket,
 *   output: { schema: Triage },
 * });
 * if ((output!.department.confidence ?? 0) < 0.6) {
 *   // route to a human
 * }
 * ```
 *
 * The question helpers are the same for every model that speaks the protocol,
 * so one decision schema works with each of them. {@link systemOne} is the
 * plugin for any server that speaks the protocol; `systemOne.typeSafe()` and
 * the other constructors set it up for the known ones.
 *
 * This package is a preview: its API may change in any minor release.
 *
 * @module
 */

import {
  GenkitError,
  modelRef,
  type ActionMetadata,
  type ModelReference,
} from 'genkit';
import { logger } from 'genkit/logging';
import {
  genkitPluginV2,
  modelActionMetadata,
  type GenkitPluginV2,
} from 'genkit/plugin';
import {
  Client,
  HeaderError,
  REQUEST_TIMEOUT_MS,
  buildHeaders,
  type Route,
  type Unwrap,
} from './client.js';
import {
  SystemOneConfigSchema,
  modelInfo,
  newModel,
  type ModelDescription,
  type SystemOneConfig,
} from './model.js';
import { ollamaRoute } from './ollama.js';
import { schema } from './questions.js';
import { pathEscape, unwrapWorkersAI, workersAIRoute } from './workersai.js';

export type { Route, Unwrap } from './client.js';
export { SystemOneConfigSchema, type SystemOneConfig } from './model.js';
export {
  answerKind,
  choice,
  label,
  level,
  margin,
  noul,
  ranked,
  responseInfo,
  score,
  type Answer,
  type ChoiceAnswer,
  type ChoiceOption,
  type ChoiceQuestion,
  type Info,
  type NoulAnswer,
  type NoulQuestion,
  type OptionGuidance,
  type Question,
  type QuestionSchema,
  type ScoreAnswer,
  type ScoreQuestion,
  type YesNo,
} from './questions.js';

/**
 * Whether models read one kind of media. Unset keeps the setting it
 * overrides: for a model, the plugin's; for the plugin, refusing the media.
 */
export type MediaSupport = 'supported' | 'unsupported';

/** Describes one decision model. */
export interface ModelSpec {
  /** Names the model in the Dev UI. Unset means one built from the ID. */
  label?: string;
  /**
   * Override the plugin's media settings for this model. Unset keeps the
   * plugin's, so an entry that only sets a label changes nothing else.
   */
  images?: MediaSupport;
  audio?: MediaSupport;
  video?: MediaSupport;
}

/**
 * The settings of a plugin for a server that speaks System One. For a server
 * with no constructor of its own, set them:
 *
 * ```ts
 * systemOne({
 *   provider: 'liquid',
 *   baseUrl: 'https://api.liquid.ai/decisions',
 *   modelsPath: '/v1/models',
 *   apiKey: process.env.LIQUID_API_KEY,
 *   images: 'supported',
 * });
 * ```
 *
 * The constructors, such as `systemOne.typeSafe()`, take the same settings,
 * over the ones they set for their server.
 */
export interface SystemOneOptions {
  /**
   * Names the plugin and prefixes its model names. It is required, cannot
   * contain a slash, and must differ from every other plugin's name, so a
   * host that serves chat models through another plugin needs a name of its
   * own here, such as openrouter-decisions.
   */
  provider: string;
  /**
   * The server's root, which `path` and `modelsPath` are joined to: an http
   * or https URL with a host, checked when the plugin is created. It is
   * required, unless a constructor gives a default.
   */
  baseUrl?: string;
  /**
   * The request path. Unset means `/v1/systemone`, and `/` means `baseUrl`
   * itself, for a base URL that is the whole endpoint.
   */
  path?: string;
  /**
   * The path of the server's model listing. When unset, no listing is asked
   * for, and the Dev UI shows the models in `models`.
   */
  modelsPath?: string;
  /**
   * Sent as a bearer token. A constructor reads it from the environment
   * variable it names when this is unset; a server that takes no key, such as
   * a local one, is reached with none. `false` takes the key from each
   * request's `config.apiKey` only, as the anthropic and google-genai plugins
   * do, and refuses a request without one; the model listing then goes out
   * with no key. A key in `config.apiKey` is recorded in traces and the Dev UI
   * like any other config field, so where traces are kept, prefer the
   * environment variable or this option. With per-request keys, give every
   * fallback reference a config with its own key, or set `isolateConfig`:
   * a reference without a config inherits the request's, so the first
   * server's key would go to the next server's host.
   */
  apiKey?: string | false;
  /**
   * The models known ahead, keyed by the ID the server uses or by the full
   * model name, `provider/ID`. They are listed in the Dev UI whether or not
   * the server has a listing. Any other ID resolves too, as a model with no
   * description. A constructor's models stay: these are added to them, and
   * an entry with one of their IDs replaces that model's whole spec, as in
   * Go. Naming a constructor's model by its full name is refused.
   */
  models?: Record<string, ModelSpec>;
  /**
   * Whether the models read each kind of media. Unset refuses a request with
   * media of that kind before it is sent, since media is an extension to the
   * protocol, and a server that does not know a field can drop it and answer
   * without the media. `'supported'` sends it, for a server that takes it,
   * and leaves a model that cannot read it to the server to refuse. A model's
   * {@link ModelSpec} overrides each.
   */
  images?: MediaSupport;
  audio?: MediaSupport;
  video?: MediaSupport;
  /**
   * Sends the requests: the escape hatch to the transport, such as a proxy
   * or client middleware. The global `fetch` when unset.
   */
  fetch?: typeof fetch;
  /**
   * Bounds one attempt, in milliseconds: a positive number; 0 is refused.
   * Unset means 30 seconds with the global `fetch`, and no bound of the
   * plugin's with a `fetch` of your own, which keeps its own timeouts.
   */
  timeoutMs?: number;
  /**
   * Sent on every request. The Authorization header the API key makes
   * replaces one given here. A name or value HTTP cannot carry is refused
   * when the plugin is created (FAILED_PRECONDITION), as is such a key; a
   * per-request key in `config.apiKey` that HTTP cannot carry fails that
   * request (INVALID_ARGUMENT), without a retry.
   */
  headers?: Record<string, string>;
  /**
   * For a server whose wire is not the native one, builds the request it
   * takes from the model ID and the native body, which it may change: the
   * path to append to `path`, joined with a slash when it does not start
   * with one, and the body to send. When unset, the native body goes to
   * `path`.
   */
  route?: Route;
  /**
   * For a server that wraps its responses, returns the native response from
   * the body the server sent, parsed as JSON. An error it throws is reported
   * under `provider`, as UNKNOWN unless it is a `GenkitError`.
   */
  unwrap?: Unwrap;
}

/**
 * What a constructor sets for its server beyond the options: where its key
 * and base URL come from.
 */
interface Preset {
  /** Names the server in model labels; unset means the provider. */
  label?: string;
  /** The variable the key is read from. A server that names one requires a key. */
  apiKeyEnv?: string;
  /** The default base URL, and the variable that overrides it. */
  baseUrl?: string;
  baseUrlEnv?: string;
  /**
   * Fills the `{account}` segment of the base URL, from `accountEnv` when it
   * is unset. A server that names `accountEnv` requires an account while its
   * base URL has that segment.
   */
  account?: string;
  accountEnv?: string;
}

/** A plugin for a server that speaks System One. */
export type SystemOnePlugin = GenkitPluginV2 & {
  /**
   * A reference to one of the plugin's models with a config, for the places
   * that take a reference rather than a name, such as a fallback list. The ID
   * is the server's or the full model name, as a key of `models` is.
   */
  modelRef(
    id: string,
    config?: SystemOneConfig
  ): ModelReference<typeof SystemOneConfigSchema>;
};

/**
 * A model listing is bounded by a timeout, since it serves the Dev UI and
 * must not hold it up when the server is slow. A listing is kept for a few
 * minutes, and a failed one is tried again after a minute.
 */
const LISTING_TIMEOUT_MS = 5_000;
const LISTING_TTL_MS = 5 * 60_000;
const LISTING_RETRY_MS = 60_000;

/** The longest timeout a timer takes: 2^31 - 1 milliseconds. */
const MAX_TIMEOUT_MS = 2_147_483_647;

/** Connection and framing headers, which only the transport sets. */
const TRANSPORT_HEADERS = [
  'connection',
  'content-length',
  'expect',
  'host',
  'keep-alive',
  'transfer-encoding',
  'upgrade',
];

/**
 * An environment variable, trimmed, so a key read from a file that ends in a
 * newline works. An explicit `apiKey` is not trimmed: it is checked as given.
 * Unset or blank is undefined.
 */
function env(name: string): string | undefined {
  return process.env[name]?.trim() || undefined;
}

function misconfigured(message: string): GenkitError {
  return new GenkitError({ status: 'FAILED_PRECONDITION', message });
}

/** A non-empty path with the leading slash that joins it to the base URL. */
function rooted(path: string): string {
  if (path === '' || path.startsWith('/')) return path;
  return `/${path}`;
}

/** The options without the ones set to undefined, which keep the defaults. */
function given<T extends object>(options: T | undefined): Partial<T> {
  return Object.fromEntries(
    Object.entries(options ?? {}).filter(([, value]) => value !== undefined)
  ) as Partial<T>;
}

/** Most IDs a listing contributes, and the longest ID taken from one. */
const LISTING_MAX_MODELS = 1_000;
const LISTING_MAX_ID_LENGTH = 256;

function newPlugin(
  options: SystemOneOptions,
  preset: Preset,
  presetModels: Record<string, ModelSpec> = {}
): SystemOnePlugin {
  const provider = options.provider;
  if (!provider) {
    throw misconfigured(
      "systemone: set provider, the plugin's name and the prefix of its models"
    );
  }
  if (provider.includes('/')) {
    throw misconfigured(
      `systemone: provider ${JSON.stringify(provider)} contains a slash, which separates it from the model ID`
    );
  }
  let apiKey = options.apiKey === false ? undefined : options.apiKey;
  const keyDeferred = options.apiKey === false;
  if (preset.apiKeyEnv && !keyDeferred) {
    apiKey = apiKey || env(preset.apiKeyEnv);
    if (!apiKey) {
      throw misconfigured(
        `${provider}: set apiKey or the ${preset.apiKeyEnv} environment variable, or pass apiKey: false to give it per request`
      );
    }
  }
  let baseUrl =
    options.baseUrl ||
    (preset.baseUrlEnv ? env(preset.baseUrlEnv) : undefined) ||
    preset.baseUrl ||
    '';
  if (!baseUrl) {
    throw misconfigured(
      `${provider}: set baseUrl, the root of the server's System One API`
    );
  }
  if (preset.accountEnv && baseUrl.includes('{account}')) {
    const account = preset.account || env(preset.accountEnv);
    if (!account) {
      throw misconfigured(
        `${provider}: pass an account ID or set the ${preset.accountEnv} environment variable`
      );
    }
    let segment: string;
    try {
      segment = pathEscape(account);
    } catch {
      throw misconfigured(
        `${provider}: the account ID is not valid text (a lone UTF-16 surrogate)`
      );
    }
    // A function, so that $& or $1 in an account ID stays literal.
    baseUrl = baseUrl.replaceAll('{account}', () => segment);
  }
  // Go finds a bad base URL only at the first request; this port checks it
  // here, so a typo fails when the app starts. The URL is never quoted: it
  // can carry a password.
  let parsed: URL | undefined;
  try {
    parsed = new URL(baseUrl);
  } catch {
    parsed = undefined;
  }
  if (
    !parsed ||
    (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') ||
    !parsed.hostname
  ) {
    throw misconfigured(
      `${provider}: baseUrl must be an http or https URL with a host`
    );
  }
  // Paths are appended to the base URL as text, so a query or a fragment
  // would swallow them, and credentials in it would end up in error messages
  // of the transport; a key goes in apiKey or headers instead.
  if (
    parsed.username ||
    parsed.password ||
    baseUrl.includes('?') ||
    baseUrl.includes('#')
  ) {
    throw misconfigured(
      `${provider}: baseUrl cannot carry credentials, a query, or a fragment`
    );
  }
  if (
    options.timeoutMs !== undefined &&
    !(
      Number.isInteger(options.timeoutMs) &&
      options.timeoutMs >= 1 &&
      options.timeoutMs <= MAX_TIMEOUT_MS
    )
  ) {
    throw misconfigured(
      `${provider}: timeoutMs must be a whole number of milliseconds from 1 to ${MAX_TIMEOUT_MS}`
    );
  }
  let path = rooted(options.path || '/v1/systemone');
  if (path === '/') path = '';

  // The plugin's headers and key go on every request, so one that HTTP
  // cannot carry is a setup to fix now, not a failure at the first request.
  try {
    buildHeaders(options.headers, apiKey, true);
  } catch (e) {
    throw misconfigured(
      `${provider}: ${e instanceof HeaderError ? e.message : 'a header cannot be sent'}`
    );
  }
  // Headers that describe the connection or the framing are the transport's
  // to set: given here they would break every request, or be dropped.
  for (const name of Object.keys(options.headers ?? {})) {
    if (TRANSPORT_HEADERS.includes(name.toLowerCase())) {
      throw misconfigured(
        `${provider}: the ${name} header is the transport's to set, and cannot be given in headers`
      );
    }
  }

  const trim = (id: string) =>
    id.startsWith(`${provider}/`) ? id.slice(provider.length + 1) : id;
  // As in Go: the options' models are added to the preset's, an entry with a
  // preset model's ID replaces its whole spec, and naming a preset model by
  // its full name is a mistake, since the two keys would describe one model.
  const models = new Map<string, ModelSpec>(Object.entries(presetModels));
  const named = new Set<string>();
  for (const [key, spec] of Object.entries(options.models ?? {})) {
    const id = trim(key);
    if (key !== id && Object.hasOwn(presetModels, id)) {
      throw misconfigured(
        `${provider}: models names ${key}, which is already defined by the preset as ${id}; use the bare ID`
      );
    }
    if (named.has(id)) {
      throw misconfigured(
        `${provider}: models names ${id} twice, by its ID and by its full name`
      );
    }
    named.add(id);
    models.set(id, given(spec));
  }

  const client = new Client({
    // The global fetch is looked up per request, so one installed later, as a
    // test or an instrumentation library does, is used.
    fetch: options.fetch ?? ((input, init) => globalThis.fetch(input, init)),
    timeoutMs:
      options.timeoutMs ?? (options.fetch ? undefined : REQUEST_TIMEOUT_MS),
    baseUrl: baseUrl.replace(/\/+$/, ''),
    apiKey,
    headers: { ...options.headers },
    endpoint: {
      name: provider,
      path,
      modelsPath: rooted(options.modelsPath ?? ''),
      route: options.route,
      unwrap: options.unwrap,
    },
  });

  const describe = (id: string): ModelDescription => {
    const spec = models.get(id) ?? {};
    const reads = (plugin?: MediaSupport, model?: MediaSupport) =>
      (model ?? plugin) === 'supported';
    return {
      label: spec.label || `${preset.label || provider} - ${id}`,
      reads: {
        images: reads(options.images, spec.images),
        audio: reads(options.audio, spec.audio),
        video: reads(options.video, spec.video),
      },
      keyPerRequest: keyDeferred,
    };
  };

  // The listing serves every caller until it goes stale, and concurrent
  // listings share one fetch.
  let listed: ActionMetadata[] | undefined;
  let listedUntil = 0;
  let listing: Promise<ActionMetadata[]> | undefined;
  let listingWarned = false;

  const listedIds = async (): Promise<{ ids: string[]; ttl: number }> => {
    if (!client.endpoint.modelsPath) return { ids: [], ttl: LISTING_TTL_MS };
    try {
      const listedModels = await client.listModels(
        AbortSignal.timeout(LISTING_TIMEOUT_MS)
      );
      const ids = listedModels
        .map((m) => m?.id || m?.name)
        .filter(
          (id): id is string =>
            typeof id === 'string' &&
            id !== '' &&
            id.length <= LISTING_MAX_ID_LENGTH
        )
        .slice(0, LISTING_MAX_MODELS);
      return { ids, ttl: LISTING_TTL_MS };
    } catch (e) {
      const message = `${provider}: model listing failed, advertising the configured models: ${(e as Error)?.message ?? e}`;
      // Said once at warn level, since it hides models from the Dev UI, and
      // again at debug level after that.
      if (listingWarned) {
        logger.debug(message);
      } else {
        listingWarned = true;
        logger.warn(message);
      }
      return { ids: [], ttl: LISTING_RETRY_MS };
    }
  };

  const list = async (): Promise<ActionMetadata[]> => {
    if (listed && Date.now() < listedUntil) return [...listed];
    listing ??= (async () => {
      const { ids, ttl } = await listedIds();
      const all = [...new Set([...models.keys(), ...ids])].sort();
      const actions = all.map((id) =>
        modelActionMetadata({
          name: `${provider}/${id}`,
          info: modelInfo(describe(id)),
          configSchema: SystemOneConfigSchema,
        })
      );
      listed = actions;
      listedUntil = Date.now() + ttl;
      return actions;
    })().finally(() => {
      listing = undefined;
    });
    return [...(await listing)];
  };

  const plugin = genkitPluginV2({
    name: provider,
    // No actions are registered up front: models resolve by name.
    init: () => [],
    // The ID is forwarded as given; whether it exists is for the server to
    // say when the first request is made.
    resolve: (actionType, id) => {
      if (actionType !== 'model') return undefined;
      return newModel(client, `${provider}/${id}`, id, describe(id));
    },
    list,
  });
  return Object.assign(plugin, {
    modelRef(id: string, config?: SystemOneConfig) {
      const modelId = trim(id);
      return modelRef({
        name: `${provider}/${modelId}`,
        configSchema: SystemOneConfigSchema,
        info: modelInfo(describe(modelId)),
        ...(config ? { config } : {}),
      });
    },
  });
}

/**
 * A plugin for any server that speaks System One. It serves the server's
 * decision models under `provider`, by the IDs the server uses: a model the
 * server calls d1 is `provider/d1`, and one a gateway calls liquid/d1 is
 * `provider/liquid/d1`. Any ID resolves, so a model the server adds later
 * works without a change here.
 *
 * It throws without a provider or a base URL, or without a key or an account
 * ID for a server that requires one, since every request would fail, and when
 * `models` names one model twice.
 */
function systemOnePlugin(options: SystemOneOptions): SystemOnePlugin {
  return newPlugin(options, {});
}

/**
 * TypeSafe AI's own API, which serves jev, the first System One model. Its
 * models are `typesafe/<id>`, such as `typesafe/jev-1.13.0`, and read text and
 * JSON only. The API key comes from `TYPESAFE_API_KEY` and the base URL from
 * `TYPESAFE_BASE_URL`, the variables TypeSafe's SDKs read. Pin a version in
 * production: confidence thresholds tuned against one release do not carry
 * over to the next.
 */
function typeSafe(options?: Partial<SystemOneOptions>): SystemOnePlugin {
  return newPlugin(
    {
      provider: 'typesafe',
      modelsPath: '/v1/models',
      ...given(options),
    },
    {
      label: 'TypeSafe',
      apiKeyEnv: 'TYPESAFE_API_KEY',
      baseUrl: 'https://api.typesafe.ai',
      baseUrlEnv: 'TYPESAFE_BASE_URL',
    },
    { 'jev-latest': {}, 'jev-1.13.0': {} }
  );
}

/**
 * OpenRouter's Decisions API, which serves several vendors' decision models
 * under one key. Its models are `openrouter-decisions/<ID>` by OpenRouter's
 * ID, such as `openrouter-decisions/liquid/d1` and
 * `openrouter-decisions/typesafe/jev-1.13`. The name is not openrouter, which
 * a plugin for OpenRouter's chat models would have, so both serve one app. The
 * API key comes from `OPENROUTER_API_KEY`. The Dev UI lists the models
 * OpenRouter lists as decision models, and the best known of them when the
 * listing fails, and a request's cost is reported in the response's usage, as
 * `usage.custom.cost`.
 *
 * Its models are text only here. OpenRouter's Decisions API takes images in
 * the state, not in the images field, and it can route a request to a model or
 * a provider that drops them and answers without them, so a request with media
 * is refused before it is sent.
 */
function openRouter(options?: Partial<SystemOneOptions>): SystemOnePlugin {
  return newPlugin(
    {
      provider: 'openrouter-decisions',
      path: '/api/alpha/decisions',
      modelsPath: '/api/v1/models?output_modalities=decisions',
      ...given(options),
    },
    {
      label: 'OpenRouter',
      apiKeyEnv: 'OPENROUTER_API_KEY',
      baseUrl: 'https://openrouter.ai',
    },
    {
      'cloudflare/clef': {},
      'cloudflare/clef-flash': {},
      'liquid/d1': {},
      'typesafe/jev-1.13': {},
      '~typesafe/jev-latest': {},
    }
  );
}

/**
 * A local Ollama server, which serves the decision models it has pulled, such
 * as Clef, and takes no key. Its models are `ollama-decisions/<name>` by
 * Ollama's name, such as `ollama-decisions/clef`. The name is not ollama,
 * which the plugin for Ollama's chat models has, so both serve one app. The
 * server is `http://localhost:11434` unless `baseUrl` says otherwise. The Dev
 * UI lists Clef, and `models` adds the other models pulled.
 *
 * Ollama takes images as raw base64 rather than data URLs, which this plugin
 * converts, and requires text in the state, so a request with no text, such as
 * one of images alone, is refused before it is sent.
 */
function ollama(options?: Partial<SystemOneOptions>): SystemOnePlugin {
  return newPlugin(
    {
      provider: 'ollama-decisions',
      route: ollamaRoute,
      images: 'supported',
      ...given(options),
    },
    { label: 'Ollama', baseUrl: 'http://localhost:11434' },
    { clef: {} }
  );
}

/**
 * Cloudflare Workers AI for one account. It serves Cloudflare's own decision
 * models, Clef and Clef-flash, which read images, and Clef-omni, which reads
 * images, audio, and video, and partners' models such as jev. Its models are
 * `cloudflare-decisions/<ID>` by the Workers AI ID:
 * `cloudflare-decisions/@cf/cloudflare/clef`,
 * `cloudflare-decisions/@cf/cloudflare/clef-flash`,
 * `cloudflare-decisions/@cf/cloudflare/clef-omni`, and
 * `cloudflare-decisions/typesafe/jev`. The API token comes from
 * `CLOUDFLARE_API_TOKEN`, and the account ID from `CLOUDFLARE_ACCOUNT_ID` when
 * `accountId` is unset. A `baseUrl` with no `{account}` segment, such as a
 * proxy's, needs no account ID.
 *
 * Workers AI names one model per release, with no versions, so a threshold
 * tuned against a model can move when Cloudflare updates it.
 */
function workersAI(
  accountId?: string,
  options?: Partial<SystemOneOptions>
): SystemOnePlugin {
  return newPlugin(
    {
      provider: 'cloudflare-decisions',
      path: '/ai/run',
      route: workersAIRoute,
      unwrap: unwrapWorkersAI,
      images: 'supported',
      ...given(options),
    },
    {
      label: 'Workers AI',
      apiKeyEnv: 'CLOUDFLARE_API_TOKEN',
      baseUrl: 'https://api.cloudflare.com/client/v4/accounts/{account}',
      account: accountId,
      accountEnv: 'CLOUDFLARE_ACCOUNT_ID',
    },
    {
      '@cf/cloudflare/clef': {},
      '@cf/cloudflare/clef-flash': {},
      '@cf/cloudflare/clef-omni': { audio: 'supported', video: 'supported' },
      'typesafe/jev': { images: 'unsupported' },
    }
  );
}

/**
 * The System One plugin: `systemOne(options)` for any server that speaks the
 * protocol, constructors for the known ones, and `systemOne.schema()`, which
 * builds the output schema of questions known only at run time.
 */
export const systemOne = Object.assign(systemOnePlugin, {
  typeSafe,
  openRouter,
  ollama,
  workersAI,
  schema,
});

export default systemOne;
