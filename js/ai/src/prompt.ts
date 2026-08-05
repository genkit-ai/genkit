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

import {
  GenkitError,
  defineActionAsync,
  getContext,
  isAction,
  stripUndefinedProps,
  type Action,
  type ActionAsyncParams,
  type ActionContext,
  type JSONSchema7,
  type z,
} from '@genkit-ai/core';
import { lazy } from '@genkit-ai/core/async';
import { logger } from '@genkit-ai/core/logging';
import type { Registry } from '@genkit-ai/core/registry';
import { toJsonSchema } from '@genkit-ai/core/schema';
import { SPAN_TYPE_ATTR, runInNewSpan } from '@genkit-ai/core/tracing';
import { Message as DpMessage, PromptFunction } from 'dotprompt';
import { existsSync, readFileSync, readdirSync } from 'fs';
import type Handlebars from 'handlebars';
import { basename, join, resolve } from 'path';
import type { DocumentData } from './document.js';
import {
  generate,
  generateStream,
  toGenerateActionOptions,
  toGenerateRequest,
  type GenerateOptions,
  type GenerateResponse,
  type GenerateStreamResponse,
  type OutputOptions,
  type ToolChoice,
} from './generate.js';
import { Message } from './message.js';
import {
  GenerateActionOptionsSchema,
  MiddlewareRef,
  type GenerateActionOptions,
  type GenerateRequest,
  type GenerateRequestSchema,
  type GenerateResponseChunkSchema,
  type GenerateResponseSchema,
  type MessageData,
  type ModelAction,
  type ModelArgument,
  type ModelMiddleware,
  type ModelReference,
  type Part,
} from './model.js';
import { getCurrentSession, type Session } from './session.js';
import type { ToolAction, ToolArgument } from './tool.js';

/**
 * Prompt action.
 */
export type PromptAction<I extends z.ZodTypeAny = z.ZodTypeAny> = Action<
  I,
  typeof GenerateRequestSchema,
  z.ZodNever
> & {
  __action: {
    metadata: {
      type: 'prompt';
    };
  };
  __executablePrompt: ExecutablePrompt<I>;
};

export function isPromptAction(action: Action): action is PromptAction {
  return action.__action.metadata?.type === 'prompt';
}

/**
 * Prompt action.
 */
export type ExecutablePromptAction<I extends z.ZodTypeAny = z.ZodTypeAny> =
  Action<
    I,
    typeof GenerateResponseSchema,
    typeof GenerateResponseChunkSchema
  > & {
    __action: {
      metadata: {
        type: 'executablePrompt';
      };
    };
    __executablePrompt: ExecutablePrompt<I>;
  };

/**
 * Configuration for a prompt action.
 */
export interface PromptConfig<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> {
  name: string;
  variant?: string;
  model?: ModelArgument<CustomOptions>;
  config?: z.infer<CustomOptions>;
  description?: string;
  input?: {
    schema?: I;
    jsonSchema?: JSONSchema7;
  };
  system?: string | Part | Part[] | PartsResolver<z.infer<I>>;
  prompt?: string | Part | Part[] | PartsResolver<z.infer<I>>;
  messages?: string | MessageData[] | MessagesResolver<z.infer<I>>;
  docs?: DocumentData[] | DocsResolver<z.infer<I>>;
  output?: OutputOptions<O>;
  maxTurns?: number;
  returnToolRequests?: boolean;
  metadata?: Record<string, any>;
  tools?: ToolArgument[];
  toolChoice?: ToolChoice;
  use?: (ModelMiddleware | MiddlewareRef)[];
  context?: ActionContext;
}

/**
 * Generate options of a prompt.
 */
export type PromptGenerateOptions<
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> = Omit<GenerateOptions<O, CustomOptions>, 'prompt' | 'system'>;

/**
 * A prompt that can be executed as a function.
 */
export interface ExecutablePrompt<
  I = undefined,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> {
  /** Prompt reference. */
  ref: { name: string; metadata?: Record<string, any> };

  /**
   * Generates a response by rendering the prompt template with given user input and then calling the model.
   *
   * @param input Prompt inputs.
   * @param opt Options for the prompt template, including user input variables and custom model configuration options.
   * @returns the model response as a promise of `GenerateStreamResponse`.
   */
  (
    input?: I,
    opts?: PromptGenerateOptions<O, CustomOptions>
  ): Promise<GenerateResponse<z.infer<O>>>;

  /**
   * Generates a response by rendering the prompt template with given user input and then calling the model.
   * @param input Prompt inputs.
   * @param opt Options for the prompt template, including user input variables and custom model configuration options.
   * @returns the model response as a promise of `GenerateStreamResponse`.
   */
  stream(
    input?: I,
    opts?: PromptGenerateOptions<O, CustomOptions>
  ): GenerateStreamResponse<z.infer<O>>;

  /**
   * Renders the prompt template based on user input.
   *
   * @param opt Options for the prompt template, including user input variables and custom model configuration options.
   * @returns a `GenerateOptions` object to be used with the `generate()` function from @genkit-ai/ai.
   */
  render(
    input?: I,
    opts?: PromptGenerateOptions<O, CustomOptions>
  ): Promise<GenerateOptions<O, CustomOptions>>;

  /**
   * Returns the prompt usable as a tool.
   */
  asTool(): Promise<ToolAction>;
}

export type PartsResolver<I, S = any> = (
  input: I,
  options: {
    state?: S;
    context: ActionContext;
  }
) => Part[] | Promise<string | Part | Part[]>;

export type MessagesResolver<I, S = any> = (
  input: I,
  options: {
    history?: MessageData[];
    state?: S;
    context: ActionContext;
  }
) => MessageData[] | Promise<MessageData[]>;

export type DocsResolver<I, S = any> = (
  input: I,
  options: {
    context: ActionContext;
    state?: S;
  }
) => DocumentData[] | Promise<DocumentData[]>;

interface PromptCache {
  userPrompt?: PromptFunction;
  system?: PromptFunction;
  messages?: PromptFunction;
}

/**
 * Defines a prompt which can be used to generate content or render a request.
 *
 * @returns The new `ExecutablePrompt`.
 */
export function definePrompt<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  options: PromptConfig<I, O, CustomOptions>
): ExecutablePrompt<z.infer<I>, O, CustomOptions> {
  return definePromptAsync(
    registry,
    `${options.name}${options.variant ? `.${options.variant}` : ''}`,
    Promise.resolve(options),
    options.metadata
  );
}

function definePromptAsync<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  name: string,
  optionsPromise: PromiseLike<PromptConfig<I, O, CustomOptions>>,
  metadata?: Record<string, any>
): ExecutablePrompt<z.infer<I>, O, CustomOptions> {
  const promptCache = {} as PromptCache;

  const renderOptionsFn = async (
    input: z.infer<I>,
    renderOptions: PromptGenerateOptions<O, CustomOptions> | undefined
  ): Promise<GenerateOptions> => {
    return await runInNewSpan(
      {
        metadata: {
          name: 'render',
          input,
        },
        labels: {
          [SPAN_TYPE_ATTR]: 'promptTemplate',
        },
      },
      async (metadata) => {
        const messages: MessageData[] = [];
        renderOptions = { ...renderOptions }; // make a copy, we will be trimming
        const session = getCurrentSession(registry);
        const resolvedOptions = await optionsPromise;

        // order of these matters:
        await renderSystemPrompt(
          registry,
          session,
          input,
          messages,
          resolvedOptions,
          promptCache,
          renderOptions
        );
        await renderMessages(
          registry,
          session,
          input,
          messages,
          resolvedOptions,
          renderOptions,
          promptCache
        );
        await renderUserPrompt(
          registry,
          session,
          input,
          messages,
          resolvedOptions,
          promptCache,
          renderOptions
        );

        let docs: DocumentData[] | undefined;
        if (typeof resolvedOptions.docs === 'function') {
          docs = await resolvedOptions.docs(input, {
            state: session?.getCustom(),

            context: renderOptions?.context || getContext() || {},
          });
        } else {
          docs = resolvedOptions.docs;
        }

        const opts: GenerateOptions = stripUndefinedProps({
          model: resolvedOptions.model,
          maxTurns: resolvedOptions.maxTurns,
          messages,
          docs,
          tools: resolvedOptions.tools,
          returnToolRequests: resolvedOptions.returnToolRequests,
          toolChoice: resolvedOptions.toolChoice,
          context: resolvedOptions.context,
          output: resolvedOptions.output,
          use: resolvedOptions.use,
          ...stripUndefinedProps(renderOptions),
          config: {
            ...resolvedOptions?.config,
            ...renderOptions?.config,
          },
          metadata: resolvedOptions.metadata?.metadata
            ? {
                prompt: resolvedOptions.metadata?.metadata,
              }
            : undefined,
        });

        // Fix for issue #3348: Preserve AbortSignal object
        // AbortSignal needs its prototype chain and shouldn't be processed by stripUndefinedProps
        if (renderOptions?.abortSignal) {
          opts.abortSignal = renderOptions.abortSignal;
        }
        // if config is empty and it was not explicitly passed in, we delete it, don't want {}
        if (Object.keys(opts.config).length === 0 && !renderOptions?.config) {
          delete opts.config;
        }
        metadata.output = opts;
        return opts;
      }
    );
  };
  const rendererActionConfig = lazy(() =>
    optionsPromise.then((options: PromptConfig<I, O, CustomOptions>) => {
      const metadata = promptMetadata(options);
      return {
        name: `${options.name}${options.variant ? `.${options.variant}` : ''}`,
        inputJsonSchema: options.input?.jsonSchema,
        inputSchema: options.input?.schema,
        description: options.description,
        actionType: 'prompt',
        metadata,
        fn: async (
          input: z.infer<I>
        ): Promise<GenerateRequest<z.ZodTypeAny>> => {
          return toGenerateRequest(
            registry,
            await renderOptionsFn(input, undefined)
          );
        },
      } as ActionAsyncParams<any, any, any>;
    })
  );
  const rendererAction = defineActionAsync(
    registry,
    'prompt',
    name,
    rendererActionConfig,
    (action) => {
      (action as PromptAction<I>).__executablePrompt =
        executablePrompt as never as ExecutablePrompt<z.infer<I>>;
    }
  ) as Promise<PromptAction<I>>;

  const executablePromptActionConfig = lazy(() =>
    optionsPromise.then((options: PromptConfig<I, O, CustomOptions>) => {
      const metadata = promptMetadata(options);
      return {
        name: `${options.name}${options.variant ? `.${options.variant}` : ''}`,
        inputJsonSchema: options.input?.jsonSchema,
        inputSchema: options.input?.schema,
        outputSchema: GenerateActionOptionsSchema,
        description: options.description,
        actionType: 'executable-prompt',
        metadata,
        fn: async (input: z.infer<I>): Promise<GenerateActionOptions> => {
          return await toGenerateActionOptions(
            registry,
            await renderOptionsFn(input, undefined)
          );
        },
      } as ActionAsyncParams<any, any, any>;
    })
  );

  defineActionAsync(
    registry,
    'executable-prompt',
    name,
    executablePromptActionConfig,
    (action) => {
      (action as ExecutablePromptAction<I>).__executablePrompt =
        executablePrompt as never as ExecutablePrompt<z.infer<I>>;
    }
  ) as Promise<ExecutablePromptAction<I>>;

  const executablePrompt = wrapInExecutablePrompt({
    registry,
    name,
    renderOptionsFn,
    rendererAction,
    metadata,
  });

  return executablePrompt;
}

function promptMetadata(options: PromptConfig<any, any, any>) {
  const metadata = {
    ...options.metadata,
    prompt: {
      ...options.metadata?.prompt,
      config: options.config,
      input: {
        schema: options.input ? toJsonSchema(options.input) : undefined,
      },
      name: options.name.includes('.')
        ? options.name.split('.')[0]
        : options.name,
      model: modelName(options.model),
      tools: (options.tools ?? []).map(toolName).filter(Boolean) as string[],
      toolChoice: options.toolChoice,
      use: (options.use ?? []).map(toMiddlewareMetadata).filter(Boolean) as {
        name: string;
        config?: any;
      }[],
    },
    type: 'prompt',
  };

  if (options.variant) {
    metadata.prompt.variant = options.variant;
  }

  return metadata;
}

function wrapInExecutablePrompt<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(wrapOpts: {
  registry: Registry;
  name: string;
  renderOptionsFn: (
    input: z.infer<I>,
    renderOptions: PromptGenerateOptions<O, CustomOptions> | undefined
  ) => Promise<GenerateOptions>;
  rendererAction: Promise<PromptAction<I>>;
  metadata?: Record<string, any>;
}) {
  const executablePrompt = (async (
    input?: I,
    opts?: PromptGenerateOptions<O, CustomOptions>
  ): Promise<GenerateResponse<z.infer<O>>> => {
    return await runInNewSpan(
      wrapOpts.registry,
      {
        metadata: {
          name: (await wrapOpts.rendererAction).__action.name,
          input,
        },
        labels: {
          [SPAN_TYPE_ATTR]: 'dotprompt',
        },
      },
      async (metadata) => {
        const output = await generate(wrapOpts.registry, {
          ...(await wrapOpts.renderOptionsFn(input, opts)),
        });
        metadata.output = output;
        return output;
      }
    );
  }) as ExecutablePrompt<z.infer<I>, O, CustomOptions>;

  executablePrompt.ref = { name: wrapOpts.name, metadata: wrapOpts.metadata };

  executablePrompt.render = async (
    input?: I,
    opts?: PromptGenerateOptions<O, CustomOptions>
  ): Promise<GenerateOptions<O, CustomOptions>> => {
    return {
      ...(await wrapOpts.renderOptionsFn(input, opts)),
    } as GenerateOptions<O, CustomOptions>;
  };

  executablePrompt.stream = (
    input?: I,
    opts?: PromptGenerateOptions<O, CustomOptions>
  ): GenerateStreamResponse<z.infer<O>> => {
    return generateStream(
      wrapOpts.registry,
      wrapOpts.renderOptionsFn(input, opts)
    );
  };

  executablePrompt.asTool = async (): Promise<ToolAction<I, O>> => {
    return (await wrapOpts.rendererAction) as unknown as ToolAction<I, O>;
  };
  return executablePrompt;
}

async function renderSystemPrompt<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  session: Session | undefined,
  input: z.infer<I>,
  messages: MessageData[],
  options: PromptConfig<I, O, CustomOptions>,
  promptCache: PromptCache,
  renderOptions: PromptGenerateOptions<O, CustomOptions> | undefined
) {
  if (typeof options.system === 'function') {
    messages.push({
      role: 'system',
      content: normalizeParts(
        await options.system(input, {
          state: session?.getCustom(),
          context: renderOptions?.context || getContext() || {},
        })
      ),
    });
  } else if (typeof options.system === 'string') {
    // memoize compiled prompt
    if (!promptCache.system) {
      promptCache.system = await registry.dotprompt.compile(options.system);
    }
    messages.push({
      role: 'system',
      content: await renderDotpromptToParts(
        registry,
        promptCache.system,
        input,
        session,
        options,
        renderOptions
      ),
    });
  } else if (options.system) {
    messages.push({
      role: 'system',
      content: normalizeParts(options.system),
    });
  }
}

async function renderMessages<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  session: Session | undefined,
  input: z.infer<I>,
  messages: MessageData[],
  options: PromptConfig<I, O, CustomOptions>,
  renderOptions: PromptGenerateOptions<O, CustomOptions>,
  promptCache: PromptCache
) {
  if (options.messages) {
    if (typeof options.messages === 'function') {
      messages.push(
        ...(await options.messages(input, {
          state: session?.getCustom(),
          context: renderOptions?.context || getContext() || {},
          history: renderOptions?.messages,
        }))
      );
    } else if (typeof options.messages === 'string') {
      // memoize compiled prompt
      if (!promptCache.messages) {
        promptCache.messages = await registry.dotprompt.compile(
          options.messages
        );
      }
      const rendered = await promptCache.messages({
        input,
        context: {
          ...(renderOptions?.context || getContext()),
          state: session?.getCustom(),
        },
        messages: renderOptions?.messages?.map((m) =>
          Message.parseData(m)
        ) as DpMessage[],
      });
      messages.push(...(rendered.messages as MessageData[]));
    } else {
      messages.push(...options.messages);
    }
  } else {
    if (renderOptions.messages) {
      messages.push(...renderOptions.messages);
    }
  }
  if (renderOptions?.messages) {
    // delete messages from opts so that we don't override messages downstream
    delete renderOptions.messages;
  }
}

async function renderUserPrompt<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  session: Session | undefined,
  input: z.infer<I>,
  messages: MessageData[],
  options: PromptConfig<I, O, CustomOptions>,
  promptCache: PromptCache,
  renderOptions: PromptGenerateOptions<O, CustomOptions> | undefined
) {
  if (typeof options.prompt === 'function') {
    messages.push({
      role: 'user',
      content: normalizeParts(
        await options.prompt(input, {
          state: session?.getCustom(),
          context: renderOptions?.context || getContext() || {},
        })
      ),
    });
  } else if (typeof options.prompt === 'string') {
    // memoize compiled prompt
    if (!promptCache.userPrompt) {
      promptCache.userPrompt = await registry.dotprompt.compile(options.prompt);
    }
    messages.push({
      role: 'user',
      content: await renderDotpromptToParts(
        registry,
        promptCache.userPrompt,
        input,
        session,
        options,
        renderOptions
      ),
    });
  } else if (options.prompt) {
    messages.push({
      role: 'user',
      content: normalizeParts(options.prompt),
    });
  }
}

function modelName(
  modelArg: ModelArgument<any> | undefined
): string | undefined {
  if (modelArg === undefined) {
    return undefined;
  }
  if (typeof modelArg === 'string') {
    return modelArg;
  }
  if ((modelArg as ModelReference<any>).name) {
    return (modelArg as ModelReference<any>).name;
  }
  return (modelArg as ModelAction).__action.name;
}

function toolName(tool: ToolArgument): string | undefined {
  if (typeof tool === 'string') {
    return tool;
  }
  if (isAction(tool)) {
    return tool.__action.name;
  }
  if (isExecutablePrompt(tool)) {
    return tool.ref.name;
  }
  return undefined;
}

function toMiddlewareMetadata(
  middleware: any
): { name: string; config?: any } | undefined {
  if (typeof middleware === 'string') {
    return { name: middleware };
  }
  if (
    typeof middleware === 'object' &&
    middleware !== null &&
    middleware.name
  ) {
    return { name: middleware.name, config: middleware.config };
  }
  return undefined;
}

function normalizeParts(parts: string | Part | Part[]): Part[] {
  if (Array.isArray(parts)) return parts;
  if (typeof parts === 'string') {
    return [
      {
        text: parts,
      },
    ];
  }
  return [parts as Part];
}

async function renderDotpromptToParts<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  promptFn: PromptFunction,
  input: any,
  session: Session | undefined,
  options: PromptConfig<I, O, CustomOptions>,
  renderOptions: PromptGenerateOptions<O, CustomOptions> | undefined
): Promise<Part[]> {
  const renderred = await promptFn({
    input,
    context: {
      ...(renderOptions?.context || getContext()),
      state: session?.getCustom(),
    },
  });
  if (renderred.messages.length !== 1) {
    throw new Error('parts tempate must produce only one message');
  }
  return renderred.messages[0].content;
}

/**
 * Checks whether the provided object is an executable prompt.
 */
export function isExecutablePrompt(obj: any): obj is ExecutablePrompt {
  return (
    !!(obj as ExecutablePrompt)?.render &&
    !!(obj as ExecutablePrompt)?.asTool &&
    !!(obj as ExecutablePrompt)?.stream
  );
}

export function loadPromptFolder(
  registry: Registry,
  dir = './prompts',
  ns: string
): void {
  const promptsPath = resolve(dir);
  if (existsSync(promptsPath)) {
    loadPromptFolderRecursively(registry, dir, ns, '');
  }
}
export function loadPromptFolderRecursively(
  registry: Registry,
  dir: string,
  ns: string,
  subDir: string
): void {
  const promptsPath = resolve(dir);
  const dirEnts = readdirSync(join(promptsPath, subDir), {
    withFileTypes: true,
  });
  for (const dirEnt of dirEnts) {
    const parentPath = join(promptsPath, subDir);
    const fileName = dirEnt.name;
    if (dirEnt.isFile() && fileName.endsWith('.prompt')) {
      if (fileName.startsWith('_')) {
        const partialName = fileName.substring(1, fileName.length - 7);
        definePartial(
          registry,
          partialName,
          readFileSync(join(parentPath, fileName), {
            encoding: 'utf8',
          })
        );
        logger.debug(
          `Registered Dotprompt partial "${partialName}" from "${join(parentPath, fileName)}"`
        );
      } else {
        // If this prompt is in a subdirectory, we need to include that
        // in the namespace to prevent naming conflicts.
        loadPrompt(
          registry,
          promptsPath,
          fileName,
          subDir ? `${subDir}/` : '',
          ns
        );
      }
    } else if (dirEnt.isDirectory()) {
      loadPromptFolderRecursively(registry, dir, ns, join(subDir, fileName));
    }
  }
}

export function definePartial(
  registry: Registry,
  name: string,
  source: string
) {
  registry.dotprompt.definePartial(name, source);
}

export function defineHelper(
  registry: Registry,
  name: string,
  fn: Handlebars.HelperDelegate
) {
  registry.dotprompt.defineHelper(name, fn);
}

function loadPrompt(
  registry: Registry,
  path: string,
  filename: string,
  prefix = '',
  ns = 'dotprompt'
): void {
  let name = `${prefix ?? ''}${basename(filename, '.prompt')}`;
  let variant: string | null = null;
  if (name.includes('.')) {
    const parts = name.split('.');
    name = parts[0];
    variant = parts[1];
  }
  const source = readFileSync(join(path, prefix ?? '', filename), 'utf8');
  const parsedPrompt = registry.dotprompt.parse(source);
  definePromptAsync(
    registry,
    registryDefinitionKey(name, variant ?? undefined, ns),
    // We use a lazy promise here because we only want prompt loaded when it's first used.
    // This is important because otherwise the loading may happen before the user has configured
    // all the schemas, etc., which will result in dotprompt.renderMetadata errors.
    lazy(async () => {
      const promptMetadata =
        await registry.dotprompt.renderMetadata(parsedPrompt);
      if (variant) {
        promptMetadata.variant = variant;
      }

      // dotprompt can set null description on the schema, which can confuse downstream schema consumers
      if (promptMetadata.output?.schema?.description === null) {
        delete promptMetadata.output.schema.description;
      }
      if (promptMetadata.input?.schema?.description === null) {
        delete promptMetadata.input.schema.description;
      }

      const metadata = {
        ...promptMetadata.metadata,
        type: 'prompt',
        prompt: {
          ...promptMetadata,
          use: (Array.isArray(promptMetadata.raw?.['use'])
            ? promptMetadata.raw?.['use']
            : []
          )
            .map(toMiddlewareMetadata)
            .filter(Boolean),
          toolChoice: promptMetadata.raw?.['toolChoice'],
          template: parsedPrompt.template,
        },
      };
      if (promptMetadata.raw?.['metadata']) {
        metadata['metadata'] = { ...promptMetadata.raw?.['metadata'] };
      }

      return {
        name: registryDefinitionKey(name, variant ?? undefined, ns),
        model: promptMetadata.model,
        config: promptMetadata.config,
        tools: promptMetadata.tools,
        description: promptMetadata.description,
        output: {
          jsonSchema: promptMetadata.output?.schema,
          format: promptMetadata.output?.format,
        },
        input: {
          jsonSchema: promptMetadata.input?.schema,
        },
        metadata,
        maxTurns: promptMetadata.raw?.['maxTurns'],
        toolChoice: promptMetadata.raw?.['toolChoice'],
        returnToolRequests: promptMetadata.raw?.['returnToolRequests'],
        use: promptMetadata.raw?.['use'],
        messages: parsedPrompt.template,
      };
    })
  );
}

export async function prompt<
  I = undefined,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  name: string,
  options?: { variant?: string; dir?: string | null }
): Promise<ExecutablePrompt<I, O, CustomOptions>> {
  return await lookupPrompt<I, O, CustomOptions>(
    registry,
    name,
    options?.variant
  );
}

function registryLookupKey(name: string, variant?: string, ns?: string) {
  return `/prompt/${registryDefinitionKey(name, variant, ns)}`;
}

async function lookupPrompt<
  I = undefined,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  name: string,
  variant?: string
): Promise<ExecutablePrompt<I, O, CustomOptions>> {
  const registryPrompt = await registry.lookupAction(
    registryLookupKey(name, variant)
  );
  if (registryPrompt) {
    return (registryPrompt as PromptAction)
      .__executablePrompt as never as ExecutablePrompt<I, O, CustomOptions>;
  }
  throw new GenkitError({
    status: 'NOT_FOUND',
    message: `Prompt ${name + (variant ? ` (variant ${variant})` : '')} not found`,
  });
}

function registryDefinitionKey(name: string, variant?: string, ns?: string) {
  // "ns/prompt.variant" where ns and variant are optional
  return `${ns ? `${ns}/` : ''}${name}${variant ? `.${variant}` : ''}`;
}
