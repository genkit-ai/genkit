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
  GenkitAI,
  defineHelper,
  definePartial,
  definePrompt,
  defineTool,
  generate,
  loadPromptFolder,
  modelRef,
  prompt,
  type BaseDataPointSchema,
  type Document,
  type EmbedderInfo,
  type ExecutablePrompt,
  type GenerateOptions,
  type GenerateRequest,
  type GenerateResponse,
  type GenerateResponseChunk,
  type GenerateResponseData,
  type GenerateStreamResponse,
  type ModelArgument,
  type ModelReference,
  type PromptConfig,
  type PromptGenerateOptions,
  type RetrieverAction,
  type RetrieverInfo,
  type ToolAction,
  type ToolConfig,
} from '@genkit-ai/ai';
import {
  defineEmbedder,
  type EmbedderAction,
  type EmbedderFn,
} from '@genkit-ai/ai/embedder';
import {
  defineEvaluator,
  evaluate,
  type EvaluatorAction,
  type EvaluatorFn,
} from '@genkit-ai/ai/evaluator';
import { configureFormats } from '@genkit-ai/ai/formats';
import {
  defineBackgroundModel,
  defineGenerateAction,
  defineModel,
  type BackgroundModelAction,
  type DefineBackgroundModelOptions,
  type DefineModelOptions,
  type GenerateResponseChunkData,
  type ModelAction,
} from '@genkit-ai/ai/model';
import {
  RankedDocument,
  RerankerParams,
  defineReranker,
  rerank,
  type RerankerFn,
  type RerankerInfo,
} from '@genkit-ai/ai/reranker';
import {
  IndexerParams,
  RetrieverParams,
  defineIndexer,
  defineRetriever,
  defineSimpleRetriever,
  index,
  retrieve,
  type IndexerAction,
  type IndexerFn,
  type RetrieverFn,
  type SimpleRetrieverOptions,
} from '@genkit-ai/ai/retriever';
import {
  dynamicTool,
  type MultipartToolAction,
  type MultipartToolFn,
  type ToolFn,
} from '@genkit-ai/ai/tool';
import {
  ActionFnArg,
  ActionRunOptions,
  GenkitError,
  ReflectionServer,
  defineBidiFlow,
  defineDynamicActionProvider,
  defineFlow,
  defineJsonSchema,
  defineSchema,
  isAction,
  isBackgroundAction,
  isDevEnv,
  registerBackgroundAction,
  setClientHeader,
  type Action,
  type ActionContext,
  type DapConfig,
  type DapFn,
  type DynamicActionProviderAction,
  type FlowConfig,
  type FlowFn,
  type JSONSchema,
  type StreamingCallback,
  type z,
} from '@genkit-ai/core';
import { Channel } from '@genkit-ai/core/async';
import type { HasRegistry } from '@genkit-ai/core/registry';
import type {
  BaseEvalDataPointSchema,
  EvalResponses,
  EvaluatorParams,
} from './evaluator.js';
import { logger } from './logging.js';
import {
  isPluginV2,
  type GenkitPlugin,
  type GenkitPluginV2,
  type ResolvableAction,
} from './plugin.js';
import { Registry, type ActionType } from './registry.js';
import { SPAN_TYPE_ATTR, runInNewSpan } from './tracing.js';

/**
 * @deprecated use `ai.definePrompt({messages: fn})`
 */
export type PromptFn<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
> = (input: z.infer<I>) => Promise<GenerateRequest<CustomOptionsSchema>>;

/**
 * Options for initializing Genkit.
 */
export interface GenkitOptions {
  /** List of plugins to load. */
  plugins?: (GenkitPlugin | GenkitPluginV2)[];
  /** Directory where dotprompts are stored. Set to `null` to disable automatic prompt loading. */
  promptDir?: string | null;
  /** Default model to use if no model is specified. */
  model?: ModelArgument<any>;
  /** Additional runtime context data for flows and tools. */
  context?: ActionContext;
  /** Display name that will be shown in developer tooling. */
  name?: string;
  /** Additional attribution information to include in the x-goog-api-client header. */
  clientHeader?: string;
}

/**
 * `Genkit` encapsulates a single Genkit instance including the {@link Registry}, {@link ReflectionServer}, {@link FlowServer}, and configuration.
 *
 * Do not instantiate this class directly. Use {@link genkit}.
 *
 * Registry keeps track of actions, flows, tools, and many other components. Reflection server exposes an API to inspect the registry and trigger executions of actions in the registry. Flow server exposes flows as HTTP endpoints for production use.
 *
 * There may be multiple Genkit instances in a single codebase.
 */
export class Genkit extends GenkitAI implements HasRegistry {
  /** Developer-configured options. */
  readonly options: GenkitOptions;
  /** Reflection server for this registry. May be null if not started. */
  private reflectionServer: ReflectionServer | null = null;
  /** List of flows that have been registered in this instance. */
  readonly flows: Action<any, any, any>[] = [];

  get apiStability() {
    return this.registry.apiStability;
  }

  constructor(options?: GenkitOptions) {
    const registry = new Registry();
    super(registry);
    this.options = options || {};
    if (this.options.context) {
      this.registry.context = this.options.context;
    }
    this.configure();
    if (isDevEnv() && !disableReflectionApi) {
      this.reflectionServer = new ReflectionServer(this.registry, {
        configuredEnvs: ['dev'],
        name: this.options.name,
      });
      this.reflectionServer.start().catch((e) => logger.error);
    }
    if (options?.clientHeader) {
      setClientHeader(options?.clientHeader);
    }
  }

  /**
   * Defines and registers a flow function.
   */
  defineFlow<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
    S extends z.ZodTypeAny = z.ZodTypeAny,
  >(
    config: FlowConfig<I, O, S> | string,
    fn: FlowFn<I, O, S>
  ): Action<I, O, S> {
    const flow = defineFlow(this.registry, config, fn);
    this.flows.push(flow);
    return flow;
  }

  /**
   * Defines and registers a bi-directional flow.
   */
  defineBidiFlow<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
    S extends z.ZodTypeAny = z.ZodTypeAny,
    Init extends z.ZodTypeAny = z.ZodTypeAny,
  >(
    config: FlowConfig<I, O, S, Init>,
    fn: (
      input: ActionFnArg<z.infer<S>, z.infer<I>, z.infer<Init>>
    ) => AsyncGenerator<z.infer<S>, z.infer<O>, void>
  ): Action<I, O, S, ActionRunOptions<z.infer<S>, z.infer<I>>, Init> {
    const flow = defineBidiFlow(this.registry, config, fn);
    this.flows.push(flow);
    return flow;
  }

  /**
   * Defines and registers a tool that can return multiple parts of content.
   *
   * Tools can be passed to models by name or value during `generate` calls to be called automatically based on the prompt and situation.
   */
  defineTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
    config: { multipart: true } & ToolConfig<I, O>,
    fn: MultipartToolFn<I, O>
  ): MultipartToolAction<I, O>;

  /**
   * Defines and registers a tool.
   *
   * Tools can be passed to models by name or value during `generate` calls to be called automatically based on the prompt and situation.
   */
  defineTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
    config: ToolConfig<I, O>,
    fn: ToolFn<I, O>
  ): ToolAction<I, O>;

  defineTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
    config: ({ multipart?: true } & ToolConfig<I, O>) | string,
    fn: ToolFn<I, O> | MultipartToolFn<I, O>
  ): ToolAction<I, O> | MultipartToolAction<I, O> {
    return defineTool(this.registry, config as any, fn as any);
  }

  /**
   * Defines a dynamic tool. Dynamic tools are just like regular tools ({@link Genkit.defineTool}) but will not be registered in the
   * Genkit registry and can be defined dynamically at runtime.
   */
  dynamicTool<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
    config: ToolConfig<I, O>,
    fn?: ToolFn<I, O>
  ): ToolAction<I, O> {
    return dynamicTool(config, fn) as ToolAction<I, O>;
  }

  /**
   * Defines and registers a dynamic action provider (e.g. mcp host)
   */
  defineDynamicActionProvider(
    config: DapConfig | string,
    fn: DapFn
  ): DynamicActionProviderAction {
    return defineDynamicActionProvider(this.registry, config, fn);
  }

  /**
   * Defines and registers a schema from a Zod schema.
   *
   * Defined schemas can be referenced by `name` in prompts in place of inline schemas.
   */
  defineSchema<T extends z.ZodTypeAny>(name: string, schema: T): T {
    return defineSchema(this.registry, name, schema);
  }

  /**
   * Defines and registers a schema from a JSON schema.
   *
   * Defined schemas can be referenced by `name` in prompts in place of inline schemas.
   */
  defineJsonSchema(name: string, jsonSchema: JSONSchema) {
    return defineJsonSchema(this.registry, name, jsonSchema);
  }

  /**
   * Defines a new model and adds it to the registry.
   */
  defineModel<CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny>(
    options: {
      apiVersion: 'v2';
    } & DefineModelOptions<CustomOptionsSchema>,
    runner: (
      request: GenerateRequest<CustomOptionsSchema>,
      options: ActionFnArg<GenerateResponseChunkData>
    ) => Promise<GenerateResponseData>
  ): ModelAction<CustomOptionsSchema>;

  /**
   * Defines a new model and adds it to the registry.
   */
  defineModel<CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny>(
    options: DefineModelOptions<CustomOptionsSchema>,
    runner: (
      request: GenerateRequest<CustomOptionsSchema>,
      streamingCallback?: StreamingCallback<GenerateResponseChunkData>
    ) => Promise<GenerateResponseData>
  ): ModelAction<CustomOptionsSchema>;

  /**
   * Defines a new model and adds it to the registry.
   */
  defineModel<CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny>(
    options: any,
    runner: (
      request: GenerateRequest<CustomOptionsSchema>,
      streamingCallback: any
    ) => Promise<GenerateResponseData>
  ): ModelAction<CustomOptionsSchema> {
    return defineModel(this.registry, options, runner);
  }

  /**
   * Defines a new background model and adds it to the registry.
   */
  defineBackgroundModel<
    CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
  >(
    options: DefineBackgroundModelOptions<CustomOptionsSchema>
  ): BackgroundModelAction<CustomOptionsSchema> {
    return defineBackgroundModel(this.registry, options);
  }

  /**
   * Looks up a prompt by `name` (and optionally `variant`). Can be used to lookup
   * .prompt files or prompts previously defined with {@link Genkit.definePrompt}
   */
  prompt<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
    CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
  >(
    name: string,
    options?: { variant?: string }
  ): ExecutablePrompt<z.infer<I>, O, CustomOptions> {
    return this.wrapExecutablePromptPromise(
      `${name}${options?.variant ? `.${options?.variant}` : ''}`,
      prompt(this.registry, name, {
        ...options,
        dir:
          this.options.promptDir === undefined
            ? './prompts'
            : this.options.promptDir,
      })
    );
  }

  private wrapExecutablePromptPromise<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
    CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
  >(
    name: string,
    promise: Promise<ExecutablePrompt<z.infer<I>, O, CustomOptions>>
  ) {
    const executablePrompt = (async (
      input?: I,
      opts?: PromptGenerateOptions<O, CustomOptions>
    ): Promise<GenerateResponse<z.infer<O>>> => {
      return (await promise)(input, opts);
    }) as ExecutablePrompt<z.infer<I>, O, CustomOptions>;

    executablePrompt.ref = { name };

    executablePrompt.render = async (
      input?: I,
      opts?: PromptGenerateOptions<O, CustomOptions>
    ): Promise<GenerateOptions<O, CustomOptions>> => {
      return (await promise).render(input, opts) as Promise<
        GenerateOptions<O, CustomOptions>
      >;
    };

    executablePrompt.stream = (
      input?: I,
      opts?: PromptGenerateOptions<O, CustomOptions>
    ): GenerateStreamResponse<O> => {
      let channel = new Channel<GenerateResponseChunk>();

      const generated = runInNewSpan(
        this.registry,
        {
          metadata: {
            name,
            input,
          },
          labels: {
            [SPAN_TYPE_ATTR]: 'dotprompt',
          },
        },
        () =>
          generate<O, CustomOptions>(
            this.registry,
            promise.then((action) =>
              action.render(input, {
                ...opts,
                onChunk: (chunk) => channel.send(chunk),
              })
            )
          )
      );
      generated.then(
        () => channel.close(),
        (err) => channel.error(err)
      );

      return {
        response: generated,
        stream: channel,
      };
    };

    executablePrompt.asTool = async (): Promise<ToolAction<I, O>> => {
      return (await promise).asTool() as Promise<ToolAction<I, O>>;
    };

    return executablePrompt;
  }

  /**
   * Defines and registers a prompt based on a function.
   *
   * This is an alternative to defining and importing a .prompt file, providing
   * the most advanced control over how the final request to the model is made.
   *
   * @param options - Prompt metadata including model, model params,
   * input/output schemas, etc
   * @param fn - A function that returns a {@link GenerateRequest}. Any config
   * parameters specified by the {@link GenerateRequest} will take precedence
   * over any parameters specified by `options`.
   *
   * ```ts
   * const hi = ai.definePrompt(
   *   {
   *     name: 'hi',
   *     input: {
   *       schema: z.object({
   *         name: z.string(),
   *       }),
   *     },
   *     config: {
   *       temperature: 1,
   *     },
   *   },
   *   async (input) => {
   *     return {
   *       messages: [ { role: 'user', content: [{ text: `hi ${input.name}` }] } ],
   *     };
   *   }
   * );
   * const { text } = await hi({ name: 'Genkit' });
   * ```
   */
  definePrompt<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
    CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
  >(
    options: PromptConfig<I, O, CustomOptions>,
    /** @deprecated use `options.messages` with a template string instead. */
    templateOrFn?: string | PromptFn<I>
  ): ExecutablePrompt<z.infer<I>, O, CustomOptions> {
    // For backwards compatibility...
    if (templateOrFn) {
      if (options.messages) {
        throw new GenkitError({
          status: 'INVALID_ARGUMENT',
          message:
            'Cannot specify template/function argument and `options.messages` at the same time',
        });
      }
      if (typeof templateOrFn === 'string') {
        return definePrompt(this.registry, {
          ...options,
          messages: templateOrFn,
        });
      } else {
        // it's the PromptFn
        return definePrompt(this.registry, {
          ...options,
          messages: async (input) => {
            const response = await (
              templateOrFn as PromptFn<z.infer<I>, CustomOptions>
            )(input);
            return response.messages;
          },
        });
      }
    }
    return definePrompt(this.registry, options);
  }

  /**
   * Creates a retriever action for the provided {@link RetrieverFn} implementation.
   */
  defineRetriever<OptionsType extends z.ZodTypeAny = z.ZodTypeAny>(
    options: {
      name: string;
      configSchema?: OptionsType;
      info?: RetrieverInfo;
    },
    runner: RetrieverFn<OptionsType>
  ): RetrieverAction<OptionsType> {
    return defineRetriever(this.registry, options, runner);
  }

  /**
   * defineSimpleRetriever makes it easy to map existing data into documents that
   * can be used for prompt augmentation.
   *
   * @param options Configuration options for the retriever.
   * @param handler A function that queries a datastore and returns items from which to extract documents.
   * @returns A Genkit retriever.
   */
  defineSimpleRetriever<C extends z.ZodTypeAny = z.ZodTypeAny, R = any>(
    options: SimpleRetrieverOptions<C, R>,
    handler: (query: Document, config: z.infer<C>) => Promise<R[]>
  ): RetrieverAction<C> {
    return defineSimpleRetriever(this.registry, options, handler);
  }

  /**
   * Creates an indexer action for the provided {@link IndexerFn} implementation.
   */
  defineIndexer<IndexerOptions extends z.ZodTypeAny>(
    options: {
      name: string;
      embedderInfo?: EmbedderInfo;
      configSchema?: IndexerOptions;
    },
    runner: IndexerFn<IndexerOptions>
  ): IndexerAction<IndexerOptions> {
    return defineIndexer(this.registry, options, runner);
  }

  /**
   * Creates evaluator action for the provided {@link EvaluatorFn} implementation.
   */
  defineEvaluator<
    DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema,
    EvalDataPoint extends
      typeof BaseEvalDataPointSchema = typeof BaseEvalDataPointSchema,
    EvaluatorOptions extends z.ZodTypeAny = z.ZodTypeAny,
  >(
    options: {
      name: string;
      displayName: string;
      definition: string;
      dataPointType?: DataPoint;
      configSchema?: EvaluatorOptions;
      isBilled?: boolean;
    },
    runner: EvaluatorFn<EvalDataPoint, EvaluatorOptions>
  ): EvaluatorAction {
    return defineEvaluator(this.registry, options, runner);
  }

  /**
   * Creates embedder model for the provided {@link EmbedderFn} model implementation.
   */
  defineEmbedder<ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny>(
    options: {
      name: string;
      configSchema?: ConfigSchema;
      info?: EmbedderInfo;
    },
    runner: EmbedderFn<ConfigSchema>
  ): EmbedderAction<ConfigSchema> {
    return defineEmbedder(this.registry, options, runner);
  }

  /**
   * create a handlebars helper (https://handlebarsjs.com/guide/block-helpers.html) to be used in dotprompt templates.
   */
  defineHelper(name: string, fn: Handlebars.HelperDelegate): void {
    defineHelper(this.registry, name, fn);
  }

  /**
   * Creates a handlebars partial (https://handlebarsjs.com/guide/partials.html) to be used in dotprompt templates.
   */
  definePartial(name: string, source: string): void {
    definePartial(this.registry, name, source);
  }

  /**
   *  Creates a reranker action for the provided {@link RerankerFn} implementation.
   */
  defineReranker<OptionsType extends z.ZodTypeAny = z.ZodTypeAny>(
    options: {
      name: string;
      configSchema?: OptionsType;
      info?: RerankerInfo;
    },
    runner: RerankerFn<OptionsType>
  ) {
    return defineReranker(this.registry, options, runner);
  }

  /**
   * Evaluates the given `dataset` using the specified `evaluator`.
   */
  evaluate<
    DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema,
    CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
  >(params: EvaluatorParams<DataPoint, CustomOptions>): Promise<EvalResponses> {
    return evaluate(this.registry, params);
  }

  /**
   * Reranks documents from a {@link RerankerArgument} based on the provided query.
   */
  rerank<CustomOptions extends z.ZodTypeAny>(
    params: RerankerParams<CustomOptions>
  ): Promise<Array<RankedDocument>> {
    return rerank(this.registry, params);
  }

  /**
   * Indexes `documents` using the provided `indexer`.
   */
  index<CustomOptions extends z.ZodTypeAny>(
    params: IndexerParams<CustomOptions>
  ): Promise<void> {
    return index(this.registry, params);
  }

  /**
   * Retrieves documents from the `retriever` based on the provided `query`.
   */
  retrieve<CustomOptions extends z.ZodTypeAny>(
    params: RetrieverParams<CustomOptions>
  ): Promise<Array<Document>> {
    return retrieve(this.registry, params);
  }

  /**
   * Configures the Genkit instance.
   */
  private configure() {
    const activeRegistry = this.registry;
    defineGenerateAction(activeRegistry);
    // install the default formats in the registry
    configureFormats(activeRegistry);
    const plugins = [...(this.options.plugins ?? [])];
    if (this.options.model) {
      this.registry.registerValue(
        'defaultModel',
        'defaultModel',
        toModelRef(this.options.model)
      );
    }
    if (this.options.promptDir !== null) {
      loadPromptFolder(
        this.registry,
        this.options.promptDir ?? './prompts',
        ''
      );
    }
    plugins.forEach((plugin) => {
      if (isPluginV2(plugin)) {
        logger.debug(`Registering v2 plugin ${plugin.name}...`);
        plugin.middleware?.()?.forEach((middleware) => {
          activeRegistry.registerValue(
            'middleware',
            middleware.name,
            middleware
          );
        });
        activeRegistry.registerPluginProvider(plugin.name, {
          name: plugin.name,
          async initializer() {
            logger.debug(`Initializing plugin ${plugin.name}:`);
            if (!plugin.init) return;
            const resolvedActions = await plugin.init();
            resolvedActions?.forEach((resolvedAction) => {
              registerActionV2(activeRegistry, resolvedAction, plugin);
            });
          },
          async resolver(action: ActionType, target: string) {
            if (!plugin.resolve) return;
            const resolvedAction = await plugin.resolve(action, target);
            if (resolvedAction) {
              registerActionV2(activeRegistry, resolvedAction, plugin);
            }
          },
          async listActions() {
            if (typeof plugin.list === 'function') {
              return (await plugin.list()).map((a) => {
                if (a.name.startsWith(`${plugin.name}/`)) {
                  return a;
                }
                return {
                  ...a,
                  // Apply namespace for v2 plugins.
                  name: `${plugin.name}/${a.name}`,
                };
              });
            }
            return [];
          },
        });
      } else {
        const loadedPlugin = (plugin as GenkitPlugin)(this);
        logger.debug(`Registering plugin ${loadedPlugin.name}...`);
        activeRegistry.registerPluginProvider(loadedPlugin.name, {
          name: loadedPlugin.name,
          async initializer() {
            logger.debug(`Initializing plugin ${loadedPlugin.name}:`);
            await loadedPlugin.initializer();
          },
          async resolver(action: ActionType, target: string) {
            if (loadedPlugin.resolver) {
              await loadedPlugin.resolver(action, target);
            }
          },
          async listActions() {
            if (loadedPlugin.listActions) {
              return await loadedPlugin.listActions();
            }
            return [];
          },
        });
      }
    });
  }

  /**
   * Stops all servers.
   */
  async stopServers() {
    await this.reflectionServer?.stop();
    this.reflectionServer = null;
  }
}

function registerActionV2(
  registry: Registry,
  resolvedAction: ResolvableAction,
  plugin: GenkitPluginV2
) {
  if (isBackgroundAction(resolvedAction)) {
    registerBackgroundAction(registry, resolvedAction, {
      namespace: plugin.name,
    });
  } else if (isAction(resolvedAction)) {
    if (!resolvedAction.__action.actionType) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: 'Action type is missing for ' + resolvedAction.__action.name,
      });
    }
    registry.registerAction(
      resolvedAction.__action.actionType,
      resolvedAction,
      { namespace: plugin.name }
    );
  } else {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'Unknown action type returned from plugin ' + plugin.name,
    });
  }
}

/**
 * Initializes Genkit with a set of options.
 *
 * This will create a new Genkit registry, register the provided plugins, stores, and other configuration. This
 * should be called before any flows are registered.
 */
export function genkit(options: GenkitOptions): Genkit {
  return new Genkit(options);
}

const shutdown = async () => {
  logger.debug('Shutting down all Genkit servers...');
  await ReflectionServer.stopAll();
  process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

let disableReflectionApi = false;

export function __disableReflectionApi() {
  disableReflectionApi = true;
}

/** Helper method to map ModelArgument to ModelReference */
function toModelRef(
  modelArg: ModelArgument<any> | undefined
): ModelReference<any> | undefined {
  if (modelArg === undefined) {
    return undefined;
  }
  if (typeof modelArg === 'string') {
    return modelRef({ name: modelArg });
  }
  if ((modelArg as ModelReference<any>).name) {
    return modelArg as ModelReference<any>;
  }
  const modelAction = modelArg as ModelAction;
  return modelRef({
    name: modelAction.__action.name,
  });
}
