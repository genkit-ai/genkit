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
import { TRPCError, initTRPC } from '@trpc/server';
import { z } from 'zod';
import {
  getDatasetStore,
  getEvalStore,
  runNewEvaluation,
  validateSchema,
} from '../eval';
import type { BaseRuntimeManager } from '../manager/manager';
import { AppProcessStatus } from '../manager/process-manager';
import { GenkitToolsError, type RuntimeInfo } from '../manager/types';
import { TraceDataSchema } from '../types';
import type { Action } from '../types/action';
import * as apis from '../types/apis';
import { CancelActionRequestSchema } from '../types/apis';
import type { EnvironmentVariable } from '../types/env';
import * as evals from '../types/eval';
import type { PromptFrontmatter } from '../types/prompt';
import { logger } from '../utils';
import { PageViewEvent, ToolsRequestEvent, record } from '../utils/analytics';
import { toolsPackage } from '../utils/package';
import { fromMessages } from '../utils/prompt';

const t = initTRPC.create({
  errorFormatter(opts) {
    const { shape, error } = opts;
    if (error.cause instanceof GenkitToolsError && error.cause.data) {
      return {
        ...shape,
        data: {
          ...shape.data,
          genkitErrorMessage: error.cause.data.message,
          genkitErrorDetails: error.cause.data.details,
        },
      };
    }
    return shape;
  },
});

const analyticsEventForRoute = (
  path: string,
  input: unknown,
  durationMs: number,
  status: string
) => {
  const event = new ToolsRequestEvent(path);
  event.duration = durationMs;
  event.parameters = {
    ...event.parameters,
    status,
  };

  switch (path) {
    case 'runAction':
      // set action type (flow, model, etc...)
      const splits = (input as apis.RunActionRequest).key?.split('/');
      event.parameters = {
        ...event.parameters,
        action: splits.length > 1 ? splits[1] : 'unknown',
      };
      break;
    default:
    // do nothing
  }

  return event;
};

const parseEnv = (environ: NodeJS.ProcessEnv): EnvironmentVariable[] => {
  const environmentVars: EnvironmentVariable[] = [];

  Object.entries(environ)
    .sort((a, b) => {
      // sort by name
      if (a[0] < b[0]) {
        return -1;
      }
      if (a[0] > b[0]) {
        return 1;
      }
      return 0;
    })
    .forEach(([name, value]) => {
      environmentVars.push({ name, value: value || '' });
    });

  return environmentVars;
};

/** Base handler that will send an analytics event */
const loggedProcedure = t.procedure.use(async (opts) => {
  const start = Date.now();
  const result = await opts.next();
  const durationMs = Date.now() - start;

  const analyticsEvent = analyticsEventForRoute(
    opts.path,
    opts.rawInput,
    durationMs,
    result.ok ? 'success' : 'failure'
  );

  // fire-and-forget
  record(analyticsEvent).catch((err) => {
    logger.error(`Failed to send analytics ${err}`);
  });

  return result;
});

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type
export const TOOLS_SERVER_ROUTER = (manager: BaseRuntimeManager) =>
  t.router({
    /** Retrieves all runnable actions. */
    listActions: loggedProcedure
      .input(apis.ListActionsRequestSchema)
      .query(async ({ input }): Promise<Record<string, Action>> => {
        return manager.listActions(input);
      }),

    /** Retrieves all values. */
    listValues: loggedProcedure
      .input(apis.ListValuesRequestSchema)
      .query(async ({ input }): Promise<Record<string, unknown>> => {
        return manager.listValues(input);
      }),

    /** Generate a .prompt file from messages and model config. */
    createPrompt: loggedProcedure
      .input(apis.CreatePromptRequestSchema)
      .mutation(async ({ input }) => {
        const frontmatter: PromptFrontmatter = {
          model: input.model.replace('/model/', ''),
          config: input.config,
          tools: input.tools?.map((toolDefinition) => toolDefinition.name),
        };
        return fromMessages(frontmatter, input.messages);
      }),

    /** Retrieves all traces for a given environment (e.g. dev or prod). */
    listTraces: loggedProcedure
      .input(apis.ListTracesRequestSchema)
      .query(async ({ input }) => {
        return manager.listTraces(input);
      }),

    /** Retrieves a trace for a given ID. */
    getTrace: loggedProcedure
      .input(apis.GetTraceRequestSchema)
      .query(async ({ input }) => {
        return manager.getTrace(input);
      }),

    /** Adds a trace to the trace store */
    addTrace: loggedProcedure
      .input(TraceDataSchema)
      .output(z.void())
      .mutation(async ({ input }) => {
        return manager.addTrace(input);
      }),

    /** Retrieves all eval run keys */
    listEvalRunKeys: loggedProcedure
      .input(apis.ListEvalKeysRequestSchema)
      .output(apis.ListEvalKeysResponseSchema)
      .query(async ({ input }) => {
        const store = await getEvalStore();
        const response = await store.list(input);
        return {
          evalRunKeys: response.evalRunKeys,
        };
      }),

    /** Retrieves a single eval run by ID */
    getEvalRun: loggedProcedure
      .input(apis.GetEvalRunRequestSchema)
      .output(evals.EvalRunSchema)
      .query(async ({ input }) => {
        const parts = input.name.split('/');
        const evalRunId = parts[1];
        const store = await getEvalStore();
        const evalRun = await store.load(evalRunId);
        if (!evalRun) {
          throw new TRPCError({
            code: 'NOT_FOUND',
            message: `Eval run with ${input.name} not found`,
          });
        }
        return evalRun;
      }),

    /** Deletes a single eval run by ID */
    deleteEvalRun: loggedProcedure
      .input(apis.DeleteEvalRunRequestSchema)
      .mutation(async ({ input }) => {
        const parts = input.name.split('/');
        const evalRunId = parts[1];
        const store = await getEvalStore();
        await store.delete(evalRunId);
      }),

    /** Retrieves all eval datasets */
    listDatasets: loggedProcedure
      .output(z.array(evals.DatasetMetadataSchema))
      .query(async () => {
        const response = await getDatasetStore().listDatasets();
        return response;
      }),

    /** Retrieves an existing dataset */
    getDataset: loggedProcedure
      .input(z.string())
      .output(evals.DatasetSchema)
      .query(async ({ input }) => {
        const response = await getDatasetStore().getDataset(input);
        return response;
      }),

    /** Creates a new dataset */
    createDataset: loggedProcedure
      .input(apis.CreateDatasetRequestSchema)
      .output(evals.DatasetMetadataSchema)
      .mutation(async ({ input }) => {
        const response = await getDatasetStore().createDataset(input);
        return response;
      }),

    /** Updates an exsting dataset */
    updateDataset: loggedProcedure
      .input(apis.UpdateDatasetRequestSchema)
      .output(evals.DatasetMetadataSchema)
      .mutation(async ({ input }) => {
        const response = await getDatasetStore().updateDataset(input);
        return response;
      }),

    /** Deletes an exsting dataset */
    deleteDataset: loggedProcedure
      .input(z.string())
      .output(z.void())
      .mutation(async ({ input }) => {
        const response = await getDatasetStore().deleteDataset(input);
        return response;
      }),

    /** Start new evaluation run */
    runNewEvaluation: loggedProcedure
      .input(apis.RunNewEvaluationRequestSchema)
      .output(evals.EvalRunKeySchema)
      .mutation(async ({ input }) => {
        const response = await runNewEvaluation(manager, input);
        return response;
      }),

    /** Validate given data against a target action schema */
    validateDatasetSchema: loggedProcedure
      .input(apis.ValidateDataRequestSchema)
      .output(apis.ValidateDataResponseSchema)
      .mutation(async ({ input }) => {
        const response = await validateSchema(manager, input);
        return response;
      }),

    /** Send a screen view analytics event */
    sendPageView: t.procedure
      .input(apis.PageViewSchema)
      .query(async ({ input }) => {
        await record(new PageViewEvent(input.pageTitle));
      }),

    /** Genkit Environment Information */
    getGenkitEnvironment: t.procedure.query(() => {
      return {
        cliPackageVersion: toolsPackage.version,
        //TODO(michaeldoyle): packageVersion: ???,
        environmentVars: parseEnv(process.env),
      };
    }),

    /**
     * Get the current active Genkit Runtime.
     *
     * Currently used by the Dev UI to "poll", since IDX cannot support SSE at
     * this time.
     */
    getCurrentRuntime: t.procedure.query(() => {
      return manager.getMostRecentRuntime() ?? ({} as RuntimeInfo);
    }),

    /**
     * Get all active Genkit Runtimes.
     *
     * Currently used by the Dev UI to "poll", since IDX cannot support SSE at
     * this time.
     */
    getActiveRuntimes: t.procedure.query(() => {
      return manager.listRuntimes();
    }),

    getAppProcessStatus: t.procedure.query((): AppProcessStatus => {
      if (!manager.processManager) {
        return { status: 'unconfigured' };
      }
      return manager.processManager.status();
    }),

    restartAppProcess: t.procedure.query(async () => {
      await manager.processManager?.restart();
      return true;
    }),

    killAppProcess: t.procedure.query(async () => {
      await manager.processManager?.kill();
      return true;
    }),

    /** Cancels a long-running action. */
    cancelAction: loggedProcedure
      .input(CancelActionRequestSchema)
      .mutation(async ({ input }) => {
        return manager.cancelAction(input);
      }),
  });

export type ToolsServerRouter = ReturnType<typeof TOOLS_SERVER_ROUTER>;
