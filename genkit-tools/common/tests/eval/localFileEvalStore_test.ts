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

import fs from 'fs';
import * as readline from 'readline';
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type Mock,
} from 'vitest';
import { LocalFileEvalStore } from '../../src/eval/localFileEvalStore';
import {
  EvalRunSchema,
  type EvalResult,
  type EvalRun,
  type EvalRunKey,
  type EvalStore,
} from '../../src/types/eval';

// Mock modules
vi.mock('fs/promises', () => ({
  get writeFile() {
    return fs.promises.writeFile;
  },
  get readFile() {
    return fs.promises.readFile;
  },
  get unlink() {
    return fs.promises.unlink;
  },
}));
vi.mock('readline');
vi.mock('process', () => ({
  cwd: vi.fn(() => 'store-root'),
}));

// Test Data
const EVAL_RESULTS: EvalResult[] = [
  {
    testCaseId: 'alakjdshfalsdkjh',
    input: { subject: 'Kermit the Frog', style: 'Jerry Seinfeld' },
    output: '...Kermit output...',
    context: ['...Kermit context...'],
    metrics: [{ evaluator: 'faithfulness', score: 0.5, rationale: '...' }],
    traceIds: [],
  },
  {
    testCaseId: 'lkjhasdfkljahsdf',
    input: { subject: "Doctor's office", style: 'Wanda Sykes' },
    output: '...Doctor output...',
    context: [],
    metrics: [{ evaluator: 'faithfulness', score: 0, rationale: '...' }],
    traceIds: [],
  },
];

const METRICS_METADATA = {
  faithfulness: {
    displayName: 'Faithfulness',
    definition: 'Faithfulness definition',
  },
};

const EVAL_RUN_WITH_ACTION = EvalRunSchema.parse({
  key: {
    actionRef: 'flow/tellMeAJoke',
    evalRunId: 'abc1234',
    createdAt: new Date().toISOString(),
  },
  results: EVAL_RESULTS,
  metricMetadata: METRICS_METADATA,
});

const EVAL_RUN_WITHOUT_ACTION = EvalRunSchema.parse({
  key: {
    evalRunId: 'def456',
    createdAt: new Date().toISOString(),
  },
  results: EVAL_RESULTS,
  metricMetadata: METRICS_METADATA,
});

const ALL_EVAL_RUN_KEYS = {
  [EVAL_RUN_WITH_ACTION.key.evalRunId]: EVAL_RUN_WITH_ACTION.key,
  [EVAL_RUN_WITHOUT_ACTION.key.evalRunId]: EVAL_RUN_WITHOUT_ACTION.key,
};

// Mock Helpers
const mockFsExists = (exists: (path: fs.PathLike) => boolean) => {
  fs.existsSync = vi
    .fn<(path: fs.PathLike) => boolean>()
    .mockImplementation(exists);
};

const mockReadlineForMigration = (keys: EvalRunKey[]) => {
  (readline.createInterface as Mock).mockReturnValue({
    [Symbol.asyncIterator]: async function* () {
      for (const key of keys) {
        yield JSON.stringify(key);
      }
    },
  } as any);
};

const mockIndexFile = (keys: Record<string, EvalRunKey>) => {
  fs.promises.readFile = vi.fn<any>().mockResolvedValue(JSON.stringify(keys));
};

const mockEvalRunFile = (evalRun: EvalRun) => {
  fs.promises.readFile = vi
    .fn<any>()
    .mockImplementation(async () => JSON.stringify(evalRun));
};

describe('localFileEvalStore', () => {
  let evalStore: EvalStore;

  beforeEach(async () => {
    LocalFileEvalStore.reset();
    vi.clearAllMocks();

    // Setup default mocks for a clean state before each test
    mockFsExists(() => false);
    fs.promises.writeFile = vi.fn<any>().mockResolvedValue(undefined);
    fs.promises.unlink = vi.fn<any>().mockResolvedValue(undefined);
    fs.writeFileSync = vi.fn<any>().mockImplementation(() => {});

    evalStore = await LocalFileEvalStore.getEvalStore();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('initialization', () => {
    it('uses json file by default when no index exists', async () => {
      expect(fs.writeFileSync).toHaveBeenCalledWith(
        expect.stringContaining('evals/index.json'),
        JSON.stringify({})
      );
    });

    it('migrates to json file if txt index is found', async () => {
      LocalFileEvalStore.reset();
      mockFsExists((path) => path.toString().endsWith('index.txt'));
      fs.createReadStream = vi.fn<any>().mockReturnValue({} as any);
      mockReadlineForMigration([EVAL_RUN_WITH_ACTION.key]);

      await LocalFileEvalStore.getEvalStore();

      expect(fs.promises.unlink).toHaveBeenCalledWith(
        expect.stringContaining('index.txt')
      );
      const expectedIndex = {
        [EVAL_RUN_WITH_ACTION.key.evalRunId]: EVAL_RUN_WITH_ACTION.key,
      };
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        expect.stringContaining('evals/index.json'),
        JSON.stringify(expectedIndex, null, 2)
      );
    });
  });

  describe('save', () => {
    const testSave = async (evalRun: EvalRun) => {
      await evalStore.save(evalRun);

      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        expect.stringContaining(`evals/${evalRun.key.evalRunId}.json`),
        JSON.stringify(evalRun)
      );

      const expectedIndex = { [evalRun.key.evalRunId]: evalRun.key };
      expect(fs.promises.writeFile).toHaveBeenNthCalledWith(
        2,
        expect.stringContaining('evals/index.json'),
        JSON.stringify(expectedIndex, null, 2)
      );
    };

    it('writes and updates index for eval run with actionId', async () => {
      await testSave(EVAL_RUN_WITH_ACTION);
    });

    it('persists a new evalRun file without an actionId', async () => {
      await testSave(EVAL_RUN_WITHOUT_ACTION);
    });
  });

  describe('load', () => {
    it('fetches an evalRun file by id', async () => {
      mockFsExists(() => true);
      mockEvalRunFile(EVAL_RUN_WITH_ACTION);

      const fetchedEvalRun = await evalStore.load(
        EVAL_RUN_WITH_ACTION.key.evalRunId
      );

      expect(fetchedEvalRun).toEqual(EVAL_RUN_WITH_ACTION);
    });

    it('returns undefined if file does not exist', async () => {
      mockFsExists(() => false);

      const fetchedEvalRun = await evalStore.load('non-existent-id');

      expect(fetchedEvalRun).toBeUndefined();
    });
  });

  describe('delete', () => {
    it('deletes an evalRun file and updates the index', async () => {
      mockFsExists(() => true);
      mockIndexFile(ALL_EVAL_RUN_KEYS);

      await evalStore.delete(EVAL_RUN_WITH_ACTION.key.evalRunId);

      expect(fs.promises.unlink).toHaveBeenCalledWith(
        expect.stringContaining(EVAL_RUN_WITH_ACTION.key.evalRunId)
      );

      const expectedIndex = {
        [EVAL_RUN_WITHOUT_ACTION.key.evalRunId]: EVAL_RUN_WITHOUT_ACTION.key,
      };
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        expect.stringContaining('evals/index.json'),
        JSON.stringify(expectedIndex, null, 2)
      );
    });

    it('does not throw or unlink if file does not exist', async () => {
      mockFsExists(() => false);

      await evalStore.delete(EVAL_RUN_WITH_ACTION.key.evalRunId);

      expect(fs.promises.unlink).not.toHaveBeenCalled();
    });
  });

  describe('list', () => {
    it('lists all evalRun keys from the index', async () => {
      mockFsExists(() => true);
      mockIndexFile(ALL_EVAL_RUN_KEYS);

      const { evalRunKeys } = await evalStore.list();

      expect(evalRunKeys).toHaveLength(2);
      expect(evalRunKeys).toContainEqual(EVAL_RUN_WITH_ACTION.key);
      expect(evalRunKeys).toContainEqual(EVAL_RUN_WITHOUT_ACTION.key);
    });

    it('filters evalRun keys by actionRef', async () => {
      mockFsExists(() => true);
      mockIndexFile(ALL_EVAL_RUN_KEYS);

      const { evalRunKeys } = await evalStore.list({
        filter: { actionRef: 'flow/tellMeAJoke' },
      });

      expect(evalRunKeys).toHaveLength(1);
      expect(evalRunKeys[0]).toEqual(EVAL_RUN_WITH_ACTION.key);
    });

    it('returns an empty array if the index is empty', async () => {
      mockIndexFile({});

      const { evalRunKeys } = await evalStore.list();

      expect(evalRunKeys).toEqual([]);
    });
  });
});
