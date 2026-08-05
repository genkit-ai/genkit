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
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import fs from 'fs';
import * as uuid from 'uuid';
import { LocalFileDatasetStore } from '../../src/eval/localFileDatasetStore';
import {
  CreateDatasetRequestSchema,
  UpdateDatasetRequestSchema,
} from '../../src/types/apis';
import type {
  Dataset,
  DatasetMetadata,
  DatasetStore,
} from '../../src/types/eval';

const FAKE_TIME = new Date('2024-02-03T12:05:33.243Z');

const SAMPLE_DATASET_1_V1 = [
  {
    input: 'Cats are evil',
    reference: 'Sorry no reference',
  },
  {
    input: 'Dogs are beautiful',
  },
];

const SAMPLE_DATASET_1_WITH_IDS = [
  {
    testCaseId: '1',
    input: 'Cats are evil',
    reference: 'Sorry no reference',
  },
  {
    testCaseId: '2',
    input: 'Dogs are beautiful',
  },
];

const SAMPLE_DATASET_1_V2 = [
  {
    input: 'Cats are evil',
    reference: 'Sorry no reference',
  },
  {
    input: 'Dogs are angels',
  },
  {
    input: 'Dogs are also super cute',
  },
];

const SAMPLE_DATASET_ID_1 = 'dataset-1-123456';

const SAMPLE_DATASET_METADATA_1_V1: DatasetMetadata = {
  datasetId: SAMPLE_DATASET_ID_1,
  metricRefs: [],
  size: 2,
  version: 1,
  datasetType: 'UNKNOWN',
  createTime: FAKE_TIME.toString(),
  updateTime: FAKE_TIME.toString(),
};
const SAMPLE_DATASET_METADATA_1_V2: DatasetMetadata = {
  datasetId: SAMPLE_DATASET_ID_1,
  size: 3,
  version: 2,
  datasetType: 'UNKNOWN',
  metricRefs: [],
  createTime: FAKE_TIME.toString(),
  updateTime: FAKE_TIME.toString(),
};

const CREATE_DATASET_REQUEST = CreateDatasetRequestSchema.parse({
  data: SAMPLE_DATASET_1_V1,
  datasetType: 'UNKNOWN',
});

const CREATE_DATASET_REQUEST_WITH_SCHEMA = CreateDatasetRequestSchema.parse({
  data: SAMPLE_DATASET_1_V1,
  datasetType: 'UNKNOWN',
  schema: {
    inputSchema: {
      type: 'string',
      $schema: 'http://json-schema.org/draft-07/schema#',
    },
    referenceSchema: {
      type: 'number',
      $schema: 'http://json-schema.org/draft-07/schema#',
    },
  },
  targetAction: '/flow/my-flow',
});

const UPDATE_DATASET_REQUEST = UpdateDatasetRequestSchema.parse({
  data: SAMPLE_DATASET_1_V2,
  datasetId: SAMPLE_DATASET_ID_1,
});

const SAMPLE_DATASET_ID_2 = 'dataset-2-123456';

const SAMPLE_DATASET_METADATA_2: DatasetMetadata = {
  datasetId: SAMPLE_DATASET_ID_2,
  metricRefs: [],
  size: 5,
  version: 1,
  datasetType: 'FLOW',
  createTime: FAKE_TIME.toString(),
  updateTime: FAKE_TIME.toString(),
};

jest.mock('process', () => {
  return {
    cwd: jest.fn(() => 'store-root'),
  };
});

jest.mock('uuid');
const uuidSpy = jest.spyOn(uuid, 'v4');
const TEST_CASE_ID = 'test-case-1234-1234-1234';
jest.mock('../../src/utils', () => ({
  generateTestCaseId: jest.fn(() => TEST_CASE_ID),
}));

jest.useFakeTimers({ advanceTimers: true });
jest.setSystemTime(FAKE_TIME);

describe('localFileDatasetStore', () => {
  let DatasetStore: DatasetStore;

  beforeEach(() => {
    // For storeRoot setup
    fs.existsSync = jest.fn(() => true);
    uuidSpy.mockReturnValueOnce('12345678');
    LocalFileDatasetStore.reset();
    DatasetStore = LocalFileDatasetStore.getDatasetStore() as DatasetStore;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('createDataset', () => {
    it('writes and updates index for new dataset', async () => {
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify({}) as any)
      );
      fs.existsSync = jest.fn(() => false);
      const dataset: Dataset = SAMPLE_DATASET_1_V1.map((s) => ({
        testCaseId: TEST_CASE_ID,
        ...s,
      }));

      const datasetMetadata = await DatasetStore.createDataset({
        ...CREATE_DATASET_REQUEST,
        datasetId: SAMPLE_DATASET_ID_1,
      });

      expect(fs.promises.writeFile).toHaveBeenCalledTimes(2);
      expect(fs.promises.writeFile).toHaveBeenNthCalledWith(
        1,
        expect.stringContaining(`datasets/${SAMPLE_DATASET_ID_1}.json`),
        JSON.stringify(dataset)
      );
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: SAMPLE_DATASET_METADATA_1_V1,
      };
      expect(fs.promises.writeFile).toHaveBeenNthCalledWith(
        2,
        expect.stringContaining('datasets/index.json'),
        JSON.stringify(metadataMap)
      );
      expect(datasetMetadata).toMatchObject(SAMPLE_DATASET_METADATA_1_V1);
    });

    it('writes and updates index for new dataset, testCaseIds are provided', async () => {
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify({}) as any)
      );
      fs.existsSync = jest.fn(() => false);
      const dataset: Dataset = SAMPLE_DATASET_1_V1.map((s, i) => ({
        testCaseId: TEST_CASE_ID + `index${i}`,
        ...s,
      }));

      const datasetMetadata = await DatasetStore.createDataset({
        ...CREATE_DATASET_REQUEST,
        data: dataset,
        datasetId: SAMPLE_DATASET_ID_1,
      });

      expect(fs.promises.writeFile).toHaveBeenCalledTimes(2);
      expect(fs.promises.writeFile).toHaveBeenNthCalledWith(
        1,
        expect.stringContaining(`datasets/${SAMPLE_DATASET_ID_1}.json`),
        JSON.stringify(dataset)
      );
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: SAMPLE_DATASET_METADATA_1_V1,
      };
      expect(fs.promises.writeFile).toHaveBeenNthCalledWith(
        2,
        expect.stringContaining('datasets/index.json'),
        JSON.stringify(metadataMap)
      );
      expect(datasetMetadata).toMatchObject(SAMPLE_DATASET_METADATA_1_V1);
    });

    it('creates new dataset, with schema', async () => {
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify({}) as any)
      );
      fs.existsSync = jest.fn(() => false);
      const dataset: Dataset = SAMPLE_DATASET_1_V1.map((s) => ({
        testCaseId: TEST_CASE_ID,
        ...s,
      }));

      const datasetMetadata = await DatasetStore.createDataset({
        ...CREATE_DATASET_REQUEST_WITH_SCHEMA,
        datasetId: SAMPLE_DATASET_ID_1,
      });

      expect(datasetMetadata.schema).toMatchObject({
        inputSchema: {
          type: 'string',
          $schema: 'http://json-schema.org/draft-07/schema#',
        },
        referenceSchema: {
          type: 'number',
          $schema: 'http://json-schema.org/draft-07/schema#',
        },
      });
      expect(datasetMetadata.targetAction).toEqual('/flow/my-flow');
    });

    it('creates new dataset, with metrics', async () => {
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify({}) as any)
      );
      fs.existsSync = jest.fn(() => false);
      const dataset: Dataset = SAMPLE_DATASET_1_V1.map((s) => ({
        testCaseId: TEST_CASE_ID,
        ...s,
      }));

      const datasetMetadata = await DatasetStore.createDataset({
        ...CREATE_DATASET_REQUEST,
        metricRefs: ['metric1', 'metric2'],
        datasetId: SAMPLE_DATASET_ID_1,
      });

      expect(datasetMetadata.metricRefs.sort()).toEqual(
        ['metric1', 'metric2'].sort()
      );
    });

    it('fails request if dataset already exists', async () => {
      fs.existsSync = jest.fn(() => true);

      expect(async () => {
        await DatasetStore.createDataset(CREATE_DATASET_REQUEST);
      }).rejects.toThrow();

      expect(fs.promises.writeFile).toBeCalledTimes(0);
    });

    it('fails if datasetId is invalid', async () => {
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify({}) as any)
      );
      fs.existsSync = jest.fn(() => false);

      expect(async () => {
        await DatasetStore.createDataset({
          ...CREATE_DATASET_REQUEST,
          datasetId: 'ultracool!@#$@#$%%%!#$%datasetid',
        });
      }).rejects.toThrow('Invalid datasetId');
    });

    it('does not fail if datasetId starts with capitals', async () => {
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify({}) as any)
      );
      fs.existsSync = jest.fn(() => false);

      expect(async () => {
        await DatasetStore.createDataset({
          ...CREATE_DATASET_REQUEST,
          datasetId: 'UltracoolId',
        });
      }).not.toThrow();
    });
  });

  describe('updateDataset', () => {
    it('succeeds for existing dataset -- append', async () => {
      fs.existsSync = jest.fn(() => true);
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: SAMPLE_DATASET_METADATA_1_V1,
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));
      const dataset: Dataset = [...SAMPLE_DATASET_1_WITH_IDS];
      const getDatasetSpy = jest
        .spyOn(DatasetStore, 'getDataset')
        .mockImplementation(() => Promise.resolve(dataset));

      const datasetMetadata = await DatasetStore.updateDataset({
        data: [
          {
            input: 'A new information on cat dog',
          },
        ],
        datasetId: SAMPLE_DATASET_ID_1,
      });

      expect(fs.promises.writeFile).toHaveBeenCalledTimes(2);
      expect(fs.promises.writeFile).toHaveBeenNthCalledWith(
        1,
        expect.stringContaining(`datasets/${SAMPLE_DATASET_ID_1}.json`),
        JSON.stringify([
          ...dataset,
          {
            testCaseId: TEST_CASE_ID,
            input: 'A new information on cat dog',
          },
        ])
      );
      const updatedMetadataMap = {
        [SAMPLE_DATASET_ID_1]: SAMPLE_DATASET_METADATA_1_V2,
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      expect(fs.promises.writeFile).toHaveBeenNthCalledWith(
        2,
        expect.stringContaining('datasets/index.json'),
        JSON.stringify(updatedMetadataMap)
      );
      expect(getDatasetSpy).toHaveBeenCalledTimes(1);
      expect(datasetMetadata).toMatchObject(SAMPLE_DATASET_METADATA_1_V2);
    });

    it('succeeds for existing dataset -- append and replace', async () => {
      fs.existsSync = jest.fn(() => true);
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: SAMPLE_DATASET_METADATA_1_V1,
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));
      const dataset: Dataset = [...SAMPLE_DATASET_1_WITH_IDS];
      const getDatasetSpy = jest
        .spyOn(DatasetStore, 'getDataset')
        .mockImplementation(() => Promise.resolve(dataset));

      const datasetMetadata = await DatasetStore.updateDataset({
        data: [
          {
            input: 'A new information on cat dog',
          },
          {
            testCaseId: '1',
            input: 'Other information on hot dog',
          },
        ],
        datasetId: SAMPLE_DATASET_ID_1,
      });

      expect(fs.promises.writeFile).toHaveBeenCalledTimes(2);
      expect(fs.promises.writeFile).toHaveBeenNthCalledWith(
        1,
        expect.stringContaining(`datasets/${SAMPLE_DATASET_ID_1}.json`),
        JSON.stringify([
          {
            testCaseId: '1',
            input: 'Other information on hot dog',
          },
          {
            testCaseId: '2',
            input: 'Dogs are beautiful',
          },
          {
            testCaseId: TEST_CASE_ID,
            input: 'A new information on cat dog',
          },
        ])
      );
      const updatedMetadataMap = {
        [SAMPLE_DATASET_ID_1]: SAMPLE_DATASET_METADATA_1_V2,
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      expect(fs.promises.writeFile).toHaveBeenNthCalledWith(
        2,
        expect.stringContaining('datasets/index.json'),
        JSON.stringify(updatedMetadataMap)
      );
      expect(getDatasetSpy).toHaveBeenCalledTimes(1);
      expect(datasetMetadata).toMatchObject(SAMPLE_DATASET_METADATA_1_V2);
    });

    it('succeeds for existing dataset -- with schema', async () => {
      fs.existsSync = jest.fn(() => true);
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: SAMPLE_DATASET_METADATA_1_V1,
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));

      const datasetMetadata = await DatasetStore.updateDataset({
        datasetId: SAMPLE_DATASET_ID_1,
        schema: {
          inputSchema: {
            type: 'string',
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
        },
        targetAction: '/flow/my-flow-2',
      });

      expect(datasetMetadata.schema).toMatchObject({
        inputSchema: {
          type: 'string',
          $schema: 'http://json-schema.org/draft-07/schema#',
        },
      });
      expect(datasetMetadata.targetAction).toEqual('/flow/my-flow-2');
    });

    it('succeeds for existing dataset -- with metrics', async () => {
      fs.existsSync = jest.fn(() => true);
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: SAMPLE_DATASET_METADATA_1_V1,
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));

      const datasetMetadata = await DatasetStore.updateDataset({
        datasetId: SAMPLE_DATASET_ID_1,
        metricRefs: ['metric1', 'metric2'],
      });

      expect(datasetMetadata.metricRefs.sort()).toEqual(
        ['metric1', 'metric2'].sort()
      );
    });

    it('succeeds for existing dataset -- with metrics override', async () => {
      fs.existsSync = jest.fn(() => true);
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: {
          ...SAMPLE_DATASET_METADATA_1_V1,
          metricRefs: ['metric1', 'metric2'],
        },
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));

      const datasetMetadata = await DatasetStore.updateDataset({
        datasetId: SAMPLE_DATASET_ID_1,
        metricRefs: ['metricA', 'metricB'],
      });

      expect(datasetMetadata.metricRefs.sort()).toEqual(
        ['metricA', 'metricB'].sort()
      );
    });

    it('succeeds for existing dataset -- with metrics unset', async () => {
      fs.existsSync = jest.fn(() => true);
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: {
          ...SAMPLE_DATASET_METADATA_1_V1,
          metricRefs: ['metric1', 'metric2'],
        },
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));

      const datasetMetadata = await DatasetStore.updateDataset({
        datasetId: SAMPLE_DATASET_ID_1,
      });

      expect(datasetMetadata.metricRefs.sort()).toEqual(
        ['metric1', 'metric2'].sort()
      );
    });

    it('succeeds for existing dataset -- with metrics reset', async () => {
      fs.existsSync = jest.fn(() => true);
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: {
          ...SAMPLE_DATASET_METADATA_1_V1,
          metricRefs: ['metric1', 'metric2'],
        },
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      // For index file reads
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.appendFile = jest.fn(async () => Promise.resolve(undefined));

      const datasetMetadata = await DatasetStore.updateDataset({
        datasetId: SAMPLE_DATASET_ID_1,
        metricRefs: [],
      });

      expect(datasetMetadata.metricRefs).toEqual([]);
    });

    it('fails for non existing dataset', async () => {
      fs.existsSync = jest.fn(() => false);

      expect(async () => {
        await DatasetStore.updateDataset(UPDATE_DATASET_REQUEST);
      }).rejects.toThrow();

      expect(fs.promises.writeFile).toBeCalledTimes(0);
    });
  });

  describe('listDatasets', () => {
    it('succeeds for zero datasets', async () => {
      fs.existsSync = jest.fn(() => false);

      const metadatas = await DatasetStore.listDatasets();

      expect(metadatas).toMatchObject([]);
    });

    it('succeeds for existing datasets', async () => {
      fs.existsSync = jest.fn(() => true);
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: SAMPLE_DATASET_METADATA_1_V1,
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );

      const metadatas = await DatasetStore.listDatasets();

      expect(metadatas).toMatchObject([
        SAMPLE_DATASET_METADATA_1_V1,
        SAMPLE_DATASET_METADATA_2,
      ]);
    });
  });

  describe('getDataset', () => {
    it('succeeds for existing dataset', async () => {
      const dataset: Dataset = SAMPLE_DATASET_1_V1.map((s) => ({
        ...s,
        testCaseId: 'id-x',
      }));
      fs.existsSync = jest.fn(() => true);
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(dataset) as any)
      );

      const fetchedDataset = await DatasetStore.getDataset(SAMPLE_DATASET_ID_1);

      expect(fetchedDataset).toMatchObject(dataset);
    });

    it('fails for non existing dataset', async () => {
      // TODO: Implement this.
    });
  });

  describe('deleteDataset', () => {
    it('deletes dataset and updates index', async () => {
      fs.promises.rm = jest.fn(async () => Promise.resolve());
      const metadataMap = {
        [SAMPLE_DATASET_ID_1]: SAMPLE_DATASET_METADATA_1_V1,
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );

      await DatasetStore.deleteDataset(SAMPLE_DATASET_ID_1);

      expect(fs.promises.rm).toHaveBeenCalledWith(
        expect.stringContaining(`datasets/${SAMPLE_DATASET_ID_1}.json`)
      );
      const updatedMetadataMap = {
        [SAMPLE_DATASET_ID_2]: SAMPLE_DATASET_METADATA_2,
      };
      expect(fs.promises.writeFile).toHaveBeenCalledWith(
        expect.stringContaining('datasets/index.json'),
        JSON.stringify(updatedMetadataMap)
      );
    });
  });

  describe('path traversal protection', () => {
    const MALICIOUS_IDS = [
      '../evil',
      '../../etc/passwd',
      '/etc/passwd',
      '..',
      '.',
      'foo/bar',
      'foo\\bar',
      'foo\0bar',
      '',
    ];

    beforeEach(() => {
      // Pretend every file exists so that, absent the guard, the operation
      // would proceed to touch the filesystem.
      fs.existsSync = jest.fn(() => true);
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify({}) as any)
      );
      fs.promises.writeFile = jest.fn(async () => Promise.resolve(undefined));
      fs.promises.rm = jest.fn(async () => Promise.resolve());
    });

    for (const maliciousId of MALICIOUS_IDS) {
      it(`rejects getDataset for id ${JSON.stringify(maliciousId)}`, async () => {
        await expect(DatasetStore.getDataset(maliciousId)).rejects.toThrow(
          'Invalid dataset id'
        );
        expect(fs.promises.readFile).not.toHaveBeenCalled();
      });

      it(`rejects deleteDataset for id ${JSON.stringify(maliciousId)}`, async () => {
        await expect(DatasetStore.deleteDataset(maliciousId)).rejects.toThrow(
          'Invalid dataset id'
        );
        expect(fs.promises.rm).not.toHaveBeenCalled();
      });

      it(`rejects updateDataset for id ${JSON.stringify(maliciousId)}`, async () => {
        await expect(
          DatasetStore.updateDataset({
            datasetId: maliciousId,
            data: SAMPLE_DATASET_1_V1,
          })
        ).rejects.toThrow('Invalid dataset id');
        expect(fs.promises.writeFile).not.toHaveBeenCalled();
      });
    }
  });

  describe('generateDatasetId', () => {
    it('returns ID if present', async () => {
      const id = await (
        DatasetStore as LocalFileDatasetStore
      ).generateDatasetId('some-id');

      expect(id).toBe('some-id');
    });

    it('returns full ID if nothing is provided', async () => {
      const id = await (
        DatasetStore as LocalFileDatasetStore
      ).generateDatasetId();

      expect(id).toBe('12345678');
    });

    it('throws if no unique ID is generated', async () => {
      const metadataMap = {
        ['some-id']: SAMPLE_DATASET_METADATA_1_V1,
      };
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );

      expect(async () => {
        await (DatasetStore as LocalFileDatasetStore).generateDatasetId(
          'some-id'
        );
      }).rejects.toThrow('Dataset ID not unique');
    });

    it('throws if datasetId is too long', async () => {
      expect(async () => {
        await (DatasetStore as LocalFileDatasetStore).generateDatasetId(
          'toolongdatasetid'.repeat(5)
        );
      }).rejects.toThrow('Invalid datasetId');
    });

    it('throws if datasetId is invalid', async () => {
      expect(async () => {
        await (DatasetStore as LocalFileDatasetStore).generateDatasetId(
          'my@#$%@#$%datasetid'
        );
      }).rejects.toThrow('Invalid datasetId');
    });

    it('throws if UUID is not unique', async () => {
      const metadataMap = {
        ['12345678']: SAMPLE_DATASET_METADATA_1_V1,
      };
      fs.promises.readFile = jest.fn(async () =>
        Promise.resolve(JSON.stringify(metadataMap) as any)
      );

      expect(async () => {
        await (DatasetStore as LocalFileDatasetStore).generateDatasetId();
      }).rejects.toThrow('Dataset ID not unique');
    });
  });
});
