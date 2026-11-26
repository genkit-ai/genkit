/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  AuthTypes,
  Connector,
  IpAddressTypes,
} from '@google-cloud/cloud-sql-connector';
import * as dotenv from 'dotenv';
import knex from 'knex';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import {
  Column,
  PostgresEngine,
  PostgresEngineArgs,
  VectorStoreTableArgs,
} from '../src/engine.js';

dotenv.config();

const USER_AGENT = 'genkit-google-cloud-sql-pg-js';
const CUSTOM_TABLE = 'test_table_custom';
const VECTOR_SIZE = 768;
const ID_COLUMN = 'uuid';
const CONTENT_COLUMN = 'my_content';
const EMBEDDING_COLUMN = 'my_embedding';
const METADATA_COLUMNS = [
  new Column('page', 'TEXT'),
  new Column('source', 'TEXT'),
];
const STORE_METADATA = true;

const REQUIRED_ENV_VARS = [
  'PROJECT_ID',
  'REGION',
  'INSTANCE_ID',
  'DATABASE_ID',
  'DB_USER',
  'DB_PASSWORD',
  'IP_ADDRESS',
  'EMAIL', // Add EMAIL here if it's consistently required for some tests
];

let envVarsFound = true;
let missingEnvVars: string[] = [];

function validateEnvVars() {
  missingEnvVars = REQUIRED_ENV_VARS.filter((varName) => !process.env[varName]);
  if (missingEnvVars.length > 0) {
    envVarsFound = false;
    console.warn(
      `Skipping tests due to missing environment variables: ${missingEnvVars.join(', ')}`
    );
  }
}

// Validate environment variables once at the top level
validateEnvVars();

describe('PostgresEngine Instance creation', () => {
  // Conditionally skip the entire describe block
  if (!envVarsFound) {
    describe.skip('PostgresEngine Instance creation (skipped due to missing env vars)', () => {
      test('this test will be skipped', () => {
        // This block won't execute
      });
    });
    return; // Exit the describe block early
  }

  let PEInstance: PostgresEngine;

  const poolConfig: knex.Knex.PoolConfig = {
    min: 0,
    max: 5,
  };

  test('should throw an error if only user or password are passed', async () => {
    const pgArgs: PostgresEngineArgs = {
      user: process.env.DB_USER ?? '',
    };

    async function createInstance() {
      PEInstance = await PostgresEngine.fromInstance(
        process.env.PROJECT_ID!,
        process.env.REGION!,
        process.env.INSTANCE_ID!,
        process.env.DATABASE_ID!,
        pgArgs
      );
    }

    await expect(createInstance).rejects.toBe(
      "Only one of 'user' or 'password' were specified. Either " +
        'both should be specified to use basic user/password ' +
        'authentication or neither for IAM DB authentication.'
    );
  });

  test('should create a PostgresEngine Instance using user and password', async () => {
    const pgArgs: PostgresEngineArgs = {
      user: process.env.DB_USER!,
      password: process.env.DB_PASSWORD!,
    };

    PEInstance = await PostgresEngine.fromInstance(
      process.env.PROJECT_ID!,
      process.env.REGION!,
      process.env.INSTANCE_ID!,
      process.env.DATABASE_ID!,
      pgArgs
    );

    const result = await PEInstance.testConnection();
    const currentTimestamp = result[0].currentTimestamp;
    expect(currentTimestamp).toBeDefined();

    try {
      await PEInstance.closeConnection();
    } catch (error) {
      throw new Error(`Error on closing connection: ${error}`);
    }
  });

  // Example of conditionally skipping a single test if a specific env var is missing
  (process.env.EMAIL ? test : test.skip)(
    'should create a PostgresEngine Instance with IAM email',
    async () => {
      const pgArgs: PostgresEngineArgs = {
        ipType: IpAddressTypes.PUBLIC,
        iamAccountEmail: process.env.EMAIL!, // Use ! as we checked for its existence
      };

      PEInstance = await PostgresEngine.fromInstance(
        process.env.PROJECT_ID!,
        process.env.REGION!,
        process.env.INSTANCE_ID!,
        process.env.DATABASE_ID!,
        pgArgs
      );

      const result = await PEInstance.testConnection();
      const currentTimestamp = result[0].currentTimestamp;
      expect(currentTimestamp).toBeDefined();

      try {
        await PEInstance.closeConnection();
      } catch (error) {
        throw new Error(`Error on closing connection: ${error}`);
      }
    }
  );

  test('should create a PostgresEngine Instance through from_engine method', async () => {
    PostgresEngine.connector = new Connector({ userAgent: USER_AGENT });
    const clientOpts = await PostgresEngine.connector.getOptions({
      instanceConnectionName: `${process.env.PROJECT_ID}:${process.env.REGION}:${process.env.INSTANCE_ID}`,
      ipType: IpAddressTypes.PUBLIC,
      authType: AuthTypes.PASSWORD,
    });

    const dbConfig: knex.Knex.Config<any> = {
      client: 'pg',
      connection: {
        ...clientOpts,
        password: process.env.DB_PASSWORD,
        user: process.env.DB_USER,
        database: process.env.DATABASE_ID,
      },
    };

    const engine = knex(dbConfig);
    PEInstance = await PostgresEngine.fromEngine(engine);

    const result = await PEInstance.testConnection();
    const currentTimestamp = result[0].currentTimestamp;
    expect(currentTimestamp).toBeDefined();

    try {
      await PEInstance.closeConnection();
    } catch (error) {
      throw new Error(`Error on closing connection: ${error}`);
    }
  });

  test('should throw an error if the URL passed to from_engine_args does not have the driver', async () => {
    const url = '';

    async function createInstance() {
      PEInstance = await PostgresEngine.fromEngineArgs(url);
    }

    await expect(createInstance).rejects.toBe(
      "Driver must be type 'postgresql+asyncpg'"
    );
  });

  test('should create a PostgresEngine Instance through from_engine_args using a URL', async () => {
    const url = `postgresql+asyncpg://${process.env.DB_USER}:${process.env.DB_PASSWORD}@${process.env.IP_ADDRESS}:5432/${process.env.DATABASE_ID}`;

    PEInstance = await PostgresEngine.fromEngineArgs(url, poolConfig);

    const result = await PEInstance.testConnection();
    const currentTimestamp = result[0].currentTimestamp;
    expect(currentTimestamp).toBeDefined();

    try {
      await PEInstance.closeConnection();
    } catch (error) {
      throw new Error(`Error on closing connection: ${error}`);
    }
  });
});

describe('PostgresEngine - table initialization', () => {
  // Conditionally skip the entire describe block
  if (!envVarsFound) {
    describe.skip('PostgresEngine - table initialization (skipped due to missing env vars)', () => {
      test('this test will be skipped', () => {
        // This block won't execute
      });
    });
    return; // Exit the describe block early
  }

  let PEInstance: PostgresEngine;

  beforeAll(async () => {
    const pgArgs: PostgresEngineArgs = {
      user: process.env.DB_USER!,
      password: process.env.DB_PASSWORD!,
    };

    PEInstance = await PostgresEngine.fromInstance(
      process.env.PROJECT_ID!,
      process.env.REGION!,
      process.env.INSTANCE_ID!,
      process.env.DATABASE_ID!,
      pgArgs
    );
  });

  test('should create the vectorstore table', async () => {
    const vsTableArgs: VectorStoreTableArgs = {
      contentColumn: CONTENT_COLUMN,
      embeddingColumn: EMBEDDING_COLUMN,
      idColumn: ID_COLUMN,
      metadataColumns: METADATA_COLUMNS,
      storeMetadata: STORE_METADATA,
      overwriteExisting: true,
    };

    await PEInstance.initVectorstoreTable(
      CUSTOM_TABLE,
      VECTOR_SIZE,
      vsTableArgs
    );

    const query = `SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '${CUSTOM_TABLE}';`;
    const expected = [
      { column_name: 'uuid', data_type: 'uuid' },
      { column_name: 'my_embedding', data_type: 'USER-DEFINED' },
      { column_name: 'json_metadata', data_type: 'json' },
      { column_name: 'my_content', data_type: 'text' },
      { column_name: 'page', data_type: 'text' },
      { column_name: 'source', data_type: 'text' },
    ];

    const { rows } = await PEInstance.pool.raw(query);

    rows.forEach((row: any, index: number) => {
      expect(row).toMatchObject(expected[index]);
    });
  });

  afterAll(async () => {
    // Only attempt to drop the table if the instance was successfully created
    if (PEInstance && PEInstance.pool) {
      await PEInstance.pool.raw(`DROP TABLE "${CUSTOM_TABLE}"`);
    }

    try {
      if (PEInstance) {
        await PEInstance.closeConnection();
      }
    } catch (error) {
      throw new Error(`Error on closing connection: ${error}`);
    }
  });
});
