/**
 * Copyright 2025 Google LLC
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

import * as assert from 'assert';
import { GenkitError } from 'genkit';
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import {
  ExpressClientOptions,
  GlobalClientOptions,
  RegionalClientOptions,
  VertexPluginOptions,
} from '../../src/vertexai/types.js';
import {
  API_KEY_FALSE_ERROR,
  MISSING_API_KEY_ERROR,
  NOT_SUPPORTED_IN_EXPRESS_ERROR,
  calculateApiKey,
  calculateRequestOptions,
  checkApiKey,
  checkSupportedResourceMethod,
  getApiKeyFromEnvVar,
  getDerivedOptions,
} from '../../src/vertexai/utils.js';

// Helper to assert GenkitError properties
function assertGenkitError(error: any, expectedError: GenkitError) {
  assert.ok(
    error instanceof GenkitError,
    `Expected GenkitError, got ${error.name}`
  );
  assert.strictEqual(
    error.status,
    expectedError.status,
    'Error status mismatch'
  );
  assert.strictEqual(
    error.message,
    expectedError.message,
    'Error message mismatch'
  );
}

describe('Vertex AI Utils', () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    // Reset env
    for (const key in process.env) {
      if (!originalEnv.hasOwnProperty(key)) {
        delete process.env[key];
      }
    }
    for (const key in originalEnv) {
      process.env[key] = originalEnv[key];
    }

    delete process.env.GCLOUD_PROJECT;
    delete process.env.GCLOUD_LOCATION;
    delete process.env.FIREBASE_CONFIG;
    delete process.env.GCLOUD_SERVICE_ACCOUNT_CREDS;
    delete process.env.VERTEX_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    delete process.env.GOOGLE_GENAI_API_KEY;
  });

  afterEach(() => {
    sinon.restore();
  });

  describe('getDerivedOptions', () => {
    let authInstance: sinon.SinonStubbedInstance<GoogleAuth>;
    let mockAuthClass: sinon.SinonStub;

    beforeEach(() => {
      authInstance = sinon.createStubInstance(GoogleAuth);
      authInstance.getAccessToken.resolves('test-token');
      authInstance.getProjectId.resolves(undefined); // Default
      mockAuthClass = sinon.stub().returns(authInstance);
    });

    describe('Regional Options', () => {
      it('should use options for projectId and location', async () => {
        const pluginOptions: VertexPluginOptions = {
          projectId: 'options-project',
          location: 'options-location',
        };
        const options = (await getDerivedOptions(
          pluginOptions,
          mockAuthClass as any
        )) as RegionalClientOptions;
        assert.strictEqual(options.kind, 'regional');
        assert.strictEqual(options.projectId, 'options-project');
        assert.strictEqual(options.location, 'options-location');
        assert.ok(options.authClient);
        sinon.assert.calledOnce(mockAuthClass);
        sinon.assert.notCalled(authInstance.getProjectId);
      });

      it('should use GCLOUD_PROJECT and GCLOUD_LOCATION env vars', async () => {
        process.env.GCLOUD_PROJECT = 'env-project';
        process.env.GCLOUD_LOCATION = 'env-location';
        const options = (await getDerivedOptions(
          undefined,
          mockAuthClass as any
        )) as RegionalClientOptions;
        assert.strictEqual(options.kind, 'regional');
        assert.strictEqual(options.projectId, 'env-project');
        assert.strictEqual(options.location, 'env-location');
        sinon.assert.calledOnce(mockAuthClass);
        const authOptions = mockAuthClass.lastCall.args[0];
        assert.strictEqual(authOptions.projectId, 'env-project');
        sinon.assert.notCalled(authInstance.getProjectId);
      });

      it('should use default location when only projectId is available', async () => {
        authInstance.getProjectId.resolves('default-project');
        const options = (await getDerivedOptions(
          undefined,
          mockAuthClass as any
        )) as RegionalClientOptions;
        assert.strictEqual(options.kind, 'regional');
        assert.strictEqual(options.projectId, 'default-project');
        assert.strictEqual(options.location, 'us-central1');
        sinon.assert.calledOnce(mockAuthClass);
        sinon.assert.calledOnce(authInstance.getProjectId);
      });

      it('should use FIREBASE_CONFIG for GoogleAuth constructor, but final projectId from getProjectId', async () => {
        process.env.FIREBASE_CONFIG = JSON.stringify({
          projectId: 'firebase-project',
        });
        authInstance.getProjectId.resolves('auth-client-project');

        const options = (await getDerivedOptions(
          { location: 'fb-location' },
          mockAuthClass as any
        )) as RegionalClientOptions;

        assert.strictEqual(options.kind, 'regional');
        sinon.assert.calledOnce(mockAuthClass);
        const authOptions = mockAuthClass.lastCall.args[0];
        assert.strictEqual(authOptions.projectId, 'firebase-project');
        sinon.assert.calledOnce(authInstance.getProjectId);
        assert.strictEqual(options.projectId, 'auth-client-project');
        assert.strictEqual(options.location, 'fb-location');
      });

      it('should prioritize plugin options over env vars', async () => {
        process.env.GCLOUD_PROJECT = 'env-project';
        process.env.GCLOUD_LOCATION = 'env-location';
        const pluginOptions: VertexPluginOptions = {
          projectId: 'options-project',
          location: 'options-location',
        };
        const options = (await getDerivedOptions(
          pluginOptions,
          mockAuthClass as any
        )) as RegionalClientOptions;
        assert.strictEqual(options.kind, 'regional');
        assert.strictEqual(options.projectId, 'options-project');
        assert.strictEqual(options.location, 'options-location');
      });

      it('should use GCLOUD_SERVICE_ACCOUNT_CREDS for auth', async () => {
        const creds = {
          client_email: 'clientEmail',
          private_key: 'private_key',
        };
        process.env.GCLOUD_SERVICE_ACCOUNT_CREDS = JSON.stringify(creds);
        authInstance.getProjectId.resolves('creds-project');

        const options = (await getDerivedOptions(
          { location: 'creds-location' },
          mockAuthClass as any
        )) as RegionalClientOptions;

        assert.strictEqual(options.kind, 'regional');
        assert.strictEqual(options.projectId, 'creds-project');
        assert.strictEqual(options.location, 'creds-location');
        sinon.assert.calledOnce(mockAuthClass);
        const authOptions = mockAuthClass.lastCall.args[0];
        assert.deepStrictEqual(authOptions.credentials, creds);
        assert.strictEqual(authOptions.projectId, undefined);
        sinon.assert.calledOnce(authInstance.getProjectId);
      });

      it('should throw error if projectId cannot be determined for regional', async () => {
        authInstance.getProjectId.resolves(undefined);
        await assert.rejects(
          getDerivedOptions(
            { location: 'some-location' },
            mockAuthClass as any
          ),
          /VertexAI Plugin is missing the 'project' configuration/
        );
        sinon.assert.calledOnce(mockAuthClass);
        sinon.assert.calledOnce(authInstance.getProjectId);
      });

      it('should prefer regional if location is specified, even with apiKey', async () => {
        const pluginOptions: VertexPluginOptions = {
          location: 'us-central1',
          apiKey: 'test-api-key',
          projectId: 'options-project',
        };
        const options = (await getDerivedOptions(
          pluginOptions,
          mockAuthClass as any
        )) as RegionalClientOptions;
        assert.strictEqual(options.kind, 'regional');
        assert.strictEqual(options.projectId, 'options-project');
        assert.strictEqual(options.location, 'us-central1');
        assert.ok(options.authClient);
        assert.strictEqual(options.apiKey, 'test-api-key');
        sinon.assert.calledOnce(mockAuthClass);
      });
    });

    describe('Global Options', () => {
      it('should use global options when location is global', async () => {
        const pluginOptions: VertexPluginOptions = {
          location: 'global',
          projectId: 'options-project',
        };
        const options = (await getDerivedOptions(
          pluginOptions,
          mockAuthClass as any
        )) as GlobalClientOptions;
        assert.strictEqual(options.kind, 'global');
        assert.strictEqual(options.location, 'global');
        assert.strictEqual(options.projectId, 'options-project');
        assert.ok(options.authClient);
        sinon.assert.calledOnce(mockAuthClass);
        sinon.assert.notCalled(authInstance.getProjectId);
      });

      it('should use env project for global options', async () => {
        process.env.GCLOUD_PROJECT = 'env-project';
        const pluginOptions: VertexPluginOptions = { location: 'global' };
        const options = (await getDerivedOptions(
          pluginOptions,
          mockAuthClass as any
        )) as GlobalClientOptions;
        assert.strictEqual(options.kind, 'global');
        assert.strictEqual(options.projectId, 'env-project');
        sinon.assert.calledOnce(mockAuthClass);
      });

      it('should use auth project for global options', async () => {
        authInstance.getProjectId.resolves('auth-project');
        const pluginOptions: VertexPluginOptions = { location: 'global' };
        const options = (await getDerivedOptions(
          pluginOptions,
          mockAuthClass as any
        )) as GlobalClientOptions;
        assert.strictEqual(options.kind, 'global');
        assert.strictEqual(options.projectId, 'auth-project');
        sinon.assert.calledOnce(mockAuthClass);
        sinon.assert.calledOnce(authInstance.getProjectId);
      });

      it('should throw error if projectId cannot be determined for global', async () => {
        authInstance.getProjectId.resolves(undefined);
        await assert.rejects(
          getDerivedOptions({ location: 'global' }, mockAuthClass as any),
          /VertexAI Plugin is missing the 'project' configuration/
        );
        sinon.assert.calledOnce(mockAuthClass);
        sinon.assert.calledOnce(authInstance.getProjectId);
      });

      it('should prefer global if location is global, even with apiKey', async () => {
        const pluginOptions: VertexPluginOptions = {
          location: 'global',
          apiKey: 'test-api-key',
          projectId: 'options-project',
        };
        const options = (await getDerivedOptions(
          pluginOptions,
          mockAuthClass as any
        )) as GlobalClientOptions;
        assert.strictEqual(options.kind, 'global');
        assert.strictEqual(options.projectId, 'options-project');
        assert.ok(options.authClient);
        assert.strictEqual(options.apiKey, 'test-api-key');
        sinon.assert.calledOnce(mockAuthClass);
      });
    });

    describe('Express Options', () => {
      it('should use express options with apiKey in options', async () => {
        const pluginOptions: VertexPluginOptions = { apiKey: 'key1' };
        const options = (await getDerivedOptions(
          pluginOptions,
          mockAuthClass as any
        )) as ExpressClientOptions;
        assert.strictEqual(options.kind, 'express');
        assert.strictEqual(options.apiKey, 'key1');
        sinon.assert.notCalled(mockAuthClass);
      });

      it('should use express options with apiKey false in options', async () => {
        const pluginOptions: VertexPluginOptions = { apiKey: false };
        const options = (await getDerivedOptions(
          pluginOptions,
          mockAuthClass as any
        )) as ExpressClientOptions;
        assert.strictEqual(options.kind, 'express');
        assert.strictEqual(options.apiKey, undefined);
        sinon.assert.notCalled(mockAuthClass);
      });

      it('should use VERTEX_API_KEY env var for express', async () => {
        process.env.VERTEX_API_KEY = 'key2';
        const options = (await getDerivedOptions(
          undefined,
          mockAuthClass as any
        )) as ExpressClientOptions;
        assert.strictEqual(options.kind, 'express');
        assert.strictEqual(options.apiKey, 'key2');
        sinon.assert.calledTwice(mockAuthClass); // Fallback attempts
      });

      it('should use GOOGLE_API_KEY env var for express', async () => {
        process.env.GOOGLE_API_KEY = 'key3';
        const options = (await getDerivedOptions(
          undefined,
          mockAuthClass as any
        )) as ExpressClientOptions;
        assert.strictEqual(options.kind, 'express');
        assert.strictEqual(options.apiKey, 'key3');
        sinon.assert.calledTwice(mockAuthClass); // Fallback attempts
      });
    });

    describe('Fallback Determination (No Options)', () => {
      it('should default to regional if project can be determined', async () => {
        authInstance.getProjectId.resolves('fallback-project');
        const options = (await getDerivedOptions(
          undefined,
          mockAuthClass as any
        )) as RegionalClientOptions;
        assert.strictEqual(options.kind, 'regional');
        assert.strictEqual(options.projectId, 'fallback-project');
        assert.strictEqual(options.location, 'us-central1');
        sinon.assert.calledOnce(mockAuthClass);
      });

      it('should fallback to express if regional/global fail and API key env exists', async () => {
        authInstance.getProjectId.resolves(undefined);
        process.env.GOOGLE_API_KEY = 'fallback-api-key';

        const options = (await getDerivedOptions(
          undefined,
          mockAuthClass as any
        )) as ExpressClientOptions;

        assert.strictEqual(options.kind, 'express');
        assert.strictEqual(options.apiKey, 'fallback-api-key');
        sinon.assert.calledTwice(mockAuthClass);
      });
    });

    describe('Error Scenarios', () => {
      it('should throw error if no options or env vars provide sufficient info', async () => {
        authInstance.getProjectId.resolves(undefined);
        await assert.rejects(
          getDerivedOptions(undefined, mockAuthClass as any),
          /Unable to determine client options/
        );
        sinon.assert.calledTwice(mockAuthClass);
      });
    });
  });

  describe('getApiKeyFromEnvVar', () => {
    it('should return VERTEX_API_KEY if set', () => {
      process.env.VERTEX_API_KEY = 'vertexKey';
      process.env.GOOGLE_API_KEY = 'googleKey';
      assert.strictEqual(getApiKeyFromEnvVar(), 'vertexKey');
    });

    it('should return GOOGLE_API_KEY if VERTEX_API_KEY is not set', () => {
      process.env.GOOGLE_API_KEY = 'googleKey';
      process.env.GOOGLE_GENAI_API_KEY = 'genaiKey';
      assert.strictEqual(getApiKeyFromEnvVar(), 'googleKey');
    });

    it('should return GOOGLE_GENAI_API_KEY if others are not set', () => {
      process.env.GOOGLE_GENAI_API_KEY = 'genaiKey';
      assert.strictEqual(getApiKeyFromEnvVar(), 'genaiKey');
    });

    it('should return undefined if no key env vars are set', () => {
      assert.strictEqual(getApiKeyFromEnvVar(), undefined);
    });
  });

  describe('checkApiKey', () => {
    it('should return pluginApiKey if it is a string', () => {
      assert.strictEqual(checkApiKey('pluginKey'), 'pluginKey');
    });

    it('should return undefined if pluginApiKey is false', () => {
      assert.strictEqual(checkApiKey(false), undefined);
    });

    it('should return env var key if pluginApiKey is undefined', () => {
      process.env.VERTEX_API_KEY = 'envKey';
      assert.strictEqual(checkApiKey(undefined), 'envKey');
    });

    it('should throw MISSING_API_KEY_ERROR if no key found', () => {
      try {
        checkApiKey(undefined);
        assert.fail('Should have thrown');
      } catch (e) {
        assertGenkitError(e, MISSING_API_KEY_ERROR);
      }
    });

    it('should not throw if pluginApiKey is false, even if no env var', () => {
      assert.doesNotThrow(() => {
        checkApiKey(false);
      });
    });
  });

  describe('calculateApiKey', () => {
    it('should use requestApiKey when provided', () => {
      assert.strictEqual(calculateApiKey(undefined, 'reqKey'), 'reqKey');
      assert.strictEqual(calculateApiKey('pluginKey', 'reqKey'), 'reqKey');
      assert.strictEqual(calculateApiKey(false, 'reqKey'), 'reqKey');
    });

    it('should use pluginApiKey if requestApiKey is undefined', () => {
      assert.strictEqual(calculateApiKey('pluginKey', undefined), 'pluginKey');
    });

    it('should use env key if plugin and request keys are undefined', () => {
      process.env.VERTEX_API_KEY = 'envKey';
      assert.strictEqual(calculateApiKey(undefined, undefined), 'envKey');
    });

    it('should prioritize pluginApiKey over env keys', () => {
      process.env.VERTEX_API_KEY = 'envKey';
      assert.strictEqual(calculateApiKey('pluginKey', undefined), 'pluginKey');
    });

    it('should throw MISSING_API_KEY_ERROR if no key is found', () => {
      try {
        calculateApiKey(undefined, undefined);
        assert.fail('Should have thrown');
      } catch (e) {
        assertGenkitError(e, MISSING_API_KEY_ERROR);
      }
    });

    it('should throw API_KEY_FALSE_ERROR if pluginApiKey is false and requestApiKey is undefined', () => {
      try {
        calculateApiKey(false, undefined);
        assert.fail('Should have thrown');
      } catch (e) {
        assertGenkitError(e, API_KEY_FALSE_ERROR);
      }
    });

    it('should not use env keys if pluginApiKey is false', () => {
      process.env.VERTEX_API_KEY = 'envKey';
      try {
        calculateApiKey(false, undefined);
        assert.fail('Should have thrown');
      } catch (e) {
        assertGenkitError(e, API_KEY_FALSE_ERROR);
      }
    });
  });

  describe('checkSupportedResourceMethod', () => {
    const expressOptions: ExpressClientOptions = {
      kind: 'express',
      apiKey: 'testKey',
    };
    const regionalOptions: RegionalClientOptions = {
      kind: 'regional',
      location: 'us-central1',
      projectId: 'testProject',
      authClient: {} as any,
    };

    it('should allow empty resourcePath', () => {
      assert.doesNotThrow(() => {
        checkSupportedResourceMethod({
          clientOptions: expressOptions,
          resourcePath: '',
        });
      });
    });

    it('should allow supported methods for Express', () => {
      const supported = [
        'countTokens',
        'generateContent',
        'streamGenerateContent',
      ];
      supported.forEach((method) => {
        assert.doesNotThrow(() => {
          checkSupportedResourceMethod({
            clientOptions: expressOptions,
            resourceMethod: method,
          });
        }, `Express should support ${method}`);
      });
    });

    it('should throw NOT_SUPPORTED_IN_EXPRESS_ERROR for unsupported methods in Express', () => {
      const unsupported = [
        'predict',
        'predictLongRunning',
        'fetchPredictOperation',
        'listModels',
      ];
      unsupported.forEach((method) => {
        try {
          checkSupportedResourceMethod({
            clientOptions: expressOptions,
            resourceMethod: method,
          });
          assert.fail(`Should have thrown for Express method ${method}`);
        } catch (e) {
          assertGenkitError(e, NOT_SUPPORTED_IN_EXPRESS_ERROR);
        }
      });
    });

    it('should allow any method for non-Express options', () => {
      const methods = [
        'countTokens',
        'generateContent',
        'streamGenerateContent',
        'predict',
        'predictLongRunning',
        'fetchPredictOperation',
      ];
      methods.forEach((method) => {
        assert.doesNotThrow(() => {
          checkSupportedResourceMethod({
            clientOptions: regionalOptions,
            resourceMethod: method,
          });
        }, `Regional should support ${method}`);
      });
    });
  });
  describe('calculateRequestOptions', () => {
    const regionalClientOptions: RegionalClientOptions = {
      kind: 'regional',
      location: 'us-central1',
      projectId: 'testProject',
      authClient: {} as any,
    };
    const globalClientOptions: GlobalClientOptions = {
      kind: 'global',
      location: 'global',
      projectId: 'testProject',
      authClient: {} as any,
    };
    const expressClientOptions: ExpressClientOptions = {
      kind: 'express',
      apiKey: 'testKey',
    };
    it('should do nothing if no overrides', () => {
      const newOptions = calculateRequestOptions(regionalClientOptions, {});
      assert.deepStrictEqual(newOptions, regionalClientOptions);
    });
    it('should override location to regional', () => {
      const newOptions = calculateRequestOptions(globalClientOptions, {
        location: 'us-west1',
      });
      assert.strictEqual(newOptions.kind, 'regional');
      assert.strictEqual(newOptions.location, 'us-west1');
    });
    it('should override location to global', () => {
      const newOptions = calculateRequestOptions(regionalClientOptions, {
        location: 'global',
      });
      assert.strictEqual(newOptions.kind, 'global');
      assert.strictEqual(newOptions.location, 'global');
    });
    it('should override apiKey for express', () => {
      const newOptions = calculateRequestOptions(expressClientOptions, {
        apiKey: 'newKey',
      });
      assert.strictEqual(newOptions.apiKey, 'newKey');
    });
    it('should override apiKey for regional', () => {
      const newOptions = calculateRequestOptions(regionalClientOptions, {
        apiKey: 'newKey',
      });
      assert.strictEqual(newOptions.apiKey, 'newKey');
    });
    it('should override apiKey for global', () => {
      const newOptions = calculateRequestOptions(globalClientOptions, {
        apiKey: 'newKey',
      });
      assert.strictEqual(newOptions.apiKey, 'newKey');
    });
  });
});
