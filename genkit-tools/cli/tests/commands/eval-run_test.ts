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

import { describe, expect, it, jest } from '@jest/globals';
import { evalExtractData } from '../../src/commands/eval-extract-data';
import { evalFlow } from '../../src/commands/eval-flow';
import { evalRun } from '../../src/commands/eval-run';
import { mcp } from '../../src/commands/mcp';

jest.mock('../../src/mcp/server', () => ({
  startMcpServer: jest.fn(),
}));

describe('eval:run', () => {
  it("fails if dataset isn't passed", () => {
    expect(() => {
      evalRun
        .exitOverride()
        .configureOutput({
          writeOut: () => {},
          writeErr: () => {},
        })
        .parse(['node', 'eval:run']);
    }).toThrowError(new Error("error: missing required argument 'dataset'"));
  });

  it('fails if invalid output-format is passed', () => {
    expect(() => {
      evalRun
        .exitOverride()
        .configureOutput({
          writeOut: () => {},
          writeErr: () => {},
        })
        .parse(['node', 'eval:run', 'data.json', '--output-format', 'xml']);
    }).toThrow(/option '--output-format <format>' argument 'xml' is invalid/);
  });

  it('keeps --batchSize as a hidden alias for --batch-size on eval:run and eval:flow', () => {
    for (const cmd of [evalRun, evalFlow]) {
      cmd
        .exitOverride()
        .configureOutput({ writeOut: () => {}, writeErr: () => {} });
      const canonical = cmd.options.find((o) => o.long === '--batch-size');
      const alias = cmd.options.find((o) => o.long === '--batchSize');
      expect(canonical?.hidden).toBe(false);
      expect(alias?.hidden).toBe(true);
      expect(alias?.attributeName()).toBe(canonical?.attributeName());

      expect(cmd.opts().batchSize).toBeUndefined();
      cmd.parseOptions(['--batchSize', '4']);
      expect(cmd.opts().batchSize).toBe(4);
      expect(() => cmd.parseOptions(['--batchSize', '0'])).toThrow(
        /Must be a positive integer\./
      );
      cmd.setOptionValue('batchSize', undefined);
    }
  });

  it('keeps --maxRows as a hidden alias for --max-rows on eval:extract-data', () => {
    evalExtractData
      .exitOverride()
      .configureOutput({ writeOut: () => {}, writeErr: () => {} });
    const canonical = evalExtractData.options.find(
      (o) => o.long === '--max-rows'
    );
    const alias = evalExtractData.options.find((o) => o.long === '--maxRows');
    expect(canonical?.hidden).toBe(false);
    expect(alias?.hidden).toBe(true);
    expect(alias?.attributeName()).toBe(canonical?.attributeName());

    expect(evalExtractData.opts().maxRows).toBe(100);
    evalExtractData.parseOptions(['--maxRows', '25']);
    expect(evalExtractData.opts().maxRows).toBe(25);
    expect(() => evalExtractData.parseOptions(['--maxRows', '0'])).toThrow(
      /Must be a positive integer\./
    );
    evalExtractData.setOptionValueWithSource('maxRows', 100, 'default');
  });

  it('keeps --explicitProjectRoot as a hidden alias for --explicit-project-root on mcp', () => {
    const canonical = mcp.options.find(
      (o) => o.long === '--explicit-project-root'
    );
    const alias = mcp.options.find((o) => o.long === '--explicitProjectRoot');
    expect(canonical?.hidden).toBe(false);
    expect(alias?.hidden).toBe(true);
    expect(alias?.attributeName()).toBe(canonical?.attributeName());

    expect(mcp.opts().explicitProjectRoot).toBe(false);
    mcp.parseOptions(['--explicitProjectRoot']);
    expect(mcp.opts().explicitProjectRoot).toBe(true);
    mcp.setOptionValueWithSource('explicitProjectRoot', false, 'default');
  });
});
