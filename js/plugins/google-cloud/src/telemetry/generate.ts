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

import { ValueType } from '@opentelemetry/api';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { createHash } from 'crypto';
import {
  GENKIT_VERSION,
  type GenerateRequestData,
  type GenerateResponseData,
  type GenerationUsage,
  type MediaPart,
  type Part,
  type ToolRequestPart,
  type ToolResponsePart,
} from 'genkit';
import { logger } from 'genkit/logging';
import { toDisplayPath } from 'genkit/tracing';
import {
  MetricCounter,
  MetricHistogram,
  internalMetricNamespaceWrap,
  type Telemetry,
} from '../metrics.js';
import {
  createCommonLogAttributes,
  extractErrorName,
  extractOuterFeatureNameFromPath,
  truncate,
  truncatePath,
} from '../utils.js';

type SharedDimensions = {
  modelName?: string;
  featureName?: string;
  path?: string;
  status?: string;
  source?: string;
  sourceVersion?: string;
};

class GenerateTelemetry implements Telemetry {
  /**
   * Wraps the declared metrics in a Genkit-specific, internal namespace.
   */
  private _N = internalMetricNamespaceWrap.bind(null, 'ai');

  private actionCounter = new MetricCounter(this._N('generate/requests'), {
    description: 'Counts calls to genkit generate actions.',
    valueType: ValueType.INT,
  });

  private latencies = new MetricHistogram(this._N('generate/latency'), {
    description: 'Latencies when interacting with a Genkit model.',
    valueType: ValueType.DOUBLE,
    unit: 'ms',
  });

  private inputCharacters = new MetricCounter(
    this._N('generate/input/characters'),
    {
      description: 'Counts input characters to any Genkit model.',
      valueType: ValueType.INT,
    }
  );

  private inputTokens = new MetricCounter(this._N('generate/input/tokens'), {
    description: 'Counts input tokens to a Genkit model.',
    valueType: ValueType.INT,
  });

  private inputImages = new MetricCounter(this._N('generate/input/images'), {
    description: 'Counts input images to a Genkit model.',
    valueType: ValueType.INT,
  });

  private outputCharacters = new MetricCounter(
    this._N('generate/output/characters'),
    {
      description: 'Counts output characters from a Genkit model.',
      valueType: ValueType.INT,
    }
  );

  private outputTokens = new MetricCounter(this._N('generate/output/tokens'), {
    description: 'Counts output tokens from a Genkit model.',
    valueType: ValueType.INT,
  });

  private thinkingTokens = new MetricCounter(
    this._N('generate/thinking/tokens'),
    {
      description: 'Counts thinking tokens from a Genkit model.',
      valueType: ValueType.INT,
    }
  );

  private outputImages = new MetricCounter(this._N('generate/output/images'), {
    description: 'Count output images from a Genkit model.',
    valueType: ValueType.INT,
  });

  tick(
    span: ReadableSpan,
    logInputAndOutput: boolean,
    projectId?: string
  ): void {
    const attributes = span.attributes;
    const modelName = truncate(attributes['genkit:name'] as string, 1024);
    const path = (attributes['genkit:path'] as string) || '';
    const input =
      'genkit:input' in attributes
        ? (JSON.parse(
            attributes['genkit:input']! as string
          ) as GenerateRequestData)
        : undefined;
    const output =
      'genkit:output' in attributes
        ? (JSON.parse(
            attributes['genkit:output']! as string
          ) as GenerateResponseData)
        : undefined;

    const errName = extractErrorName(span.events);
    let featureName = truncate(
      (attributes['genkit:metadata:flow:name'] ||
        extractOuterFeatureNameFromPath(path)) as string
    );
    if (!featureName || featureName === '<unknown>') {
      featureName = 'generate';
    }

    const sessionId = attributes['genkit:sessionId'] as string;
    const threadName = attributes['genkit:threadName'] as string;

    if (input) {
      this.recordGenerateActionMetrics(modelName, featureName, path, {
        response: output,
        errName,
      });
      this.recordGenerateActionConfigLogs(
        span,
        modelName,
        featureName,
        path,
        input,
        projectId,
        sessionId,
        threadName
      );

      if (logInputAndOutput) {
        this.recordGenerateActionInputLogs(
          span,
          modelName,
          featureName,
          path,
          input,
          projectId,
          sessionId,
          threadName
        );
      }
    }

    if (output && logInputAndOutput) {
      this.recordGenerateActionOutputLogs(
        span,
        modelName,
        featureName,
        path,
        output,
        projectId,
        sessionId,
        threadName
      );
    }
  }

  private recordGenerateActionMetrics(
    modelName: string,
    featureName: string,
    path: string,
    opts: {
      response?: GenerateResponseData;
      errName?: string;
    }
  ) {
    this.doRecordGenerateActionMetrics(modelName, opts.response?.usage || {}, {
      featureName,
      path,
      latencyMs: opts.response?.latencyMs,
      errName: opts.errName,
      source: 'ts',
      sourceVersion: GENKIT_VERSION,
    });
  }

  private recordGenerateActionConfigLogs(
    span: ReadableSpan,
    model: string,
    featureName: string,
    qualifiedPath: string,
    input: GenerateRequestData,
    projectId?: string,
    sessionId?: string,
    threadName?: string
  ) {
    const path = truncatePath(toDisplayPath(qualifiedPath));
    const sharedMetadata = {
      ...createCommonLogAttributes(span, projectId),
      model,
      path,
      qualifiedPath,
      featureName,
      sessionId,
      threadName,
    };
    logger.info(`Config[${path}, ${model}]`, {
      ...sharedMetadata,
      maxOutputTokens: input.config?.maxOutputTokens,
      stopSequences: input.config?.stopSequences, // array
      source: 'ts',
      sourceVersion: GENKIT_VERSION,
    });
  }

  private recordGenerateActionInputLogs(
    span: ReadableSpan,
    model: string,
    featureName: string,
    qualifiedPath: string,
    input: GenerateRequestData,
    projectId?: string,
    sessionId?: string,
    threadName?: string
  ) {
    const path = truncatePath(toDisplayPath(qualifiedPath));
    const sharedMetadata = {
      ...createCommonLogAttributes(span, projectId),
      model,
      path,
      qualifiedPath,
      featureName,
      sessionId,
      threadName,
    };

    const messages = input.messages.length;
    input.messages.forEach((msg, msgIdx) => {
      const parts = msg.content.length;
      msg.content.forEach((part, partIdx) => {
        const partCounts = this.toPartCounts(partIdx, parts, msgIdx, messages);
        logger.info(`Input[${path}, ${model}] ${partCounts}`, {
          ...sharedMetadata,
          content: this.toPartLogContent(part),
          role: msg.role,
          partIndex: partIdx,
          totalParts: parts,
          messageIndex: msgIdx,
          totalMessages: messages,
        });
      });
    });
  }

  private recordGenerateActionOutputLogs(
    span: ReadableSpan,
    model: string,
    featureName: string,
    qualifiedPath: string,
    output: GenerateResponseData,
    projectId?: string,
    sessionId?: string,
    threadName?: string
  ) {
    const path = truncatePath(toDisplayPath(qualifiedPath));
    const sharedMetadata = {
      ...createCommonLogAttributes(span, projectId),
      model,
      path,
      qualifiedPath,
      featureName,
      sessionId,
      threadName,
    };
    const message = output.message || output.candidates?.[0]?.message!;

    if (message?.content) {
      const parts = message.content.length;
      message.content.forEach((part, partIdx) => {
        const partCounts = this.toPartCounts(partIdx, parts, 0, 1);
        const initial = output.finishMessage
          ? { finishMessage: truncate(output.finishMessage) }
          : {};
        logger.info(`Output[${path}, ${model}] ${partCounts}`, {
          ...initial,
          ...sharedMetadata,
          content: this.toPartLogContent(part),
          role: message.role,
          partIndex: partIdx,
          totalParts: parts,
          candidateIndex: 0,
          totalCandidates: 1,
          messageIndex: 0,
          finishReason: output.finishReason,
        });
      });
    }
  }

  private toPartCounts(
    partOrdinal: number,
    parts: number,
    msgOrdinal: number,
    messages: number
  ): string {
    if (parts > 1 && messages > 1) {
      return `(part ${this.xOfY(partOrdinal, parts)} in message ${this.xOfY(
        msgOrdinal,
        messages
      )})`;
    }
    if (parts > 1) {
      return `(part ${this.xOfY(partOrdinal, parts)})`;
    }
    if (messages > 1) {
      return `(message ${this.xOfY(msgOrdinal, messages)})`;
    }
    return '';
  }

  private xOfY(x: number, y: number): string {
    return `${x + 1} of ${y}`;
  }

  private toPartLogContent(part: Part): string {
    if (part.text) {
      return truncate(part.text);
    }
    if (part.data) {
      return truncate(JSON.stringify(part.data));
    }
    if (part.media) {
      return this.toPartLogMedia(part);
    }
    if (part.toolRequest) {
      return this.toPartLogToolRequest(part);
    }
    if (part.toolResponse) {
      return this.toPartLogToolResponse(part);
    }
    if (part.custom) {
      return truncate(JSON.stringify(part.custom));
    }
    return '<unknown format>';
  }

  private toPartLogMedia(part: MediaPart): string {
    if (part.media.url.startsWith('data:')) {
      const splitIdx = part.media.url.indexOf('base64,');
      if (splitIdx < 0) {
        return '<unknown media format>';
      }
      const prefix = part.media.url.substring(0, splitIdx + 7);
      const hashedContent = createHash('sha256')
        .update(part.media.url.substring(splitIdx + 7))
        .digest('hex');
      return `${prefix}<sha256(${hashedContent})>`;
    }
    return truncate(part.media.url);
  }

  private toPartLogToolRequest(part: ToolRequestPart): string {
    const inputText =
      typeof part.toolRequest.input === 'string'
        ? part.toolRequest.input
        : JSON.stringify(part.toolRequest.input);
    return truncate(
      `Tool request: ${part.toolRequest.name}, ref: ${part.toolRequest.ref}, input: ${inputText}`
    );
  }

  private toPartLogToolResponse(part: ToolResponsePart): string {
    const outputText =
      typeof part.toolResponse.output === 'string'
        ? part.toolResponse.output
        : JSON.stringify(part.toolResponse.output);
    return truncate(
      `Tool response: ${part.toolResponse.name}, ref: ${part.toolResponse.ref}, output: ${outputText}`
    );
  }

  /**
   * Records all metrics associated with performing a GenerateAction.
   */
  private doRecordGenerateActionMetrics(
    modelName: string,
    usage: GenerationUsage,
    dimensions: {
      featureName?: string;
      path?: string;
      latencyMs?: number;
      errName?: string;
      source?: string;
      sourceVersion: string;
    }
  ) {
    const shared: SharedDimensions = {
      modelName: modelName,
      featureName: dimensions.featureName,
      path: dimensions.path,
      source: dimensions.source,
      sourceVersion: dimensions.sourceVersion,
      status: dimensions.errName ? 'failure' : 'success',
    };

    this.actionCounter.add(1, {
      error: dimensions.errName,
      ...shared,
    });

    this.latencies.record(dimensions.latencyMs, shared);

    // inputs
    this.inputTokens.add(usage.inputTokens, shared);
    this.inputCharacters.add(usage.inputCharacters, shared);
    this.inputImages.add(usage.inputImages, shared);

    // outputs
    this.outputTokens.add(usage.outputTokens, shared);
    this.outputCharacters.add(usage.outputCharacters, shared);
    this.outputImages.add(usage.outputImages, shared);

    // thoughts
    this.thinkingTokens.add(usage.thoughtsTokens, shared);
  }
}

const generateTelemetry = new GenerateTelemetry();
export { generateTelemetry };
