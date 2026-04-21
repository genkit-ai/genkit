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

import { GenkitError, ToolRequest, z } from 'genkit';
import {
  CandidateData,
  MessageData,
  ModelReference,
  Part,
  TextPart,
  ToolDefinition,
} from 'genkit/model';
import { JSONPath } from 'jsonpath-plus';
import {
  FunctionCallingMode,
  FunctionDeclaration,
  GenerateContentCandidate as GeminiCandidate,
  Content as GeminiContent,
  Part as GeminiPart,
  PartialArg,
  Schema,
  SchemaType,
  VideoMetadata,
  isObject,
} from './types.js';

export function toGeminiTool(tool: ToolDefinition): FunctionDeclaration {
  const declaration: FunctionDeclaration = {
    name: tool.name.replace(/\//g, '__'), // Gemini throws on '/' in tool name
    description: tool.description,
    parameters: toGeminiSchemaProperty(tool.inputSchema),
  };
  return declaration;
}

function toGeminiSchemaProperty(property?: ToolDefinition['inputSchema']) {
  if (!property || !property.type) {
    return undefined;
  }
  const baseSchema: Schema = {};
  if (property.description) {
    baseSchema.description = property.description;
  }
  if (property.enum) {
    baseSchema.enum = property.enum;
  }
  if (property.nullable) {
    baseSchema.nullable = property.nullable;
  }
  let propertyType;
  // nullable schema can ALSO be defined as, for example, type=['string','null']
  if (Array.isArray(property.type)) {
    const types = property.type as string[];
    if (types.includes('null')) {
      baseSchema.nullable = true;
    }
    // grab the type that's not `null`
    propertyType = types.find((t) => t !== 'null');
  } else {
    propertyType = property.type;
  }
  if (propertyType === 'object') {
    const nestedProperties = {};
    if (property.properties) {
      Object.keys(property.properties).forEach((key) => {
        nestedProperties[key] = toGeminiSchemaProperty(
          property.properties[key]
        );
      });
    }
    return {
      ...baseSchema,
      type: SchemaType.OBJECT,
      properties: nestedProperties,
      required: property.required,
    };
  } else if (propertyType === 'array') {
    return {
      ...baseSchema,
      type: SchemaType.ARRAY,
      items: toGeminiSchemaProperty(property.items),
    };
  } else {
    const schemaType = SchemaType[propertyType.toUpperCase()] as SchemaType;
    if (!schemaType) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Unsupported property type ${propertyType.toUpperCase()}`,
      });
    }
    return {
      ...baseSchema,
      type: schemaType,
    };
  }
}

function toGeminiMedia(part: Part): GeminiPart {
  let media: GeminiPart;
  if (part.media?.url.startsWith('data:')) {
    // Inline data
    const dataUrl = part.media.url;
    const b64Data = dataUrl.substring(dataUrl.indexOf(',')! + 1);
    const contentType =
      part.media.contentType ||
      dataUrl.substring(dataUrl.indexOf(':')! + 1, dataUrl.indexOf(';'));
    media = { inlineData: { mimeType: contentType, data: b64Data } };
  } else {
    // File data
    if (!part.media?.contentType) {
      throw Error(
        'Must supply a `contentType` when sending File URIs to Gemini.'
      );
    }
    media = {
      fileData: {
        mimeType: part.media.contentType,
        fileUri: part.media.url,
      },
    };
  }

  // Video metadata
  if (part.metadata?.videoMetadata) {
    let videoMetadata = part.metadata.videoMetadata as VideoMetadata;
    media.videoMetadata = { ...videoMetadata };
  }

  // Media resolution
  if (part.metadata?.mediaResolution) {
    media.mediaResolution = { ...part.metadata.mediaResolution };
  }

  return maybeAddGeminiThoughtSignatureAndMetadata(part, media);
}

function toGeminiToolRequest(part: Part): GeminiPart {
  if (!part.toolRequest?.input) {
    throw Error('Invalid ToolRequestPart: input was missing.');
  }
  let functionCall: GeminiPart['functionCall'] = {
    name: part.toolRequest.name,
    args: part.toolRequest.input,
  };
  if (part.toolRequest.ref) {
    functionCall.id = part.toolRequest.ref;
  }
  return maybeAddGeminiThoughtSignatureAndMetadata(part, { functionCall });
}

function toGeminiToolResponse(part: Part): GeminiPart {
  if (!part.toolResponse?.output) {
    throw Error('Invalid ToolResponsePart: output was missing.');
  }
  const functionResponse: GeminiPart['functionResponse'] = {
    name: part.toolResponse.name,
    response: {
      name: part.toolResponse.name,
      content: part.toolResponse.output,
    },
  };
  if (part.toolResponse.content) {
    functionResponse.parts = part.toolResponse.content.map(toGeminiPart);
  }
  if (part.toolResponse.ref) {
    functionResponse.id = part.toolResponse.ref;
  }
  return maybeAddGeminiThoughtSignatureAndMetadata(part, {
    functionResponse,
  });
}

function toGeminiReasoning(part: Part): GeminiPart {
  const out: GeminiPart = { thought: true };
  if (part.reasoning?.length) {
    out.text = part.reasoning;
  }
  return maybeAddGeminiThoughtSignatureAndMetadata(part, out);
}

function toGeminiCustom(part: Part): GeminiPart {
  if (part.custom?.codeExecutionResult) {
    return maybeAddGeminiThoughtSignatureAndMetadata(part, {
      codeExecutionResult: part.custom.codeExecutionResult,
    });
  }
  if (part.custom?.executableCode) {
    return maybeAddGeminiThoughtSignatureAndMetadata(part, {
      executableCode: part.custom.executableCode,
    });
  }
  if (part.custom?.geminiToolCall) {
    return maybeAddGeminiThoughtSignatureAndMetadata(part, {
      toolCall: part.custom.geminiToolCall as any,
    });
  }
  if (part.custom?.geminiToolResponse) {
    return maybeAddGeminiThoughtSignatureAndMetadata(part, {
      toolResponse: part.custom.geminiToolResponse as any,
    });
  }
  throw new Error('Unsupported Custom Part type');
}

function toGeminiText(part: Part): GeminiPart {
  return maybeAddGeminiThoughtSignatureAndMetadata(part, {
    text: part.text ?? '',
  });
}

/**
 * Propagates thought signatures and plugin-specific part metadata from Genkit to Gemini.
 *
 * We nest arbitrary plugin-specific metadata under `part.metadata.metadata` to safely isolate it
 * from Genkit's other metadata fields (e.g. videoMetadata) that are parsed individually.
 */
function maybeAddGeminiThoughtSignatureAndMetadata(
  part: Part,
  geminiPart: GeminiPart
): GeminiPart {
  let updatedPart = geminiPart;

  if (part.metadata) {
    const { thoughtSignature, metadata } = part.metadata;

    if (thoughtSignature) {
      updatedPart = {
        ...updatedPart,
        thoughtSignature: thoughtSignature as string,
      };
    }

    if (isObject(metadata) && Object.keys(metadata).length > 0) {
      updatedPart = {
        ...updatedPart,
        partMetadata: metadata,
      };
    }
  }

  return updatedPart;
}

function toGeminiPart(part: Part): GeminiPart {
  if (typeof part.text === 'string') {
    return toGeminiText(part);
  }
  if (part.media) {
    return toGeminiMedia(part);
  }
  if (part.toolRequest) {
    return toGeminiToolRequest(part);
  }
  if (part.toolResponse) {
    return toGeminiToolResponse(part);
  }
  if (typeof part.reasoning === 'string') {
    return toGeminiReasoning(part);
  }
  if (part.custom) {
    return toGeminiCustom(part);
  }

  throw new Error('Unsupported Part type ' + JSON.stringify(part));
}

function toGeminiRole(
  role: MessageData['role'],
  model?: ModelReference<z.ZodTypeAny>
): string {
  switch (role) {
    case 'user':
      return 'user';
    case 'model':
      return 'model';
    case 'system':
      if (model?.info?.supports?.systemRole) {
        // We should have already pulled out the supported system messages,
        // anything remaining is unsupported; throw an error.
        throw new Error(
          'system role is only supported for a single message in the first position'
        );
      } else {
        throw new Error('system role is not supported');
      }
    case 'tool':
      return 'function';
    default:
      return 'user';
  }
}

export function toGeminiMessage(
  message: MessageData,
  model?: ModelReference<z.ZodTypeAny>
): GeminiContent {
  let sortedParts = message.content;
  if (message.role === 'tool') {
    sortedParts = [...message.content].sort((a, b) => {
      const aRef = a.toolResponse?.ref;
      const bRef = b.toolResponse?.ref;
      if (!aRef && !bRef) return 0;
      if (!aRef) return 1;
      if (!bRef) return -1;
      return parseInt(aRef, 10) - parseInt(bRef, 10);
    });
  }
  return {
    role: toGeminiRole(message.role, model),
    parts: sortedParts.map(toGeminiPart),
  };
}

export function toGeminiSystemInstruction(message: MessageData): GeminiContent {
  return {
    role: 'user',
    parts: message.content.map(toGeminiPart),
  };
}

/**
 * Converts mode from either genkit tool choice (lowercase)
 * or functionCallingConfig (uppercase).
 * @param from The mode to convert from
 * @returns
 */
export function toGeminiFunctionModeEnum(
  from?: string
  //genkitMode: 'auto' | 'required' | 'none'
): FunctionCallingMode | undefined {
  if (from === undefined) {
    return undefined;
  }
  switch (from) {
    case 'MODE_UNSPECIFIED': {
      return FunctionCallingMode.MODE_UNSPECIFIED;
    }
    case 'required':
    case 'ANY': {
      return FunctionCallingMode.ANY;
    }
    case 'auto':
    case 'AUTO': {
      return FunctionCallingMode.AUTO;
    }
    case 'none':
    case 'NONE': {
      return FunctionCallingMode.NONE;
    }
    default:
      throw new Error(`unsupported function calling mode: ${from}`);
  }
}

function fromGeminiFinishReason(
  reason: GeminiCandidate['finishReason']
): CandidateData['finishReason'] {
  if (!reason) return 'unknown';
  switch (reason) {
    case 'STOP':
      return 'stop';
    case 'MAX_TOKENS':
      return 'length';
    case 'SAFETY': // blocked for safety
    case 'RECITATION': // blocked for reciting training data
    case 'LANGUAGE': // blocked for using an unsupported language
    case 'BLOCKLIST': // blocked for forbidden terms
    case 'PROHIBITED_CONTENT': // blocked for potentially containing prohibited content
    case 'SPII': // blocked for potentially containing Sensitive Personally Identifiable Information
      return 'blocked';
    case 'MALFORMED_FUNCTION_CALL':
    case 'MISSING_THOUGHT_SIGNATURE':
    case 'OTHER':
      return 'other';
    default:
      return 'unknown';
  }
}

/**
 * Propagates thought signatures and plugin-specific part metadata from Gemini to Genkit.
 *
 * We nest the Gemini `partMetadata` response under `part.metadata.metadata` to cleanly
 * separate arbitrary API metadata from Genkit's top-level native Part configurations.
 */
function maybeAddThoughtSignatureAndMetadata(
  geminiPart: GeminiPart,
  part: Part
): Part {
  let metadata: Record<string, any> = { ...part?.metadata };
  let hasMetadata = Object.keys(metadata).length > 0;

  if (geminiPart.thoughtSignature) {
    metadata.thoughtSignature = geminiPart.thoughtSignature;
    hasMetadata = true;
  }

  if (geminiPart.partMetadata) {
    metadata.metadata = { ...geminiPart.partMetadata };
    hasMetadata = true;
  }

  if (hasMetadata) {
    return {
      ...part,
      metadata,
    };
  }

  return part;
}

function fromGeminiThought(part: GeminiPart): Part {
  return maybeAddThoughtSignatureAndMetadata(part, {
    reasoning: part.text || '',
  });
}

function fromGeminiInlineData(part: GeminiPart): Part {
  // Check if the required properties exist
  if (
    !part.inlineData ||
    !part.inlineData.hasOwnProperty('mimeType') ||
    !part.inlineData.hasOwnProperty('data')
  ) {
    throw new Error('Invalid GeminiPart: missing required properties');
  }
  const { mimeType, data } = part.inlineData;
  // Combine data and mimeType into a data URL
  const dataUrl = `data:${mimeType};base64,${data}`;

  return maybeAddThoughtSignatureAndMetadata(part, {
    media: {
      url: dataUrl,
      contentType: mimeType,
    },
  });
}

function fromGeminiFileData(part: GeminiPart): Part {
  if (
    !part.fileData ||
    !part.fileData.hasOwnProperty('mimeType') ||
    !part.fileData.hasOwnProperty('fileUri')
  ) {
    throw new Error(
      'Invalid Gemini File Data Part: missing required properties'
    );
  }

  return maybeAddThoughtSignatureAndMetadata(part, {
    media: {
      url: part.fileData?.fileUri,
      contentType: part.fileData?.mimeType,
    },
  });
}

/**
 * Applies Gemini partial args to the target object.
 *
 * https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/Content#PartialArg
 */
export function applyGeminiPartialArgs(
  target: object,
  partialArgs: PartialArg[]
) {
  for (const partialArg of partialArgs) {
    if (!partialArg.jsonPath) {
      continue;
    }
    let value: boolean | string | number | null | undefined;
    if (partialArg.boolValue !== undefined) {
      value = partialArg.boolValue;
    } else if (partialArg.nullValue !== undefined) {
      value = null;
    } else if (partialArg.numberValue !== undefined) {
      value = partialArg.numberValue;
    } else if (partialArg.stringValue !== undefined) {
      value = partialArg.stringValue;
    }
    if (value === undefined) {
      continue;
    }

    let current: any = target;
    const path = JSONPath.toPathArray(partialArg.jsonPath);
    // ex: for path '$.data[0][0]' toPathArray returns: ['$', 'data', '0', '0']
    // we skip the first (root) reference and dereference the rest.
    for (let i = 1; i < path.length - 1; i++) {
      const key = path[i];
      const nextKey = path[i + 1];
      if (current[key] === undefined) {
        if (!isNaN(parseInt(nextKey, 10))) {
          current[key] = [];
        } else {
          current[key] = {};
        }
      }
      current = current[key];
    }

    const finalKey = path[path.length - 1];
    if (
      partialArg.stringValue !== undefined &&
      typeof current[finalKey] === 'string'
    ) {
      current[finalKey] += partialArg.stringValue;
    } else {
      current[finalKey] = value as any;
    }
  }
}

function fromGeminiFunctionCall(
  part: GeminiPart,
  previousChunks?: CandidateData[]
): Part {
  if (!part.functionCall) {
    throw Error(
      'Invalid Gemini Function Call Part: missing function call data'
    );
  }
  const req: Partial<ToolRequest> = {
    name: part.functionCall.name,
    input: part.functionCall.args,
  };

  if (part.functionCall.id) {
    req.ref = part.functionCall.id;
  }

  if (part.functionCall.willContinue) {
    req.partial = true;
  }

  handleFunctionCallPartials(req, part, previousChunks);

  const toolRequest: Part = { toolRequest: req as ToolRequest };

  return maybeAddThoughtSignatureAndMetadata(part, toolRequest);
}

function handleFunctionCallPartials(
  req: Partial<ToolRequest>,
  part: GeminiPart,
  previousChunks?: CandidateData[]
) {
  if (!part.functionCall) {
    throw Error(
      'Invalid Gemini Function Call Part: missing function call data'
    );
  }

  // we try to find if there's a previous partial tool request part.
  const prevPart = previousChunks?.at(-1)?.message.content?.at(-1);
  const prevPartialToolRequestPart =
    prevPart?.toolRequest && prevPart?.toolRequest.partial
      ? prevPart
      : undefined;

  // if the current functionCall has partialArgs, we try to apply the diff to the
  // potentially including the previous partial part.
  if (part.functionCall.partialArgs) {
    const newInput = prevPartialToolRequestPart?.toolRequest?.input
      ? JSON.parse(JSON.stringify(prevPartialToolRequestPart.toolRequest.input))
      : {};
    applyGeminiPartialArgs(newInput, part.functionCall.partialArgs);
    req.input = newInput;
  }

  // If there's a previous partial part, we copy some fields over, because the
  // API will not return these.
  if (prevPartialToolRequestPart) {
    if (!req.name) {
      req.name = prevPartialToolRequestPart.toolRequest.name;
    }
    if (!req.ref) {
      req.ref = prevPartialToolRequestPart.toolRequest.ref;
    }
    // This is a special case for the final partial function call chunk from the API,
    // it will have nothing... so we need to make sure to copy the input
    // from the previous.
    if (req.input === undefined) {
      req.input = prevPartialToolRequestPart.toolRequest.input;
    }
  }
}

function fromGeminiFunctionResponse(part: GeminiPart): Part {
  if (!part.functionResponse) {
    throw new Error(
      'Invalid Gemini Function Call Part: missing function call data'
    );
  }

  let output: unknown = part.functionResponse.response;
  if (isObject(output) && 'name' in output && 'content' in output) {
    output = output.content;
  }

  const toolResponse: Part = {
    toolResponse: {
      name: part.functionResponse.name.replace(/__/g, '/'), // restore slashes
      output,
    },
  };
  if (part.functionResponse.id) {
    toolResponse.toolResponse!.ref = part.functionResponse.id;
  }
  return maybeAddThoughtSignatureAndMetadata(part, toolResponse);
}

function fromExecutableCode(part: GeminiPart): Part {
  if (!part.executableCode) {
    throw new Error('Invalid GeminiPart: missing executableCode');
  }
  return maybeAddThoughtSignatureAndMetadata(part, {
    custom: {
      executableCode: {
        language: part.executableCode.language,
        code: part.executableCode.code,
      },
    },
  });
}

function fromCodeExecutionResult(part: GeminiPart): Part {
  if (!part.codeExecutionResult) {
    throw new Error('Invalid GeminiPart: missing codeExecutionResult');
  }
  return maybeAddThoughtSignatureAndMetadata(part, {
    custom: {
      codeExecutionResult: {
        outcome: part.codeExecutionResult.outcome,
        output: part.codeExecutionResult.output,
      },
    },
  });
}

function fromGeminiText(part: GeminiPart): Part {
  return maybeAddThoughtSignatureAndMetadata(part, {
    text: part.text,
  } as TextPart);
}

function fromGeminiToolCall(part: GeminiPart): Part {
  if (!part.toolCall) {
    throw new Error(
      'Invalid Gemini Function Call Part: missing tool call data'
    );
  }
  return maybeAddThoughtSignatureAndMetadata(part, {
    custom: {
      geminiToolCall: part.toolCall,
    },
  });
}

function fromGeminiToolResponse(part: GeminiPart): Part {
  if (!part.toolResponse) {
    throw new Error(
      'Invalid Gemini Function Call Part: missing tool response data'
    );
  }
  return maybeAddThoughtSignatureAndMetadata(part, {
    custom: {
      geminiToolResponse: part.toolResponse,
    },
  });
}

function fromGeminiPart(
  part: GeminiPart,
  previousChunks?: CandidateData[]
): Part {
  if (part.thought) return fromGeminiThought(part as any);
  if (typeof part.text === 'string') return fromGeminiText(part);
  if (part.inlineData) return fromGeminiInlineData(part);
  if (part.fileData) return fromGeminiFileData(part);
  if (part.functionCall) return fromGeminiFunctionCall(part, previousChunks);
  if (part.functionResponse) return fromGeminiFunctionResponse(part);
  if (part.toolCall) return fromGeminiToolCall(part);
  if (part.toolResponse) return fromGeminiToolResponse(part);
  if (part.executableCode) return fromExecutableCode(part);
  if (part.codeExecutionResult) return fromCodeExecutionResult(part);
  if (part.thoughtSignature) return fromGeminiThought(part as any);

  throw new Error('Unsupported GeminiPart type ' + JSON.stringify(part));
}

export function fromGeminiCandidate(
  candidate: GeminiCandidate,
  previousChunks?: CandidateData[]
): CandidateData {
  const parts = candidate.content?.parts || [];
  const genkitCandidate: CandidateData = {
    index: candidate.index || 0,
    message: {
      role: 'model',
      content: parts
        // the model sometimes returns empty parts, ignore those.
        .filter((p) => Object.keys(p).length > 0)
        .map((part) => fromGeminiPart(part, previousChunks)),
    },
    finishReason: fromGeminiFinishReason(candidate.finishReason),
    finishMessage: candidate.finishMessage,
    custom: {
      safetyRatings: candidate.safetyRatings,
      citationMetadata: candidate.citationMetadata,
    },
  };

  return genkitCandidate;
}

export const TEST_ONLY = { fromGeminiPart, toGeminiPart };
