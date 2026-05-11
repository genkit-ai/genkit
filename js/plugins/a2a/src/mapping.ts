/**
 * Copyright 2026 Google LLC
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

import type {
  Artifact as A2AArtifact,
  Part as A2APart,
  DataPart,
  FilePart,
  FileWithBytes,
  FileWithUri,
  TextPart,
} from '@a2a-js/sdk';
import type { Part as GenkitPart } from 'genkit';

/**
 * Set of keys that identify a valid Genkit Part.
 */
const GENKIT_PART_KEYS = new Set([
  'text',
  'media',
  'toolRequest',
  'toolResponse',
  'data',
  'custom',
]);

/**
 * Returns true if the value looks like a Genkit Part (has at least one known key).
 */
function isGenkitPart(value: unknown): value is GenkitPart {
  return (
    value !== null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.keys(value as object).some((k) => GENKIT_PART_KEYS.has(k))
  );
}

// ---------------------------------------------------------------------------
// Genkit → A2A
// ---------------------------------------------------------------------------

/**
 * Maps a Genkit Part to an A2A Part.
 */
export function mapGenkitPartToA2A(part: GenkitPart): A2APart {
  if (part.text !== undefined) {
    return { kind: 'text', text: part.text } as TextPart;
  }

  if (part.media) {
    const url = part.media.url;
    const mimeType = part.media.contentType;

    if (url.startsWith('data:')) {
      const match = url.match(/^data:([^;]+);base64,(.+)$/);
      if (match) {
        return {
          kind: 'file',
          file: {
            bytes: match[2],
            mimeType: match[1] || mimeType,
            name: 'inline_file',
          },
        } as FilePart;
      }
    }
    return {
      kind: 'file',
      file: {
        uri: url,
        mimeType,
        name: 'remote_file',
      },
    } as FilePart;
  }

  if (part.toolRequest) {
    return {
      kind: 'data',
      data: {
        id: part.toolRequest.ref || crypto.randomUUID(),
        name: part.toolRequest.name,
        args: part.toolRequest.input,
      },
      metadata: { genkit_type: 'function_call' },
    } as DataPart;
  }

  if (part.toolResponse) {
    return {
      kind: 'data',
      data: {
        id: part.toolResponse.ref || crypto.randomUUID(),
        name: part.toolResponse.name,
        response: part.toolResponse.output,
      },
      metadata: { genkit_type: 'function_response' },
    } as DataPart;
  }

  // Fallback: wrap entire part as data
  return { kind: 'data', data: part } as DataPart;
}

// ---------------------------------------------------------------------------
// A2A → Genkit
// ---------------------------------------------------------------------------

/**
 * Maps an A2A Part to a Genkit Part.
 *
 * Text parts that were serialized as JSON by mapGenkitPartToA2A are restored
 * only when they parse to a valid Genkit Part object.
 */
export function mapA2APartToGenkit(part: A2APart): GenkitPart {
  if (part.kind === 'text') {
    const textPart = part as TextPart;
    // Attempt to restore Genkit parts that were JSON-serialized as text
    try {
      const parsed = JSON.parse(textPart.text);
      if (isGenkitPart(parsed)) {
        return parsed;
      }
    } catch {
      // Not JSON — just a regular text part
    }
    return { text: textPart.text };
  }

  if (part.kind === 'data') {
    const dataPart = part as DataPart;

    if (dataPart.metadata?.genkit_type === 'function_call') {
      const data = dataPart.data as {
        id?: string;
        name: string;
        args?: unknown;
      };
      return {
        toolRequest: {
          ref: data.id,
          name: data.name,
          input: data.args,
        },
      };
    }

    if (dataPart.metadata?.genkit_type === 'function_response') {
      const data = dataPart.data as {
        id?: string;
        name: string;
        response?: unknown;
      };
      return {
        toolResponse: {
          ref: data.id,
          name: data.name,
          output: data.response,
        },
      };
    }

    // Check if the data itself is a Genkit Part
    if (isGenkitPart(dataPart.data)) {
      return dataPart.data as GenkitPart;
    }

    return { data: dataPart.data };
  }

  if (part.kind === 'file') {
    const filePart = part as FilePart;
    const file = filePart.file;

    if ('bytes' in file) {
      const bytesFile = file as FileWithBytes;
      return {
        media: {
          url: `data:${bytesFile.mimeType || 'application/octet-stream'};base64,${bytesFile.bytes}`,
          contentType: bytesFile.mimeType,
        },
      };
    } else if ('uri' in file) {
      const uriFile = file as FileWithUri;
      return {
        media: {
          url: uriFile.uri,
          contentType: uriFile.mimeType,
        },
      };
    }
  }

  // Fallback
  return { text: JSON.stringify(part) };
}

// ---------------------------------------------------------------------------
// Artifact mapping
// ---------------------------------------------------------------------------

/** Genkit Artifact type (from Genkit Agent) */
export interface GenkitArtifact {
  name?: string;
  parts: GenkitPart[];
  metadata?: Record<string, unknown>;
}

/**
 * Maps a Genkit Artifact to an A2A Artifact.
 * Artifact.name is required — throws if absent.
 */
export function mapGenkitArtifactToA2A(artifact: GenkitArtifact): A2AArtifact {
  if (!artifact.name) {
    throw new Error(
      'Artifact.name is required when using the A2A adapter. ' +
        'Set a unique name on each artifact to serve as its A2A artifactId.'
    );
  }

  const { a2a: a2aMeta, ...restMetadata } = (artifact.metadata || {}) as {
    a2a?: Record<string, unknown>;
    [key: string]: unknown;
  };
  const a2aOverrides = (a2aMeta || {}) as Record<string, unknown>;

  return {
    artifactId: artifact.name,
    name: (a2aOverrides.name as string) ?? artifact.name,
    ...(a2aOverrides.description !== undefined && {
      description: a2aOverrides.description as string,
    }),
    parts: artifact.parts.map(mapGenkitPartToA2A),
    ...(Object.keys(restMetadata).length > 0 && { metadata: restMetadata }),
  } as A2AArtifact;
}

/**
 * Maps an A2A Artifact to a Genkit Artifact.
 * The A2A artifactId becomes Genkit name (deduplication key).
 */
export function mapA2AArtifactToGenkit(artifact: A2AArtifact): GenkitArtifact {
  const a2aMeta: Record<string, unknown> = {};
  if (artifact.name !== undefined) a2aMeta.name = artifact.name;
  if (artifact.description !== undefined)
    a2aMeta.description = artifact.description;
  if ((artifact as any).extensions !== undefined)
    a2aMeta.extensions = (artifact as any).extensions;
  if (artifact.metadata !== undefined) a2aMeta.metadata = artifact.metadata;

  return {
    name: artifact.artifactId,
    parts: artifact.parts.map(mapA2APartToGenkit),
    ...(Object.keys(a2aMeta).length > 0 && { metadata: { a2a: a2aMeta } }),
  };
}
