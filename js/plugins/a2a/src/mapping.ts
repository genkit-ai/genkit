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
  DataPart as A2ADataPart,
  FilePart as A2AFilePart,
  Message as A2AMessage,
  Part as A2APart,
  TextPart as A2ATextPart,
} from '@a2a-js/sdk';
import type { ToolRequestPart, ToolResponsePart } from 'genkit';
import type { AgentInput, MessageData, Part } from 'genkit/beta';

/**
 * Metadata keys used to carry Genkit-specific information across the A2A wire.
 *
 * A2A's part model (`text` | `file` | `data`) is narrower than Genkit's, so we
 * encode the extra Genkit semantics in part `metadata` under a `genkit:`
 * namespace. This keeps a Genkit ↔ Genkit round-trip lossless while remaining
 * interoperable with generic A2A clients (which simply ignore the metadata and
 * see plain text / file / data parts).
 */
export const A2A_METADATA = {
  /**
   * Discriminates the Genkit part flavor a `DataPart`/`TextPart` originated
   * from: `reasoning`, `toolRequest`, `toolResponse`, `data`, or `custom`.
   */
  TYPE: 'genkit:type',
  /**
   * Present on a `text` part that was a Genkit `reasoning` part (ADK-style
   * "thought" flag). The reasoning text is carried in the text field.
   */
  REASONING: 'genkit:reasoning',
  /**
   * Present on a `toolRequest` data part that paused the turn as an interrupt.
   * Carries the interrupt metadata (or `true`).
   */
  INTERRUPT: 'genkit:interrupt',
  /**
   * Present on an inbound resume `toolRequest` data part to indicate the tool
   * should be *restarted* (re-run) rather than responded to. Carries the
   * `resumed` metadata (or `true`).
   */
  RESTART: 'genkit:restart',
  /**
   * Preserves the original Genkit part `metadata` so it survives the
   * round-trip (the `genkit:*` keys above are kept separate from it).
   */
  PART_METADATA: 'genkit:partMetadata',
} as const;

/**
 * Genkit part "flavors" encoded in `A2A_METADATA.TYPE`.
 */
export const GenkitPartType = {
  REASONING: 'reasoning',
  TOOL_REQUEST: 'toolRequest',
  TOOL_RESPONSE: 'toolResponse',
  DATA: 'data',
  CUSTOM: 'custom',
} as const;

type Metadata = Record<string, unknown>;

/**
 * Reads the preserved original Genkit part metadata from an A2A part's
 * metadata blob (stored under {@link A2A_METADATA.PART_METADATA}).
 */
function readPartMetadata(meta?: Metadata): Metadata | undefined {
  const preserved = meta?.[A2A_METADATA.PART_METADATA];
  if (preserved && typeof preserved === 'object' && !Array.isArray(preserved)) {
    return preserved as Metadata;
  }
  return undefined;
}

/**
 * Builds the A2A part `metadata` object for a Genkit part, merging the
 * type discriminator, any extra `genkit:*` keys, and the preserved original
 * part metadata. Returns `undefined` when there is nothing to attach.
 */
function buildPartMetadata(
  type: string,
  partMetadata?: Metadata,
  extra?: Metadata
): Metadata | undefined {
  const meta: Metadata = { [A2A_METADATA.TYPE]: type, ...extra };
  if (partMetadata && Object.keys(partMetadata).length > 0) {
    meta[A2A_METADATA.PART_METADATA] = partMetadata;
  }
  return meta;
}

// ---------------------------------------------------------------------------
// Genkit Part -> A2A Part
// ---------------------------------------------------------------------------

/**
 * Determines whether a Genkit media url is an inline `data:` URI (vs a remote
 * resource), so it can be mapped to A2A `FileWithBytes` instead of
 * `FileWithUri`.
 */
function parseDataUri(
  url: string
): { mimeType?: string; bytes: string } | undefined {
  const match = /^data:([^;,]*)?(;base64)?,(.*)$/s.exec(url);
  if (!match) return undefined;
  const [, mimeType, isBase64, data] = match;
  // Only base64 inline data maps cleanly to A2A `bytes`; percent-encoded text
  // data uris are left as a uri reference.
  if (!isBase64) return undefined;
  return { mimeType: mimeType || undefined, bytes: data };
}

/**
 * Maps a single Genkit {@link Part} to an A2A {@link A2APart}, or `undefined`
 * when the part has no A2A representation (e.g. an empty/resource part).
 */
export function genkitPartToA2A(part: Part): A2APart | undefined {
  const partMetadata = part.metadata as Metadata | undefined;

  if (typeof part.text === 'string') {
    const meta = buildPartMetadata('text', partMetadata);
    // 'text' is the default flavor; drop the discriminator to keep plain text
    // parts clean for generic A2A clients.
    if (meta && Object.keys(meta).length === 1) {
      const out: A2ATextPart = { kind: 'text', text: part.text };
      if (partMetadata && Object.keys(partMetadata).length > 0) {
        out.metadata = { [A2A_METADATA.PART_METADATA]: partMetadata };
      }
      return out;
    }
    return { kind: 'text', text: part.text, metadata: meta };
  }

  if (typeof part.reasoning === 'string') {
    return {
      kind: 'text',
      text: part.reasoning,
      metadata: buildPartMetadata(GenkitPartType.REASONING, partMetadata, {
        [A2A_METADATA.REASONING]: true,
      }),
    };
  }

  if (part.media) {
    const dataUri = parseDataUri(part.media.url);
    const meta = buildPartMetadata('file', partMetadata);
    if (dataUri) {
      return {
        kind: 'file',
        file: {
          bytes: dataUri.bytes,
          mimeType: part.media.contentType ?? dataUri.mimeType,
        },
        metadata: meta,
      };
    }
    return {
      kind: 'file',
      file: {
        uri: part.media.url,
        mimeType: part.media.contentType,
      },
      metadata: meta,
    };
  }

  if (part.toolRequest) {
    const interrupt = (partMetadata as Metadata | undefined)?.interrupt;
    return {
      kind: 'data',
      data: { ...part.toolRequest } as Record<string, unknown>,
      metadata: buildPartMetadata(
        GenkitPartType.TOOL_REQUEST,
        partMetadata,
        interrupt !== undefined
          ? { [A2A_METADATA.INTERRUPT]: interrupt }
          : undefined
      ),
    };
  }

  if (part.toolResponse) {
    return {
      kind: 'data',
      data: { ...part.toolResponse } as Record<string, unknown>,
      metadata: buildPartMetadata(GenkitPartType.TOOL_RESPONSE, partMetadata),
    };
  }

  if (part.data !== undefined) {
    return {
      kind: 'data',
      data: (typeof part.data === 'object' && part.data !== null
        ? part.data
        : { value: part.data }) as Record<string, unknown>,
      metadata: buildPartMetadata(GenkitPartType.DATA, partMetadata),
    };
  }

  if (part.custom) {
    return {
      kind: 'data',
      data: part.custom as Record<string, unknown>,
      metadata: buildPartMetadata(GenkitPartType.CUSTOM, partMetadata),
    };
  }

  // Empty / resource parts have no clean A2A representation.
  return undefined;
}

/**
 * Maps an array of Genkit {@link Part}s to A2A {@link A2APart}s, dropping
 * parts that have no A2A representation.
 */
export function genkitPartsToA2A(parts?: Part[]): A2APart[] {
  return (parts ?? [])
    .map((p) => genkitPartToA2A(p))
    .filter((p): p is A2APart => p !== undefined);
}

// ---------------------------------------------------------------------------
// A2A Part -> Genkit Part
// ---------------------------------------------------------------------------

/**
 * Reattaches preserved Genkit part metadata to a freshly-built Genkit part.
 */
function withPartMetadata<T extends Part>(part: T, meta?: Metadata): T {
  const preserved = readPartMetadata(meta);
  return preserved ? ({ ...part, metadata: preserved } as T) : part;
}

/**
 * Maps a single A2A {@link A2APart} to a Genkit {@link Part}.
 *
 * Uses the `genkit:type` discriminator (when present) to reconstruct the exact
 * Genkit part flavor; otherwise falls back to a structural interpretation so
 * parts produced by generic (non-Genkit) A2A clients still map sensibly.
 */
export function a2aPartToGenkit(part: A2APart): Part | undefined {
  const meta = part.metadata as Metadata | undefined;
  const type = meta?.[A2A_METADATA.TYPE];

  if (part.kind === 'text') {
    const textPart = part as A2ATextPart;
    if (type === GenkitPartType.REASONING || meta?.[A2A_METADATA.REASONING]) {
      return withPartMetadata({ reasoning: textPart.text }, meta);
    }
    return withPartMetadata({ text: textPart.text }, meta);
  }

  if (part.kind === 'file') {
    const filePart = part as A2AFilePart;
    const file = filePart.file;
    if ('bytes' in file) {
      const mimeType = file.mimeType ?? 'application/octet-stream';
      return withPartMetadata(
        {
          media: {
            url: `data:${mimeType};base64,${file.bytes}`,
            contentType: file.mimeType,
          },
        },
        meta
      );
    }
    return withPartMetadata(
      { media: { url: file.uri, contentType: file.mimeType } },
      meta
    );
  }

  if (part.kind === 'data') {
    const dataPart = part as A2ADataPart;
    const data = dataPart.data as Record<string, unknown>;

    if (type === GenkitPartType.TOOL_REQUEST) {
      const interrupt = meta?.[A2A_METADATA.INTERRUPT];
      const base = withPartMetadata(
        { toolRequest: data as Part['toolRequest'] } as Part,
        meta
      );
      if (interrupt !== undefined) {
        return {
          ...base,
          metadata: { ...(base.metadata as Metadata), interrupt },
        };
      }
      return base;
    }

    if (type === GenkitPartType.TOOL_RESPONSE) {
      return withPartMetadata(
        { toolResponse: data as Part['toolResponse'] } as Part,
        meta
      );
    }

    if (type === GenkitPartType.CUSTOM) {
      return withPartMetadata({ custom: data } as Part, meta);
    }

    // Default (including `genkit:type === 'data'` and generic A2A data parts).
    return withPartMetadata({ data } as Part, meta);
  }

  return undefined;
}

/**
 * Maps an array of A2A {@link A2APart}s to Genkit {@link Part}s, dropping
 * parts that have no Genkit representation.
 */
export function a2aPartsToGenkit(parts?: A2APart[]): Part[] {
  return (parts ?? [])
    .map((p) => a2aPartToGenkit(p))
    .filter((p): p is Part => p !== undefined);
}

// ---------------------------------------------------------------------------
// Message-level mapping
// ---------------------------------------------------------------------------

/**
 * Maps an A2A role to a Genkit message role.
 */
function a2aRoleToGenkit(role: A2AMessage['role']): MessageData['role'] {
  return role === 'agent' ? 'model' : 'user';
}

/**
 * Maps a Genkit message role to an A2A role (`user` for everything but
 * `model`, which becomes `agent`).
 */
export function genkitRoleToA2A(role: MessageData['role']): A2AMessage['role'] {
  return role === 'model' ? 'agent' : 'user';
}

/**
 * Maps an A2A {@link A2AMessage} to a Genkit {@link MessageData}.
 */
export function a2aMessageToGenkit(message: A2AMessage): MessageData {
  return {
    role: a2aRoleToGenkit(message.role),
    content: a2aPartsToGenkit(message.parts),
  };
}

/**
 * Maps a Genkit {@link MessageData} to A2A parts (the body of an A2A message).
 */
export function genkitMessageToA2AParts(message: MessageData): A2APart[] {
  return genkitPartsToA2A(message.content);
}

// ---------------------------------------------------------------------------
// Resume detection (A2A input message -> Genkit resume payload)
// ---------------------------------------------------------------------------

/**
 * Inspects an incoming A2A message that targets a task currently paused in the
 * `input-required` state and builds the Genkit {@link AgentInput} for the next
 * turn.
 *
 * Tool-response data parts become `resume.respond` entries; tool-request data
 * parts tagged with {@link A2A_METADATA.RESTART} become `resume.restart`
 * entries. When the message carries no such parts it is treated as a fresh
 * user turn (`{ message }`).
 */
export function a2aMessageToResumeInput(message: A2AMessage): AgentInput {
  const respond: ToolResponsePart[] = [];
  const restart: ToolRequestPart[] = [];

  for (const part of message.parts) {
    if (part.kind !== 'data') continue;
    const meta = part.metadata as Metadata | undefined;
    const type = meta?.[A2A_METADATA.TYPE];
    const data = (part as A2ADataPart).data as Record<string, unknown>;

    if (type === GenkitPartType.TOOL_RESPONSE) {
      respond.push({
        toolResponse: data as ToolResponsePart['toolResponse'],
      });
    } else if (
      type === GenkitPartType.TOOL_REQUEST &&
      meta?.[A2A_METADATA.RESTART] !== undefined
    ) {
      const resumed = meta[A2A_METADATA.RESTART];
      restart.push({
        toolRequest: data as ToolRequestPart['toolRequest'],
        metadata: { resumed: resumed === true ? true : resumed },
      });
    }
  }

  if (respond.length === 0 && restart.length === 0) {
    return { message: a2aMessageToGenkit(message) };
  }

  return {
    resume: {
      ...(respond.length > 0 && { respond }),
      ...(restart.length > 0 && { restart }),
    },
  };
}
