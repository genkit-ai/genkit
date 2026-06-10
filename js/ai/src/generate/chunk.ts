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

import { GenkitError } from '@genkit-ai/core';
import { extractJson } from '../extract.js';
import type {
  GenerateResponseChunkData,
  Part,
  Role,
  ToolRequestPart,
} from '../model.js';

/** A function that parses a {@link GenerateResponseChunk} into a typed output value. */
export type ChunkParser<T = unknown> = (chunk: GenerateResponseChunk<T>) => T;

/**
 * Represents a single chunk of a streaming model response. Provides convenience
 * accessors for extracting text, media, tool requests, and accumulated output
 * from the chunk and all preceding chunks in the stream.
 */
export class GenerateResponseChunk<T = unknown>
  implements GenerateResponseChunkData
{
  /** The index of the message this chunk corresponds to, starting with `0` for the first model response of the generation. */
  index: number;
  /** The role of the message this chunk corresponds to. Will always be `model` or `tool`. */
  role: Role;
  /** The content generated in this chunk. */
  content: Part[];
  /** Custom model-specific data for this chunk. */
  custom?: unknown;
  /** Accumulated chunks for partial output extraction. */
  previousChunks?: GenerateResponseChunkData[];
  /** The parser to be used to parse `output` from this chunk. */
  parser?: ChunkParser<T>;

  constructor(
    data: GenerateResponseChunkData,
    options: {
      previousChunks?: GenerateResponseChunkData[];
      role: Role;
      index: number;
      parser?: ChunkParser<T>;
    }
  ) {
    this.content = data.content || [];
    this.custom = data.custom;
    this.previousChunks = options.previousChunks
      ? [...options.previousChunks]
      : undefined;
    this.index = options.index;
    this.role = options.role;
    this.parser = options.parser;
  }

  /**
   * Concatenates all `text` parts present in the chunk with no delimiter.
   * @returns A string of all concatenated text parts.
   */
  get text(): string {
    return this.content.map((part) => part.text || '').join('');
  }

  /**
   * Concatenates all `reasoning` parts present in the chunk with no delimiter.
   * @returns A string of all concatenated reasoning parts.
   */
  get reasoning(): string {
    return this.content.map((part) => part.reasoning || '').join('');
  }

  /**
   * Concatenates all `text` parts of all chunks from the response thus far.
   * @returns A string of all concatenated chunk text content.
   */
  get accumulatedText(): string {
    return this.previousText + this.text;
  }

  /**
   * Concatenates all `text` parts of all preceding chunks.
   */
  get previousText(): string {
    if (!this.previousChunks)
      throw new GenkitError({
        status: 'FAILED_PRECONDITION',
        message: 'Cannot compose accumulated text without previous chunks.',
      });

    return this.previousChunks
      ?.map((c) => c.content.map((p) => p.text || '').join(''))
      .join('');
  }

  /**
   * Returns the first media part detected in the chunk. Useful for extracting
   * (for example) an image from a generation expected to create one.
   * @returns The first detected `media` part in the chunk.
   */
  get media(): { url: string; contentType?: string } | null {
    return this.content.find((part) => part.media)?.media || null;
  }

  /**
   * Returns the first detected `data` part of a chunk.
   * @returns The first `data` part detected in the chunk (if any).
   */
  get data(): T | null {
    return this.content.find((part) => part.data)?.data as T | null;
  }

  /**
   * Returns all tool request found in this chunk.
   * @returns Array of all tool request found in this chunk.
   */
  get toolRequests(): ToolRequestPart[] {
    return this.content.filter(
      (part) => !!part.toolRequest
    ) as ToolRequestPart[];
  }

  /**
   * Parses the chunk into the desired output format using the parser associated
   * with the generate request, or falls back to naive JSON parsing otherwise.
   */
  get output(): T | null {
    if (this.parser) return this.parser(this);
    return this.data || extractJson(this.accumulatedText);
  }

  toJSON(): GenerateResponseChunkData {
    const data = {
      role: this.role,
      index: this.index,
      content: this.content,
    } as GenerateResponseChunkData;
    if (this.custom) {
      data.custom = this.custom;
    }
    return data;
  }
}
