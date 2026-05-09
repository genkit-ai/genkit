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

import { z } from '@genkit-ai/core';
import type { Embedding } from './embedder.js';
import { PartSchema, type Part } from './parts.js';

// We need both metadata and embedMetadata because they can
// contain the same fields (e.g. video start/stop) with different values.
export const DocumentDataSchema = z.object({
  content: z.array(PartSchema),
  metadata: z.record(z.string(), z.any()).optional(),
});
export type DocumentData = z.infer<typeof DocumentDataSchema>;

function deepCopy<T>(value: T): T {
  if (value === undefined) {
    return value;
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

/**
 * Document represents document content along with its metadata that can be embedded, indexed or
 * retrieved. Each document can contain multiple parts (for example text and an image)
 */
export class Document implements DocumentData {
  content: Part[];
  metadata?: Record<string, any>;

  constructor(data: DocumentData) {
    this.content = deepCopy(data.content);
    this.metadata = deepCopy(data.metadata);
  }

  static fromParts(content: Part[], metadata?: Record<string, any>) {
    return new Document({
      content,
      metadata,
    });
  }

  static fromText(text: string, metadata?: Record<string, any>) {
    return new Document({
      content: [{ text }],
      metadata,
    });
  }

  // Construct a Document from a single media item
  static fromMedia(
    url: string,
    contentType?: string,
    metadata?: Record<string, unknown>
  ) {
    return new Document({
      content: [
        {
          media: {
            contentType,
            url,
          },
        },
      ],
      metadata,
    });
  }

  // Construct a Document from content
  static fromData(
    data: string,
    dataType?: string,
    metadata?: Record<string, unknown>
  ) {
    if (dataType === 'text') {
      return this.fromText(data, metadata);
    }
    return this.fromMedia(data, dataType, metadata);
  }

  /**
   * Concatenates all `text` parts present in the document with no delimiter.
   * @returns A string of all concatenated text parts.
   */
  get text(): string {
    return this.content.map((part) => part.text || '').join('');
  }

  /**
   * Media array getter.
   * @returns the array of media parts.
   */
  get media(): { url: string; contentType?: string }[] {
    return this.content
      .filter((part) => part.media && !part.text)
      .map((part) => part.media!);
  }

  /**
   * Gets the first item in the document. Either text or media url.
   */
  get data(): string {
    //
    if (this.text) {
      return this.text;
    }
    if (this.media) {
      return this.media[0].url;
    }
    return '';
  }

  /**
   * Gets the contentType of the data that is returned by data()
   */
  get dataType(): string | undefined {
    if (this.text) {
      return 'text';
    }
    if (this.media && this.media[0].contentType) {
      return this.media[0].contentType;
    }
    return undefined;
  }

  toJSON(): DocumentData {
    return {
      content: deepCopy(this.content),
      metadata: deepCopy(this.metadata),
    } as DocumentData;
  }

  /**
   * Embedders may return multiple embeddings for a single document.
   * But storage still requires a 1:1 relationship. So we create an
   * array of Documents from a single document - one per embedding.
   * @param embeddings The embeddings to create the documents from.
   * @returns an array of documents based on this document and the embeddings.
   */
  getEmbeddingDocuments(embeddings: Embedding[]): Document[] {
    const documents: Document[] = [];
    for (const embedding of embeddings) {
      const jsonDoc = this.toJSON();
      if (embedding.metadata) {
        if (!jsonDoc.metadata) {
          jsonDoc.metadata = {};
        }
        jsonDoc.metadata.embedMetadata = embedding.metadata;
      }
      documents.push(new Document(jsonDoc));
    }
    checkUniqueDocuments(documents);
    return documents;
  }
}

// Unique documents are important because we key
// our vector storage on the Md5 hash of the JSON.stringify(document)
// So if we have multiple duplicate documents with
// different embeddings, we will either skip or overwrite
// those entries and lose embedding information.
// Export and boolean return value for testing only.
export function checkUniqueDocuments(documents: Document[]): boolean {
  const seen = new Set();
  for (const doc of documents) {
    const serialized = JSON.stringify(doc);
    if (seen.has(serialized)) {
      console.warn(
        'Warning: embedding documents are not unique. Are you missing embed metadata?'
      );
      return false;
    }
    seen.add(serialized);
  }
  return true;
}
