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

import type { MediaPart, Part } from 'genkit';
import { extractMimeType } from './utils.js';

// Helpers for mapping `toolResponse.content` (multipart tool responses) to
// Gemini function responses. The plugin declares
// `supports.toolResponseContent`, so it must deliver every part or degrade it
// to text, never fail the request or drop it silently.

/** The MIME type of a media part, from `contentType` or its URL. */
export function mediaMimeType(part: MediaPart): string {
  return part.media.contentType || extractMimeType(part.media.url);
}

/**
 * Document types Gemini accepts in function results: PDF, CSV and Office /
 * OpenDocument files.
 */
export function isGeminiDocumentMimeType(mimeType: string): boolean {
  return (
    mimeType === 'application/pdf' ||
    mimeType === 'text/csv' ||
    mimeType === 'application/msword' ||
    mimeType === 'application/vnd.ms-excel' ||
    mimeType === 'application/vnd.ms-powerpoint' ||
    mimeType.startsWith('application/vnd.openxmlformats-officedocument.') ||
    mimeType.startsWith('application/vnd.oasis.opendocument.')
  );
}

/**
 * Whether Gemini accepts a media type in a function result. Unknown types
 * (empty string, e.g. a URL without an extension) are passed through and left
 * to the API.
 */
export function isGeminiToolResultMimeType(mimeType: string): boolean {
  return (
    !mimeType ||
    mimeType.startsWith('image/') ||
    mimeType.startsWith('audio/') ||
    mimeType.startsWith('video/') ||
    mimeType.startsWith('text/') ||
    isGeminiDocumentMimeType(mimeType)
  );
}

/** Text standing in for a tool response part Gemini can't take as is. */
export function toolContentNote(part: Part): string {
  if (part.resource) return `Resource: ${part.resource.uri}`;
  if (part.media) {
    return `[${mediaMimeType(part as MediaPart) || 'media'} content omitted: not supported by the model]`;
  }
  return JSON.stringify(part.data ?? part);
}

/**
 * Decodes the text of an inline (`data:`) media URL. Malformed payloads are
 * returned as is rather than failing the request.
 */
export function decodeDataUrlText(url: string): string {
  const comma = url.indexOf(',');
  if (comma === -1) return '';
  const data = url.substring(comma + 1);
  try {
    return url.substring(0, comma).endsWith(';base64')
      ? new TextDecoder().decode(
          Uint8Array.from(atob(data), (c) => c.charCodeAt(0))
        )
      : decodeURIComponent(data);
  } catch {
    return data;
  }
}
