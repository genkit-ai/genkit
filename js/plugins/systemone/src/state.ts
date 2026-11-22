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

import {
  GenkitError,
  type DocumentData,
  type GenerateRequest,
  type MessageData,
  type Part,
} from 'genkit';
import { Options } from './protocol.js';

/** Which kinds of media a model reads besides text and JSON. */
export interface Reads {
  images: boolean;
  audio: boolean;
  video: boolean;
}

/**
 * The media a request carries, as base64 data URLs by kind, each in the order
 * its parts appear, messages first and documents after. Media is an extension
 * some servers make to the protocol, in the images, audio, and videos fields.
 */
export interface Media {
  images: string[];
  audio: string[];
  videos: string[];
}

/**
 * JSON text sent verbatim, which a `stateJSON` text part becomes, so a number
 * too large for a double reaches the model unchanged. The request body is
 * serialized with {@link stringifyBody}, which writes it in place.
 */
export class RawJson {
  constructor(readonly text: string) {}
}

/**
 * Serializes a request body with every {@link RawJson} in it written as its
 * text, and every {@link Options} as an object whose keys keep their order.
 */
export function stringifyBody(body: unknown): string {
  const raws: string[] = [];
  const token = `systemone-raw-${Math.random().toString(36).slice(2)}-`;
  const text = JSON.stringify(body, (_key, value) => {
    if (value instanceof RawJson) {
      raws.push(value.text);
      return `${token}${raws.length - 1}`;
    }
    if (value instanceof Options) {
      const members = value.entries.map(
        ([key, description]) =>
          `${JSON.stringify(key)}:${stringifyBody(description)}`
      );
      raws.push(`{${members.join(',')}}`);
      return `${token}${raws.length - 1}`;
    }
    return value;
  });
  if (raws.length === 0) return text;
  return text.replace(
    new RegExp(`"${token}(\\d+)"`, 'g'),
    (_match, i: string) => raws[Number(i)]
  );
}

function invalidArgument(message: string): GenkitError {
  return new GenkitError({
    status: 'INVALID_ARGUMENT',
    message: `systemone: ${message}`,
  });
}

/**
 * Gathers the system messages into the text that goes in front of every
 * question's instructions. The model has no system role: the state is what it
 * evaluates and the questions are what it is told, so a system message belongs
 * with the questions, never in the state. Only text can be instructions; loop
 * plumbing left in a history is skipped.
 */
export function systemPreamble(messages: MessageData[]): string {
  const texts: string[] = [];
  for (const msg of messages) {
    if (msg.role !== 'system') continue;
    let text = '';
    for (const part of msg.content) {
      if (isFormatInstructions(part)) continue;
      if (typeof part.text !== 'string') {
        throw invalidArgument(
          `a ${partKindName(part)} part cannot be instructions; a system message is text only`
        );
      }
      text += part.text;
    }
    text = text.trim();
    if (text !== '') texts.push(text);
  }
  return texts.join('\n\n');
}

/**
 * Turns the request into the state and the media: one message as its value,
 * several as `{role, content}` records, and with documents attached an object
 * of both, so a question can name the context it is about. System messages are
 * instructions, not state, and are left out.
 *
 * Media parts of a kind the model reads become the request's media and
 * contribute nothing to the state; a media part of any other kind is refused.
 * A message of media alone is still a turn, with the empty string as its
 * content, so the turns around it keep their roles and a request of that one
 * message sends the empty string as its state.
 */
export function buildState(
  req: GenerateRequest,
  stateJSON: boolean,
  reads: Reads
): { state: unknown; media: Media } {
  const b = new StateBuilder(reads);
  const records: { role: string; content: unknown }[] = [];
  for (const msg of req.messages) {
    if (msg.role === 'system') continue;
    const collected = b.mediaCount();
    let value = b.messageValue(msg.content, stateJSON);
    if (value === undefined) {
      if (b.mediaCount() === collected) continue;
      value = '';
    }
    records.push({ role: msg.role, content: value });
  }
  if (records.length === 0) {
    throw invalidArgument(
      'the request carries no state; a system message is instructions, so pass a prompt or messages too'
    );
  }
  const docs: unknown[] = [];
  for (const doc of req.docs ?? []) {
    const value = b.documentValue(doc);
    if (value !== undefined) docs.push(value);
  }
  if (docs.length > 0) {
    return { state: { messages: records, context: docs }, media: b.media };
  }
  if (records.length === 1) {
    return { state: records[0].content, media: b.media };
  }
  return { state: records, media: b.media };
}

/**
 * Reads the state out of messages and documents, and collects the media it
 * meets on the way.
 */
class StateBuilder {
  readonly media: Media = { images: [], audio: [], videos: [] };

  constructor(private readonly reads: Reads) {}

  mediaCount(): number {
    return (
      this.media.images.length +
      this.media.audio.length +
      this.media.videos.length
    );
  }

  /**
   * A message's contribution to the state: the value of its one part, or the
   * values of all of them. A message with nothing to contribute yields
   * undefined.
   */
  messageValue(content: Part[], stateJSON: boolean): unknown {
    const values: unknown[] = [];
    for (const part of content) {
      const value = this.partValue(part, stateJSON);
      if (value !== undefined) values.push(value);
    }
    switch (values.length) {
      case 0:
        return undefined;
      case 1:
        return values[0];
      default:
        return values;
    }
  }

  /**
   * A part's contribution to the state. Text is sent as is, or as the JSON it
   * holds when the config asks, kept verbatim so a number too large for a
   * double reaches the model unchanged; a data part is sent as its value;
   * media is collected and contributes nothing. Loop plumbing left in a
   * history is skipped rather than sent as state, and any other kind of part
   * is refused rather than stringified.
   */
  private partValue(part: Part, stateJSON: boolean): unknown {
    if (isFormatInstructions(part)) return undefined;
    if (typeof part.text === 'string') {
      if (!stateJSON) return part.text;
      try {
        JSON.parse(part.text);
      } catch (e) {
        throw invalidArgument(
          `stateJSON is set but the text is not JSON: ${(e as Error).message}`
        );
      }
      return new RawJson(part.text);
    }
    if (part.data !== undefined) return part.data;
    if (part.media) {
      this.collect(part.media.url, part.media.contentType);
      return undefined;
    }
    throw invalidArgument(
      `a ${partKindName(part)} part cannot be state; the model takes text, JSON, and media only`
    );
  }

  /**
   * Adds a media part to the request's media, or refuses it when the model
   * does not read its kind.
   */
  private collect(url: string, partContentType?: string) {
    const { dataUrl, contentType } = mediaUrl(url, partContentType);
    let reads: boolean;
    let media: string[];
    if (contentType.startsWith('image/')) {
      reads = this.reads.images;
      media = this.media.images;
    } else if (contentType.startsWith('audio/')) {
      reads = this.reads.audio;
      media = this.media.audio;
    } else if (contentType.startsWith('video/')) {
      reads = this.reads.video;
      media = this.media.videos;
    } else {
      throw invalidArgument(
        `a ${contentType || 'media'} part is not an image, audio, or video; the models take no other media`
      );
    }
    if (!reads) {
      throw invalidArgument(`the model reads no ${contentType} media`);
    }
    media.push(dataUrl);
  }

  /**
   * A document's contribution to the context: its content, read as a
   * message's is, or its content with its metadata when it has any, so a
   * passage keeps its title and source for a question to refer to. Text stays
   * text whatever the config says, since a retrieved passage is prose, not a
   * rendered template. A document of media alone with no metadata contributes
   * nothing beside its media.
   */
  documentValue(doc: DocumentData): unknown {
    const content = this.messageValue(doc.content, false);
    if (!doc.metadata || Object.keys(doc.metadata).length === 0) {
      return content;
    }
    // A document of media alone sends its content as null, as Go does.
    return { content: content ?? null, metadata: doc.metadata };
  }
}

/**
 * A media part as the media fields take it: a base64 data URL, and its media
 * type. A base64 payload goes out as it is, under the part's media type, since
 * the server decodes it anyway; any other payload is percent-decoded (RFC
 * 2397) and encoded. The scheme, the base64 token, and the media type are read
 * without regard to case. A remote URL is refused: the servers take inline
 * media only, and fetching it here would hide a network call, so the caller
 * downloads it first.
 */
export function mediaUrl(
  url: string,
  partContentType?: string
): { dataUrl: string; contentType: string } {
  const colon = url.indexOf(':');
  const scheme = colon < 0 ? '' : url.slice(0, colon).toLowerCase();
  if (scheme === 'http' || scheme === 'https' || scheme === 'gs') {
    // The URL is not quoted, since it can carry a token; its origin says
    // which one it is.
    let origin = scheme;
    try {
      origin = new URL(url).origin;
    } catch {
      // A URL that does not parse is named by its scheme alone.
    }
    throw invalidArgument(
      `the media at ${origin} is a URL, and the models take inline media only; download it first and pass it as a data URL`
    );
  }
  if (scheme !== 'data') {
    throw invalidArgument('a media part must carry its content as a data URL');
  }
  const rest = url.slice(colon + 1);
  const comma = rest.indexOf(',');
  if (comma < 0) {
    throw invalidArgument('reading the media: the data URL has no comma');
  }
  const header = rest.slice(0, comma);
  const payload = rest.slice(comma + 1);
  const mediaType = header.split(';')[0];
  const contentType = (partContentType || mediaType).toLowerCase();
  const prefix = `data:${contentType};base64,`;
  if (header.toLowerCase().endsWith(';base64')) {
    return { dataUrl: prefix + payload, contentType };
  }
  return {
    dataUrl: prefix + percentDecode(payload).toString('base64'),
    contentType,
  };
}

/** The bytes of a percent-encoded payload; a malformed escape is refused. */
function percentDecode(payload: string): Buffer {
  const bytes: number[] = [];
  const text = Buffer.from(payload, 'utf8');
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== 0x25 /* % */) {
      bytes.push(text[i]);
      continue;
    }
    const hex = text.subarray(i + 1, i + 3).toString('latin1');
    if (!/^[0-9a-fA-F]{2}$/.test(hex)) {
      throw invalidArgument(`reading the media: invalid URL escape "%${hex}"`);
    }
    bytes.push(parseInt(hex, 16));
    i += 2;
  }
  return Buffer.from(bytes);
}

/**
 * Reports whether the part is the output instructions the generate loop
 * injects. The loop injects them for the enum format, and a history recorded
 * from another model's turn keeps them in that turn's messages; either way the
 * part is plumbing, not instructions or state.
 */
function isFormatInstructions(part: Part): boolean {
  return part.metadata?.purpose === 'output';
}

/** Names a part's kind for an error message. */
function partKindName(part: Part): string {
  if (part.media) return 'media';
  if (part.toolRequest) return 'tool request';
  if (part.toolResponse) return 'tool response';
  if (part.reasoning !== undefined) return 'reasoning';
  if (part.resource) return 'resource';
  if (part.data !== undefined) return 'data';
  return 'custom';
}
