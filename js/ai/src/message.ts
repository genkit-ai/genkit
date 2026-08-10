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

import { extractJson } from './extract.js';
import type {
  MessageData,
  Part,
  ToolRequestPart,
  ToolResponsePart,
} from './model.js';

export type MessageParser<T = unknown> = (message: Message) => T;

/**
 * Message represents a single role's contribution to a generation. Each message
 * can contain multiple parts (for example text and an image), and each generation
 * can contain multiple messages.
 */
export class Message<T = unknown> implements MessageData {
  role: MessageData['role'];
  content: Part[];
  metadata?: Record<string, any>;
  parser?: MessageParser<T>;

  static parseData(
    lenientMessage:
      | string
      | (MessageData & { content: string | Part | Part[]; role: string })
      | MessageData,
    defaultRole: MessageData['role'] = 'user'
  ): MessageData {
    if (typeof lenientMessage === 'string') {
      return { role: defaultRole, content: [{ text: lenientMessage }] };
    }
    return {
      ...lenientMessage,
      content: Message.parseContent(lenientMessage.content),
    };
  }

  static parse(
    lenientMessage: string | (MessageData & { content: string }) | MessageData
  ): Message {
    return new Message(Message.parseData(lenientMessage));
  }

  static parseContent(lenientPart: string | Part | (string | Part)[]): Part[] {
    if (typeof lenientPart === 'string') {
      return [{ text: lenientPart }];
    } else if (Array.isArray(lenientPart)) {
      return lenientPart.map((p) => (typeof p === 'string' ? { text: p } : p));
    } else {
      return [lenientPart];
    }
  }

  constructor(message: MessageData, options?: { parser?: MessageParser<T> }) {
    this.role = message.role;
    this.content = message.content;
    this.metadata = message.metadata;
    this.parser = options?.parser;
  }

  /**
   * Attempts to parse the content of the message according to the supplied
   * output parser. Without a parser, returns `data` contained in the message or
   * tries to parse JSON from the text of the message.
   *
   * @returns The structured output contained in the message.
   */
  get output(): T {
    return this.parser?.(this) || this.data || extractJson<T>(this.text);
  }

  toolResponseParts(): ToolResponsePart[] {
    const res = this.content.filter((part) => !!part.toolResponse);
    return res as ToolResponsePart[];
  }

  /**
   * Concatenates all `text` parts present in the message with no delimiter.
   * @returns A string of all concatenated text parts.
   */
  get text(): string {
    return this.content.map((part) => part.text || '').join('');
  }

  /**
   * Concatenates all `reasoning` parts present in the message with no delimiter.
   * @returns A string of all concatenated reasoning parts.
   */
  get reasoning(): string {
    return this.content.map((part) => part.reasoning || '').join('');
  }

  /**
   * Returns the first media part detected in the message. Useful for extracting
   * (for example) an image from a generation expected to create one.
   * @returns The first detected `media` part in the message.
   */
  get media(): { url: string; contentType?: string } | null {
    return this.content.find((part) => part.media)?.media || null;
  }

  /**
   * Returns the first detected `data` part of a message.
   * @returns The first `data` part detected in the message (if any).
   */
  get data(): T | null {
    return this.content.find((part) => part.data)?.data as T | null;
  }

  /**
   * Returns all tool request found in this message.
   * @returns Array of all tool request found in this message.
   */
  get toolRequests(): ToolRequestPart[] {
    return this.content.filter(
      (part) => !!part.toolRequest
    ) as ToolRequestPart[];
  }

  /**
   * Returns all tool requests annotated with interrupt metadata.
   * @returns Array of all interrupt tool requests.
   */
  get interrupts(): ToolRequestPart[] {
    return this.toolRequests.filter((t) => !!t.metadata?.interrupt);
  }

  /**
   * Converts the Message to a plain JS object.
   * @returns Plain JS object representing the data contained in the message.
   */
  toJSON(): MessageData {
    const out: MessageData = {
      role: this.role,
      content: [...this.content],
    };
    if (this.metadata) out.metadata = this.metadata;
    return out;
  }
}
