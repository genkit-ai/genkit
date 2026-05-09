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

import type { SpanData } from '@genkit-ai/tools-common';
import type { Response } from 'express';

/**
 * Event type for span broadcasts.
 */
export interface SpanEvent {
  type: 'span_start' | 'span_end';
  traceId: string;
  span: SpanData;
}

/**
 * Broadcast manager for SSE connections.
 * Tracks active connections per traceId and broadcasts updates.
 */
export class BroadcastManager {
  private connections: Map<string, Set<Response>> = new Map();

  /**
   * Register a new SSE connection for a traceId.
   */
  subscribe(traceId: string, response: Response): void {
    if (!this.connections.has(traceId)) {
      this.connections.set(traceId, new Set());
    }
    this.connections.get(traceId)!.add(response);

    // Clean up when connection closes
    response.on('close', () => {
      this.unsubscribe(traceId, response);
    });
  }

  /**
   * Remove a connection from subscriptions.
   */
  unsubscribe(traceId: string, response: Response): void {
    const connections = this.connections.get(traceId);
    if (connections) {
      connections.delete(response);
      if (connections.size === 0) {
        this.connections.delete(traceId);
      }
    }
  }

  /**
   * Broadcast span updates to all subscribers of a traceId.
   */
  broadcast(traceId: string, event: SpanEvent): void {
    const connections = this.connections.get(traceId);
    if (!connections || connections.size === 0) {
      return;
    }

    const data = JSON.stringify(event);
    const messageToSend = `data: ${data}\n\n`;

    // Note: response.write() doesn't throw on dead sockets in Express - it
    // returns false and the 'close' event handles cleanup. No try/catch needed.
    for (const connection of connections) {
      connection.write(messageToSend);
    }
  }

  /**
   * Close all connections for a traceId.
   */
  close(traceId: string): void {
    const connections = this.connections.get(traceId);
    if (connections) {
      for (const connection of connections) {
        try {
          connection.end();
        } catch (error) {
          // Ignore errors when closing
        }
      }
      this.connections.delete(traceId);
    }
  }

  /**
   * Get the number of active connections for a traceId.
   * Useful for testing.
   */
  getConnectionCount(traceId: string): number {
    return this.connections.get(traceId)?.size ?? 0;
  }

  /**
   * Check if a traceId has any active connections.
   * Useful for testing.
   */
  hasConnections(traceId: string): boolean {
    return this.connections.has(traceId);
  }
}
