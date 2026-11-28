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

import { createServer, type IncomingMessage, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';

/**
 * A fake System One server: it answers every question it is sent, records the
 * requests, and serves a model listing. A choice picks its first option by
 * name, a score answers 1.3 on a three-level scale, and a noul answers 0.93.
 */
export class FakeServer {
  /** The listing a GET returns; a 500 when unset. */
  models?: string;
  /** The usage a reply carries. */
  usage: Record<string, unknown> = { input_tokens: 12, output_tokens: 3 };
  /** Wraps a reply in a server's envelope. */
  wrap?: (reply: Record<string, unknown>) => unknown;
  /**
   * Replies with these first, in order, before answering; one with a
   * `delayMs` waits that long before it replies.
   */
  failures: {
    status: number;
    body?: string;
    headers?: Record<string, string>;
    delayMs?: number;
  }[] = [];
  /** Replaces the answers to the questions it names. */
  answers?: Record<string, unknown>;

  readonly requests: { url: string; headers: IncomingMessage['headers'] }[] =
    [];
  readonly bodies: any[] = [];
  readonly rawBodies: string[] = [];
  readonly listings: string[] = [];
  /** The Authorization header of each listing request. */
  readonly listingAuth: (string | undefined)[] = [];
  private server?: Server;

  get url(): string {
    const { port } = this.server!.address() as AddressInfo;
    return `http://127.0.0.1:${port}`;
  }

  async start(): Promise<this> {
    this.server = createServer(async (req, res) => {
      if (req.method === 'GET') {
        this.listings.push(req.url ?? '');
        this.listingAuth.push(req.headers.authorization);
        if (this.models === undefined) {
          res.writeHead(500).end();
          return;
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(this.models);
        return;
      }
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk as Buffer);
      const raw = Buffer.concat(chunks).toString('utf8');
      this.requests.push({ url: req.url ?? '', headers: req.headers });
      this.rawBodies.push(raw);
      const body = JSON.parse(raw);
      this.bodies.push(body);
      const failure = this.failures.shift();
      if (failure) {
        if (failure.delayMs) {
          await new Promise((resolve) => setTimeout(resolve, failure.delayMs));
        }
        if (res.destroyed) return;
        res.writeHead(failure.status, failure.headers ?? {});
        res.end(failure.body ?? '');
        return;
      }
      const native =
        body?.input && typeof body.input === 'object' ? body.input : body;
      const answers: Record<string, unknown> = {};
      for (const [id, q] of Object.entries<any>(native.questions ?? {})) {
        switch (q.type) {
          case 'choice': {
            const keys = Object.keys(q.criteria ?? {}).sort();
            if (keys.length === 0) {
              res
                .writeHead(400)
                .end('{"error":{"message":"a choice needs criteria"}}');
              return;
            }
            const probabilities: Record<string, number> = {};
            keys.forEach((k, i) => {
              probabilities[k] = i === 0 ? 1 - 0.1 * (keys.length - 1) : 0.1;
            });
            answers[id] = {
              type: 'choice',
              choice: keys[0],
              probabilities,
              confidence: 0.6,
            };
            break;
          }
          case 'score': {
            const legend: Record<string, unknown> = {};
            (q.criteria ?? []).forEach((level: unknown, i: number) => {
              legend[String(i)] = level;
            });
            answers[id] = {
              type: 'score',
              score: 1.3,
              legend,
              probabilities: { '0': 0, '1': 0.7, '2': 0.3 },
              confidence: 0.54,
            };
            break;
          }
          case 'noul':
            answers[id] = { type: 'noul', noul: 0.93 };
            break;
        }
      }
      const reply: Record<string, unknown> = {
        model: native.model ?? body.model,
        answers: { ...answers, ...this.answers },
        usage: this.usage,
      };
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(this.wrap ? this.wrap(reply) : reply));
    });
    await new Promise<void>((resolve) =>
      this.server!.listen(0, '127.0.0.1', resolve)
    );
    return this;
  }

  async stop(): Promise<void> {
    await new Promise<void>((resolve) => this.server?.close(() => resolve()));
  }
}
