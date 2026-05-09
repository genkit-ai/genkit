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

import * as http from 'http';
import { AddressInfo } from 'net';

export interface MockResponse {
  statusCode?: number;
  body?: any;
  headers?: http.OutgoingHttpHeaders;
  stream?: boolean;
  chunks?: any[]; // For streaming
}

export class FakeOpenAIServer {
  private server: http.Server;
  private port: number = 0;
  private responses: MockResponse[] = [];
  public requests: { headers: http.IncomingHttpHeaders; body: any }[] = [];
  private expectedApiKey?: string;

  constructor(expectedApiKey?: string) {
    this.expectedApiKey = expectedApiKey;
    this.server = http.createServer(async (req, res) => {
      let body = '';
      req.on('data', (chunk) => {
        body += chunk.toString();
      });

      await new Promise<void>((resolve) => req.on('end', resolve));

      const parsedBody = body ? JSON.parse(body) : {};
      this.requests.push({ headers: req.headers, body: parsedBody });

      if (this.expectedApiKey) {
        const authHeader = req.headers['authorization'];
        if (!authHeader || authHeader !== `Bearer ${this.expectedApiKey}`) {
          res.writeHead(401, { 'Content-Type': 'application/json' });
          res.end(
            JSON.stringify({
              error: {
                message: 'Incorrect API key provided',
                type: 'invalid_request_error',
                param: null,
                code: 'invalid_api_key',
              },
            })
          );
          return;
        }
      }

      const response = this.responses.shift();
      if (!response) {
        // Default response if nothing queued
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(
          JSON.stringify({
            choices: [
              {
                message: { role: 'assistant', content: 'default response' },
                finish_reason: 'stop',
              },
            ],
          })
        );
        return;
      }

      const statusCode = response.statusCode || 200;
      const headers = response.headers || {
        'Content-Type': 'application/json',
      };

      if (response.stream) {
        headers['Content-Type'] = 'text/event-stream';
        headers['Cache-Control'] = 'no-cache';
        headers['Connection'] = 'keep-alive';
        res.writeHead(statusCode, headers);

        if (response.chunks) {
          for (const chunk of response.chunks) {
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          }
        }
        res.write('data: [DONE]\n\n');
        res.end();
      } else {
        res.writeHead(statusCode, headers);
        res.end(JSON.stringify(response.body));
      }
    });
  }

  async start() {
    await new Promise<void>((resolve) => {
      this.server.listen(0, () => {
        this.port = (this.server.address() as AddressInfo).port;
        resolve();
      });
    });
  }

  stop() {
    this.server.close();
  }

  get baseUrl() {
    return `http://localhost:${this.port}/v1`;
  }

  setNextResponse(response: MockResponse) {
    this.responses.push(response);
  }
}
