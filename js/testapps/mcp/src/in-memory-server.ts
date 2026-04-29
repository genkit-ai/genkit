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

import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

// Create a local MCP server for testing multipart images without spawning processes
const customMcpServer = new McpServer({
  name: 'custom_mcp_server',
  version: '1.0.0',
});

// A 1x1 black PNG
const IMAGE_B64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';

customMcpServer.registerTool(
  'get_test_image',
  { description: 'Returns an image' },
  async () => {
    return {
      content: [
        {
          type: 'image',
          data: IMAGE_B64,
          mimeType: 'image/png',
        },
        {
          type: 'text',
          text: 'Successfully returned the test image',
        },
      ],
    };
  }
);

customMcpServer.registerTool(
  'error_tool',
  { description: 'A tool that always returns an error' },
  async () => {
    return {
      isError: true,
      content: [
        {
          type: 'text',
          text: 'This is a simulated error from the tool.',
        },
      ],
    };
  }
);

customMcpServer.registerTool(
  'resource_tool',
  { description: 'A tool that returns an embedded resource' },
  async () => {
    return {
      content: [
        {
          type: 'resource',
          resource: {
            uri: 'test://resource/1',
            text: 'This is some text inside an embedded resource.',
            mimeType: 'text/plain',
          },
        },
      ],
    };
  }
);

const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
customMcpServer.connect(serverTransport);

export { clientTransport };
