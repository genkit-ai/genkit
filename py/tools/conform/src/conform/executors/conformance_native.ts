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
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @fileoverview Generic native conformance executor for JS/TS plugins.
 *
 * Protocol: JSONL-over-stdio.
 *
 * 1. Receives --plugin <name> as a CLI argument.
 * 2. Initializes the matching plugin via a registry map.
 * 3. Prints {"ready": true}\n to stdout.
 * 4. Reads one JSON line from stdin per request.
 * 5. Calls ai.generate() natively.
 * 6. Writes one JSON line to stdout with the response.
 * 7. Repeats until stdin closes.
 *
 * Driven by the Python `conform` tool:
 *
 *   conform check-model --runtime js --runner native
 */

import { parseArgs } from 'node:util';
import * as readline from 'readline';
import { genkit, type Genkit } from 'genkit';
import { googleAI, vertexAI } from '@genkit-ai/google-genai';
import { anthropic } from '@genkit-ai/anthropic';
import { ollama } from 'genkitx-ollama';
import type {
    GenerateResponseData,
    GenerateResponseChunkData,
    MessageData,
    ToolDefinition,
} from '@genkit-ai/ai';


interface NativeRequest {
    model: string;
    input: {
        messages: MessageData[];
        tools?: ToolDefinition[];
        output?: {
            format?: string;
            schema?: Record<string, unknown>;
            constrained?: boolean;
        };
        config?: Record<string, unknown>;
    };
    stream: boolean;
}

interface NativeResponse {
    response: Record<string, unknown> | null;
    chunks: Record<string, unknown>[];
    error?: string;
}


type PluginInitFunc = () => Genkit;

const pluginRegistry: Record<string, PluginInitFunc> = {
    'google-genai': () => genkit({ plugins: [googleAI()] }),
    'vertex-ai': () => genkit({ plugins: [vertexAI()] }),
    'anthropic': () => genkit({ plugins: [anthropic()] }),
    'ollama': () => genkit({ plugins: [ollama()] }),
};


async function handleRequest(
    ai: Genkit,
    req: NativeRequest
): Promise<NativeResponse> {
    try {
        const generateOptions: Record<string, unknown> = {
            model: req.model,
            messages: req.input.messages,
            returnToolRequests: true,
        };

        // Tools — define placeholder tools with the schema from the test definition.
        if (req.input.tools && req.input.tools.length > 0) {
            const tools = req.input.tools.map((toolDef) =>
                ai.defineTool(
                    {
                        name: toolDef.name,
                        description: toolDef.description ?? '',
                        // Pass the JSON schema from the test definition so that
                        // provider plugins receive a valid input_schema (e.g.
                        // Anthropic requires `type: "object"` to be present).
                        inputJsonSchema: toolDef.inputSchema ?? {
                            type: 'object' as const,
                            properties: {},
                        },
                    },
                    async () => '21C'
                )
            );
            generateOptions.tools = tools;
        }

        // Output format — pass format and optional jsonSchema.
        if (req.input.output?.format) {
            const outputConfig: Record<string, unknown> = {
                format: req.input.output.format,
            };
            if (req.input.output.schema) {
                outputConfig.jsonSchema = req.input.output.schema;
            }
            generateOptions.output = outputConfig;
        }

        // Config.
        if (req.input.config) {
            generateOptions.config = req.input.config;
        }

        // Execute.
        let responseData: GenerateResponseData;
        const chunks: GenerateResponseChunkData[] = [];

        if (req.stream) {
            const { response: respPromise, stream } = await ai.generateStream(
                generateOptions as Parameters<typeof ai.generateStream>[0]
            );
            for await (const chunk of stream) {
                chunks.push(chunk.toJSON());
            }
            const resp = await respPromise;
            responseData = resp.toJSON();
        } else {
            const resp = await ai.generate(
                generateOptions as Parameters<typeof ai.generate>[0]
            );
            responseData = resp.toJSON();
        }

        return {
            response: responseData as unknown as Record<string, unknown>,
            chunks: chunks as unknown as Record<string, unknown>[],
        };
    } catch (err: unknown) {
        const errMsg = err instanceof Error ? err.message : String(err);
        return {
            response: null,
            chunks: [],
            error: errMsg,
        };
    }
}


async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            plugin: { type: 'string' },
        },
        strict: true,
    });

    const pluginName = values.plugin;
    if (!pluginName) {
        process.stderr.write(
            `error: --plugin is required\navailable plugins: ${Object.keys(pluginRegistry).join(', ')}\n`
        );
        process.exit(1);
    }

    const initFn = pluginRegistry[pluginName];
    if (!initFn) {
        process.stderr.write(
            `error: unknown plugin "${pluginName}"\navailable plugins: ${Object.keys(pluginRegistry).join(', ')}\n`
        );
        process.exit(1);
    }

    // Initialize plugin.
    const ai = initFn();

    // Signal readiness.
    process.stdout.write(JSON.stringify({ ready: true }) + '\n');

    // Read requests from stdin, one JSON line at a time.
    const rl = readline.createInterface({
        input: process.stdin,
        crlfDelay: Infinity,
    });

    for await (const line of rl) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        let req: NativeRequest;
        try {
            req = JSON.parse(trimmed);
        } catch {
            const errResp: NativeResponse = {
                response: null,
                chunks: [],
                error: `Invalid request JSON: ${trimmed.slice(0, 200)}`,
            };
            process.stdout.write(JSON.stringify(errResp) + '\n');
            continue;
        }

        const result = await handleRequest(ai, req);
        process.stdout.write(JSON.stringify(result) + '\n');
    }
}

main().catch((err) => {
    process.stderr.write(`Fatal error: ${err}\n`);
    process.exit(1);
});
