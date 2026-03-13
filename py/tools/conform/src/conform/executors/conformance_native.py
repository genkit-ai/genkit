#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

r"""Python native conformance executor (JSONL-over-stdio).

Protocol: JSONL-over-stdio (same as Go and JS executors).

1. Receives ``--plugin <name>`` as a CLI argument.
2. Initializes the matching plugin via a registry map.
3. Prints ``{"ready": true}\\n`` to stdout.
4. Reads one JSON line from stdin per request.
5. Calls ``ai.generate()`` natively.
6. Writes one JSON line to stdout with the response.
7. Repeats until stdin closes.

Driven by the Python ``conform`` tool::

    conform check-model --runtime python --runner native
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import traceback
from typing import Any, cast

import structlog

from genkit.ai import Genkit
from genkit.codec import dump_dict

# ---------------------------------------------------------------------------
# Plugin registry — maps conform plugin names to Genkit init functions.
# Each function returns a configured ``Genkit`` instance.
# ---------------------------------------------------------------------------


def _init_google_genai() -> Genkit:
    """Initialize the Google GenAI plugin."""
    from genkit.plugins.google_genai import GoogleAI

    return Genkit(plugins=[GoogleAI()])


def _init_vertex_ai() -> Genkit:
    """Initialize the Vertex AI plugin."""
    from genkit.plugins.vertex_ai import ModelGardenPlugin

    return Genkit(plugins=[ModelGardenPlugin()])


def _init_anthropic() -> Genkit:
    """Initialize the Anthropic plugin."""
    from genkit.plugins.anthropic import Anthropic

    return Genkit(plugins=[Anthropic()])


def _init_ollama() -> Genkit:
    """Initialize the Ollama plugin."""
    from genkit.plugins.ollama import Ollama
    from genkit.plugins.ollama.models import ModelDefinition

    return Genkit(
        plugins=[
            Ollama(
                models=[
                    ModelDefinition(name='llama3.2'),
                    ModelDefinition(name='llama3.2:1b'),
                ],
            ),
        ],
    )


def _init_amazon_bedrock() -> Genkit:
    """Initialize the Amazon Bedrock plugin."""
    from genkit.plugins.amazon_bedrock import AmazonBedrock

    return Genkit(plugins=[AmazonBedrock()])


def _init_compat_oai() -> Genkit:
    """Initialize the OpenAI compat plugin."""
    from genkit.plugins.compat_oai import OpenAI

    return Genkit(plugins=[OpenAI()])


def _init_deepseek() -> Genkit:
    """Initialize the DeepSeek plugin."""
    from genkit.plugins.deepseek import DeepSeek

    return Genkit(plugins=[DeepSeek()])


def _init_mistral() -> Genkit:
    """Initialize the Mistral plugin."""
    from genkit.plugins.mistral import Mistral

    return Genkit(plugins=[Mistral()])


def _init_xai() -> Genkit:
    """Initialize the xAI plugin."""
    from genkit.plugins.xai import XAI

    return Genkit(plugins=[XAI()])


def _init_cohere() -> Genkit:
    """Initialize the Cohere plugin."""
    from genkit.plugins.cohere import Cohere

    return Genkit(plugins=[Cohere()])


def _init_cloudflare_workers_ai() -> Genkit:
    """Initialize the Cloudflare Workers AI plugin."""
    from genkit.plugins.cloudflare_workers_ai import CloudflareWorkersAI

    return Genkit(plugins=[CloudflareWorkersAI()])


def _init_huggingface() -> Genkit:
    """Initialize the Hugging Face plugin."""
    from genkit.plugins.huggingface import HuggingFace

    return Genkit(plugins=[HuggingFace()])


def _init_microsoft_foundry() -> Genkit:
    """Initialize the Microsoft Foundry plugin."""
    from genkit.plugins.microsoft_foundry import MicrosoftFoundry

    return Genkit(plugins=[MicrosoftFoundry()])


PLUGIN_REGISTRY: dict[str, Any] = {
    'google-genai': _init_google_genai,
    'vertex-ai': _init_vertex_ai,
    'anthropic': _init_anthropic,
    'ollama': _init_ollama,
    'amazon-bedrock': _init_amazon_bedrock,
    'compat-oai': _init_compat_oai,
    'deepseek': _init_deepseek,
    'mistral': _init_mistral,
    'xai': _init_xai,
    'cohere': _init_cohere,
    'cloudflare-workers-ai': _init_cloudflare_workers_ai,
    'huggingface': _init_huggingface,
    'microsoft-foundry': _init_microsoft_foundry,
}

# ---------------------------------------------------------------------------
# Ephemeral tool registration (for tool conformance tests).
# ---------------------------------------------------------------------------


def _register_ephemeral_tool(
    ai: Genkit,
    name: str,
    tool_def: dict[str, Any],
) -> None:
    """Register a no-op tool so generate() can resolve its definition."""
    from genkit.core.action.types import ActionKind

    registry = ai.registry
    with registry._lock:  # pyright: ignore[reportPrivateUsage]
        tool_entries = registry._entries.get(  # pyright: ignore[reportPrivateUsage]
            cast('ActionKind', ActionKind.TOOL),
            {},
        )
        if name in tool_entries:
            return

    description = tool_def.get('description', '')
    input_schema = tool_def.get('inputSchema')

    async def noop_tool(input_data: Any, ctx: Any) -> dict[str, str]:  # noqa: ANN401
        """No-op tool stub for conformance testing."""
        return {'result': 'noop'}

    action = registry.register_action(
        kind=cast('ActionKind', ActionKind.TOOL),
        name=name,
        fn=noop_tool,
        description=description,
        metadata={'tool': {'type': 'tool'}},
    )
    if input_schema:
        action._input_schema = input_schema  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Request handling.
# ---------------------------------------------------------------------------


async def handle_request(
    ai: Genkit,
    req: dict[str, Any],
) -> dict[str, Any]:
    """Handle a single JSONL request."""
    try:
        model_name = req['model']
        input_data = req.get('input', {})
        stream = req.get('stream', False)

        messages = input_data.get('messages', [])
        output_config = input_data.get('output')
        tools_defs = input_data.get('tools')
        config = input_data.get('config')

        # Convert raw message dicts to Message objects.
        from genkit.core.typing import Message, OutputConfig, Part

        msg_objects = [Message.model_validate(m) for m in messages]

        # Separate system messages.
        system_parts: list[Part] | None = None
        non_system_messages: list[Message] = []
        for msg in msg_objects:
            if msg.role == 'system':
                system_parts = msg.content
            else:
                non_system_messages.append(msg)

        # Build output config.
        output_obj: OutputConfig | None = None
        if output_config:
            output_obj = OutputConfig.model_validate(output_config)

        # Register ephemeral tools.
        tool_names: list[str] | None = None
        if tools_defs:
            tool_names = []
            for tdef in tools_defs:
                t_name = tdef['name']
                tool_names.append(t_name)
                _register_ephemeral_tool(ai, t_name, tdef)

        chunks: list[dict[str, Any]] = []

        if stream:
            stream_iter, response_future = ai.generate_stream(
                model=model_name,
                system=system_parts,
                messages=non_system_messages,
                tools=tool_names,
                config=config,
                output=output_obj,
                return_tool_requests=True,
            )
            async for chunk in stream_iter:
                chunks.append(cast(dict[str, Any], dump_dict(chunk)))
            result = await response_future
        else:
            result = await ai.generate(
                model=model_name,
                system=system_parts,
                messages=non_system_messages,
                tools=tool_names,
                config=config,
                output=output_obj,
                return_tool_requests=True,
            )

        response = cast(dict[str, Any], dump_dict(result))
        return {'response': response, 'chunks': chunks}

    except Exception:
        return {
            'response': None,
            'chunks': [],
            'error': traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the JSONL-over-stdio loop."""
    # CRITICAL: Redirect all logging to stderr BEFORE plugin initialization.
    # Plugins use structlog (via genkit.core.logging) which defaults to stdout.
    # Since this executor uses JSONL-over-stdio, ANY non-JSON line on stdout
    # causes the conform runner to fail with "invalid JSON".
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format='%(message)s',
    )
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )

    parser = argparse.ArgumentParser(
        description='Python native conformance executor',
    )
    parser.add_argument(
        '--plugin',
        required=True,
        choices=sorted(PLUGIN_REGISTRY.keys()),
        help='Plugin name to initialize.',
    )
    args = parser.parse_args()

    init_fn = PLUGIN_REGISTRY[args.plugin]
    ai = init_fn()

    # Signal readiness.
    sys.stdout.write(json.dumps({'ready': True}) + '\n')
    sys.stdout.flush()

    # Read requests from stdin, one JSON line at a time.
    for line in sys.stdin:
        trimmed = line.strip()
        if not trimmed:
            continue

        try:
            req = json.loads(trimmed)
        except json.JSONDecodeError:
            err_resp = {
                'response': None,
                'chunks': [],
                'error': f'Invalid request JSON: {trimmed[:200]}',
            }
            sys.stdout.write(json.dumps(err_resp) + '\n')
            sys.stdout.flush()
            continue

        result = await handle_request(ai, req)
        sys.stdout.write(json.dumps(result) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    asyncio.run(main())
