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

"""Map Genkit model calls onto OpenTelemetry GenAI attribute names.

Pure helpers, no OpenTelemetry imports, so the leftover a later reader
joins to — provider, model, finish reason, output type — can be tested
without spinning up a tracer.
"""

from __future__ import annotations

from enum import Enum


class GenAiAttr:
    """Canonical ``gen_ai.*`` attribute names."""

    OPERATION_NAME = 'gen_ai.operation.name'
    PROVIDER_NAME = 'gen_ai.provider.name'

    REQUEST_MODEL = 'gen_ai.request.model'
    REQUEST_TEMPERATURE = 'gen_ai.request.temperature'
    REQUEST_TOP_P = 'gen_ai.request.top_p'
    REQUEST_TOP_K = 'gen_ai.request.top_k'
    REQUEST_MAX_TOKENS = 'gen_ai.request.max_tokens'
    REQUEST_STOP_SEQUENCES = 'gen_ai.request.stop_sequences'
    REQUEST_FREQUENCY_PENALTY = 'gen_ai.request.frequency_penalty'
    REQUEST_PRESENCE_PENALTY = 'gen_ai.request.presence_penalty'
    REQUEST_SEED = 'gen_ai.request.seed'
    REQUEST_CHOICE_COUNT = 'gen_ai.request.choice.count'

    OUTPUT_TYPE = 'gen_ai.output.type'

    RESPONSE_FINISH_REASONS = 'gen_ai.response.finish_reasons'

    TOKEN_TYPE = 'gen_ai.token.type'

    USAGE_INPUT_TOKENS = 'gen_ai.usage.input_tokens'
    USAGE_OUTPUT_TOKENS = 'gen_ai.usage.output_tokens'
    USAGE_REASONING_OUTPUT_TOKENS = 'gen_ai.usage.reasoning.output_tokens'
    USAGE_CACHE_READ_INPUT_TOKENS = 'gen_ai.usage.cache_read.input_tokens'

    TOOL_NAME = 'gen_ai.tool.name'
    TOOL_TYPE = 'gen_ai.tool.type'

    INPUT_MESSAGES = 'gen_ai.input.messages'
    OUTPUT_MESSAGES = 'gen_ai.output.messages'
    SYSTEM_INSTRUCTIONS = 'gen_ai.system_instructions'

    ERROR_TYPE = 'error.type'


class GenkitAttr:
    """Non-reserved ``genkit.*`` attributes.

    Kept out of ``gen_ai.*`` so a GenAI-aware backend never tries to
    render raw Genkit payloads as spec message content.
    """

    ACTION_TYPE = 'genkit.action.type'
    INPUT = 'genkit.input'
    OUTPUT = 'genkit.output'


class GenAiOperation:
    """Well-known values for ``gen_ai.operation.name``."""

    CHAT = 'chat'
    EXECUTE_TOOL = 'execute_tool'


class GenAiMetric:
    """Canonical ``gen_ai.*`` metric instrument names."""

    TOKEN_USAGE = 'gen_ai.client.token.usage'
    OPERATION_DURATION = 'gen_ai.client.operation.duration'


GEN_AI_SEMCONV_VERSION = '1.38.0'

GEN_AI_OPERATION_DETAILS_EVENT = 'gen_ai.client.inference.operation.details'

CAPTURE_CONTENT_ENV_VAR = 'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'


class ContentCapturingMode(Enum):
    """Where captured GenAI message content is recorded.

    Content may contain PII and is often large, so the default is
    ``NO_CONTENT``. ``EVENT_ONLY`` keeps structured content on a
    dedicated log event (preferred in production). ``SPAN_ONLY`` puts
    it on span attributes as a JSON string — easy to eyeball in a
    trace UI, but subject to backend attribute limits. ``SPAN_AND_EVENT``
    does both.

    Env tokens are the spec's UPPER_SNAKE names
    (``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, ``SPAN_AND_EVENT``).
    """

    NO_CONTENT = 'NO_CONTENT'
    SPAN_ONLY = 'SPAN_ONLY'
    EVENT_ONLY = 'EVENT_ONLY'
    SPAN_AND_EVENT = 'SPAN_AND_EVENT'


def parse_content_capturing_mode(raw: str | None) -> ContentCapturingMode | None:
    """Parse a spec ``ContentCapturingMode`` env token.

    Null/empty → ``NO_CONTENT``. A known token (case-insensitive,
    surrounding whitespace ignored) → that mode. Unknown → ``None``
    so the caller can warn and fall back.
    """
    if raw is None or not raw.strip():
        return ContentCapturingMode.NO_CONTENT
    token = raw.strip().upper()
    for mode in ContentCapturingMode:
        if mode.value == token:
            return mode
    return None


def split_model_name(name: str) -> tuple[str | None, str]:
    """Split a fully qualified model name into ``(prefix, model)``.

    ``googleai/gemini-flash-latest`` → ``('googleai', 'gemini-flash-latest')``.
    A name without a ``/`` yields a ``None`` prefix.
    """
    index = name.find('/')
    if index < 0:
        return None, name
    return name[:index], name[index + 1 :]


def derive_provider_name(prefix: str | None) -> str | None:
    """Derive ``gen_ai.provider.name`` from a model-name prefix.

    Known plugin prefixes map to the spec's well-known provider names
    so a GenAI-aware backend (Jaeger, Cloud Trace) can group them.
    Unknown prefixes pass through lowercased so a custom plugin still
    gets a discriminator. No prefix → omit the attribute.
    """
    if prefix is None or prefix == '':
        return None
    key = prefix.lower()
    if key in {'googleai', 'google-genai', 'google_genai'}:
        return 'gcp.gemini'
    if key in {'vertexai', 'vertex-ai', 'vertex_ai'}:
        return 'gcp.vertex_ai'
    if key == 'openai':
        return 'openai'
    if key == 'anthropic':
        return 'anthropic'
    return key


def map_finish_reason(genkit_reason: str | None, *, failed: bool) -> str:
    """Map a Genkit finish reason to the GenAI ``finish_reasons`` value.

    ``failed`` picks the fallback for ambiguous reasons (``other`` /
    ``unknown`` / missing): ``error`` when the span failed, otherwise
    ``stop``.
    """
    if genkit_reason == 'stop':
        return 'stop'
    if genkit_reason == 'length':
        return 'length'
    if genkit_reason == 'blocked':
        return 'content_filter'
    if genkit_reason == 'interrupted':
        # An interrupted turn still produced a usable leftover; treat it
        # as a normal stop rather than an error.
        return 'stop'
    if failed:
        return 'error'
    return 'stop'


def derive_output_type(*, format: str | None = None, content_type: str | None = None) -> str | None:
    """Derive ``gen_ai.output.type`` from an output format / content type.

    ``json`` when JSON was requested, ``text`` when a text format was
    requested, otherwise omit the attribute.
    """
    fmt = format.lower() if format else None
    ctype = content_type.lower() if content_type else None
    if fmt == 'json' or (ctype is not None and 'json' in ctype):
        return 'json'
    if fmt == 'text' or (ctype is not None and ctype.startswith('text/')):
        return 'text'
    return None


def as_int(value: object) -> int | None:
    """Coerce a config/usage value to ``int``, or ``None`` if not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def as_double(value: object) -> float | None:
    """Coerce a config value to ``float``, or ``None`` if not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def as_string_list(value: object) -> list[str] | None:
    """Coerce a config value to ``list[str]``, or ``None``.

    A list of any element type is stringified; a single scalar becomes
    a one-element list.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
