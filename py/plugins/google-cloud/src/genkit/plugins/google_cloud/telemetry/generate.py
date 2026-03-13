# Copyright 2025 Google LLC
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

"""Generate action telemetry for GCP.

This module tracks generate action metrics (tokens, latencies) and logs,
matching the JavaScript implementation in telemetry/generate.ts and Go
implementation in googlecloud/generate.go.

When It Fires:
    The generate telemetry handler is called for spans where:
    - genkit:type = "action"
    - genkit:metadata:subtype = "model"

Metrics Recorded:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Metric Name                          │ Type      │ Description          │
    ├──────────────────────────────────────┼───────────┼──────────────────────┤
    │ genkit/ai/generate/requests          │ Counter   │ Model call count     │
    │ genkit/ai/generate/latency           │ Histogram │ Response time (ms)   │
    │ genkit/ai/generate/input/tokens      │ Counter   │ Input token count    │
    │ genkit/ai/generate/input/characters  │ Counter   │ Input char count     │
    │ genkit/ai/generate/input/images      │ Counter   │ Input image count    │
    │ genkit/ai/generate/input/videos      │ Counter   │ Input video count    │
    │ genkit/ai/generate/input/audio       │ Counter   │ Input audio count    │
    │ genkit/ai/generate/output/tokens     │ Counter   │ Output token count   │
    │ genkit/ai/generate/output/characters │ Counter   │ Output char count    │
    │ genkit/ai/generate/output/images     │ Counter   │ Output image count   │
    │ genkit/ai/generate/output/videos     │ Counter   │ Output video count   │
    │ genkit/ai/generate/output/audio      │ Counter   │ Output audio count   │
    │ genkit/ai/generate/thinking/tokens   │ Counter   │ Thinking token count │
    └──────────────────────────────────────┴───────────┴──────────────────────┘

Metric Dimensions:
    All metrics include these dimensions:
    - modelName: The model name (e.g., "gemini-2.0-flash")
    - featureName: The outer flow/feature name
    - path: The qualified Genkit path
    - status: "success" or "failure"
    - error: Error name (only on failure)
    - source: "py" (language identifier)
    - sourceVersion: Genkit version

Logs Recorded:
    1. Config logs (always): Model configuration (maxOutputTokens, stopSequences)
    2. Input logs (when log_input_and_output=True): Per-message, per-part input
    3. Output logs (when log_input_and_output=True): Per-part output content

Log Format:
    - Config[path, model] - Model configuration
    - Input[path, model] (part X of Y) - Input content with part indices
    - Output[path, model] (part X of Y) - Output content with part indices

Media Handling:
    - Data URLs (base64) are hashed with SHA-256 to avoid logging large content
    - Format: "data:image/png;base64,<sha256(hash)>"

GCP Documentation:
    Cloud Monitoring Metrics:
        - Custom Metrics: https://cloud.google.com/monitoring/custom-metrics
        - Quotas: https://cloud.google.com/monitoring/quotas
        - Note: Rate limit is 1 point per 5 seconds per time series

    OpenTelemetry:
        - Python Metrics SDK: https://opentelemetry-python.readthedocs.io/en/stable/sdk/metrics.html

Cross-Language Parity:
    - JavaScript: js/plugins/google-cloud/src/telemetry/generate.ts
    - Go: go/plugins/googlecloud/generate.go
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import Any

import structlog
from opentelemetry import metrics
from opentelemetry.sdk.trace import ReadableSpan

from genkit.core import GENKIT_VERSION

from .gcp_logger import gcp_logger
from .utils import (
    create_common_log_attributes,
    extract_error_name,
    extract_outer_feature_name_from_path,
    to_display_path,
    truncate,
    truncate_path,
)

logger = structlog.get_logger(__name__)

# Lazy-initialized metrics
_action_counter: metrics.Counter | None = None
_latency: metrics.Histogram | None = None
_input_characters: metrics.Counter | None = None
_input_tokens: metrics.Counter | None = None
_input_images: metrics.Counter | None = None
_input_videos: metrics.Counter | None = None
_input_audio: metrics.Counter | None = None
_output_characters: metrics.Counter | None = None
_output_tokens: metrics.Counter | None = None
_output_images: metrics.Counter | None = None
_output_videos: metrics.Counter | None = None
_output_audio: metrics.Counter | None = None
_thinking_tokens: metrics.Counter | None = None


def _get_meter() -> metrics.Meter:
    return metrics.get_meter('genkit')


def _get_action_counter() -> metrics.Counter:
    global _action_counter
    if _action_counter is None:
        _action_counter = _get_meter().create_counter(
            'genkit/ai/generate/requests',
            description='Counts calls to genkit generate actions.',
            unit='1',
        )
    return _action_counter


def _get_latency() -> metrics.Histogram:
    global _latency
    if _latency is None:
        _latency = _get_meter().create_histogram(
            'genkit/ai/generate/latency',
            description='Latencies when interacting with a Genkit model.',
            unit='ms',
        )
    return _latency


def _get_input_characters() -> metrics.Counter:
    global _input_characters
    if _input_characters is None:
        _input_characters = _get_meter().create_counter(
            'genkit/ai/generate/input/characters',
            description='Counts input characters to any Genkit model.',
            unit='1',
        )
    return _input_characters


def _get_input_tokens() -> metrics.Counter:
    global _input_tokens
    if _input_tokens is None:
        _input_tokens = _get_meter().create_counter(
            'genkit/ai/generate/input/tokens',
            description='Counts input tokens to a Genkit model.',
            unit='1',
        )
    return _input_tokens


def _get_input_images() -> metrics.Counter:
    global _input_images
    if _input_images is None:
        _input_images = _get_meter().create_counter(
            'genkit/ai/generate/input/images',
            description='Counts input images to a Genkit model.',
            unit='1',
        )
    return _input_images


def _get_input_videos() -> metrics.Counter:
    """Get or create the input videos counter (Go parity)."""
    global _input_videos
    if _input_videos is None:
        _input_videos = _get_meter().create_counter(
            'genkit/ai/generate/input/videos',
            description='Counts input videos to a Genkit model.',
            unit='1',
        )
    return _input_videos


def _get_input_audio() -> metrics.Counter:
    """Get or create the input audio counter (Go parity)."""
    global _input_audio
    if _input_audio is None:
        _input_audio = _get_meter().create_counter(
            'genkit/ai/generate/input/audio',
            description='Counts input audio files to a Genkit model.',
            unit='1',
        )
    return _input_audio


def _get_output_characters() -> metrics.Counter:
    global _output_characters
    if _output_characters is None:
        _output_characters = _get_meter().create_counter(
            'genkit/ai/generate/output/characters',
            description='Counts output characters from a Genkit model.',
            unit='1',
        )
    return _output_characters


def _get_output_tokens() -> metrics.Counter:
    global _output_tokens
    if _output_tokens is None:
        _output_tokens = _get_meter().create_counter(
            'genkit/ai/generate/output/tokens',
            description='Counts output tokens from a Genkit model.',
            unit='1',
        )
    return _output_tokens


def _get_output_images() -> metrics.Counter:
    global _output_images
    if _output_images is None:
        _output_images = _get_meter().create_counter(
            'genkit/ai/generate/output/images',
            description='Count output images from a Genkit model.',
            unit='1',
        )
    return _output_images


def _get_output_videos() -> metrics.Counter:
    """Get or create the output videos counter (Go parity)."""
    global _output_videos
    if _output_videos is None:
        _output_videos = _get_meter().create_counter(
            'genkit/ai/generate/output/videos',
            description='Counts output videos from a Genkit model.',
            unit='1',
        )
    return _output_videos


def _get_output_audio() -> metrics.Counter:
    """Get or create the output audio counter (Go parity)."""
    global _output_audio
    if _output_audio is None:
        _output_audio = _get_meter().create_counter(
            'genkit/ai/generate/output/audio',
            description='Counts output audio files from a Genkit model.',
            unit='1',
        )
    return _output_audio


def _get_thinking_tokens() -> metrics.Counter:
    global _thinking_tokens
    if _thinking_tokens is None:
        _thinking_tokens = _get_meter().create_counter(
            'genkit/ai/generate/thinking/tokens',
            description='Counts thinking tokens from a Genkit model.',
            unit='1',
        )
    return _thinking_tokens


class GenerateTelemetry:
    """Telemetry handler for Genkit generate actions (model calls)."""

    def tick(
        self,
        span: ReadableSpan,
        log_input_and_output: bool,
        project_id: str | None = None,
    ) -> None:
        """Record telemetry for a generate action span.

        Args:
            span: The span to record telemetry for.
            log_input_and_output: Whether to log input/output.
            project_id: Optional GCP project ID.
        """
        attrs = span.attributes or {}
        model_name = truncate(str(attrs.get('genkit:name', '<unknown>')), 1024)
        path = str(attrs.get('genkit:path', ''))

        # Parse input and output from span attributes
        input_data: dict[str, Any] | None = None
        output_data: dict[str, Any] | None = None

        input_json = attrs.get('genkit:input')
        if input_json and isinstance(input_json, str):
            with contextlib.suppress(json.JSONDecodeError):
                input_data = json.loads(input_json)

        output_json = attrs.get('genkit:output')
        if output_json and isinstance(output_json, str):
            with contextlib.suppress(json.JSONDecodeError):
                output_data = json.loads(output_json)

        err_name = extract_error_name(list(span.events))
        feature_name = truncate(
            str(attrs.get('genkit:metadata:flow:name', '')) or extract_outer_feature_name_from_path(path)
        )
        if not feature_name or feature_name == '<unknown>':
            feature_name = 'generate'

        session_id = str(attrs.get('genkit:sessionId', '')) or None
        thread_name = str(attrs.get('genkit:threadName', '')) or None

        if input_data:
            self._record_generate_action_metrics(model_name, feature_name, path, output_data, err_name)
            self._record_generate_action_config_logs(
                span, model_name, feature_name, path, input_data, project_id, session_id, thread_name
            )

            if log_input_and_output:
                self._record_generate_action_input_logs(
                    span, model_name, feature_name, path, input_data, project_id, session_id, thread_name
                )

        if output_data and log_input_and_output:
            self._record_generate_action_output_logs(
                span, model_name, feature_name, path, output_data, project_id, session_id, thread_name
            )

    def _record_generate_action_metrics(
        self,
        model_name: str,
        feature_name: str,
        path: str,
        response: dict[str, Any] | None,
        err_name: str | None,
    ) -> None:
        """Record metrics for a generate action.

        Records all generate metrics matching JS/Go parity:
        - requests, latency
        - input: tokens, characters, images, videos, audio
        - output: tokens, characters, images, videos, audio
        - thinking tokens
        """
        usage = response.get('usage', {}) if response else {}
        latency_ms = response.get('latencyMs') if response else None

        # Note: modelName uses 1024 char limit (matching JS/Go), other dimensions use 256
        shared = {
            'modelName': model_name[:1024],
            'featureName': feature_name[:256],
            'path': path[:256],
            'source': 'py',
            'sourceVersion': GENKIT_VERSION,
            'status': 'failure' if err_name else 'success',
        }

        error_dims = {'error': err_name[:256]} if err_name else {}
        _get_action_counter().add(1, {**shared, **error_dims})

        if latency_ms is not None:
            _get_latency().record(latency_ms, shared)

        # Input metrics
        if usage.get('inputTokens'):
            _get_input_tokens().add(int(usage['inputTokens']), shared)
        if usage.get('inputCharacters'):
            _get_input_characters().add(int(usage['inputCharacters']), shared)
        if usage.get('inputImages'):
            _get_input_images().add(int(usage['inputImages']), shared)
        if usage.get('inputVideos'):
            _get_input_videos().add(int(usage['inputVideos']), shared)
        if usage.get('inputAudio'):
            _get_input_audio().add(int(usage['inputAudio']), shared)

        # Output metrics
        if usage.get('outputTokens'):
            _get_output_tokens().add(int(usage['outputTokens']), shared)
        if usage.get('outputCharacters'):
            _get_output_characters().add(int(usage['outputCharacters']), shared)
        if usage.get('outputImages'):
            _get_output_images().add(int(usage['outputImages']), shared)
        if usage.get('outputVideos'):
            _get_output_videos().add(int(usage['outputVideos']), shared)
        if usage.get('outputAudio'):
            _get_output_audio().add(int(usage['outputAudio']), shared)

        # Thinking tokens
        if usage.get('thoughtsTokens'):
            _get_thinking_tokens().add(int(usage['thoughtsTokens']), shared)

    def _record_generate_action_config_logs(
        self,
        span: ReadableSpan,
        model: str,
        feature_name: str,
        qualified_path: str,
        input_data: dict[str, Any],
        project_id: str | None,
        session_id: str | None,
        thread_name: str | None,
    ) -> None:
        """Log generate action configuration."""
        path = truncate_path(to_display_path(qualified_path))
        metadata = {
            **create_common_log_attributes(span, project_id),
            'model': model,
            'path': path,
            'qualifiedPath': qualified_path,
            'featureName': feature_name,
            'source': 'py',
            'sourceVersion': GENKIT_VERSION,
        }
        if session_id:
            metadata['sessionId'] = session_id
        if thread_name:
            metadata['threadName'] = thread_name

        config = input_data.get('config', {})
        if config.get('maxOutputTokens'):
            metadata['maxOutputTokens'] = config['maxOutputTokens']
        if config.get('stopSequences'):
            metadata['stopSequences'] = config['stopSequences']

        gcp_logger.log_structured(f'Config[{path}, {model}]', metadata)

    def _record_generate_action_input_logs(
        self,
        span: ReadableSpan,
        model: str,
        feature_name: str,
        qualified_path: str,
        input_data: dict[str, Any],
        project_id: str | None,
        session_id: str | None,
        thread_name: str | None,
    ) -> None:
        """Log generate action input messages."""
        path = truncate_path(to_display_path(qualified_path))
        base_metadata = {
            **create_common_log_attributes(span, project_id),
            'model': model,
            'path': path,
            'qualifiedPath': qualified_path,
            'featureName': feature_name,
        }
        if session_id:
            base_metadata['sessionId'] = session_id
        if thread_name:
            base_metadata['threadName'] = thread_name

        messages = input_data.get('messages', [])
        total_messages = len(messages)

        for msg_idx, msg in enumerate(messages):
            role = msg.get('role', 'user')
            content = msg.get('content', [])
            total_parts = len(content)

            for part_idx, part in enumerate(content):
                part_counts = self._to_part_counts(part_idx, total_parts, msg_idx, total_messages)
                metadata = {
                    **base_metadata,
                    'content': self._to_part_log_content(part),
                    'role': role,
                    'partIndex': part_idx,
                    'totalParts': total_parts,
                    'messageIndex': msg_idx,
                    'totalMessages': total_messages,
                }
                gcp_logger.log_structured(f'Input[{path}, {model}] {part_counts}', metadata)

    def _record_generate_action_output_logs(
        self,
        span: ReadableSpan,
        model: str,
        feature_name: str,
        qualified_path: str,
        output_data: dict[str, Any],
        project_id: str | None,
        session_id: str | None,
        thread_name: str | None,
    ) -> None:
        """Log generate action output."""
        path = truncate_path(to_display_path(qualified_path))
        base_metadata = {
            **create_common_log_attributes(span, project_id),
            'model': model,
            'path': path,
            'qualifiedPath': qualified_path,
            'featureName': feature_name,
        }
        if session_id:
            base_metadata['sessionId'] = session_id
        if thread_name:
            base_metadata['threadName'] = thread_name

        message = output_data.get('message') or (output_data.get('candidates', [{}])[0].get('message'))
        if not message or not message.get('content'):
            return

        content = message.get('content', [])
        total_parts = len(content)
        finish_reason = output_data.get('finishReason')
        finish_message = output_data.get('finishMessage')

        for part_idx, part in enumerate(content):
            part_counts = self._to_part_counts(part_idx, total_parts, 0, 1)
            metadata = {
                **base_metadata,
                'content': self._to_part_log_content(part),
                'role': message.get('role', 'model'),
                'partIndex': part_idx,
                'totalParts': total_parts,
                'candidateIndex': 0,
                'totalCandidates': 1,
                'messageIndex': 0,
                'finishReason': finish_reason,
            }
            if finish_message:
                metadata['finishMessage'] = truncate(finish_message)

            gcp_logger.log_structured(f'Output[{path}, {model}] {part_counts}', metadata)

    def _to_part_counts(
        self,
        part_ordinal: int,
        parts: int,
        msg_ordinal: int,
        messages: int,
    ) -> str:
        """Format part counts for log messages."""
        if parts > 1 and messages > 1:
            return f'(part {self._x_of_y(part_ordinal, parts)} in message {self._x_of_y(msg_ordinal, messages)})'
        if parts > 1:
            return f'(part {self._x_of_y(part_ordinal, parts)})'
        if messages > 1:
            return f'(message {self._x_of_y(msg_ordinal, messages)})'
        return ''

    def _x_of_y(self, x: int, y: int) -> str:
        """Format 'X of Y' string."""
        return f'{x + 1} of {y}'

    def _to_part_log_content(self, part: dict[str, Any]) -> str:
        """Convert a part to log-safe content."""
        if part.get('text'):
            return truncate(str(part['text']))
        if part.get('data'):
            return truncate(json.dumps(part['data']))
        if part.get('media'):
            return self._to_part_log_media(part)
        if part.get('toolRequest'):
            return self._to_part_log_tool_request(part)
        if part.get('toolResponse'):
            return self._to_part_log_tool_response(part)
        if part.get('custom'):
            return truncate(json.dumps(part['custom']))
        return '<unknown format>'

    def _to_part_log_media(self, part: dict[str, Any]) -> str:
        """Convert media part to log-safe content."""
        media = part.get('media', {})
        url = media.get('url', '')

        if url.startswith('data:'):
            split_idx = url.find('base64,')
            if split_idx < 0:
                return '<unknown media format>'
            prefix = url[: split_idx + 7]
            hashed = hashlib.sha256(url[split_idx + 7 :].encode()).hexdigest()
            return f'{prefix}<sha256({hashed})>'

        return truncate(url)

    def _to_part_log_tool_request(self, part: dict[str, Any]) -> str:
        """Convert tool request part to log-safe content."""
        req = part.get('toolRequest', {})
        name = req.get('name', '')
        ref = req.get('ref', '')
        input_val = req.get('input', '')
        if not isinstance(input_val, str):
            input_val = json.dumps(input_val)
        return truncate(f'Tool request: {name}, ref: {ref}, input: {input_val}')

    def _to_part_log_tool_response(self, part: dict[str, Any]) -> str:
        """Convert tool response part to log-safe content."""
        resp = part.get('toolResponse', {})
        name = resp.get('name', '')
        ref = resp.get('ref', '')
        output_val = resp.get('output', '')
        if not isinstance(output_val, str):
            output_val = json.dumps(output_val)
        return truncate(f'Tool response: {name}, ref: {ref}, output: {output_val}')


# Singleton instance
generate_telemetry = GenerateTelemetry()
