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

"""AI monitoring metrics for Genkit.

This module provides lazy-initialized OpenTelemetry metrics for AI operations.
Metrics are exported to Google Cloud Monitoring with the workload.googleapis.com
prefix by default.

Metrics Defined:
    Input metrics:
        - genkit/ai/generate/input/tokens
        - genkit/ai/generate/input/characters
        - genkit/ai/generate/input/images
        - genkit/ai/generate/input/videos
        - genkit/ai/generate/input/audio

    Output metrics:
        - genkit/ai/generate/output/tokens
        - genkit/ai/generate/output/characters
        - genkit/ai/generate/output/images
        - genkit/ai/generate/output/videos
        - genkit/ai/generate/output/audio

    Thinking metrics:
        - genkit/ai/generate/thinking/tokens

See Also:
    - Cloud Monitoring Custom Metrics: https://cloud.google.com/monitoring/custom-metrics
    - Workload Metrics: https://cloud.google.com/monitoring/api/metrics_other
"""

import contextlib
import json
import re

import structlog
from opentelemetry import metrics
from opentelemetry.sdk.trace import ReadableSpan

logger = structlog.get_logger(__name__)

meter = metrics.get_meter('genkit')


def _metric(name: str, desc: str, unit: str = '1') -> tuple[str, str, str]:
    """Create metric name with genkit/ai/ prefix.

    Args:
        name: Metric name
        desc: Metric description
        unit: Metric unit (default: '1')

    Returns:
        Tuple of (prefixed_name, description, unit)
    """
    return f'genkit/ai/{name}', desc, unit


# Metric caches for lazy initialization
_counter_cache: dict[str, metrics.Counter] = {}
_histogram_cache: dict[str, metrics.Histogram] = {}


def _get_counter(name: str, desc: str, unit: str = '1') -> metrics.Counter:
    """Get or create counter metric with lazy initialization.

    Args:
        name: Metric name
        desc: Metric description
        unit: Metric unit (default: '1')

    Returns:
        OpenTelemetry Counter metric
    """
    if name not in _counter_cache:
        _counter_cache[name] = meter.create_counter(name, description=desc, unit=unit)
    return _counter_cache[name]


def _get_histogram(name: str, desc: str, unit: str = '1') -> metrics.Histogram:
    """Get or create histogram metric with lazy initialization.

    Args:
        name: Metric name
        desc: Metric description
        unit: Metric unit (default: '1')

    Returns:
        OpenTelemetry Histogram metric
    """
    if name not in _histogram_cache:
        _histogram_cache[name] = meter.create_histogram(name, description=desc, unit=unit)
    return _histogram_cache[name]


# Metric definitions
def _requests() -> metrics.Counter:
    return _get_counter(*_metric('generate/requests', 'Generate requests'))


def _failures() -> metrics.Counter:
    return _get_counter(*_metric('generate/failures', 'Generate failures'))


def _latency() -> metrics.Histogram:
    return _get_histogram(*_metric('generate/latency', 'Generate latency', 'ms'))


def _input_tokens() -> metrics.Counter:
    return _get_counter(*_metric('generate/input/tokens', 'Input tokens'))


def _output_tokens() -> metrics.Counter:
    return _get_counter(*_metric('generate/output/tokens', 'Output tokens'))


def _input_characters() -> metrics.Counter:
    return _get_counter(*_metric('generate/input/characters', 'Input characters'))


def _output_characters() -> metrics.Counter:
    return _get_counter(*_metric('generate/output/characters', 'Output characters'))


def _input_images() -> metrics.Counter:
    return _get_counter(*_metric('generate/input/images', 'Input images'))


def _output_images() -> metrics.Counter:
    return _get_counter(*_metric('generate/output/images', 'Output images'))


def _input_videos() -> metrics.Counter:
    return _get_counter(*_metric('generate/input/videos', 'Input videos'))


def _output_videos() -> metrics.Counter:
    return _get_counter(*_metric('generate/output/videos', 'Output videos'))


def _input_audio() -> metrics.Counter:
    return _get_counter(*_metric('generate/input/audio', 'Input audio'))


def _output_audio() -> metrics.Counter:
    return _get_counter(*_metric('generate/output/audio', 'Output audio'))


def record_generate_metrics(span: ReadableSpan) -> None:
    """Record AI monitoring metrics from a model action span.

    Args:
        span: OpenTelemetry span containing model execution data
    """
    attrs = span.attributes
    if not attrs:
        return

    # Check if this is a model action
    if attrs.get('genkit:type') != 'action' or attrs.get('genkit:metadata:subtype') != 'model':
        return

    # Extract dimensions
    model = str(attrs.get('genkit:name', '<unknown>'))[:1000]
    path = str(attrs.get('genkit:path', ''))[:1000]
    source = _extract_feature_name(path)
    is_error = not span.status.is_ok
    error = 'error' if is_error else 'none'

    dimensions = {'model': model, 'source': source, 'error': error}

    try:
        _requests().add(1, dimensions)
        if is_error:
            _failures().add(1, dimensions)

        # Latency
        latency_ms = None
        if span.end_time and span.start_time:
            latency_ms = (span.end_time - span.start_time) / 1_000_000
            _latency().record(latency_ms, dimensions)

        usage = {}
        output_json = attrs.get('genkit:output')
        if output_json and isinstance(output_json, str):
            try:
                output_data = json.loads(output_json)
                usage = output_data.get('usage', {})
            except (json.JSONDecodeError, AttributeError):
                pass

        usage_metrics = {
            'inputTokens': _input_tokens,
            'outputTokens': _output_tokens,
            'inputCharacters': _input_characters,
            'outputCharacters': _output_characters,
            'inputImages': _input_images,
            'outputImages': _output_images,
            'inputVideos': _input_videos,
            'outputVideos': _output_videos,
            'inputAudio': _input_audio,
            'outputAudio': _output_audio,
        }

        for key, metric_fn in usage_metrics.items():
            value = usage.get(key)
            if value is not None:
                with contextlib.suppress(ValueError, TypeError):
                    metric_fn().add(int(value), dimensions)

    except Exception as e:
        logger.warning('Error recording metrics', error=str(e))


def _extract_feature_name(path: str) -> str:
    """Extract feature name from Genkit action path.

    Args:
        path: Genkit action path in format '/{name,t:type}' or '/{outer,t:flow}/{inner,t:flow}'

    Returns:
        Extracted feature name or '<unknown>' if path cannot be parsed
    """
    if not path:
        return '<unknown>'

    parts = path.split('/')
    if len(parts) < 2:
        return '<unknown>'

    match = re.match(r'\{([^,}]+)', parts[1])
    return match.group(1) if match else '<unknown>'
