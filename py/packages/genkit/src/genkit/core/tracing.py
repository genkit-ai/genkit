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


"""Telemetry and tracing functionality for the Genkit framework.

This module provides functionality for collecting and exporting telemetry data
from Genkit operations. It uses OpenTelemetry for tracing and exports span
data to a telemetry server for monitoring and debugging purposes.

The module includes:
    - A custom span exporter for sending trace data to a telemetry server
    - Utility functions for converting and formatting trace attributes
    - Configuration for development environment tracing
"""

import traceback
from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from genkit.core.environment import is_dev_environment
from genkit.core.logging import get_logger
from genkit.core.trace.default_exporter import create_span_processor, init_telemetry_server_exporter
from genkit.core.trace.types import GenkitSpan
from genkit.core.typing import SpanMetadata

ATTR_PREFIX = 'genkit'
logger = get_logger(__name__)


tracer = trace_api.get_tracer('genkit-tracer', 'v1')


def init_provider() -> TracerProvider:
    """Inits and returns the tracer global provider."""
    tracer_provider = trace_api.get_tracer_provider()

    if tracer_provider is None or not isinstance(tracer_provider, TracerProvider):  # pyright: ignore[reportUnnecessaryComparison]
        tracer_provider = TracerProvider()
        trace_api.set_tracer_provider(tracer_provider)
        # pyrefly: ignore[missing-attribute] - LoggingInstrumentor has instrument() method
        LoggingInstrumentor().instrument(set_logging_format=True)
        logger.debug('Creating a new global tracer provider for telemetry.')

    if not isinstance(tracer_provider, TracerProvider):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f'The current trace provider is not an instance of TracerProvider.  It is of type: {type(tracer_provider)}'
        )

    return tracer_provider


def add_custom_exporter(exporter: SpanExporter | None, name: str = 'last') -> None:
    """Adds custom span exporter to current tracer provider.

    Args:
        exporter: Custom or dedicated span exporter.
        name: Name of the span exporter. Only for logging purposes.
    """
    current_provider = init_provider()

    try:
        if exporter is None:
            logger.warn(f'{name} exporter is None')
            return

        processor = create_span_processor(exporter)
        current_provider.add_span_processor(processor)
        logger.debug(f'{name} exporter added successfully.')
    except Exception:
        logger.error(f'tracing.add_custom_exporter: failed to add exporter {name}')
        logger.exception('Failed to add custom exporter')


if is_dev_environment():
    add_custom_exporter(init_telemetry_server_exporter(), 'local_telemetry_server')


@contextmanager
def run_in_new_span(
    metadata: SpanMetadata,
    labels: dict[str, str] | None = None,
    links: list[trace_api.Link] | None = None,
) -> Generator[GenkitSpan, None, None]:
    """Starts a new span context under the current trace.

    This method provides a contexmanager for working with Genkit spans. The
    context object is a `GenkitSpan`, which is a light wrapper on OpenTelemetry
    span object, with handling for genkit attributes.
    """
    with tracer.start_as_current_span(name=metadata.name, links=links) as ot_span:
        span = GenkitSpan(ot_span, labels)
        try:
            yield span
            span.set_genkit_attribute('status', 'success')
        except Exception as e:
            logger.debug(f'Error in run_in_new_span: {e!s}')
            logger.debug(traceback.format_exc())
            span.set_genkit_attribute('status', 'error')
            span.set_status(status=trace_api.StatusCode.ERROR, description=str(e))
            span.record_exception(e)
            raise e
