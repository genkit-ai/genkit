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

"""Metrics exporting functionality for GCP telemetry.

This module contains the metric exporter wrapper that adjusts start
times for Google Cloud Monitoring compatibility.
"""

from collections.abc import Callable

from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.sdk.metrics import (
    Counter,
    Histogram,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    UpDownCounter,
)
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    MetricExporter,
    MetricExportResult,
    MetricsData,
)

from .constants import METRIC_START_TIME_ADJUSTMENT_NS


class GenkitMetricExporter(MetricExporter):
    """Metric exporter wrapper that adjusts start times for GCP compatibility.

    Cloud Monitoring does not support delta metrics for custom metrics and will
    convert any DELTA aggregations to CUMULATIVE ones on export. There is implicit
    overlap in the start/end times that the Metric reader sends -- the end_time
    of the previous export becomes the start_time of the current export.

    This wrapper adds a microsecond to start times to ensure discrete export
    timeframes and prevent data being overwritten.

    This matches the JavaScript MetricExporterWrapper in gcpOpenTelemetry.ts.
    """

    def __init__(
        self,
        exporter: CloudMonitoringMetricsExporter,
        error_handler: Callable[[Exception], None] | None = None,
    ) -> None:
        """Initialize the metric exporter wrapper.

        Args:
            exporter: The underlying CloudMonitoringMetricsExporter.
            error_handler: Optional callback for export errors.
        """
        self._exporter = exporter
        self._error_handler = error_handler

        # Force DELTA temporality for all instrument types to match JS implementation.
        delta = AggregationTemporality.DELTA
        self._preferred_temporality = {
            Counter: delta,
            UpDownCounter: delta,
            Histogram: delta,
            ObservableCounter: delta,
            ObservableUpDownCounter: delta,
            ObservableGauge: delta,
        }

        self._preferred_aggregation = getattr(exporter, '_preferred_aggregation', None)

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,
        **kwargs: object,
    ) -> MetricExportResult:
        """Export metrics after adjusting start times.

        Modifies start times of each data point to ensure no overlap with
        previous exports when GCP converts DELTA to CUMULATIVE.

        Args:
            metrics_data: The metrics data to export.
            timeout_millis: Export timeout in milliseconds.
            **kwargs: Additional arguments for base class compatibility.

        Returns:
            The export result from the wrapped exporter.
        """
        # Modify start times before export
        self._modify_start_times(metrics_data)

        try:
            return self._exporter.export(metrics_data, timeout_millis, **kwargs)
        except Exception as e:
            if self._error_handler:
                self._error_handler(e)
            raise

    def _modify_start_times(self, metrics_data: MetricsData) -> None:
        """Add 1ms to start times to prevent overlap.

        Args:
            metrics_data: The metrics data to modify in-place.
        """
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    for data_point in metric.data.data_points:
                        # Add 1 millisecond to start time
                        if hasattr(data_point, 'start_time_unix_nano'):
                            # Modifying frozen dataclass via workaround
                            object.__setattr__(
                                data_point,
                                'start_time_unix_nano',
                                data_point.start_time_unix_nano + METRIC_START_TIME_ADJUSTMENT_NS,
                            )

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        """Delegate force flush to wrapped exporter.

        Args:
            timeout_millis: Timeout in milliseconds.

        Returns:
            True if flush succeeded.
        """
        if hasattr(self._exporter, 'force_flush'):
            return self._exporter.force_flush(timeout_millis)
        return True

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
        """Delegate shutdown to wrapped exporter.

        Args:
            timeout_millis: Timeout in milliseconds.
            **kwargs: Additional arguments for base class compatibility.
        """
        self._exporter.shutdown(timeout_millis, **kwargs)
