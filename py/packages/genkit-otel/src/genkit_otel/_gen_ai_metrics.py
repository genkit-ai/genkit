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

"""The two spec-defined GenAI client metrics, recorded per model operation."""

from __future__ import annotations

from opentelemetry.metrics import Histogram, Meter
from opentelemetry.util.types import AttributeValue

from genkit_otel._gen_ai_attributes import GenAiAttr, GenAiMetric

# Explicit token-count buckets the spec recommends for the token-usage
# histogram. The duration histogram uses the SDK default seconds buckets.
TOKEN_BUCKETS = [
    1.0,
    4.0,
    16.0,
    64.0,
    256.0,
    1024.0,
    4096.0,
    16384.0,
    65536.0,
    262144.0,
    1048576.0,
    4194304.0,
    16777216.0,
    67108864.0,
]


class GenAiMetrics:
    """Token-usage and operation-duration histograms.

    Instruments are created from the meter so nothing is allocated until
    the first model call, and so this stays a no-op when the SDK is not
    initialized (the meter returns non-recording instruments).
    """

    def __init__(self, meter: Meter) -> None:
        self._token_usage = _create_histogram(
            meter,
            name=GenAiMetric.TOKEN_USAGE,
            unit='{token}',
            description='Number of input and output tokens used by the model.',
            boundaries=TOKEN_BUCKETS,
        )
        self._operation_duration = _create_histogram(
            meter,
            name=GenAiMetric.OPERATION_DURATION,
            unit='s',
            description='Duration of a GenAI model operation.',
            boundaries=None,
        )

    def record_token_usage(
        self,
        *,
        base_attributes: dict[str, AttributeValue],
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        """Record input/output token counts, one point per non-null count."""
        if input_tokens is not None:
            self._token_usage.record(
                input_tokens,
                {**base_attributes, GenAiAttr.TOKEN_TYPE: 'input'},
            )
        if output_tokens is not None:
            self._token_usage.record(
                output_tokens,
                {**base_attributes, GenAiAttr.TOKEN_TYPE: 'output'},
            )

    def record_duration(self, seconds: float, attributes: dict[str, AttributeValue]) -> None:
        """Record operation duration in seconds.

        Recorded for both successful and failed operations (failures
        carry ``error.type`` in ``attributes``).
        """
        self._operation_duration.record(seconds, attributes)


def _create_histogram(
    meter: Meter,
    *,
    name: str,
    unit: str,
    description: str,
    boundaries: list[float] | None,
) -> Histogram:
    if boundaries is not None:
        try:
            return meter.create_histogram(
                name,
                unit=unit,
                description=description,
                explicit_bucket_boundaries_advisory=boundaries,
            )
        except TypeError:
            pass
    return meter.create_histogram(name, unit=unit, description=description)
