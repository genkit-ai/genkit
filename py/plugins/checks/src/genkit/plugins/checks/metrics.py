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

"""Safety policy metrics for Google Checks AI Safety.

This module defines the safety policy types supported by the Checks
``classifyContent`` API. Each metric corresponds to a specific category
of content safety violation.

See:
    https://developers.google.com/checks/guides/guardrails-api
"""

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum

from pydantic import BaseModel, Field


class ChecksEvaluationMetricType(StrEnum):
    """Safety policy types supported by the Checks classifyContent API.

    Each value maps to a ``policy_type`` in the Checks API request body.
    """

    DANGEROUS_CONTENT = 'DANGEROUS_CONTENT'
    """The model facilitates, promotes, or enables access to harmful goods,
    services, and activities."""

    PII_SOLICITING_RECITING = 'PII_SOLICITING_RECITING'
    """The model reveals an individual's personal information and data."""

    HARASSMENT = 'HARASSMENT'
    """The model generates content that is malicious, intimidating, bullying,
    or abusive towards another individual."""

    SEXUALLY_EXPLICIT = 'SEXUALLY_EXPLICIT'
    """The model generates content that is sexually explicit in nature."""

    HATE_SPEECH = 'HATE_SPEECH'
    """The model promotes violence, hatred, discrimination on the basis of
    race, religion, etc."""

    MEDICAL_INFO = 'MEDICAL_INFO'
    """The model facilitates harm by providing health advice or guidance."""

    VIOLENCE_AND_GORE = 'VIOLENCE_AND_GORE'
    """The model generates content that contains gratuitous, realistic
    descriptions of violence or gore."""

    OBSCENITY_AND_PROFANITY = 'OBSCENITY_AND_PROFANITY'
    """The model generates content that contains vulgar, profane, or
    offensive language."""


class ChecksEvaluationMetricConfig(BaseModel):
    """Configuration for a single Checks evaluation metric with optional threshold.

    Attributes:
        type: The safety policy type.
        threshold: Optional confidence threshold for the policy. Lower values
            are stricter. If not set, the API default is used.
    """

    type: ChecksEvaluationMetricType
    threshold: float | None = Field(
        default=None,
        description='Optional confidence threshold. Lower values are stricter.',
    )


# A metric can be either a plain enum value (uses API defaults) or a config
# object with an explicit threshold.
ChecksEvaluationMetric = ChecksEvaluationMetricType | ChecksEvaluationMetricConfig
