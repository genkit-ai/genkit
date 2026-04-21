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

"""Environment detection for Genkit runtime."""

import os

from genkit._core._compat import StrEnum

# Environment variable name
GENKIT_ENV = 'GENKIT_ENV'


class GenkitEnvironment(StrEnum):
    """Genkit runtime environments."""

    DEV = 'dev'
    PROD = 'prod'


def is_dev_environment() -> bool:
    """Check if running in development mode (GENKIT_ENV=dev)."""
    return os.getenv(GENKIT_ENV) == GenkitEnvironment.DEV


def get_current_environment() -> GenkitEnvironment:
    """Get current environment, defaults to PROD."""
    env = os.getenv(GENKIT_ENV)
    if env == GenkitEnvironment.DEV:
        return GenkitEnvironment.DEV
    return GenkitEnvironment.PROD
