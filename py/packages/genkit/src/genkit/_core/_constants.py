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

"""Module containing various core constants."""

import importlib.metadata
from typing import Final


def package_version() -> str:
    """Installed ``genkit`` package version, or ``0.0.0`` if it is not installed."""
    try:
        return importlib.metadata.version('genkit')
    except importlib.metadata.PackageNotFoundError:
        return '0.0.0'


# Sent over HTTP in the client header and on the OTLP log scope.
GENKIT_VERSION: Final[str] = package_version()

GENKIT_CLIENT_HEADER: Final[str] = f'genkit-python/{GENKIT_VERSION}'
"""Client identifier plugins send to providers, e.g. `genkit-python/1.0.0`."""
