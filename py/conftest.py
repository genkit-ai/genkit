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

"""Workspace-wide pytest configuration."""

import structlog


def pytest_configure() -> None:
    """Use plain tracebacks in structlog during tests to skip Rich/Pygments lexing."""
    processors = [
        structlog.dev.ConsoleRenderer(
            colors=False,
            exception_formatter=structlog.dev.plain_traceback,
        )
        if isinstance(processor, structlog.dev.ConsoleRenderer)
        else processor
        for processor in structlog.get_config()['processors']
    ]
    structlog.configure(processors=processors)
