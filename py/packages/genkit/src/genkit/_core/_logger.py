# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Internal logger for genkit core. Not part of public API."""

import structlog
from structlog.typing import FilteringBoundLogger


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog bound logger with a concrete return type for checkers."""
    return structlog.get_logger(name)
