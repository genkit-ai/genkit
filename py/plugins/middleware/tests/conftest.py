# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for middleware plugin unit tests."""

import pytest

from genkit._core._registry import Registry
from genkit.middleware import MiddlewareContext


@pytest.fixture
def ctx() -> MiddlewareContext:
    return MiddlewareContext(registry=Registry(), enqueue_parts=lambda _parts: None)
