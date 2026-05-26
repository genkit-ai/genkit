# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for middleware plugin unit tests."""

import pytest

from genkit._core._registry import Registry
from genkit.middleware import GenerateMiddlewareContext


@pytest.fixture
def ctx() -> GenerateMiddlewareContext:
    return GenerateMiddlewareContext(registry=Registry())
