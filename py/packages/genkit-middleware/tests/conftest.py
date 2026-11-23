# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for middleware plugin unit tests."""

import pytest

from genkit.exp import Genkit
from genkit.middleware import MiddlewareContext


@pytest.fixture
def ctx() -> MiddlewareContext:
    return MiddlewareContext(ai=Genkit())
