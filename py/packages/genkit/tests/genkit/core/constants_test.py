# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for package version constants."""

import importlib.metadata

from genkit._core._constants import GENKIT_CLIENT_HEADER, GENKIT_VERSION


def test_genkit_version_comes_from_the_installed_package() -> None:
    assert GENKIT_VERSION == importlib.metadata.version('genkit')
    assert GENKIT_CLIENT_HEADER == f'genkit-python/{GENKIT_VERSION}'
