# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Internal logger for genkit core. Not part of public API."""

import structlog

get_logger = structlog.get_logger
