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

"""Pytest configuration for the cross-adapter web tests.

Django's request and response classes need configured settings before use.
The guard keeps this a no-op when the genkit-django tests configured them first.
"""

import django
from django.conf import settings


def pytest_configure() -> None:
    """Configure Django before any test imports run."""
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            DATABASES={},
            INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
            SECRET_KEY='test-secret-key',
            ALLOWED_HOSTS=['*'],
            DEFAULT_CHARSET='utf-8',
            USE_TZ=True,
        )
        django.setup()
