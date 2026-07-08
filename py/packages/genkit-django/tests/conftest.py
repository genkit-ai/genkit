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

"""Pytest configuration for the Django plugin tests.

Configures the minimum Django settings the plugin's tests need. Django's
``HttpRequest``, ``StreamingHttpResponse``, and ``AsyncClient`` all require a
configured settings module before they can be imported or used.
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
            ROOT_URLCONF=__name__,
            SECRET_KEY='test-secret-key',
            ALLOWED_HOSTS=['*'],
            DEFAULT_CHARSET='utf-8',
            USE_TZ=True,
        )
        django.setup()


# Required by Django; tests build their own urlpatterns via override_settings.
urlpatterns: list = []
