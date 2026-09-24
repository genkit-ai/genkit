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

"""A2UI protocol constants and shared shapes."""

from typing import Any, Literal

A2UI_MIME_TYPE = 'application/a2ui+json'
A2UI_CATALOG_VALUE_TYPE = 'a2ui-catalog'
DEFAULT_CATALOG_ID = 'basic'
DEFAULT_VERSION = 'v0.9'
SupportedVersion = Literal['v0.9', 'v0.9.1']
BASIC_CATALOG_ID = 'https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json'
SURFACE_ID_PLACEHOLDER = 'SURFACE_ID'

ValidateMode = Literal['strict', 'warn', 'off']
Envelope = dict[str, Any]

SURFACE_KEYS = ('createSurface', 'updateComponents', 'updateDataModel', 'deleteSurface')
