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

"""URL config for the django-hello sample."""

from django.urls import path

# pyrefly resolves imports from py/ root, so it can't see this sibling sample
# package; at runtime uvicorn is launched from samples/django-hello/ and finds it.
from recipes.views import say_hi  # pyrefly: ignore[missing-import]

urlpatterns = [
    path('chat', say_hi),
]
