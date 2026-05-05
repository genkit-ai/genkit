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

"""Tests for Vertex AI tuned Gemini endpoint routing helpers."""

from types import SimpleNamespace

import pytest

from genkit.plugins.google_genai.models.gemini import (
    is_tuned_gemini_name,
    resolve_vertex_model_name,
)


@pytest.mark.parametrize(
    'name,expected',
    [
        ('endpoints/1234567890', True),
        ('projects/p/locations/us-central1/endpoints/9', True),
        ('gemini-2.5-flash', False),
        ('imagen-4.0-generate-001', False),
        ('projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash', False),
        ('', False),
    ],
)
def test_is_tuned_gemini_name(name: str, expected: bool) -> None:
    """is_tuned_gemini_name recognises both short and fully qualified forms."""
    assert is_tuned_gemini_name(name) is expected


def _vertex_client(project: str = 'my-proj', location: str = 'us-central1') -> SimpleNamespace:
    return SimpleNamespace(_api_client=SimpleNamespace(vertexai=True, project=project, location=location))


def _googleai_client() -> SimpleNamespace:
    return SimpleNamespace(_api_client=SimpleNamespace(vertexai=False, project=None, location=None))


def test_resolve_short_form_expands_with_project_and_location() -> None:
    """Short-form endpoints/ID is expanded to the full resource path on Vertex."""
    client = _vertex_client()
    got = resolve_vertex_model_name(client, 'endpoints/9876')
    assert got == 'projects/my-proj/locations/us-central1/endpoints/9876'


def test_resolve_full_form_passes_through() -> None:
    """Fully qualified projects/.../endpoints/... paths are returned unchanged."""
    client = _vertex_client()
    name = 'projects/other/locations/us-east1/endpoints/42'
    assert resolve_vertex_model_name(client, name) == name


def test_resolve_non_tuned_name_passes_through() -> None:
    """Non-tuned names (e.g. gemini-2.5-flash) are unchanged so the SDK transformer still applies."""
    client = _vertex_client()
    assert resolve_vertex_model_name(client, 'gemini-2.5-flash') == 'gemini-2.5-flash'


def test_resolve_on_googleai_backend_is_noop() -> None:
    """Tuned endpoints are a Vertex-only concept; on GoogleAI, leave the name alone."""
    client = _googleai_client()
    assert resolve_vertex_model_name(client, 'endpoints/1') == 'endpoints/1'


def test_resolve_without_project_or_location_leaves_name() -> None:
    """If the client lacks project/location, defer to the SDK rather than building a bad path."""
    client = _vertex_client(project='', location='')
    assert resolve_vertex_model_name(client, 'endpoints/1') == 'endpoints/1'
