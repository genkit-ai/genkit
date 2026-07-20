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

"""Tests for Vertex AI multi-region location and per-request location support."""

import os
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_genai import VertexAI
from genkit_google_genai.constants import (
    is_multi_regional_location,
    multi_regional_base_url,
    vertex_api_host,
)
from genkit_google_genai.evaluators.evaluation import EvaluatorFactory
from genkit_google_genai.models import gemini as gemini_module
from genkit_google_genai.models.gemini import GeminiConfigSchema, GeminiModel
from google import genai
from google.genai import types as genai_types
from google.genai.types import HttpOptions

from genkit import GenkitError, Message, ModelRequest, Part, Role, TextPart

US_REP_URL = 'https://aiplatform.us.rep.googleapis.com'
EU_REP_URL = 'https://aiplatform.eu.rep.googleapis.com'


def _text_request(config: dict[str, Any] | None = None) -> ModelRequest[Any]:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
        config=config,
    )


@pytest.fixture(autouse=True)
def _reset_adc_project_cache():
    """Reset the module-level ADC project cache between tests.

    The probed flag matters as much as the value: leaving it set would make a
    later test silently skip the ADC lookup it means to exercise.
    """
    gemini_module._adc_project_cache = None
    gemini_module._adc_project_probed = False
    yield
    gemini_module._adc_project_cache = None
    gemini_module._adc_project_probed = False


class TestMultiRegionConstants:
    """Tests for multi-region helpers."""

    def test_is_multi_regional_location(self) -> None:
        """Only 'us' and 'eu' are multi-regions."""
        assert is_multi_regional_location('us') is True
        assert is_multi_regional_location('eu') is True
        assert is_multi_regional_location('us-central1') is False
        assert is_multi_regional_location('global') is False
        assert is_multi_regional_location(None) is False

    def test_vertex_api_host(self) -> None:
        """Host selection covers regional, multi-regional, and global."""
        assert vertex_api_host('us-central1') == 'us-central1-aiplatform.googleapis.com'
        assert vertex_api_host('global') == 'aiplatform.googleapis.com'
        assert vertex_api_host('us') == 'aiplatform.us.rep.googleapis.com'
        assert vertex_api_host('eu') == 'aiplatform.eu.rep.googleapis.com'

    def test_multi_regional_base_url(self) -> None:
        """Multi-region base URLs use the rep hosts, no trailing slash."""
        assert multi_regional_base_url('us') == US_REP_URL
        assert multi_regional_base_url('eu') == EU_REP_URL


class TestVertexAIPluginLocation:
    """Tests for plugin-level location resolution."""

    def test_multi_region_sets_base_url(self) -> None:
        """A multi-region location routes to the rep endpoint."""
        with patch('genkit_google_genai.google.genai.client.Client'):
            plugin = VertexAI(project='p', location='us')
        assert plugin._location == 'us'
        assert plugin._client_kwargs['http_options'].base_url == US_REP_URL
        assert plugin._base_url_pinned is False

    def test_regional_location_leaves_base_url_unset(self) -> None:
        """Regional locations keep SDK-derived endpoints."""
        with patch('genkit_google_genai.google.genai.client.Client'):
            plugin = VertexAI(project='p', location='europe-west1')
        assert plugin._client_kwargs['http_options'].base_url is None

    def test_global_location_leaves_base_url_unset(self) -> None:
        """Global location keeps the SDK-derived endpoint."""
        with patch('genkit_google_genai.google.genai.client.Client'):
            plugin = VertexAI(project='p', location='global')
        assert plugin._client_kwargs['http_options'].base_url is None

    def test_explicit_base_url_wins_over_multi_region(self) -> None:
        """An explicit base_url is not clobbered by multi-region routing."""
        with patch('genkit_google_genai.google.genai.client.Client'):
            plugin = VertexAI(project='p', location='us', base_url='https://example.com/')
        assert plugin._client_kwargs['http_options'].base_url == 'https://example.com/'
        assert plugin._base_url_pinned is True

    def test_http_options_base_url_wins_over_multi_region(self) -> None:
        """A base_url inside http_options is not clobbered either."""
        with patch('genkit_google_genai.google.genai.client.Client'):
            plugin = VertexAI(project='p', location='eu', http_options={'base_url': 'https://example.com/'})
        assert plugin._client_kwargs['http_options'].base_url == 'https://example.com/'

    def test_camel_case_base_url_wins_over_multi_region(self) -> None:
        """A camelCase baseUrl inside http_options is honored, not clobbered."""
        with patch('genkit_google_genai.google.genai.client.Client'):
            # cast: the camelCase alias is what a JS-shaped config passes; the
            # SDK accepts it at runtime but HttpOptionsDict only spells snake_case.
            plugin = VertexAI(project='p', location='us', http_options=cast(Any, {'baseUrl': 'https://example.com/'}))
        assert plugin._client_kwargs['http_options'].base_url == 'https://example.com/'
        assert plugin._base_url_pinned is True

    def test_empty_base_url_treated_as_unset(self) -> None:
        """An empty base_url string does not defeat multi-region routing."""
        with patch('genkit_google_genai.google.genai.client.Client'):
            plugin = VertexAI(project='p', location='us', base_url='')
        assert plugin._client_kwargs['http_options'].base_url == US_REP_URL

    def test_caller_http_options_not_mutated(self) -> None:
        """Plugin-derived settings never leak into the caller's HttpOptions."""
        shared = HttpOptions()
        with patch('genkit_google_genai.google.genai.client.Client'):
            plugin_us = VertexAI(project='p', location='us', http_options=shared)
            plugin_eu = VertexAI(project='p', location='eu', http_options=shared)
        assert shared.base_url is None
        assert shared.headers is None
        assert plugin_us._client_kwargs['http_options'].base_url == US_REP_URL
        assert plugin_eu._client_kwargs['http_options'].base_url == EU_REP_URL
        assert plugin_eu._base_url_pinned is False

    def test_location_env_fallback_google_cloud_location(self) -> None:
        """GOOGLE_CLOUD_LOCATION is consulted when location is not passed."""
        env = {'GCLOUD_PROJECT': 'p', 'GOOGLE_CLOUD_LOCATION': 'eu'}
        with patch.dict(os.environ, env, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                plugin = VertexAI()
        assert plugin._location == 'eu'
        assert plugin._client_kwargs['http_options'].base_url == EU_REP_URL

    def test_location_env_fallback_chain(self) -> None:
        """GCLOUD_LOCATION is consulted after GOOGLE_CLOUD_LOCATION."""
        with patch.dict(os.environ, {'GCLOUD_PROJECT': 'p', 'GCLOUD_LOCATION': 'europe-west1'}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                assert VertexAI()._location == 'europe-west1'
        env = {'GCLOUD_PROJECT': 'p', 'GOOGLE_CLOUD_LOCATION': 'us-east1', 'GCLOUD_LOCATION': 'europe-west1'}
        with patch.dict(os.environ, env, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                assert VertexAI()._location == 'us-east1'

    def test_location_defaults_to_us_central1(self) -> None:
        """Without an explicit location or env vars, us-central1 is used."""
        with patch.dict(os.environ, {'GCLOUD_PROJECT': 'p'}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                plugin = VertexAI()
        assert plugin._location == 'us-central1'


class TestVertexAIPluginProject:
    """Tests for plugin-level project resolution."""

    def test_google_cloud_project_env_fallback(self) -> None:
        """GOOGLE_CLOUD_PROJECT is consulted after GCLOUD_PROJECT."""
        with patch.dict(os.environ, {'GOOGLE_CLOUD_PROJECT': 'env-p'}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                plugin = VertexAI(location='us')
        assert plugin._project == 'env-p'
        assert plugin._client_kwargs['project'] == 'env-p'

    def test_regional_resolves_project_from_adc(self) -> None:
        """A regional plugin resolves the project via ADC when none is configured.

        Evaluator registration in init() needs a concrete project, so leaving
        it unresolved would fail a deployment that relies purely on ADC.
        """
        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                with patch('genkit_google_genai.google.google_auth_default', return_value=(None, 'adc-p')):
                    plugin = VertexAI(location='us-central1')
        assert plugin._project == 'adc-p'
        assert plugin._client_kwargs['project'] == 'adc-p'

    def test_regional_without_adc_project_does_not_raise(self) -> None:
        """A regional plugin with no resolvable project still constructs.

        Unlike multi-regions, regional endpoints are usable in express mode and
        the SDK raises its own error later, so construction must not fail here.
        """
        from google.auth.exceptions import DefaultCredentialsError

        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                with patch(
                    'genkit_google_genai.google.google_auth_default', side_effect=DefaultCredentialsError('no adc')
                ):
                    plugin = VertexAI(location='us-central1')
        assert plugin._project is None

    def test_api_key_skips_adc_probe(self) -> None:
        """Express mode (api_key, no project) does not probe ADC."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                with patch('genkit_google_genai.google.google_auth_default') as mock_adc:
                    plugin = VertexAI(location='us-central1', api_key='k')
        mock_adc.assert_not_called()
        assert plugin._project is None

    def test_explicit_project_skips_adc_probe(self) -> None:
        """An explicit project short-circuits the ADC probe."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                with patch('genkit_google_genai.google.google_auth_default') as mock_adc:
                    VertexAI(project='p', location='us-central1')
        mock_adc.assert_not_called()

    def test_multi_region_resolves_project_from_adc(self) -> None:
        """With no project configured, a multi-region plugin resolves it via ADC."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                with patch('genkit_google_genai.google.google_auth_default', return_value=(None, 'adc-p')):
                    plugin = VertexAI(location='us')
        assert plugin._project == 'adc-p'
        assert plugin._client_kwargs['project'] == 'adc-p'

    def test_multi_region_without_project_raises(self) -> None:
        """A multi-region plugin with no resolvable project fails fast."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                with patch('genkit_google_genai.google.google_auth_default', return_value=(None, None)):
                    with pytest.raises(ValueError, match='multi-region'):
                        VertexAI(location='eu')

    def test_multi_region_no_adc_raises_friendly_error(self) -> None:
        """A missing ADC setup surfaces as the friendly ValueError, not a raw auth error."""
        from google.auth.exceptions import DefaultCredentialsError

        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                with patch(
                    'genkit_google_genai.google.google_auth_default', side_effect=DefaultCredentialsError('no adc')
                ):
                    with pytest.raises(ValueError, match='multi-region'):
                        VertexAI(location='eu')

    def test_multi_region_uses_credentials_project_id(self) -> None:
        """Explicit credentials carrying project_id avoid the ADC probe."""
        creds = MagicMock()
        creds.project_id = 'creds-p'
        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                with patch('genkit_google_genai.google.google_auth_default') as mock_adc:
                    plugin = VertexAI(location='us', credentials=creds)
        mock_adc.assert_not_called()
        assert plugin._project == 'creds-p'

    def test_multi_region_with_pinned_base_url_still_resolves_project(self) -> None:
        """A pinned base_url does not bypass multi-region project resolution.

        The SDK skips its own ADC project lookup whenever a base_url is set,
        regardless of who set it, so the plugin must resolve the project even
        when the caller pinned the URL.
        """
        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                with patch('genkit_google_genai.google.google_auth_default', return_value=(None, 'adc-p')):
                    plugin = VertexAI(location='us', base_url='https://example.com/')
        assert plugin._project == 'adc-p'
        assert plugin._client_kwargs['project'] == 'adc-p'
        assert plugin._client_kwargs['http_options'].base_url == 'https://example.com/'

    def test_multi_region_with_pinned_base_url_without_project_raises(self) -> None:
        """A pinned base_url plus multi-region still fails fast without a project."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                with patch('genkit_google_genai.google.google_auth_default', return_value=(None, None)):
                    with pytest.raises(ValueError, match='multi-region'):
                        VertexAI(location='eu', base_url='https://example.com/')


class TestRealClientHonorsPinnedBaseUrl:
    """Pin the google-genai contract the plugin's routing relies on.

    Every other test mocks genai.Client, so nothing else would notice if an
    SDK upgrade stopped honoring a user-supplied http_options.base_url (today
    the SDK re-applies user http_options after computing its own default
    endpoint). Constructing the client needs no network or ADC because project
    and location are given explicitly.
    """

    def test_real_client_keeps_multi_region_base_url(self) -> None:
        """A real client keeps the rep base_url the plugin pins for 'us'."""
        with patch.dict(os.environ, {}, clear=True):
            client = genai.Client(
                vertexai=True, project='p', location='us', http_options=HttpOptions(base_url=US_REP_URL)
            )
        assert client._api_client._http_options.base_url == US_REP_URL
        assert client._api_client.project == 'p'
        assert client._api_client.location == 'us'


class TestEvaluatorApiHost:
    """Tests for evaluator endpoint host selection."""

    def test_regional_host(self) -> None:
        """Regional locations use the {location}-aiplatform pattern."""
        factory = EvaluatorFactory(project_id='p', location='us-central1')
        assert factory._api_host() == 'us-central1-aiplatform.googleapis.com'

    def test_global_rejected(self) -> None:
        """The evaluation service is regional; 'global' is rejected."""
        factory = EvaluatorFactory(project_id='p', location='global')
        with pytest.raises(GenkitError, match='does not support'):
            factory._api_host()

    def test_multi_region_rejected(self) -> None:
        """The evaluation service is regional; multi-regions are rejected."""
        factory = EvaluatorFactory(project_id='p', location='eu')
        with pytest.raises(GenkitError, match='does not support'):
            factory._api_host()


class TestGeminiConfigSchemaLocation:
    """Tests for the per-request location config field."""

    def test_location_field(self) -> None:
        """The schema accepts a location override."""
        assert GeminiConfigSchema(location='eu').location == 'eu'
        assert GeminiConfigSchema().location is None


def _vertex_plugin(location: str = 'us-central1', base_url: str | None = None, project: str | None = 'p') -> VertexAI:
    with patch('genkit_google_genai.google.genai.client.Client'):
        return VertexAI(project=project, location=location, base_url=base_url)


def _vertex_model(location: str = 'us-central1', base_url: str | None = None, project: str | None = 'p') -> GeminiModel:
    plugin = _vertex_plugin(location, base_url, project)
    client = MagicMock()
    client.vertexai = True
    return GeminiModel(
        'gemini-2.5-flash',
        client,
        client_kwargs=plugin._client_kwargs,
        base_url_pinned=plugin._base_url_pinned,
    )


def _plugin_client(model: GeminiModel) -> MagicMock:
    """The mock client the model was constructed with, typed for assertions."""
    return cast(MagicMock, model._client)


class TestResolveRequestClient:
    """Tests for per-request client resolution in GeminiModel."""

    @pytest.mark.asyncio
    async def test_no_overrides_returns_plugin_client(self) -> None:
        """Without overrides the plugin client is reused."""
        model = _vertex_model()
        assert await model._resolve_request_client(_text_request()) is model._client

    @pytest.mark.asyncio
    async def test_no_overrides_on_multi_region_plugin_returns_plugin_client(self) -> None:
        """A multi-region plugin without overrides also skips temp-client creation."""
        model = _vertex_model(location='us')
        assert await model._resolve_request_client(_text_request()) is model._client

    @pytest.mark.asyncio
    async def test_location_override_creates_client_with_location(self) -> None:
        """A regional override is passed through with plugin settings intact."""
        model = _vertex_model()
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            await model._resolve_request_client(_text_request({'location': 'europe-west1'}))
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs['location'] == 'europe-west1'
        assert kwargs['project'] == 'p'
        assert kwargs['http_options'].base_url is None
        assert 'x-goog-api-client' in kwargs['http_options'].headers

    @pytest.mark.asyncio
    async def test_multi_region_override_sets_rep_base_url(self) -> None:
        """A multi-region override routes to the rep endpoint."""
        model = _vertex_model()
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            await model._resolve_request_client(_text_request({'location': 'eu'}))
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs['location'] == 'eu'
        assert kwargs['http_options'].base_url == EU_REP_URL

    @pytest.mark.asyncio
    async def test_api_version_override_keeps_multi_region_base_url(self) -> None:
        """An apiVersion-only override on a multi-region plugin keeps the rep endpoint."""
        model = _vertex_model(location='us')
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            await model._resolve_request_client(_text_request({'api_version': 'v1'}))
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs['location'] == 'us'
        assert kwargs['http_options'].api_version == 'v1'
        assert kwargs['http_options'].base_url == US_REP_URL

    @pytest.mark.asyncio
    async def test_regional_override_on_multi_region_plugin_clears_rep_url(self) -> None:
        """Overriding a multi-region plugin with a region drops the rep endpoint."""
        model = _vertex_model(location='us')
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            await model._resolve_request_client(_text_request({'location': 'europe-west1'}))
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs['location'] == 'europe-west1'
        assert kwargs['http_options'].base_url is None

    @pytest.mark.asyncio
    async def test_pinned_base_url_survives_location_override(self) -> None:
        """A plugin-pinned base URL (proxy) is preserved across a location override."""
        model = _vertex_model(base_url='https://corp-proxy.example.com/')
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            await model._resolve_request_client(_text_request({'location': 'us'}))
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs['location'] == 'us'
        assert kwargs['http_options'].base_url == 'https://corp-proxy.example.com/'

    @pytest.mark.asyncio
    async def test_pinned_base_url_survives_api_version_override(self) -> None:
        """A plugin-pinned base URL is preserved across an apiVersion override."""
        model = _vertex_model(base_url='https://corp-proxy.example.com/')
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            await model._resolve_request_client(_text_request({'api_version': 'v1'}))
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs['http_options'].base_url == 'https://corp-proxy.example.com/'
        assert kwargs['http_options'].api_version == 'v1'

    @pytest.mark.asyncio
    async def test_base_url_override_wins(self) -> None:
        """An explicit per-request base_url beats multi-region derivation."""
        model = _vertex_model()
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            await model._resolve_request_client(_text_request({'location': 'us', 'base_url': 'https://example.com/'}))
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs['http_options'].base_url == 'https://example.com/'

    @pytest.mark.asyncio
    async def test_multi_region_override_resolves_missing_project(self) -> None:
        """A multi-region override with no project falls back to ADC."""
        model = _vertex_model()
        model._client_kwargs = dict(model._client_kwargs, project=None)
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            with patch('genkit_google_genai.models.gemini.google_auth_default', return_value=(None, 'adc-p')):
                await model._resolve_request_client(_text_request({'location': 'eu'}))
        assert mock_ctor.call_args.kwargs['project'] == 'adc-p'

    @pytest.mark.asyncio
    async def test_multi_region_override_without_project_raises(self) -> None:
        """A multi-region override with no resolvable project fails fast."""
        model = _vertex_model()
        model._client_kwargs = dict(model._client_kwargs, project=None)
        with patch('genkit_google_genai.models.gemini.google_auth_default', return_value=(None, None)):
            with pytest.raises(GenkitError, match='project is required'):
                await model._resolve_request_client(_text_request({'location': 'eu'}))

    @pytest.mark.asyncio
    async def test_multi_region_override_no_adc_raises_friendly_error(self) -> None:
        """DefaultCredentialsError surfaces as the friendly GenkitError."""
        from google.auth.exceptions import DefaultCredentialsError

        model = _vertex_model()
        model._client_kwargs = dict(model._client_kwargs, project=None)
        with patch(
            'genkit_google_genai.models.gemini.google_auth_default', side_effect=DefaultCredentialsError('no adc')
        ):
            with pytest.raises(GenkitError, match='project is required'):
                await model._resolve_request_client(_text_request({'location': 'eu'}))

    @pytest.mark.asyncio
    async def test_failed_adc_probe_is_not_repeated(self) -> None:
        """A missing ADC setup is probed once, not on every overridden request.

        ADC resolution does blocking file and metadata-server IO, so an
        environment without ADC (express mode, say) must not pay for a probe
        per request.
        """
        from google.auth.exceptions import DefaultCredentialsError

        model = _vertex_model()
        model._client_kwargs = dict(model._client_kwargs, project=None)
        with patch('genkit_google_genai.models.gemini.genai.Client'):
            with patch(
                'genkit_google_genai.models.gemini.google_auth_default',
                side_effect=DefaultCredentialsError('no adc'),
            ) as mock_adc:
                await model._resolve_request_client(_text_request({'api_version': 'v1'}))
                await model._resolve_request_client(_text_request({'api_version': 'v1'}))
        assert mock_adc.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_adc_project_is_not_repeated(self) -> None:
        """ADC resolving to no project is also cached, not re-probed."""
        model = _vertex_model()
        model._client_kwargs = dict(model._client_kwargs, project=None)
        with patch('genkit_google_genai.models.gemini.genai.Client'):
            with patch('genkit_google_genai.models.gemini.google_auth_default', return_value=(None, None)) as mock_adc:
                await model._resolve_request_client(_text_request({'api_version': 'v1'}))
                await model._resolve_request_client(_text_request({'api_version': 'v1'}))
        assert mock_adc.call_count == 1

    @pytest.mark.asyncio
    async def test_successful_adc_probe_is_not_repeated(self) -> None:
        """A successful resolution stays cached across requests."""
        model = _vertex_model()
        model._client_kwargs = dict(model._client_kwargs, project=None)
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            with patch(
                'genkit_google_genai.models.gemini.google_auth_default', return_value=(None, 'adc-p')
            ) as mock_adc:
                await model._resolve_request_client(_text_request({'api_version': 'v1'}))
                await model._resolve_request_client(_text_request({'api_version': 'v1'}))
        assert mock_adc.call_count == 1
        assert mock_ctor.call_args.kwargs['project'] == 'adc-p'

    @pytest.mark.asyncio
    async def test_api_version_override_prefills_adc_project(self) -> None:
        """Any override on an ADC-project plugin resolves the project off-loop."""
        model = _vertex_model()
        model._client_kwargs = dict(model._client_kwargs, project=None)
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            with patch('genkit_google_genai.models.gemini.google_auth_default', return_value=(None, 'adc-p')):
                await model._resolve_request_client(_text_request({'api_version': 'v1'}))
        assert mock_ctor.call_args.kwargs['project'] == 'adc-p'

    @pytest.mark.asyncio
    async def test_base_url_override_prefills_adc_project(self) -> None:
        """A per-request base_url override keeps project/location resolution intact."""
        model = _vertex_model()
        model._client_kwargs = dict(model._client_kwargs, project=None)
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            with patch('genkit_google_genai.models.gemini.google_auth_default', return_value=(None, 'adc-p')):
                await model._resolve_request_client(_text_request({'base_url': 'https://corp-proxy.example.com/'}))
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs['project'] == 'adc-p'
        assert kwargs['http_options'].base_url == 'https://corp-proxy.example.com/'

    @pytest.mark.asyncio
    async def test_express_mode_override_skips_adc_probe(self) -> None:
        """Express mode (api_key) never resolves a project for an override.

        The SDK rejects api_key and project together, so a resolved ADC
        project on the developer's machine would make an otherwise valid
        apiVersion override fail client construction.
        """
        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                plugin = VertexAI(api_key='k', location='us-central1')
        client = MagicMock()
        client.vertexai = True
        model = GeminiModel(
            'gemini-2.5-flash',
            client,
            client_kwargs=plugin._client_kwargs,
            base_url_pinned=plugin._base_url_pinned,
        )
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            with patch('genkit_google_genai.models.gemini.google_auth_default', return_value=(None, 'adc-p')) as adc:
                await model._resolve_request_client(_text_request({'api_version': 'v1'}))
        adc.assert_not_called()
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs['api_key'] == 'k'
        assert kwargs.get('project') is None

    @pytest.mark.asyncio
    async def test_express_mode_multi_region_override_raises(self) -> None:
        """A multi-region override in express mode fails with a clear error."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('genkit_google_genai.google.genai.client.Client'):
                plugin = VertexAI(api_key='k', location='us-central1')
        client = MagicMock()
        client.vertexai = True
        model = GeminiModel(
            'gemini-2.5-flash',
            client,
            client_kwargs=plugin._client_kwargs,
            base_url_pinned=plugin._base_url_pinned,
        )
        with patch('genkit_google_genai.models.gemini.google_auth_default') as adc:
            with pytest.raises(GenkitError, match='express'):
                await model._resolve_request_client(_text_request({'location': 'eu'}))
        adc.assert_not_called()

    @pytest.mark.asyncio
    async def test_credentials_project_id_used_before_adc(self) -> None:
        """A credentials object carrying project_id avoids the ADC probe."""
        model = _vertex_model()
        creds = MagicMock()
        creds.project_id = 'creds-p'
        model._client_kwargs = dict(model._client_kwargs, project=None, credentials=creds)
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            with patch('genkit_google_genai.models.gemini.google_auth_default') as mock_adc:
                await model._resolve_request_client(_text_request({'location': 'eu'}))
        mock_adc.assert_not_called()
        assert mock_ctor.call_args.kwargs['project'] == 'creds-p'

    @pytest.mark.asyncio
    async def test_location_ignored_for_googleai_backend(self) -> None:
        """Location overrides are ignored for the Gemini API backend."""
        client = MagicMock()
        client.vertexai = False
        model = GeminiModel('gemini-2.5-flash', client, client_kwargs={'vertexai': False, 'api_key': 'k'})
        assert await model._resolve_request_client(_text_request({'location': 'eu'})) is client

    @pytest.mark.asyncio
    async def test_api_key_override_for_googleai_backend(self) -> None:
        """A per-request api_key override replaces the plugin key and drops credentials."""
        client = MagicMock()
        client.vertexai = False
        model = GeminiModel(
            'gemini-2.5-flash',
            client,
            client_kwargs={'vertexai': False, 'api_key': 'plugin-key', 'credentials': MagicMock()},
        )
        with patch('genkit_google_genai.models.gemini.genai.Client') as mock_ctor:
            await model._resolve_request_client(_text_request({'api_key': 'override-key'}))
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs['api_key'] == 'override-key'
        assert kwargs['credentials'] is None


class TestPluginModelWiring:
    """The plugin must pass its client kwargs into the models it constructs."""

    @pytest.mark.asyncio
    async def test_vertexai_action_passes_client_kwargs(self) -> None:
        """The model action constructs GeminiModel with the plugin's kwargs."""
        plugin = _vertex_plugin(location='us')
        action = plugin._resolve_model('vertexai/gemini-2.5-flash')
        with patch('genkit_google_genai.google.GeminiModel') as mock_model:
            mock_model.return_value.generate = AsyncMock(return_value=MagicMock())
            await action._fn(_text_request(), MagicMock())
        kwargs = mock_model.call_args.kwargs
        assert kwargs['client_kwargs'] is plugin._client_kwargs
        assert kwargs['base_url_pinned'] is plugin._base_url_pinned

    @pytest.mark.asyncio
    async def test_googleai_action_passes_client_kwargs(self) -> None:
        """The GoogleAI model action also passes the plugin's kwargs."""
        from genkit_google_genai import GoogleAI

        with patch('genkit_google_genai.google.genai.client.Client'):
            plugin = GoogleAI(api_key='k')
        action = plugin._resolve_model('googleai/gemini-2.5-flash')
        with patch('genkit_google_genai.google.GeminiModel') as mock_model:
            mock_model.return_value.generate = AsyncMock(return_value=MagicMock())
            await action._fn(_text_request(), MagicMock())
        kwargs = mock_model.call_args.kwargs
        assert kwargs['client_kwargs'] is plugin._client_kwargs


class TestGenerateUsesResolvedClient:
    """generate() must issue the API call on the override-resolved client."""

    @pytest.mark.asyncio
    async def test_generate_calls_temp_client(self) -> None:
        """A location override routes the generate call through the temp client."""
        model = _vertex_model()
        response = genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(parts=[genai_types.Part(text='ok')], role='model'),
                    finish_reason=genai_types.FinishReason.STOP,
                )
            ]
        )
        temp_client = MagicMock()
        temp_client.aio.models.generate_content = AsyncMock(return_value=response)
        ctx = MagicMock()
        ctx.is_streaming = False
        with patch('genkit_google_genai.models.gemini.genai.Client', return_value=temp_client):
            result = await model.generate(_text_request({'location': 'europe-west1'}), ctx)
        temp_client.aio.models.generate_content.assert_awaited_once()
        _plugin_client(model).aio.models.generate_content.assert_not_called()
        assert result.message is not None
        assert result.message.content[0].root.text == 'ok'

    @pytest.mark.asyncio
    async def test_streaming_generate_calls_temp_client(self) -> None:
        """A location override routes the streaming call through the temp client."""
        model = _vertex_model()
        chunk = genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(parts=[genai_types.Part(text='ok')], role='model'),
                    finish_reason=genai_types.FinishReason.STOP,
                )
            ]
        )

        async def _stream():
            yield chunk

        temp_client = MagicMock()
        temp_client.aio.models.generate_content_stream = AsyncMock(return_value=_stream())
        ctx = MagicMock()
        ctx.is_streaming = True
        with patch('genkit_google_genai.models.gemini.genai.Client', return_value=temp_client):
            await model.generate(_text_request({'location': 'europe-west1'}), ctx)
        temp_client.aio.models.generate_content_stream.assert_awaited_once()
        _plugin_client(model).aio.models.generate_content_stream.assert_not_called()
        ctx.send_chunk.assert_called()

    @pytest.mark.asyncio
    async def test_generate_threads_resolved_client_into_message_building(self) -> None:
        """generate() passes the resolved client to _build_messages (cache ops)."""
        model = _vertex_model()
        response = genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(parts=[genai_types.Part(text='ok')], role='model'),
                    finish_reason=genai_types.FinishReason.STOP,
                )
            ]
        )
        temp_client = MagicMock()
        temp_client.aio.models.generate_content = AsyncMock(return_value=response)
        ctx = MagicMock()
        ctx.is_streaming = False
        contents = [genai_types.Content(parts=[genai_types.Part(text='hi')], role='user')]
        with patch('genkit_google_genai.models.gemini.genai.Client', return_value=temp_client):
            with patch.object(model, '_build_messages', AsyncMock(return_value=(contents, None))) as mock_build:
                await model.generate(_text_request({'location': 'europe-west1'}), ctx)
        assert mock_build.call_args.kwargs['client'] is temp_client


class TestCachedContentClientRouting:
    """Context-cache operations must run on the request-resolved client."""

    @pytest.mark.asyncio
    async def test_retrieve_cached_content_uses_passed_client(self) -> None:
        """Cache list/create run on the passed client, not the plugin client."""
        model = _vertex_model()

        async def _no_pages():
            return
            yield

        cache_client = MagicMock()
        cache_client.aio.caches.list = AsyncMock(return_value=_no_pages())
        cache_client.aio.caches.create = AsyncMock(return_value=genai_types.CachedContent(name='caches/x'))
        contents = [genai_types.Content(parts=[genai_types.Part(text='hi')], role='user')]
        cache = await model._retrieve_cached_content(
            request=_text_request(),
            model_name='gemini-2.0-flash',
            cache_config={'ttl_seconds': 60},
            contents=contents,
            client=cache_client,
        )
        cache_client.aio.caches.list.assert_awaited_once()
        cache_client.aio.caches.create.assert_awaited_once()
        _plugin_client(model).aio.caches.list.assert_not_called()
        _plugin_client(model).aio.caches.create.assert_not_called()
        assert cache.name == 'caches/x'


class TestLocationConfigThroughPipeline:
    """Regression tests: the location key must never reach the API config."""

    @pytest.mark.asyncio
    async def test_location_stripped_from_generate_content_config(self) -> None:
        """A config with location converts to GenerateContentConfig without error."""
        model = _vertex_model()
        request = _text_request({'location': 'eu', 'temperature': 0.5})
        cfg = await model._genkit_to_googleai_cfg(request=request)
        assert cfg is not None
        assert cfg.temperature == 0.5
        assert not hasattr(cfg, 'location')

    @pytest.mark.asyncio
    async def test_typed_config_location_stripped(self) -> None:
        """Same for a typed GeminiConfigSchema config."""
        model = _vertex_model()
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
            config=GeminiConfigSchema(location='us', temperature=0.1),
        )
        cfg = await model._genkit_to_googleai_cfg(request=request)
        assert cfg is not None
        assert not hasattr(cfg, 'location')
