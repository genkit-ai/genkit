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

"""Typed construction of every exported config class, imported from the package root.

This file is the type-checker contract: pyright, pyrefly and ty must accept
the flat snake_case keyword arguments and dict-literal list entries here with
no suppressions. The runtime asserts pin the camelCase config dump and the
HTTP body the plugin sends.
"""

import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
import pytest
from genkit_google_genai import (
    AntigravityConfig,
    DeepResearchConfig,
    GeminiConfig,
    GeminiImageConfig,
    GeminiTtsConfig,
    GemmaConfig,
    GoogleAI,
    HarmBlockMethod,
    HarmBlockThreshold,
    HarmCategory,
    ImageAspectRatio,
    ImageResizeMode,
    ImageSize,
    LyriaConfig,
    McpServerConfig,
    PersonGeneration,
    PhishBlockThreshold,
    ProminentPeople,
    ResponseModality,
    SafetySetting,
    SearchType,
    ThinkingLevel,
    VeoConfig,
    VideoCompressionQuality,
)
from pydantic import BaseModel, ValidationError
from typing_extensions import assert_type

from genkit import Genkit


def _wire(config: BaseModel) -> dict[str, object]:
    return config.model_dump(by_alias=True, exclude_none=True, mode='json')


def test_empty_construction() -> None:
    """Every field is optional, so a bare constructor type-checks and validates."""
    for cls in (
        GeminiConfig,
        GeminiTtsConfig,
        GeminiImageConfig,
        GemmaConfig,
        VeoConfig,
        LyriaConfig,
        AntigravityConfig,
        DeepResearchConfig,
    ):
        assert _wire(cls()) == {}


def test_gemini_config_flat_kwargs() -> None:
    """GeminiConfig takes flat snake_case kwargs; the dump is flat camelCase."""
    config = GeminiConfig(
        version='gemini-2.5-flash-001',
        base_url='https://kitchen.example',
        api_version='v1beta',
        location='us-central1',
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_output_tokens=1000,
        stop_sequences=['END'],
        seed=7,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        response_logprobs=True,
        logprobs=3,
        response_modalities=['TEXT'],
        context_cache=True,
        code_execution=True,
        google_search=True,
        google_search_types=['web', 'image'],
        google_search_exclude_domains=['example.com'],
        google_search_blocking_confidence='BLOCK_HIGH_AND_ABOVE',
        google_search_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        google_search_end_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
        url_context=True,
        safety_settings=[
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH', 'method': 'SEVERITY'},
            SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='OFF'),
        ],
        file_search_store_names=['fileSearchStores/menu'],
        file_search_metadata_filter='cuisine=thai',
        file_search_top_k=3,
        thinking_level='HIGH',
        include_thoughts=True,
        extra={'labels': {'team': 'kitchen'}},
    )

    assert _wire(config) == {
        'version': 'gemini-2.5-flash-001',
        'baseUrl': 'https://kitchen.example',
        'apiVersion': 'v1beta',
        'location': 'us-central1',
        'temperature': 0.7,
        'topP': 0.9,
        'topK': 40,
        'maxOutputTokens': 1000,
        'stopSequences': ['END'],
        'seed': 7,
        'presencePenalty': 0.1,
        'frequencyPenalty': 0.2,
        'responseLogprobs': True,
        'logprobs': 3,
        'responseModalities': ['TEXT'],
        'contextCache': True,
        'codeExecution': True,
        'googleSearch': True,
        'googleSearchTypes': ['web', 'image'],
        'googleSearchExcludeDomains': ['example.com'],
        'googleSearchBlockingConfidence': 'BLOCK_HIGH_AND_ABOVE',
        'googleSearchStartTime': '2026-01-01T00:00:00Z',
        'googleSearchEndTime': '2026-02-01T00:00:00Z',
        'urlContext': True,
        'safetySettings': [
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH', 'method': 'SEVERITY'},
            {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'OFF'},
        ],
        'fileSearchStoreNames': ['fileSearchStores/menu'],
        'fileSearchMetadataFilter': 'cuisine=thai',
        'fileSearchTopK': 3,
        'thinkingLevel': 'HIGH',
        'includeThoughts': True,
        'extra': {'labels': {'team': 'kitchen'}},
    }
    # Same instance from camelCase wire JSON, as a .prompt file writes it.
    assert GeminiConfig.model_validate(_wire(config)) == config


def test_gemini_tts_config_flat_voice() -> None:
    """GeminiTtsConfig takes a voice name, or speakers mapped to voice names."""
    single = GeminiTtsConfig(voice='Kore', language_code='en-US')
    duet = GeminiTtsConfig(temperature=0.3, speakers={'Chef': 'Kore', 'Server': 'Puck'})

    assert _wire(single) == {'voice': 'Kore', 'languageCode': 'en-US'}
    assert _wire(duet) == {'temperature': 0.3, 'speakers': {'Chef': 'Kore', 'Server': 'Puck'}}


def test_gemini_image_config_flat_image() -> None:
    """GeminiImageConfig takes the image settings as flat fields."""
    config = GeminiImageConfig(
        response_modalities=['TEXT', 'IMAGE'],
        aspect_ratio='16:9',
        image_size='2K',
        output_mime_type='image/jpeg',
        output_compression_quality=80,
        person_generation='ALLOW_ADULT',
        prominent_people='BLOCK_PROMINENT_PEOPLE',
    )

    assert _wire(config) == {
        'responseModalities': ['TEXT', 'IMAGE'],
        'aspectRatio': '16:9',
        'imageSize': '2K',
        'outputMimeType': 'image/jpeg',
        'outputCompressionQuality': 80,
        'personGeneration': 'ALLOW_ADULT',
        'prominentPeople': 'BLOCK_PROMINENT_PEOPLE',
    }


def test_gemma_config_snake_case_kwargs() -> None:
    """GemmaConfig inherits GeminiConfig fields and keeps an unbounded temperature."""
    config = GemmaConfig(temperature=2.5, max_output_tokens=256, top_k=20)

    assert _wire(config) == {'temperature': 2.5, 'maxOutputTokens': 256, 'topK': 20}


def test_veo_config_snake_case_kwargs() -> None:
    """VeoConfig takes snake_case kwargs; the dump is camelCase."""
    config = VeoConfig(
        number_of_videos=1,
        generate_audio=True,
        fps=24,
        output_gcs_uri='gs://kitchen/promo.mp4',
        pubsub_topic='projects/p/topics/renders',
        compression_quality='OPTIMIZED',
        resize_mode='PAD',
        labels={'team': 'kitchen'},
        last_frame={'uri': 'gs://kitchen/last.png'},
        reference_images=[{'uri': 'gs://kitchen/ref.png'}],
        mask={'uri': 'gs://kitchen/mask.png'},
        webhook_config={'url': 'https://kitchen.example/hook'},
        extra={'parameters': {'storageUri': 'gs://kitchen'}},
        negative_prompt='no logos',
        aspect_ratio='16:9',
        person_generation='allow_adult',
        duration_seconds=8,
        resolution='720p',
        seed=3,
        enhance_prompt=True,
        base_url='https://kitchen.example',
        api_version='v1',
        location='us-central1',
    )

    assert _wire(config) == {
        'numberOfVideos': 1,
        'generateAudio': True,
        'fps': 24,
        'outputGcsUri': 'gs://kitchen/promo.mp4',
        'pubsubTopic': 'projects/p/topics/renders',
        'compressionQuality': 'OPTIMIZED',
        'resizeMode': 'PAD',
        'labels': {'team': 'kitchen'},
        'lastFrame': {'uri': 'gs://kitchen/last.png'},
        'referenceImages': [{'uri': 'gs://kitchen/ref.png'}],
        'mask': {'uri': 'gs://kitchen/mask.png'},
        'webhookConfig': {'url': 'https://kitchen.example/hook'},
        'extra': {'parameters': {'storageUri': 'gs://kitchen'}},
        'negativePrompt': 'no logos',
        'aspectRatio': '16:9',
        'personGeneration': 'allow_adult',
        'durationSeconds': 8,
        'resolution': '720p',
        'seed': 3,
        'enhancePrompt': True,
        'baseUrl': 'https://kitchen.example',
        'apiVersion': 'v1',
        'location': 'us-central1',
    }


def test_lyria_config_snake_case_kwargs() -> None:
    """LyriaConfig takes snake_case kwargs; the dump is camelCase."""
    config = LyriaConfig(
        base_url='https://kitchen.example',
        api_version='v1beta',
        timeout=30000,
        custom_headers={'x-team': 'kitchen'},
        response_modalities=['audio'],
    )

    assert _wire(config) == {
        'baseUrl': 'https://kitchen.example',
        'apiVersion': 'v1beta',
        'timeout': 30000,
        'customHeaders': {'x-team': 'kitchen'},
        'responseModalities': ['audio'],
    }


def test_antigravity_config_snake_case_kwargs() -> None:
    """AntigravityConfig takes snake_case kwargs; the dump is camelCase."""
    config = AntigravityConfig(
        base_url='https://kitchen.example',
        api_version='v1beta',
        timeout=30000,
        custom_headers={'x-team': 'kitchen'},
        previous_interaction_id='int-1',
        store=True,
        environment='sandbox',
        response_modalities=['text'],
    )

    assert _wire(config) == {
        'baseUrl': 'https://kitchen.example',
        'apiVersion': 'v1beta',
        'timeout': 30000,
        'customHeaders': {'x-team': 'kitchen'},
        'previousInteractionId': 'int-1',
        'store': True,
        'environment': 'sandbox',
        'responseModalities': ['text'],
    }


def test_deep_research_config_flat_kwargs() -> None:
    """DeepResearchConfig takes flat file search fields and dict-literal MCP servers."""
    config = DeepResearchConfig(
        thinking_summaries='auto',
        previous_interaction_id='int-1',
        collaborative_planning=True,
        google_search=True,
        file_search_store_names=['fileSearchStores/menu'],
        mcp_servers=[
            {'name': 'crm', 'url': 'https://crm.example/mcp', 'allowed_tools': ['lookup']},
            McpServerConfig(name='pos', url='https://pos.example/mcp'),
        ],
    )

    assert _wire(config) == {
        'thinkingSummaries': 'auto',
        'previousInteractionId': 'int-1',
        'collaborativePlanning': True,
        'googleSearch': True,
        'fileSearchStoreNames': ['fileSearchStores/menu'],
        'mcpServers': [
            {'name': 'crm', 'url': 'https://crm.example/mcp', 'allowed_tools': ['lookup']},
            {'name': 'pos', 'url': 'https://pos.example/mcp'},
        ],
    }


def _fake_gemini_api(bodies: list[dict[str, Any]]) -> Callable[..., Awaitable[httpx.Response]]:
    """Stand-in for httpx.AsyncClient.send that answers generateContent and records each POST body."""

    async def send(_client: httpx.AsyncClient, request: httpx.Request, **_: object) -> httpx.Response:
        if request.method == 'GET':
            return httpx.Response(200, json={'models': []}, request=request)
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'ok'}]}, 'finishReason': 'STOP'}]},
            request=request,
        )

    return send


@pytest.mark.asyncio
async def test_flat_config_request_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flat GeminiConfig reaches the generateContent body in the nested API shape.

    google-genai camelCases the fields it types and forwards the nested
    dicts Genkit hands it as-is, so keys inside thinkingConfig and fileSearch
    go out snake_case. The API accepts both spellings.
    """
    # 1. Route google-genai's HTTP calls to a fake API
    bodies: list[dict[str, Any]] = []
    monkeypatch.setattr(httpx.AsyncClient, 'send', _fake_gemini_api(bodies))
    ai = Genkit(plugins=[GoogleAI(api_key='fake-key')])

    # 2. Generate with a flat config
    config = GeminiConfig(
        temperature=0.4,
        max_output_tokens=500,
        safety_settings=[{'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH'}],
        thinking_budget=1024,
        include_thoughts=True,
        file_search_store_names=['fileSearchStores/menu'],
    )
    response = await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.', config=config)

    # 3. Check the body the plugin sent
    assert response.text == 'ok'
    assert bodies == [
        {
            'contents': [{'parts': [{'text': 'Suggest a dish.'}], 'role': 'user'}],
            'generationConfig': {
                'maxOutputTokens': 500,
                'temperature': 0.4,
                'thinkingConfig': {'include_thoughts': True, 'thinking_budget': 1024},
            },
            'safetySettings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH'}],
            'tools': [{'fileSearch': {'file_search_store_names': ['fileSearchStores/menu']}}],
        }
    ]


@pytest.mark.parametrize(
    ('tool_choice', 'tool_config'),
    [
        ('auto', {'functionCallingConfig': {'mode': 'AUTO'}}),
        ('required', {'functionCallingConfig': {'mode': 'ANY', 'allowedFunctionNames': ['lookup_menu']}}),
        # 'none' used to be ignored, leaving Gemini free to call the tools.
        ('none', {'functionCallingConfig': {'mode': 'NONE'}}),
        (None, None),
    ],
    ids=['auto', 'required', 'none', 'unset'],
)
@pytest.mark.asyncio
async def test_tool_choice_request_body(
    monkeypatch: pytest.MonkeyPatch,
    tool_choice: Literal['auto', 'required', 'none'] | None,
    tool_config: dict[str, Any] | None,
) -> None:
    """Genkit's tool_choice goes out as toolConfig.functionCallingConfig; 'required' names the request's tools."""
    # 1. Route google-genai's HTTP calls to a fake API and register one tool
    bodies: list[dict[str, Any]] = []
    monkeypatch.setattr(httpx.AsyncClient, 'send', _fake_gemini_api(bodies))
    ai = Genkit(plugins=[GoogleAI(api_key='fake-key')])

    @ai.tool(name='lookup_menu')
    async def lookup_menu(dish: str) -> str:
        return 'ramen'

    # 2. Generate with the tool choice
    await ai.generate(
        model='googleai/gemini-2.5-flash',
        prompt='Is the pho in stock?',
        tools=['lookup_menu'],
        tool_choice=tool_choice,
    )

    # 3. The body carries the mapped mode, or no toolConfig for unset
    assert bodies[-1].get('toolConfig') == tool_config


@pytest.mark.asyncio
async def test_tool_choice_without_function_tools_sends_no_tool_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built-in tools alone don't get a function calling mode; there are no functions to govern."""
    bodies: list[dict[str, Any]] = []
    monkeypatch.setattr(httpx.AsyncClient, 'send', _fake_gemini_api(bodies))
    ai = Genkit(plugins=[GoogleAI(api_key='fake-key')])

    await ai.generate(
        model='googleai/gemini-2.5-flash', prompt='hi', tool_choice='required', config=GeminiConfig(google_search=True)
    )

    assert 'toolConfig' not in bodies[-1]


def test_choice_fields_are_closed_literals() -> None:
    """Choice fields take plain strings from a closed set.

    assert_type makes pyright, pyrefly and ty check the declared set, so
    GeminiConfig(thinking_level='ULTRA') fails all three. Runtime rejects the
    same value from JSON.
    """
    # 1. Static: the field type is the closed Literal, not str
    assert_type(GeminiConfig().thinking_level, Literal['MINIMAL', 'LOW', 'MEDIUM', 'HIGH'] | None)
    assert_type(GeminiConfig().google_search_blocking_confidence, PhishBlockThreshold | None)
    assert_type(GeminiConfig().google_search_types, list[SearchType] | None)
    assert_type(GeminiConfig().response_modalities, list[ResponseModality] | None)
    assert_type(GeminiConfig().google_search, bool | None)
    assert_type(GeminiImageConfig().aspect_ratio, ImageAspectRatio | None)
    assert_type(GeminiImageConfig().image_size, ImageSize | None)
    assert_type(GeminiImageConfig().person_generation, PersonGeneration | None)
    assert_type(GeminiImageConfig().prominent_people, ProminentPeople | None)
    assert_type(VeoConfig().compression_quality, VideoCompressionQuality | None)
    assert_type(VeoConfig().resize_mode, ImageResizeMode | None)
    safety = SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE', method='SEVERITY')
    assert_type(safety['category'], HarmCategory)
    assert_type(safety['threshold'], HarmBlockThreshold)
    assert_type(safety.get('method'), HarmBlockMethod | None)

    # 2. Runtime: an unknown value fails validation
    with pytest.raises(ValidationError, match='thinking_level|thinkingLevel'):
        GeminiConfig.model_validate({'thinkingLevel': 'ULTRA'})


def test_exported_literal_aliases_annotate_user_code() -> None:
    """The root Literal aliases type a caller's own helpers; values stay plain strings."""

    # 1. A menu helper typed with the exported aliases
    def plan_config(level: ThinkingLevel, threshold: HarmBlockThreshold) -> GeminiConfig:
        return GeminiConfig(
            thinking_level=level,
            safety_settings=[{'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': threshold}],
        )

    # 2. Call it with plain strings
    config = plan_config('LOW', 'BLOCK_ONLY_HIGH')

    # 3. The dump carries the same strings
    assert _wire(config) == {
        'thinkingLevel': 'LOW',
        'safetySettings': [{'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_ONLY_HIGH'}],
    }


@pytest.mark.parametrize(
    ('config', 'tool'),
    [
        (
            DeepResearchConfig(file_search_store_names=['fileSearchStores/menu']),
            {'type': 'file_search', 'file_search_store_names': ['fileSearchStores/menu']},
        ),
        (
            DeepResearchConfig(
                file_search_store_names=['fileSearchStores/menu'],
                file_search_top_k=5,
                file_search_metadata_filter='cuisine=thai',
            ),
            {
                'type': 'file_search',
                'file_search_store_names': ['fileSearchStores/menu'],
                'top_k': 5,
                'metadata_filter': 'cuisine=thai',
            },
        ),
    ],
    ids=['store-names-only', 'top-k-and-filter'],
)
@pytest.mark.asyncio
async def test_deep_research_file_search_create_body(
    monkeypatch: pytest.MonkeyPatch, config: DeepResearchConfig, tool: dict[str, Any]
) -> None:
    """The Interactions create body carries the file_search tool in snake_case, without unset keys."""
    # 1. Route the Interactions call to a fake API
    bodies: list[dict[str, Any]] = []

    async def send(_client: httpx.AsyncClient, request: httpx.Request, **_: object) -> httpx.Response:
        if request.method == 'GET':
            return httpx.Response(200, json={'models': []}, request=request)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'id': 'int-1', 'status': 'in_progress'}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    ai = Genkit(plugins=[GoogleAI(api_key='fake-key')])

    # 2. Start a Deep Research job with file search attached
    await ai.generate_operation(
        model='googleai/deep-research-pro-preview-12-2025',
        prompt='Research pho shops.',
        config=config,
    )

    # 3. Check the create body
    assert bodies == [
        {
            'agent': 'deep-research-pro-preview-12-2025',
            'agent_config': {'type': 'deep-research'},
            'background': True,
            'input': [{'content': [{'text': 'Research pho shops.', 'type': 'text'}], 'type': 'user_input'}],
            'tools': [tool],
        }
    ]
