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
snake_case keyword arguments and typed nested values here with no
suppressions. The runtime asserts pin the camelCase config dump and the
HTTP body the plugin sends.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx
import pytest
from genkit_google_genai import (
    AntigravityConfig,
    DeepResearchConfig,
    FileSearchConfig,
    FunctionCallingConfig,
    FunctionCallingMode,
    GeminiConfig,
    GeminiImageConfig,
    GeminiTtsConfig,
    GemmaConfig,
    GoogleAI,
    HarmBlockMethod,
    HarmBlockThreshold,
    HarmCategory,
    ImageAspectRatio,
    ImageConfig,
    ImageSize,
    LyriaConfig,
    McpServerConfig,
    MultiSpeakerVoiceConfig,
    PrebuiltVoiceConfig,
    SafetySetting,
    SpeakerVoiceConfig,
    SpeechConfig,
    ThinkingConfig,
    ThinkingLevel,
    VeoConfig,
    VoiceConfig,
)
from google.genai import types as genai_types
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
        ThinkingConfig,
        FunctionCallingConfig,
        FileSearchConfig,
        ImageConfig,
        VoiceConfig,
        PrebuiltVoiceConfig,
        SpeakerVoiceConfig,
        MultiSpeakerVoiceConfig,
        SpeechConfig,
    ):
        assert _wire(cls()) == {}


def test_gemini_config_snake_case_kwargs() -> None:
    """GeminiConfig takes snake_case kwargs and typed nested models; the dump is camelCase."""
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
        candidate_count=1,
        response_logprobs=True,
        logprobs=3,
        response_modalities=['TEXT'],
        context_cache=True,
        code_execution=True,
        google_search=genai_types.GoogleSearch(exclude_domains=['example.com']),
        url_context=True,
        safety_settings=[
            SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_ONLY_HIGH',
                method='SEVERITY',
            )
        ],
        function_calling_config=FunctionCallingConfig(
            mode='ANY',
            allowed_function_names=['order_dish'],
            stream_function_call_arguments=True,
        ),
        file_search=FileSearchConfig(
            file_search_store_names=['fileSearchStores/menu'],
            metadata_filter='cuisine=thai',
            top_k=3,
        ),
        thinking_config=ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024,
            thinking_level='HIGH',
        ),
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
        'candidateCount': 1,
        'responseLogprobs': True,
        'logprobs': 3,
        'responseModalities': ['TEXT'],
        'contextCache': True,
        'codeExecution': True,
        'googleSearch': {'excludeDomains': ['example.com']},
        'urlContext': True,
        'safetySettings': [
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH', 'method': 'SEVERITY'}
        ],
        'functionCallingConfig': {
            'mode': 'ANY',
            'allowedFunctionNames': ['order_dish'],
            'streamFunctionCallArguments': True,
        },
        'fileSearch': {
            'fileSearchStoreNames': ['fileSearchStores/menu'],
            'metadataFilter': 'cuisine=thai',
            'topK': 3,
        },
        'thinkingConfig': {'includeThoughts': True, 'thinkingBudget': 1024, 'thinkingLevel': 'HIGH'},
        'extra': {'labels': {'team': 'kitchen'}},
    }
    # Same instance from camelCase wire JSON.
    assert GeminiConfig.model_validate(_wire(config)) == config


def test_gemini_tts_config_nested_speech() -> None:
    """GeminiTtsConfig nests SpeechConfig -> VoiceConfig -> PrebuiltVoiceConfig by field name."""
    kore = VoiceConfig(prebuilt_voice_config=PrebuiltVoiceConfig(voice_name='Kore'))
    config = GeminiTtsConfig(
        temperature=0.3,
        speech_config=SpeechConfig(
            language_code='en-US',
            multi_speaker_voice_config=MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    SpeakerVoiceConfig(speaker='Chef', voice_config=kore),
                    SpeakerVoiceConfig(
                        speaker='Server',
                        voice_config=VoiceConfig(prebuilt_voice_config=PrebuiltVoiceConfig(voice_name='Puck')),
                    ),
                ]
            ),
        ),
    )

    assert _wire(config) == {
        'temperature': 0.3,
        'speechConfig': {
            'languageCode': 'en-US',
            'multiSpeakerVoiceConfig': {
                'speakerVoiceConfigs': [
                    {'speaker': 'Chef', 'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Kore'}}},
                    {'speaker': 'Server', 'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Puck'}}},
                ]
            },
        },
    }
    single = GeminiTtsConfig(speech_config=SpeechConfig(voice_config=kore))
    assert _wire(single) == {'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Kore'}}}}


def test_gemini_image_config_nested_image() -> None:
    """GeminiImageConfig nests ImageConfig by field name."""
    config = GeminiImageConfig(
        response_modalities=['TEXT', 'IMAGE'],
        image_config=ImageConfig(
            aspect_ratio='16:9',
            image_size='2K',
            output_mime_type='image/png',
            output_compression_quality=80,
            person_generation='ALLOW_ADULT',
            prominent_people=genai_types.ProminentPeople.BLOCK_PROMINENT_PEOPLE,
            image_output_options=genai_types.ImageConfigImageOutputOptions(mime_type='image/jpeg'),
        ),
    )

    assert _wire(config) == {
        'responseModalities': ['TEXT', 'IMAGE'],
        'imageConfig': {
            'aspectRatio': '16:9',
            'imageSize': '2K',
            'outputMimeType': 'image/png',
            'outputCompressionQuality': 80,
            'personGeneration': 'ALLOW_ADULT',
            'prominentPeople': 'BLOCK_PROMINENT_PEOPLE',
            'imageOutputOptions': {'mimeType': 'image/jpeg'},
        },
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
        compression_quality=genai_types.VideoCompressionQuality.OPTIMIZED,
        resize_mode=genai_types.ImageResizeMode.PAD,
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


def test_deep_research_config_nested_kwargs() -> None:
    """DeepResearchConfig nests FileSearchConfig and McpServerConfig by field name."""
    config = DeepResearchConfig(
        thinking_summaries='auto',
        previous_interaction_id='int-1',
        collaborative_planning=True,
        google_search=True,
        file_search=FileSearchConfig(file_search_store_names=['fileSearchStores/menu']),
        mcp_servers=[McpServerConfig(name='crm', url='https://crm.example/mcp', allowed_tools=['lookup'])],
    )

    assert _wire(config) == {
        'thinkingSummaries': 'auto',
        'previousInteractionId': 'int-1',
        'collaborativePlanning': True,
        'googleSearch': True,
        'fileSearch': {'fileSearchStoreNames': ['fileSearchStores/menu']},
        'mcpServers': [{'name': 'crm', 'url': 'https://crm.example/mcp', 'allowedTools': ['lookup']}],
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
async def test_typed_nested_config_request_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typed nested GeminiConfig reaches the generateContent body.

    google-genai camelCases the fields it types and forwards the nested
    dicts Genkit hands it as-is, so keys inside thinkingConfig and fileSearch
    go out snake_case. The API accepts both spellings.
    """
    # 1. Route google-genai's HTTP calls to a fake API
    bodies: list[dict[str, Any]] = []
    monkeypatch.setattr(httpx.AsyncClient, 'send', _fake_gemini_api(bodies))
    ai = Genkit(plugins=[GoogleAI(api_key='fake-key')])

    # 2. Generate with a config built from typed nested models
    config = GeminiConfig(
        temperature=0.4,
        max_output_tokens=500,
        safety_settings=[
            SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_ONLY_HIGH',
            )
        ],
        thinking_config=ThinkingConfig(thinking_budget=1024, thinking_level='HIGH'),
        function_calling_config=FunctionCallingConfig(mode='AUTO'),
        file_search=FileSearchConfig(file_search_store_names=['fileSearchStores/menu']),
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
                'thinkingConfig': {'thinking_budget': 1024, 'thinking_level': 'HIGH'},
            },
            'safetySettings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH'}],
            'toolConfig': {'functionCallingConfig': {'mode': 'AUTO'}},
            'tools': [{'fileSearch': {'file_search_store_names': ['fileSearchStores/menu']}}],
        }
    ]


def test_choice_fields_are_closed_literals() -> None:
    """Choice fields take plain strings from a closed set.

    assert_type makes pyright, pyrefly and ty check the declared set, so a
    typed ThinkingConfig(thinking_level='ULTRA') fails all three. Runtime
    rejects the same value from JSON.
    """
    # 1. Static: the field type is the closed Literal, not str
    assert_type(ThinkingConfig().thinking_level, Literal['MINIMAL', 'LOW', 'MEDIUM', 'HIGH'] | None)
    assert_type(FunctionCallingConfig().mode, FunctionCallingMode | None)
    assert_type(ImageConfig().aspect_ratio, ImageAspectRatio | None)
    assert_type(ImageConfig().image_size, ImageSize | None)
    safety = SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE')
    assert_type(safety.category, HarmCategory)
    assert_type(safety.threshold, HarmBlockThreshold)
    assert_type(safety.method, HarmBlockMethod | None)

    # 2. Runtime: an unknown value fails validation
    with pytest.raises(ValidationError, match='thinking_level|thinkingLevel'):
        ThinkingConfig.model_validate({'thinkingLevel': 'ULTRA'})


def test_exported_literal_aliases_annotate_user_code() -> None:
    """The root Literal aliases type a caller's own helpers; values stay plain strings."""

    # 1. A menu helper typed with the exported aliases
    def plan_config(level: ThinkingLevel, mode: FunctionCallingMode) -> GeminiConfig:
        return GeminiConfig(
            thinking_config=ThinkingConfig(thinking_level=level),
            function_calling_config=FunctionCallingConfig(mode=mode),
        )

    # 2. Call it with plain strings
    config = plan_config('LOW', 'AUTO')

    # 3. The dump carries the same strings
    assert _wire(config) == {'thinkingConfig': {'thinkingLevel': 'LOW'}, 'functionCallingConfig': {'mode': 'AUTO'}}


def test_deep_research_file_search_uses_the_gemini_class() -> None:
    """Deep Research shares FileSearchConfig: store names are optional and unknown keys raise."""
    # 1. Store names are no longer required
    assert DeepResearchConfig(file_search=FileSearchConfig(top_k=5)).file_search == FileSearchConfig(top_k=5)

    # 2. A typo fails instead of riding to the Interactions API
    with pytest.raises(ValidationError, match='top_kk'):
        DeepResearchConfig.model_validate({'file_search': {'file_search_store_names': ['s'], 'top_kk': 5}})


@pytest.mark.parametrize(
    ('file_search', 'tool'),
    [
        (
            FileSearchConfig(file_search_store_names=['fileSearchStores/menu']),
            {'type': 'file_search', 'file_search_store_names': ['fileSearchStores/menu']},
        ),
        (
            FileSearchConfig(
                file_search_store_names=['fileSearchStores/menu'], top_k=5, metadata_filter='cuisine=thai'
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
    monkeypatch: pytest.MonkeyPatch, file_search: FileSearchConfig, tool: dict[str, Any]
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
        config=DeepResearchConfig(file_search=file_search),
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
