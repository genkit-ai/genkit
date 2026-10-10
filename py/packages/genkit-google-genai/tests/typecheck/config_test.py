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

"""Typed construction of every exported config class.

This file is the type-checker contract: pyright, pyrefly and ty must accept
snake_case keyword arguments and typed nested values here with no
suppressions. The runtime asserts pin the camelCase wire dump.
"""

from genkit_google_genai import (
    AntigravityConfig,
    DeepResearchConfig,
    GeminiConfig,
    GeminiImageConfig,
    GeminiTtsConfig,
    GemmaConfig,
    LyriaConfig,
    VeoConfig,
)
from genkit_google_genai._models._deep_research import FileSearchConfig as DeepResearchFileSearch, McpServerConfig
from genkit_google_genai._models._gemini import (
    FileSearchConfig,
    FunctionCallingConfig,
    FunctionCallingMode,
    GoogleSearchConfig,
    HarmBlockThreshold,
    HarmCategory,
    ImageAspectRatio,
    ImageConfig,
    ImageOutputOptions,
    ImageSize,
    MultiSpeakerVoiceConfig,
    PrebuiltVoiceConfig,
    SafetySettingsSchema,
    SpeakerVoiceConfig,
    SpeechConfig,
    ThinkingConfig,
    ThinkingLevel,
    VoiceConfig,
)
from genkit_google_genai._models._lyria import LyriaConfig as VertexLyriaConfig
from pydantic import BaseModel


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
        VertexLyriaConfig,
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
        ImageOutputOptions,
        GoogleSearchConfig,
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
        google_search=GoogleSearchConfig(exclude_domains=['example.com']),
        url_context=True,
        safety_settings=[
            SafetySettingsSchema(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            )
        ],
        function_calling_config=FunctionCallingConfig(
            mode=FunctionCallingMode.ANY,
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
            thinking_level=ThinkingLevel.HIGH,
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
        'safetySettings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH'}],
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
            aspect_ratio=ImageAspectRatio.RATIO_16_9,
            image_size=ImageSize.SIZE_2K,
            output_mime_type='image/png',
            output_compression_quality=80,
            person_generation='ALLOW_ADULT',
            prominent_people='BLOCK_PROMINENT_PEOPLE',
            image_output_options=ImageOutputOptions(mime_type='image/jpeg'),
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


def test_lyria_configs_snake_case_kwargs() -> None:
    """Both Lyria config classes take snake_case kwargs; the dump is camelCase."""
    interactions = LyriaConfig(
        base_url='https://kitchen.example',
        api_version='v1beta',
        timeout=30000,
        custom_headers={'x-team': 'kitchen'},
        response_modalities=['audio'],
    )
    assert _wire(interactions) == {
        'baseUrl': 'https://kitchen.example',
        'apiVersion': 'v1beta',
        'timeout': 30000,
        'customHeaders': {'x-team': 'kitchen'},
        'responseModalities': ['audio'],
    }

    vertex = VertexLyriaConfig(negative_prompt='drums', seed=1, sample_count=2, location='global')
    assert _wire(vertex) == {'negativePrompt': 'drums', 'seed': 1, 'sampleCount': 2, 'location': 'global'}


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
        file_search=DeepResearchFileSearch(file_search_store_names=['fileSearchStores/menu']),
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
