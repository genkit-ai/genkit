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

"""Built-in conformance test cases.

Each entry maps a capability name (e.g. ``tool-request``) to a test case
dict with ``name``, ``input``, ``validators``, and optional ``stream``.

These are ported 1:1 from the JS canonical source.  When adding or modifying
test cases, update BOTH files:

    JS:  genkit-tools/cli/src/commands/dev-test-model.ts — TEST_CASES
    Py:  py/tools/conform/src/conform/util_test_cases.py   (this file)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Base64-encoded PNG of the Genkit logo (used for image input tests).
# This is the same image used in the JS canonical source (20,620 chars).
# Stored in a separate file to keep this module readable.
_IMAGE_BASE64 = (Path(__file__).parent / 'genkit_logo.b64').read_text().strip()

# Type alias for test case dicts.
TestCase = dict[str, Any]

# Built-in test cases, keyed by capability name.
# These can be referenced in model-conformance.yaml via the ``supports`` list.
TEST_CASES: dict[str, TestCase] = {
    'tool-request': {
        'name': 'Tool Request Conformance',
        'input': {
            'messages': [
                {
                    'role': 'user',
                    'content': [{'text': 'What is the weather in New York? Use the tool.'}],
                },
            ],
            'tools': [
                {
                    'name': 'weather',
                    'description': 'Get the weather for a city',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {'city': {'type': 'string'}},
                        'required': ['city'],
                    },
                },
            ],
        },
        'validators': ['has-tool-request:weather'],
    },
    'structured-output': {
        'name': 'Structured Output Conformance',
        'input': {
            'messages': [
                {
                    'role': 'user',
                    'content': [{'text': 'Generate a profile for John Doe.'}],
                },
            ],
            'output': {
                'format': 'json',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'age': {'type': 'number'},
                    },
                    'required': ['name', 'age'],
                },
                'constrained': True,
            },
        },
        'validators': ['valid-json'],
    },
    'multiturn': {
        'name': 'Multiturn Conformance',
        'input': {
            'messages': [
                {'role': 'user', 'content': [{'text': 'My name is Genkit.'}]},
                {'role': 'model', 'content': [{'text': 'Hello Genkit.'}]},
                {'role': 'user', 'content': [{'text': 'What is my name?'}]},
            ],
        },
        'validators': ['text-includes:Genkit'],
    },
    'streaming-multiturn': {
        'name': 'Multiturn Conformance with streaming',
        'stream': True,
        'input': {
            'messages': [
                {'role': 'user', 'content': [{'text': 'My name is Genkit.'}]},
                {'role': 'model', 'content': [{'text': 'Hello Genkit.'}]},
                {'role': 'user', 'content': [{'text': 'What is my name?'}]},
            ],
        },
        'validators': ['stream-text-includes:Genkit'],
    },
    'streaming-tool-request': {
        'name': 'Tool Request Conformance with streaming',
        'stream': True,
        'input': {
            'messages': [
                {
                    'role': 'user',
                    'content': [{'text': 'What is the weather in New York? Use the weather tool'}],
                },
            ],
            'tools': [
                {
                    'name': 'weather',
                    'description': 'Get the weather for a city',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {'city': {'type': 'string'}},
                        'required': ['city'],
                    },
                },
            ],
        },
        'validators': ['stream-has-tool-request:weather'],
    },
    'streaming-structured-output': {
        'name': 'Structured Output Conformance with streaming',
        'stream': True,
        'input': {
            'messages': [
                {
                    'role': 'user',
                    'content': [{'text': 'Generate a movie review for John Wick'}],
                },
            ],
            'output': {
                'format': 'json',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'rating': {'type': 'number'},
                    },
                    'required': ['name', 'rating'],
                },
                'constrained': True,
            },
        },
        'validators': ['stream-valid-json'],
    },
    'system-role': {
        'name': 'System Role Conformance',
        'input': {
            'messages': [
                {
                    'role': 'system',
                    'content': [
                        {
                            'text': (
                                'IMPORTANT: your response are machine processed, always '
                                "start/prefix your response with 'RESPONSE:', "
                                "ex: 'RESPONSE: hello'"
                            ),
                        },
                    ],
                },
                {'role': 'user', 'content': [{'text': 'hello'}]},
            ],
        },
        'validators': ['text-starts-with:RESPONSE:'],
    },
    'input-image-base64': {
        'name': 'Image Input (Base64) Conformance',
        'input': {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'What text do you see in this image?'},
                        {
                            'media': {
                                'url': f'data:image/png;base64,{_IMAGE_BASE64}',
                                'contentType': 'image/png',
                            },
                        },
                    ],
                },
            ],
        },
        'validators': ['text-includes:genkit'],
    },
    'input-image-url': {
        'name': 'Image Input (URL) Conformance',
        'input': {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'What is this logo?'},
                        {
                            'media': {
                                'url': 'https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png',
                                'contentType': 'image/png',
                            },
                        },
                    ],
                },
            ],
        },
        'validators': ['text-includes:google'],
    },
    'input-video-youtube': {
        'name': 'Video Input (YouTube) Conformance',
        'input': {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Describe this video.'},
                        {
                            'media': {
                                'url': 'https://www.youtube.com/watch?v=3p1P5grjXIQ',
                                'contentType': 'video/mp4',
                            },
                        },
                    ],
                },
            ],
        },
        'validators': ['text-not-empty'],
    },
    'output-audio': {
        'name': 'Audio Output (TTS) Conformance',
        'input': {
            'messages': [{'role': 'user', 'content': [{'text': 'Say hello.'}]}],
        },
        'validators': ['valid-media:audio'],
    },
    'output-image': {
        'name': 'Image Output (Generation) Conformance',
        'input': {
            'messages': [
                {
                    'role': 'user',
                    'content': [{'text': 'Generate an image of a cat.'}],
                },
            ],
        },
        'validators': ['valid-media:image'],
    },
}
