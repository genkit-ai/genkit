#!/usr/bin/env python3
# Copyright 2025 Google LLC
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

"""Vertex AI Imagen - generate an image from a prompt. Requires GCLOUD_PROJECT and GCLOUD_LOCATION."""

from __future__ import annotations

import os

from genkit import Genkit
from genkit.plugins.google_genai import VertexAI

# 1. Initialize Genkit with Vertex AI plugin (authenticates via Application Default Credentials)
if 'GCLOUD_PROJECT' not in os.environ and 'GOOGLE_CLOUD_PROJECT' in os.environ:
    os.environ['GCLOUD_PROJECT'] = os.environ['GOOGLE_CLOUD_PROJECT']

ai = Genkit(plugins=[VertexAI()])


async def main() -> None:
    """Run Imagen generation directly without intermediate flow wrappers."""
    try:
        # 2. Generate an image using Vertex AI Imagen 3 (`number_of_images=1`)
        response = await ai.generate(
            prompt='Draw a watercolor cat wearing a top hat',
            model='vertexai/imagen-3.0-generate-002',
            config={'number_of_images': 1, 'aspect_ratio': '1:1', 'add_watermark': False},
        )
        print(response.model_dump_json(indent=2))
        # => ModelResponse containing the generated image part:
        # => message.content[0].media.url = "data:image/png;base64,iVBORw0KGgo..."
    except Exception as error:
        message = 'Set GOOGLE_CLOUD_PROJECT and Application Default Credentials before running this sample directly.'
        print(f'{message}\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
