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

"""Vertex AI Imagen - generate an image from a prompt."""

import os

from genkit import Genkit, ModelResponse
from genkit.plugins.google_genai import VertexAI

if 'GCLOUD_PROJECT' not in os.environ:
    if 'GOOGLE_CLOUD_PROJECT' in os.environ:
        os.environ['GCLOUD_PROJECT'] = os.environ['GOOGLE_CLOUD_PROJECT']
    else:
        os.environ['GCLOUD_PROJECT'] = input('Please enter your GCLOUD_PROJECT_ID: ')

ai = Genkit(plugins=[VertexAI()])


@ai.flow()
async def draw_image_with_imagen() -> ModelResponse:
    """Draw an image using Imagen model.

    Returns:
        The image.
    """
    config = {
        'number_of_images': 1,
        'language': 'en',
        'seed': 20,
        'add_watermark': False,
    }

    # pyrefly: ignore[no-matching-overload] - config dict is compatible with dict[str, object]
    return await ai.generate(
        prompt='Draw a cat in a hat',
        model='vertexai/imagen-3.0-generate-002',
        # optional config; check README for available fields
        config=config,
    )


async def main() -> None:
    """Run the Imagen sample once."""
    try:
        response = await draw_image_with_imagen()
        print(response.model_dump_json(indent=2))  # noqa: T201
    except Exception as error:
        message = 'Set GOOGLE_CLOUD_PROJECT and Application Default Credentials before running this sample directly.'
        print(f'{message}\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
