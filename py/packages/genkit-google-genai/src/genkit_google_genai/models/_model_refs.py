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

"""Shared machinery for the typed family ref constructors.

Each plugin class exposes one constructor per config family
(``gemini_model``, ``imagen_model``, ...). The return type is the contract:
``ModelRef[GeminiConfigSchema]`` tells generate-time code exactly which
config is legal, so a constructor must refuse ids whose runtime action
would validate a different schema — otherwise the ref lies and the wrong
config sails through until the API rejects it.
"""

from typing import TypeVar

from pydantic import BaseModel

from genkit import GenkitError
from genkit.embedder import EmbedderRef
from genkit.model import ModelRef, model_ref
from genkit_google_genai.models._routing import classify_family, strip_ref_prefixes

ConfigT = TypeVar('ConfigT', bound=BaseModel)

# Families with a constructor, for "use X instead" error hints.
FAMILY_METHOD = {
    'gemini': 'gemini_model',
    'tts': 'gemini_tts_model',
    'image': 'gemini_image_model',
    'gemma': 'gemma_model',
    'imagen': 'imagen_model',
    'embedder': 'embedding',
}


def wrong_family_error(*, plugin_class: str, method: str, family: str, local: str, actual: str) -> GenkitError:
    """Build the INVALID_ARGUMENT error naming the id and the way out."""
    if actual == 'embedder':
        hint = f"'{local}' is an embedder; use {plugin_class}.embedding()."
    elif actual == 'veo':
        hint = f"'{local}' is a Veo video model; it runs as a background model and has no ref constructor."
    elif actual in ('lyria', 'deep-research', 'antigravity'):
        hint = f"'{local}' has no ref constructor in this plugin."
    elif actual == 'unsupported':
        hint = f"'{local}' is not a supported model."
    elif actual in FAMILY_METHOD:
        hint = f"'{local}' is not a {family} model; use {plugin_class}.{FAMILY_METHOD[actual]}()."
    else:
        hint = f"'{local}' is not a {family} model."
    return GenkitError(status='INVALID_ARGUMENT', message=f'{plugin_class}.{method}: {hint}')


def family_model_ref(
    name: str,
    *,
    namespace: str,
    plugin_class: str,
    family: str,
    method: str,
    config_schema: type[ConfigT],
    config: ConfigT | None,
) -> ModelRef[ConfigT]:
    """Strip, gate against the closed family set, then stamp this plugin's namespace."""
    # str(None) is 'None', and gemini_model allows unknown ids, so a
    # non-string would mint a real-looking ref instead of failing here.
    if not isinstance(name, str):
        raise GenkitError(status='INVALID_ARGUMENT', message=f'{plugin_class}.{method}: model name must be a string.')
    local = strip_ref_prefixes(name)
    if not local:
        raise GenkitError(status='INVALID_ARGUMENT', message=f'{plugin_class}.{method}: model name is required.')
    actual = classify_family(local)
    # Unknown ids stay usable through gemini_model so a brand-new Gemini
    # release works before this plugin learns its name. Every other family
    # method takes only its own ids.
    allowed = actual == family or (family == 'gemini' and actual == 'unknown')
    if not allowed:
        raise wrong_family_error(plugin_class=plugin_class, method=method, family=family, local=local, actual=actual)
    return model_ref(local, config_schema=config_schema, namespace=namespace, config=config)


def family_embedder_ref(
    name: str,
    *,
    namespace: str,
    plugin_class: str,
    config: dict[str, object] | None,
    version: str | None,
) -> EmbedderRef:
    """Strip, gate, and build an EmbedderRef for ai.embed().

    This deliberately returns an EmbedderRef, not a ModelRef: an embedder id
    must never end up in generate(model=...).
    """
    if not isinstance(name, str):
        raise GenkitError(
            status='INVALID_ARGUMENT', message=f'{plugin_class}.embedding: embedder name must be a string.'
        )
    local = strip_ref_prefixes(name)
    if not local:
        raise GenkitError(status='INVALID_ARGUMENT', message=f'{plugin_class}.embedding: embedder name is required.')
    actual = classify_family(local)
    if actual not in ('embedder', 'unknown'):
        raise wrong_family_error(
            plugin_class=plugin_class, method='embedding', family='embedder', local=local, actual=actual
        )
    return EmbedderRef(name=f'{namespace}/{local}', config=config, version=version)
