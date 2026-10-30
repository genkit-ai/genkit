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

"""Amazon Bedrock plugin for Genkit.

Registers Bedrock-hosted models (Anthropic Claude, Amazon Nova, Meta Llama,
Mistral, Cohere, and others) as Genkit model actions. Text generation uses the
Bedrock Converse and ConverseStream APIs; embedders and image generation use
InvokeModel.
"""

from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import BaseModel

from genkit import GenkitError, ModelResponse, get_logger
from genkit.embedder import Embedder, EmbedderRef, EmbedRequest, EmbedResponse
from genkit.model import Model, ModelContext, ModelRef, ModelRequest
from genkit.plugin_api import (
    Action,
    ActionKind,
    Plugin,
)
from genkit_amazon_bedrock._config import (
    DEFAULT_TOTAL_TIMEOUT,
    BedrockConfig,
    ImageConfig,
    NovaCanvasConfig,
    StabilityImageConfig,
    TitanImageConfig,
)
from genkit_amazon_bedrock._embedders import (
    BedrockEmbedder,
    KnownBedrockEmbedder,
    get_embedder_info,
    is_embedding_model,
    looks_like_embedding_model,
)
from genkit_amazon_bedrock._image import (
    BedrockImageModel,
    ImageFamily,
    image_config_class,
    image_family,
    is_image_model,
)
from genkit_amazon_bedrock._model_info import (
    KnownBedrockText,
    KnownNovaCanvas,
    KnownStabilityImage,
    KnownTitanImage,
    get_model_info,
    is_rerank_model,
)
from genkit_amazon_bedrock._models import BedrockModel
from genkit_amazon_bedrock._transport import BedrockTransport

if TYPE_CHECKING:
    import boto3.session

logger = get_logger(__name__)

BEDROCK_PLUGIN_NAME = 'bedrock'


def _action_name(model_id: str) -> str:
    return f'{BEDROCK_PLUGIN_NAME}/{model_id}'


def _model_type(model_id: str) -> Literal['chat', 'image']:
    """Routes a model ID: image families to InvokeModel, the rest to Converse.

    Resolve is lazy, so an ID assumed to be chat would send
    ``amazon.nova-canvas-v1:0`` down the Converse path and fail only at call
    time. Embedders classify by ID the same way.
    """
    return 'image' if is_image_model(model_id) else 'chat'


def _config_schema(model_id: str) -> type[BedrockConfig] | type[ImageConfig]:
    """The config class for a model; resolve and list both read it so the Dev UI matches the action.

    Image models get their family's class (Titan Image, Nova Canvas,
    Stability), everything else ``BedrockConfig``.
    """
    return image_config_class(model_id) or BedrockConfig


ConfigT = TypeVar('ConfigT', bound=BaseModel)

RefFamily = Literal['text', 'embedding'] | ImageFamily
"""What a ref constructor serves, in AWS output-modality terms: text (Converse), one image family, or embedding."""

_REF_PREFIX = f'{BEDROCK_PLUGIN_NAME}/'

# Each family's constructor, and how an error names the family.
_REF_CONSTRUCTOR: dict[RefFamily, tuple[str, str]] = {
    'text': ('text', 'a text'),
    'embedding': ('embedding', 'an embedding'),
    'titan_image': ('titan_image', 'a Titan Image'),
    'nova_canvas': ('nova_canvas', 'a Nova Canvas'),
    'stability': ('stability', 'a Stability'),
}


def _strip_ref_prefix(name: str) -> str:
    """Peels ``bedrock/`` so a pasted action name is not stamped twice."""
    local = name
    while local.startswith(_REF_PREFIX):
        local = local.removeprefix(_REF_PREFIX)
    return local


def _ref_route(local: str) -> RefFamily | Literal['rerank']:
    """Routes a ref ID in ``resolve``'s order: embedder, rerank, image family, then text."""
    if looks_like_embedding_model(local):
        return 'embedding'
    if is_rerank_model(local):
        return 'rerank'
    return image_family(local) or 'text'


def _ref_local_id(name: object, *, family: RefFamily) -> str:
    """Checks ``name`` routes to ``family`` and returns it without ``bedrock/``.

    Uses the routing ``resolve`` uses, so a ref never names an action of
    another kind or claims a config class that action wouldn't register.
    """
    method = _REF_CONSTRUCTOR[family][0]
    # str(None) is 'None', and text() takes unknown IDs, so a non-string would
    # mint a real-looking ref instead of failing here.
    if not isinstance(name, str):
        raise GenkitError(status='INVALID_ARGUMENT', message=f'Bedrock.{method}: model name must be a string.')
    local = _strip_ref_prefix(name)
    if not local:
        raise GenkitError(status='INVALID_ARGUMENT', message=f'Bedrock.{method}: model name is required.')
    actual = _ref_route(local)
    if actual == family:
        return local
    if actual == 'rerank':
        hint = f"'{local}' is a rerank model; this plugin has no rerank action."
    elif actual == 'text' and family != 'embedding':
        # Image editing services and unknown IDs land here too: they route to
        # Converse, so text() is the only constructor that fits.
        hint = f"'{local}' is not {_REF_CONSTRUCTOR[family][1]} text-to-image model; text models take Bedrock.text()."
    else:
        constructor, label = _REF_CONSTRUCTOR[actual]
        hint = f"'{local}' is {label} model; use Bedrock.{constructor}()."
    raise GenkitError(status='INVALID_ARGUMENT', message=f'Bedrock.{method}: {hint}')


def _model_ref(
    name: str, *, family: RefFamily, config_schema: type[ConfigT], config: ConfigT | None
) -> ModelRef[ConfigT]:
    local = _ref_local_id(name, family=family)
    return ModelRef(name=f'{_REF_PREFIX}{local}', config_schema=config_schema, config=config)


def _require_id_list(arg: str, value: list[str] | None) -> list[str]:
    # A missing bracket (models='amazon.nova-lite-v1:0') would otherwise
    # iterate the string and list one action per character.
    if isinstance(value, str):
        raise TypeError(f'{arg}= takes a list of Bedrock model IDs, got a str. Did you mean {arg}=[{value!r}]?')
    return list(value or [])


class Bedrock(Plugin):
    """Amazon Bedrock plugin for Genkit."""

    name = BEDROCK_PLUGIN_NAME

    @classmethod
    def text(cls, name: KnownBedrockText | str, *, config: BedrockConfig | None = None) -> ModelRef[BedrockConfig]:
        """Typed ref for a text model, e.g. ``Bedrock.text('amazon.nova-lite-v1:0')``.

        "Text" is Bedrock's TEXT output modality (``ListFoundationModels``
        ``byOutputModality``), served over Converse. It includes models that
        take image or document input, like Claude and Nova Pro. Unknown IDs
        are allowed, as ``resolve`` allows them, so a new model works before
        this plugin lists it. Cross-region inference-profile IDs and ARNs are
        taken as-is. Image, embedding and rerank IDs raise INVALID_ARGUMENT
        naming the constructor to use.

        Example:
            ```python
            from genkit_amazon_bedrock import Bedrock, BedrockConfig

            # 1. Bind the model and a default config once
            claude = Bedrock.text(
                'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                config=BedrockConfig(max_output_tokens=256),
            )

            # 2. Use it like a model name
            response = await ai.generate(model=claude, prompt='Pair a wine with mushroom risotto.')
            print(response.text)
            # => A dry white such as a Pinot Grigio ...
            ```
        """
        return _model_ref(name, family='text', config_schema=BedrockConfig, config=config)

    @classmethod
    def nova_canvas(
        cls, name: KnownNovaCanvas | str, *, config: NovaCanvasConfig | None = None
    ) -> ModelRef[NovaCanvasConfig]:
        """Typed ref for Amazon Nova Canvas, e.g. ``Bedrock.nova_canvas('amazon.nova-canvas-v1:0')``.

        Takes any ID that routes to Nova Canvas, including ``us.`` profiles and
        ARNs ending in a Nova Canvas model ID. Other IDs raise
        INVALID_ARGUMENT naming the constructor to use.

        Example:
            ```python
            from genkit_amazon_bedrock import Bedrock, NovaCanvasConfig

            # 1. Typed ref; config is checked against NovaCanvasConfig
            canvas = Bedrock.nova_canvas('amazon.nova-canvas-v1:0')

            # 2. Generate
            response = await ai.generate(
                model=canvas,
                prompt='A dim sum spread on a marble table.',
                config=NovaCanvasConfig(width=512, height=512, quality='premium'),
            )
            print(response.media[0].content_type)
            # => image/png
            ```
        """
        return _model_ref(name, family='nova_canvas', config_schema=NovaCanvasConfig, config=config)

    @classmethod
    def titan_image(
        cls, name: KnownTitanImage | str, *, config: TitanImageConfig | None = None
    ) -> ModelRef[TitanImageConfig]:
        """Typed ref for Amazon Titan Image Generator (v1, v2).

        Takes any ID that routes to Titan Image. Other IDs raise
        INVALID_ARGUMENT naming the constructor to use.

        Example:
            ```python
            from genkit_amazon_bedrock import Bedrock, TitanImageConfig

            # 1. Ref with a default seed for reproducible menus
            titan = Bedrock.titan_image('amazon.titan-image-generator-v2:0', config=TitanImageConfig(seed=7))

            # 2. Generate two images
            response = await ai.generate(
                model=titan,
                prompt='A bowl of ramen on a cedar counter.',
                config={'number_of_images': 2},
            )
            print(len(response.media))
            # => 2
            ```
        """
        return _model_ref(name, family='titan_image', config_schema=TitanImageConfig, config=config)

    @classmethod
    def stability(
        cls, name: KnownStabilityImage | str, *, config: StabilityImageConfig | None = None
    ) -> ModelRef[StabilityImageConfig]:
        """Typed ref for Stable Diffusion 3.5, Stable Image Core or Stable Image Ultra.

        Takes any ID that routes to the Stability text-to-image family. The
        Stability editing services (inpaint, erase, ...) are not text-to-image
        and raise, as do IDs of other families.

        Example:
            ```python
            from genkit_amazon_bedrock import Bedrock, StabilityImageConfig

            # 1. Typed ref
            sd = Bedrock.stability('stability.sd3-5-large-v1:0')

            # 2. Widescreen jpeg
            response = await ai.generate(
                model=sd,
                prompt='A food truck at dusk, string lights.',
                config=StabilityImageConfig(aspect_ratio='16:9', output_format='jpeg'),
            )
            print(response.media[0].content_type)
            # => image/jpeg
            ```
        """
        return _model_ref(name, family='stability', config_schema=StabilityImageConfig, config=config)

    @classmethod
    def embedding(cls, name: KnownBedrockEmbedder | str) -> EmbedderRef:
        """EmbedderRef for ``ai.embed()``, e.g. ``Bedrock.embedding('amazon.titan-embed-text-v2:0')``.

        Returns an EmbedderRef, not a ModelRef: an embedding ID must never
        reach ``generate(model=...)``, and the text and image constructors
        refuse one. Takes no ``config``: Bedrock embedders ignore per-request
        options today, so a config here would be dropped silently.

        Example:
            ```python
            from genkit_amazon_bedrock import Bedrock

            # 1. Typed embedder ref
            titan = Bedrock.embedding('amazon.titan-embed-text-v2:0')

            # 2. Embed a menu item
            embeddings = await ai.embed(embedder=titan, content='Smoked salmon tartine')
            print(len(embeddings[0].embedding))
            # => 1024
            ```
        """
        local = _ref_local_id(name, family='embedding')
        return EmbedderRef(name=f'{_REF_PREFIX}{local}')

    def __init__(
        self,
        *,
        region: str | None = None,
        max_retries: int | None = None,
        read_timeout: float | None = None,
        connect_timeout: float | None = None,
        max_pool_connections: int | None = None,
        total_timeout: float | None = DEFAULT_TOTAL_TIMEOUT,
        session: 'boto3.session.Session | None' = None,
        models: list[str] | None = None,
        embedders: list[str] | None = None,
    ) -> None:
        """Initializes the Bedrock plugin.

        The AWS client knobs all default to None, meaning "use the ambient AWS
        configuration" (``AWS_MAX_ATTEMPTS``, ``AWS_RETRY_MODE``,
        ``~/.aws/config``, a session's default client config), and fall back to
        the package defaults only when that configuration says nothing.

        Args:
            region: AWS region. Defaults to the SDK resolution chain
                (``AWS_REGION``, ``AWS_DEFAULT_REGION``, ``~/.aws/config``);
                initialization fails loudly when no region resolves rather
                than silently picking one.
            max_retries: Retry limit for Bedrock API calls.
            read_timeout: Socket read timeout in seconds. Resets on every byte
                received, so it caps silence, not the call.
            connect_timeout: Socket connect timeout in seconds.
            max_pool_connections: HTTP connection pool size.
            total_timeout: Whole-call deadline in seconds, covering retries and
                the slow-dribble case the read timeout cannot see. None removes
                the deadline, leaving only the socket timeouts.
            session: Optional pre-configured ``boto3.session.Session`` for custom
                credentials or advanced SDK wiring.
            models: Bedrock model IDs to list in the Dev UI, e.g.
                ``us.anthropic.claude-sonnet-4-5-20250929-v1:0``. The route
                comes from the ID: image-generation families go through
                InvokeModel, everything else through Converse. Unlisted IDs
                still resolve dynamically.
            embedders: Bedrock embedding model IDs to register, e.g.
                ``amazon.titan-embed-text-v2:0``. As with models, unlisted IDs
                still resolve dynamically.
        """
        self.region = region
        self.max_retries = max_retries
        self.read_timeout = read_timeout
        self.connect_timeout = connect_timeout
        self.max_pool_connections = max_pool_connections
        self.total_timeout = total_timeout
        self._session = session
        self.models = _require_id_list('models', models)
        self.embedders = _require_id_list('embedders', embedders)
        self._transport = BedrockTransport(
            region=region,
            max_retries=max_retries,
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
            max_pool_connections=max_pool_connections,
            total_timeout=total_timeout,
            session=session,
        )

    async def init(self) -> list[Action]:
        """Initialize plugin.

        Builds the shared client so misconfiguration (e.g. no resolvable AWS
        region) fails at startup instead of on the first model call.

        Returns:
            Empty list (actions are lazily created via ``resolve``).
        """
        await self._transport.ensure_client()
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by its Bedrock model id.

        Any model ID resolves. Nothing is discovered: listing the catalogue
        needs a second, control-plane ``bedrock`` client and the IAM actions
        that go with it, and this plugin opens only ``bedrock-runtime``.

        Args:
            action_type: The kind of action to resolve.
            name: The Bedrock model id without the ``bedrock/`` prefix
                (``amazon.nova-lite-v1:0``, or a full inference-profile ARN).

        Returns:
            Action object if resolvable, None otherwise.
        """
        if action_type == ActionKind.EMBEDDER:
            # An unroutable embedding ID still resolves, so embed() can name it
            # as an unsupported embedder instead of the registry saying 404.
            if not looks_like_embedding_model(name):
                logger.debug('Bedrock resolve declined', model=name, kind='embedder', reason='not_an_embedder')
                return None
            return self._create_embedder_action(name)
        if action_type != ActionKind.MODEL:
            logger.debug('Bedrock resolve declined', model=name, kind=str(action_type), reason='unsupported_kind')
            return None
        if looks_like_embedding_model(name):
            # Embedding models speak InvokeModel, not Converse; resolving one
            # as a chat model only defers the failure to call time.
            logger.debug('Bedrock resolve declined', model=name, kind='model', reason='embedding_model')
            return None
        if is_rerank_model(name):
            # Same story for rerank models; the plugin has no rerank action.
            logger.debug('Bedrock resolve declined', model=name, kind='model', reason='rerank_model')
            return None
        model_type = _model_type(name)
        logger.debug('Bedrock model resolved', model=name, model_type=model_type, listed=name in self.models)
        return self._create_model_action(name, model_type)

    def _create_model_action(self, model_id: str, model_type: Literal['chat', 'image'] = 'chat') -> Action:
        model_info = get_model_info(model_id, model_type)
        config_schema = _config_schema(model_id)

        async def _generate(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
            if model_type == 'image':
                image_model = BedrockImageModel(model_id=model_id, transport=self._transport)
                return await image_model.generate(request, ctx)
            model = BedrockModel(model_id=model_id, transport=self._transport)
            return await model.generate(request, ctx)

        return Model(
            _action_name(model_id),
            _generate,
            config_schema=config_schema,
            metadata={
                'model': {
                    'label': model_info.label,
                    'stage': model_info.stage.value if model_info.stage else None,
                    'supports': (
                        model_info.supports.model_dump(by_alias=True, exclude_none=True) if model_info.supports else {}
                    ),
                },
            },
        )

    def _create_embedder_action(self, model_id: str) -> Action:
        async def _embed(request: EmbedRequest) -> EmbedResponse:
            embedder_client = BedrockEmbedder(model_id=model_id, transport=self._transport)
            return await embedder_client.embed(request)

        return Embedder(
            _action_name(model_id),
            _embed,
            info=get_embedder_info(model_id),
        )

    async def list_actions(self) -> list[Action]:
        """List configured Bedrock models and embedders.

        Only explicitly configured entries are listed, and only those this
        plugin can actually serve: an embedding or rerank ID in ``models``, or
        a chat ID in ``embedders``, would otherwise be advertised and then fail
        on use. Each model is listed with the same route ``resolve`` picks, so
        an image model carries the image config schema. A bare ``Bedrock()``
        therefore lists nothing; see ``resolve`` for why the catalogue is not
        read.

        Returns:
            Action for each configured model and embedder.
        """
        actions: list[Action] = []
        for model_id in self.models:
            if looks_like_embedding_model(model_id) or is_rerank_model(model_id):
                continue
            model_type = _model_type(model_id)
            actions.append(self._create_model_action(model_id, model_type))
        models = len(actions)
        actions.extend(
            self._create_embedder_action(model_id) for model_id in self.embedders if is_embedding_model(model_id)
        )
        logger.debug(
            'Bedrock actions listed',
            models=models,
            embedders=len(actions) - models,
            models_configured=len(self.models),
            embedders_configured=len(self.embedders),
        )
        return actions
