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

"""Firestore-backed session store for agent snapshots.

Each turn is persisted as a JSON Patch diff from its parent, with periodic
full-state checkpoints. Checkpoint state is split across shard documents so no
single write approaches Firestore's ~1 MiB document limit.

Reads and writes use only document-ID lookups (pointer + snapshot + shards), so
deployments need no secondary indexes and stay strongly consistent.

Default layout (collection ``genkit-sessions``, prefix ``global``)::

    genkit-sessions/{prefix}/snapshots/{snapshotId}
    genkit-sessions-shards/{prefix}/shards/{checkpointId}_{index}
    genkit-sessions-pointers/{prefix}/pointers/{sessionId}
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, Literal

from google.api_core import exceptions as google_exceptions
from google.cloud import firestore
from google.cloud.firestore import (
    AsyncClient,
    AsyncCollectionReference,
    AsyncDocumentReference,
    AsyncTransaction,
    DocumentSnapshot,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from genkit._ai._agents._session import (
    SessionStore,
    SnapshotStatusStream,
    SnapshotSubscriber,
    StateT,
)
from genkit._ai._agents._session_stores._util import (
    TERMINAL_STATUSES,
    SaveFn,
    apply_save,
    iterate_statuses,
    require_one_selector,
    session_id_of,
)
from genkit._ai._json_patch import apply_json_patch, diff_json
from genkit._core._error import GenkitError
from genkit._core._loop_cache import _loop_local_client
from genkit._core._typing import (
    AgentFinishReason,
    GenkitRuntimeError,
    JsonPatchOp,
    JsonPatchOperation,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
)

DEFAULT_COLLECTION = 'genkit-sessions'
DEFAULT_PREFIX = 'global'
logger = logging.getLogger(__name__)
# Favor common chat workloads: enough diffs between checkpoints to keep write
# amplification down, small enough that reconstruct stays cheap.
DEFAULT_CHECKPOINT_INTERVAL = 25
# Kept well under Firestore's 1 MiB/doc limit so a single shard or diff write
# cannot be rejected for size.
DEFAULT_SHARD_SIZE = 512 * 1024

# Ceiling for shard_size: shard documents carry the chunk plus field overhead,
# so cap below Firestore's 1 MiB document limit with headroom.
MAX_SHARD_SIZE = 1_000_000

# Firestore transactions retry internally on contention; after this many
# attempts the failure surfaces as GenkitError(status='ABORTED').
DEFAULT_TRANSACTION_MAX_ATTEMPTS = 5


@dataclass(eq=False)
class _StatusSubscription:
    """One ``on_snapshot_status_change`` call: private queue + Firestore watch."""

    queue: asyncio.Queue[SnapshotStatus | None]
    watch: Any | None = None
    last_status: SnapshotStatus | None = None
    closed: bool = False

    __hash__ = object.__hash__


@dataclass
class _StatusLoopState:
    """Per-event-loop Firestore clients + open status subscriptions."""

    subscriptions: set[_StatusSubscription] = field(default_factory=set)
    client: AsyncClient | None = None
    sync_client: firestore.Client | None = None
    owns_async_client: bool = False
    owns_sync_client: bool = False


class _ShardDoc(BaseModel):
    """Schema for a checkpoint state shard document stored in Firestore."""

    model_config = ConfigDict(extra='ignore')

    chunk: bytes

    @field_validator('chunk', mode='before')
    @classmethod
    def _coerce_chunk(cls, v: object) -> object:
        """Coerce raw Firestore binary chunk representations to bytes.

        google-cloud-firestore's gRPC deserializer returns zero-copy memoryview
        objects for binary blob fields in DocumentSnapshot.to_dict().
        """
        if isinstance(v, memoryview):
            return v.tobytes()
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        if isinstance(v, str):
            return v.encode('utf-8')
        return v


class _SnapshotWriteMeta(BaseModel):
    """Metadata written onto a snapshot doc for later reconstruction."""

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    kind: Literal['diff', 'checkpoint']
    checkpoint_id: str
    checkpoint_shard_count: int = Field(ge=1)
    # [] means checkpoint; a missing field is corrupt, not an empty path.
    segment_path: list[str]
    # Wire form is a JSON string (see ``_SnapshotDoc.state_patch``); this
    # in-memory meta keeps the parsed op list until the doc is built.
    state_patch: list[dict[str, Any]] | None = None


class _ParentChainMeta(BaseModel):
    """Parent snapshot metadata required for diff calculation."""

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    checkpoint_id: str
    checkpoint_shard_count: int = Field(ge=1)
    segment_path: list[str]


class _SnapshotDoc(BaseModel):
    """Schema for turn snapshot document stored in Firestore."""

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    snapshot_id: str
    session_id: str
    parent_id: str | None = None
    created_at: str
    updated_at: str | None = None
    status: SnapshotStatus | None = None
    heartbeat_at: str | None = None
    finish_reason: AgentFinishReason | None = None
    error: GenkitRuntimeError | None = None
    kind: Literal['diff', 'checkpoint']
    checkpoint_id: str
    checkpoint_shard_count: int = Field(ge=1)
    # [] means checkpoint; a missing field is corrupt, not an empty path.
    segment_path: list[str]
    # Stored as a JSON string so deep nested state does not hit Firestore's
    # ~20-level map nesting limit (checkpoints already avoid that via shards).
    # Typed loosely so pre-release array values reach the decode error path.
    state_patch: Any = None

    def to_session_snapshot(self, state_raw: dict[str, Any] | SessionState | None = None) -> SessionSnapshot:
        """Convert Firestore snapshot document and reconstructed state to a SessionSnapshot."""
        state = _state_from_dict(state_raw)
        return SessionSnapshot(
            snapshot_id=self.snapshot_id,
            session_id=self.session_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            heartbeat_at=self.heartbeat_at,
            status=self.status,
            finish_reason=self.finish_reason,
            error=self.error,
            state=state,
        )


class _PointerDoc(BaseModel):
    """Schema for session pointer document stored in Firestore."""

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    current_snapshot_id: str = Field(min_length=1)
    checkpoint_id: str | None = None
    checkpoint_shard_count: int | None = Field(default=None, ge=1)
    segment_path: list[str]


def _status_from_doc(doc_snapshot: DocumentSnapshot) -> SnapshotStatus | None:
    """Extract and validate the snapshot status from a Firestore document."""
    if not doc_snapshot.exists:
        return None
    status_val = (doc_snapshot.to_dict() or {}).get('status')
    if status_val is None:
        return None
    try:
        out: Any = SnapshotStatus(status_val)
    except ValueError:
        logger.warning("Unknown SnapshotStatus '%s' in Firestore document '%s'", status_val, doc_snapshot.id)
        return None
    return out


def _state_to_dict(state: SessionState | None) -> dict[str, Any]:
    """Convert a SessionState object to a dictionary representation."""
    if state is None:
        return {}
    try:
        dumped = state.model_dump(by_alias=True, exclude_none=True, mode='json')
    except Exception as e:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'session state is not storable: {e}',
        ) from e
    return dumped if isinstance(dumped, dict) else {}


def _state_from_dict(data: dict[str, Any] | SessionState | None) -> SessionState | None:
    """Parse a dictionary or SessionState object into a SessionState instance."""
    if data is None:
        return None
    if isinstance(data, SessionState):
        return data
    try:
        return SessionState.model_validate(data)
    except Exception as e:
        raise GenkitError(
            status='DATA_LOSS',
            message=f'FirestoreSessionStore: corrupt session state document: {e}',
        ) from e


def _json_dumps_utf8(value: Any) -> bytes:  # noqa: ANN401
    """Compact UTF-8 JSON bytes, or INVALID_ARGUMENT when the value is not storable."""
    try:
        return json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeEncodeError) as e:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'session state is not storable: {e}',
        ) from e


# google.api_core exception class -> gRPC status name (see _to_genkit_error).
_GOOGLE_EXC_STATUS: dict[str, str] = {
    'Aborted': 'ABORTED',
    'AlreadyExists': 'ALREADY_EXISTS',
    'Cancelled': 'CANCELLED',
    'DataLoss': 'DATA_LOSS',
    'DeadlineExceeded': 'DEADLINE_EXCEEDED',
    'FailedPrecondition': 'FAILED_PRECONDITION',
    'Forbidden': 'PERMISSION_DENIED',
    'InternalServerError': 'INTERNAL',
    'InvalidArgument': 'INVALID_ARGUMENT',
    'NotFound': 'NOT_FOUND',
    'OutOfRange': 'OUT_OF_RANGE',
    'PermissionDenied': 'PERMISSION_DENIED',
    'ResourceExhausted': 'RESOURCE_EXHAUSTED',
    'ServiceUnavailable': 'UNAVAILABLE',
    'TooManyRequests': 'RESOURCE_EXHAUSTED',
    'Unauthenticated': 'UNAUTHENTICATED',
    'Unauthorized': 'UNAUTHENTICATED',
    'Unknown': 'UNKNOWN',
}


def _to_genkit_error(e: google_exceptions.GoogleAPICallError) -> GenkitError:
    """Convert a Google API error into a ``GenkitError``.

    The status maps 1:1 from the exception class (both vocabularies are gRPC
    statuses); the server's message is preserved verbatim, with no diagnosis
    added.
    """
    status = _GOOGLE_EXC_STATUS.get(type(e).__name__, 'UNKNOWN')
    return GenkitError(
        status=status,  # type: ignore[arg-type]
        message=f'FirestoreSessionStore: Firestore rejected the transaction ({type(e).__name__}): {e.message or e}',
    )


class _UserCodeError(Exception):
    """Carrier: the mutator raised; deliver its exception verbatim."""


async def _translate_txn_errors(
    awaitable: Awaitable[SessionSnapshot | None],
) -> SessionSnapshot | None:
    """Await a transactional coroutine, converting backend failures to ``GenkitError``.

    On success this is the coroutine's return value from the attempt that
    committed.

    The store's public contract is that every failure arising from its own
    serialization, storage, or the Google libraries is a ``GenkitError`` with a
    gRPC status. Exceptions from caller-supplied mutation functions propagate
    unchanged (tunneled past library cleanup heuristics that would otherwise
    misclassify them). Two library quirks are recognized narrowly, both
    surfacing as a bare ``ValueError``: retry exhaustion (chained from the
    final ``Aborted``), and a failed rollback on a transaction that never
    obtained an ID, which masks the original error entirely.
    """
    try:
        return await awaitable
    except _UserCodeError as e:
        cause = e.__cause__
        assert cause is not None
        raise cause from None
    except ValueError as e:
        # Private-behavior dependency, deliberately narrow: the firestore
        # library wraps retry exhaustion as a bare ValueError ("Failed to
        # commit transaction in N attempts.") chained from the final Aborted.
        # If a future release changes this shape we fall through to re-raise —
        # degraded (untyped) but never mistranslated.
        if isinstance(e.__cause__, google_exceptions.Aborted):
            raise _to_genkit_error(e.__cause__) from e
        if e.__cause__ is None and 'no transaction ID' in str(e):
            # Second shape: a failure early in the transaction lifecycle
            # (e.g. begin under heavy load) triggers rollback on a
            # transaction with no ID; the library raises this unchained
            # ValueError, destroying the original error. The transaction
            # categorically did not commit, so ABORTED (retryable) is the
            # honest classification even though the root cause is lost.
            raise GenkitError(
                status='ABORTED',
                message=(
                    'FirestoreSessionStore: transaction failed before it could be '
                    f'established and the original error was masked by cleanup: {e}'
                ),
            ) from e
        raise
    except GenkitError:
        raise
    except google_exceptions.RetryError as e:
        cause = e.cause
        if isinstance(cause, google_exceptions.GoogleAPICallError):
            raise _to_genkit_error(cause) from e
        raise GenkitError(
            status='ABORTED',
            message=f'FirestoreSessionStore: Firestore retries exhausted: {e}',
        ) from e
    except google_exceptions.GoogleAPICallError as e:
        raise _to_genkit_error(e) from e


def _validate_doc_id(value: str | None, name: str) -> None:
    """Reject ids that Firestore would treat as path structure, not a document id.

    Ids become single document path segments; a ``/`` re-routes the path (an
    even segment count silently addresses a different, deeper document — a
    watch on it registers and then never fires), and ``.``/``..`` are reserved.
    Leading or trailing whitespace is also rejected so save and get agree on
    what counts as a valid id. Validating up front turns these failure modes
    into a typed error.
    """
    if (
        not value
        or value != value.strip()
        or not value.strip()
        or '/' in value
        or value in ('.', '..')
        or (value.startswith('__') and value.endswith('__'))
    ):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                f'FirestoreSessionStore: invalid {name} {value!r}: '
                "must be a non-empty Firestore document id (no '/', not '.', '..', or "
                '__reserved__, no leading/trailing whitespace).'
            ),
        )


def _patch_to_json(patch: list[JsonPatchOperation]) -> list[dict[str, Any]]:
    r"""Serialize patch ops for storage as valid RFC 6902 documents.

    Deliberately not ``model_dump(exclude_none=True)``: RFC 6902 requires the
    ``value`` member on add/replace/test operations, so an explicit ``null``
    must be stored as ``\"value\": null`` rather than an absent key.
    """
    out: list[dict[str, Any]] = []
    for op in patch:
        d: dict[str, Any] = {'op': op.op.value, 'path': op.path}
        if op.op in (JsonPatchOp.ADD, JsonPatchOp.REPLACE, JsonPatchOp.TEST):
            d['value'] = op.value
        if op.op in (JsonPatchOp.MOVE, JsonPatchOp.COPY):
            d['from'] = op.from_
        out.append(d)
    return out


def _encode_state_patch(patch: list[dict[str, Any]] | None) -> str | None:
    """Serialize a patch op list to the snapshot-doc wire form (JSON string)."""
    if patch is None:
        return None
    return json.dumps(patch, separators=(',', ':'), ensure_ascii=False)


def _state_patch_data_loss(snapshot_id: str, detail: str) -> GenkitError:
    """Build the shared DATA_LOSS error for an unreadable ``statePatch`` field."""
    return GenkitError(
        status='DATA_LOSS',
        message=(
            f"FirestoreSessionStore: snapshot '{snapshot_id}' has an unreadable "
            f"'statePatch' field ({detail}). "
            'Documents written by pre-release versions are not readable; '
            "delete the session's documents and re-create."
        ),
    )


def _decode_state_patch(raw: Any, *, snapshot_id: str) -> list[dict[str, Any]]:  # noqa: ANN401
    r"""Parse the snapshot-doc ``statePatch`` JSON string into op dicts.

    Diff segments always store a JSON string (including ``\"[]\"`` for a
    zero-change turn). ``None`` / empty string / a missing field are
    corruption, not an empty patch.
    """
    if raw is None or raw == '':
        raise _state_patch_data_loss(snapshot_id, 'missing or empty')
    if not isinstance(raw, str):
        raise _state_patch_data_loss(snapshot_id, f'expected a JSON string; found {type(raw).__name__}')
    try:
        parsed = json.loads(raw)
    except Exception as e:
        raise _state_patch_data_loss(snapshot_id, 'unparseable') from e
    if not isinstance(parsed, list):
        raise _state_patch_data_loss(snapshot_id, 'not a patch array')
    return parsed


def _patch_from_json(raw: list[dict[str, Any]] | None) -> list[JsonPatchOperation]:
    """Parse raw JSON Patch dicts into ``JsonPatchOperation`` models."""
    if not raw:
        return []
    return [JsonPatchOperation.model_validate(op) for op in raw]


def _byte_length(value: Any) -> int:  # noqa: ANN401
    """UTF-8 byte size of a compact JSON encoding of ``value``.

    Decides when a diff is too large for one Firestore document and should be
    promoted to a sharded checkpoint. Uses ``ensure_ascii=False`` so multilingual
    text is measured as the same UTF-8 bytes this store actually writes — escaping
    non-ASCII would inflate the size and over-promote diffs into checkpoints.
    """
    return len(_json_dumps_utf8(value))


class FirestoreSessionStore(SessionStore[StateT], SnapshotSubscriber, Generic[StateT]):
    """Durable Firestore store for agent session snapshots.

    Configuration is fixed at construction; to change it, create a new store instance.

    Each turn is saved as a JSON Patch diff from its parent, with periodic
    full-state checkpoints split across shard documents so no single write
    approaches Firestore's ~1 MiB limit. Session lookup is a pointer read plus
    document-ID fetches only — no secondary indexes.

    Uses Application Default Credentials by default, or the emulator when
    ``FIRESTORE_EMULATOR_HOST`` is set. Pass ``snapshot_path_prefix`` (often
    derived from call ``context``, e.g. an authenticated user id) when session
    ids may collide across tenants.

    One store instance is safe to use from more than one asyncio event loop
    (for example your app and the Dev UI). Each
    ``on_snapshot_status_change`` call gets its own Firestore listener, and
    status updates appear after the write commits. If you pass ``client`` /
    ``sync_client``, the first loop that uses them keeps those instances;
    later loops get their own clients when we can create matching ones.

    Costs and limits:
        - **Write contention**: All writes for a session update a single pointer document.
          Rates above ~1 write/sec contend on it; exhausting retries raises a retryable
          ``GenkitError(ABORTED)`` (configurable via ``transaction_max_attempts``).
        - **Diff turn updates**: Updating a diff turn (including heartbeats) re-reads up to
          ``checkpoint_interval`` parent documents to recompute the patch. At scale, these
          re-reads dominate Firestore cost ahead of writes. Lowering ``checkpoint_interval``
          trades cheaper updates for more checkpoint storage.
        - **Checkpoint updates**: Updating an existing checkpoint turn rewrites the snapshot
          and all its shard documents, amplifying heartbeats into multi-document writes.
        - **Status watches**: Watches read only the snapshot document (status field). Each
          active ``on_snapshot_status_change`` call holds its own background watch thread
          until a terminal status, stream ``aclose``, or :meth:`close`. Two subscribers on
          the same snapshot therefore use two watches.
        - **Event buffering**: Status streams buffer unread events in memory without bound if
          the subscriber consumes slower than events arrive.

    Interoperability:
        The Python, JS, and Go stores share the same concepts and storage
        model; each SDK's surface and behavioral guarantees are idiomatic to
        its language and documented per SDK. On-disk document formats are
        maintained independently and may diverge. Use a separate Firestore
        collection per SDK.

    Data lifecycle:
        Deleting a session's documents is safe at any time: lookups report
        the session as absent, and if data is written again the pointer is
        transparently recreated.
    """

    _FROZEN = frozenset({'collection', 'shard_size', 'checkpoint_interval', 'transaction_max_attempts'})

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Block post-init reassignment of construction knobs with a typed error."""
        if name in self._FROZEN and hasattr(self, '_closed'):
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=(
                    f'FirestoreSessionStore: {name!r} is fixed at construction; '
                    'create a new store instance with the desired configuration.'
                ),
            )
        super().__setattr__(name, value)

    def __init__(
        self,
        *,
        client: AsyncClient | None = None,
        sync_client: firestore.Client | None = None,
        collection: str = DEFAULT_COLLECTION,
        snapshot_path_prefix: Callable[[dict[str, Any] | None], str] | None = None,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        shard_size: int = DEFAULT_SHARD_SIZE,
        transaction_max_attempts: int = DEFAULT_TRANSACTION_MAX_ATTEMPTS,
    ) -> None:
        """Create a Firestore-backed session store.

        Realtime status watches need a sync ``Client`` (the async client cannot
        watch documents). Pass ``sync_client`` to own that lifecycle yourself;
        otherwise one is created lazily per event loop to match that loop's
        async client and closed by :meth:`close`.

        ``snapshot_path_prefix`` maps request ``context`` (for example auth) to a
        path segment that isolates tenants when session ids may collide.

        ``transaction_max_attempts`` bounds how many times a contended
        transaction is retried before failing with status ``ABORTED``. The
        default suits typical sessions; raise it for workloads with many
        concurrent writers per session.
        """
        _validate_doc_id(collection, 'collection')
        if checkpoint_interval < 1:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'FirestoreSessionStore: checkpoint_interval must be >= 1, got {checkpoint_interval}.',
            )
        if not 0 < shard_size <= MAX_SHARD_SIZE:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f'FirestoreSessionStore: shard_size must be between 1 and {MAX_SHARD_SIZE} '
                    f'bytes to stay under the Firestore document limit, got {shard_size}.'
                ),
            )
        if transaction_max_attempts < 1:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f'FirestoreSessionStore: transaction_max_attempts must be >= 1, got {transaction_max_attempts}.'
                ),
            )
        self._transaction_max_attempts = transaction_max_attempts
        self._provided_client = client
        self._provided_sync_client = sync_client
        self._provided_client_loop: asyncio.AbstractEventLoop | None = None
        self._provided_sync_loop: asyncio.AbstractEventLoop | None = None
        self._collection = collection
        self._prefix_fn = snapshot_path_prefix or (lambda _context: DEFAULT_PREFIX)
        self._checkpoint_interval = checkpoint_interval
        self._shard_size = shard_size
        # Per running event loop: clients + open per-subscribe watches.
        # Strong keys so we can prune closed loops and tear down their resources.
        self._status_by_loop: dict[asyncio.AbstractEventLoop, _StatusLoopState] = {}
        self._status_by_loop_mu = threading.Lock()
        # Last so post-init frozen-config __setattr__ can key off its presence.
        self._closed = False

    def _status_loop_state(self) -> _StatusLoopState:
        """Clients + open status subscriptions for the running event loop.

        Also tears down watches/clients left behind by closed loops so a
        notebook-style ``asyncio.run`` cell does not leak listeners.
        """
        loop = asyncio.get_running_loop()
        with self._status_by_loop_mu:
            orphan_watches, orphan_states = self._prune_closed_status_loops_locked()
            state = self._status_by_loop.get(loop)
            if state is None:
                state = _StatusLoopState()
                self._status_by_loop[loop] = state
        for watch in orphan_watches:
            with contextlib.suppress(Exception):
                watch.unsubscribe()
        for dead in orphan_states:
            self._release_loop_clients(dead)
        return state

    def _prune_closed_status_loops_locked(self) -> tuple[list[Any], list[_StatusLoopState]]:
        """Drop bags whose event loop is closed. Caller holds mu."""
        dead = [lp for lp in self._status_by_loop if lp.is_closed()]
        orphan_watches: list[Any] = []
        orphan_states: list[_StatusLoopState] = []
        for lp in dead:
            state = self._status_by_loop.pop(lp)
            for sub in state.subscriptions:
                if sub.watch is not None:
                    orphan_watches.append(sub.watch)
                    sub.watch = None
                sub.closed = True
            state.subscriptions.clear()
            orphan_states.append(state)
        return orphan_watches, orphan_states

    @staticmethod
    def _release_loop_clients(state: _StatusLoopState, *, async_clients: bool = True) -> None:
        """Close clients this store created for a loop bag."""
        if state.owns_sync_client and state.sync_client is not None:
            with contextlib.suppress(Exception):
                state.sync_client.close()
            state.sync_client = None
            state.owns_sync_client = False
        if not async_clients:
            return
        if state.owns_async_client and state.client is not None:
            close = getattr(state.client, 'close', None)
            if close is not None:
                with contextlib.suppress(Exception):
                    result = close()
                    # AsyncClient.close is async on some SDK versions; best-effort.
                    if asyncio.iscoroutine(result):
                        result.close()  # type: ignore[attr-defined]
            state.client = None
            state.owns_async_client = False

    @staticmethod
    def _try_clone_async_client(template: Any) -> AsyncClient | None:  # noqa: ANN401
        """Build a new AsyncClient matching ``template``'s project/database, if possible."""
        project = getattr(template, 'project', None)
        if not isinstance(project, str) or not project:
            return None
        kwargs: dict[str, Any] = {'project': project}
        database = getattr(template, '_database', None)
        if database is not None:
            kwargs['database'] = database
        credentials = getattr(template, '_credentials', None)
        if credentials is not None:
            kwargs['credentials'] = credentials
        client_info = getattr(template, '_client_info', None)
        if client_info is not None:
            kwargs['client_info'] = client_info
        client_options = getattr(template, '_client_options', None)
        if client_options is not None:
            kwargs['client_options'] = client_options
        try:
            return firestore.AsyncClient(**kwargs)
        except Exception:  # noqa: BLE001
            return None

    def _ensure_async_client(self) -> AsyncClient:
        """Async client for the running loop (created or bound lazily)."""
        state = self._status_loop_state()
        if state.client is not None:
            # Keep serving a bound client after close() so in-flight reads that
            # suspended across close can finish and hit the closed check.
            return state.client
        if self._closed:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message='FirestoreSessionStore: store is closed.',
            )
        loop = asyncio.get_running_loop()
        if self._provided_client is not None and self._provided_client_loop is None:
            state.client = self._provided_client
            state.owns_async_client = False
            self._provided_client_loop = loop
            return state.client
        if self._provided_client is not None:
            cloned = self._try_clone_async_client(self._provided_client)
            if cloned is not None:
                state.client = cloned
                state.owns_async_client = True
                return state.client
            # Test doubles / odd clients: share the provided instance.
            state.client = self._provided_client
            state.owns_async_client = False
            return state.client
        state.client = firestore.AsyncClient()
        state.owns_async_client = True
        return state.client

    @property
    def client(self) -> AsyncClient:
        """Async Firestore client for the running event loop."""
        return self._ensure_async_client()

    @client.setter
    def client(self, value: AsyncClient) -> None:
        """Bind an async client to the running loop (tests / advanced wiring)."""
        state = self._status_loop_state()
        state.client = value
        state.owns_async_client = False

    @property
    def sync_client(self) -> firestore.Client | None:
        """Sync watch client for the running loop, if one has been created."""
        return self._status_loop_state().sync_client

    @sync_client.setter
    def sync_client(self, value: firestore.Client | None) -> None:
        """Set/clear the running loop's sync client (tests rebind watches this way)."""
        state = self._status_loop_state()
        state.sync_client = value
        # Caller-supplied clients are never closed by the store.
        state.owns_sync_client = False

    @property
    def _subscriptions(self) -> set[_StatusSubscription]:
        """Current loop's open status subscriptions (tests and internal helpers)."""
        return self._status_loop_state().subscriptions

    @property
    def collection(self) -> str:
        """Root Firestore collection name for snapshot documents."""
        return self._collection

    @property
    def shard_size(self) -> int:
        """Max UTF-8 bytes per checkpoint shard document."""
        return self._shard_size

    @property
    def checkpoint_interval(self) -> int:
        """Diff-chain length before a new full-state checkpoint is written."""
        return self._checkpoint_interval

    @property
    def transaction_max_attempts(self) -> int:
        """How many times a contended transaction is retried before ABORTED."""
        return self._transaction_max_attempts

    @property
    def _lock(self) -> asyncio.Lock:
        """Loop-local lock for status subscription start/teardown."""
        getter = getattr(self, '_loop_lock_getter', None)
        if getter is None:
            getter = _loop_local_client(lambda: asyncio.Lock())
            object.__setattr__(self, '_loop_lock_getter', getter)
        return getter()

    def _prefix(self, context: dict[str, Any] | None) -> str:
        """Resolve and validate the tenant prefix for this call.

        The prefix is a document id in all three collection roots, so it is
        held to the same rules as snapshot and session ids — a bad prefix
        would silently re-route every path under it. Public operations
        resolve it once at entry and thread the string down so a caller
        mutating the context mid-await cannot split a write across tenants.
        """
        prefix = self._prefix_fn(context)
        _validate_doc_id(prefix, 'snapshot_path_prefix')
        return prefix

    def _snapshots_col(self, prefix: str) -> AsyncCollectionReference:
        """Return the Firestore collection reference for snapshots."""
        return self.client.collection(self.collection).document(prefix).collection('snapshots')

    def _pointers_col(self, prefix: str) -> AsyncCollectionReference:
        """Return the Firestore collection reference for session pointers."""
        return self.client.collection(f'{self.collection}-pointers').document(prefix).collection('pointers')

    def _shards_col(self, prefix: str) -> AsyncCollectionReference:
        """Return the Firestore collection reference for checkpoint shards."""
        return self.client.collection(f'{self.collection}-shards').document(prefix).collection('shards')

    def _snapshot_ref(self, snapshot_id: str, prefix: str) -> AsyncDocumentReference:
        """Return the Firestore document reference for a snapshot ID."""
        return self._snapshots_col(prefix).document(snapshot_id)

    def _pointer_ref(self, session_id: str, prefix: str) -> AsyncDocumentReference:
        """Return the Firestore document reference for a session pointer ID."""
        return self._pointers_col(prefix).document(session_id)

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> SessionSnapshot | None:
        """Load a snapshot by id, or the current tip of a session.

        Exactly one of ``snapshot_id`` or ``session_id`` must be set. Session
        lookup follows the pointer document and returns the snapshot most
        recently written for that session (last writer wins). After a branch
        (e.g. regenerate), concurrent writers make the current snapshot
        race-dependent; resume by ``snapshot_id`` when a specific branch
        matters.
        """
        require_one_selector(snapshot_id=snapshot_id, session_id=session_id)
        if snapshot_id is not None:
            _validate_doc_id(snapshot_id, 'snapshot_id')
        if session_id is not None:
            _validate_doc_id(session_id, 'session_id')
        prefix = self._prefix(context)
        transaction = self.client.transaction(read_only=True, max_attempts=self.transaction_max_attempts)

        @firestore.async_transactional
        async def read_in_transaction(transaction: AsyncTransaction) -> SessionSnapshot | None:
            if snapshot_id is not None:
                reconstructed = await self._reconstruct(transaction, snapshot_id, prefix=prefix)
                return self._to_snapshot(reconstructed) if reconstructed else None

            assert session_id is not None
            pointer_doc = await self._pointer_ref(session_id, prefix).get(transaction=transaction)
            if not pointer_doc.exists:
                return None

            try:
                pointer = _PointerDoc.model_validate(pointer_doc.to_dict() or {})
            except Exception as e:
                raise GenkitError(
                    status='DATA_LOSS',
                    message=f"FirestoreSessionStore: invalid session pointer document for '{session_id}'.",
                ) from e
            # Pointer carries the leaf's chain metadata so we can rebuild without
            # an extra snapshot-doc read when everything is present.
            current_id = pointer.current_snapshot_id
            checkpoint_id = pointer.checkpoint_id
            shard_count = pointer.checkpoint_shard_count
            if current_id and checkpoint_id and shard_count is not None:
                reconstructed = await self._reconstruct_from(
                    transaction,
                    checkpoint_id=checkpoint_id,
                    shard_count=shard_count,
                    segment_path=pointer.segment_path,
                    target_id=current_id,
                    prefix=prefix,
                )
                if reconstructed is not None:
                    result = self._to_snapshot(reconstructed)
                    if result is not None:
                        if result.session_id != session_id:
                            raise GenkitError(
                                status='DATA_LOSS',
                                message=(
                                    f"FirestoreSessionStore: session '{session_id}' pointer resolves to "
                                    f"snapshot '{result.snapshot_id}' owned by session '{result.session_id}'."
                                ),
                            )
                        return result

            if current_id:
                reconstructed = await self._reconstruct(transaction, current_id, prefix=prefix)
                result = self._to_snapshot(reconstructed)
                if result is not None and result.session_id != session_id:
                    raise GenkitError(
                        status='DATA_LOSS',
                        message=(
                            f"FirestoreSessionStore: session '{session_id}' pointer resolves to "
                            f"snapshot '{result.snapshot_id}' owned by session '{result.session_id}'."
                        ),
                    )
                return result
            return None

        return await _translate_txn_errors(read_in_transaction(transaction))

    async def save_snapshot(
        self,
        snapshot_id: str,
        fn: SaveFn,
        *,
        context: dict[str, Any] | None = None,
    ) -> SessionSnapshot | None:
        """Atomically create or update a snapshot and advance its session pointer.

        Abort, heartbeat, and finalize all share this path. A process-local lock
        can't coordinate across instances, so the snapshot write and pointer
        update commit together in one Firestore transaction. Status subscribers
        observe the new status through their own Firestore watches after that
        commit lands (there is no in-process fan-out from this method).

        Upserts (mutators that rewrite an existing snapshot) must target the
        current leaf. Rewriting an interior snapshot would break descendants
        that still apply its diff.

        Terminal statuses are absorbing: rewriting a finished snapshot's
        state or metadata is allowed, but changing its status raises
        ``FAILED_PRECONDITION``.

        A snapshot's lineage is fixed at creation: ``parent_id`` cannot change,
        and only the session's current snapshot may have its state rewritten
        (metadata such as status and heartbeat may be updated on any snapshot).
        Interior turns are therefore immutable; to remove content from history,
        delete the session's documents and re-create what should remain.
        """
        _validate_doc_id(snapshot_id, 'snapshot_id')
        # Resolve the tenant prefix before the transaction: a failing prefix
        # function must fail BEFORE anything commits, and the post-commit
        # notification must use the same prefix the write used.
        prefix = self._prefix(context)
        snap_ref = self._snapshot_ref(snapshot_id, prefix)
        transaction = self.client.transaction(max_attempts=self.transaction_max_attempts)

        @firestore.async_transactional
        async def rmw(transaction: AsyncTransaction) -> SessionSnapshot | None:
            existing_recon = await self._reconstruct(transaction, snapshot_id, prefix=prefix)
            existing = self._to_snapshot(existing_recon) if existing_recon else None
            try:
                next_snapshot = apply_save(existing=existing, snapshot_id=snapshot_id, fn=fn)
            except GenkitError:
                raise
            except BaseException as e:
                raise _UserCodeError() from e
            if next_snapshot is None:
                return None
            _validate_doc_id(next_snapshot.session_id, 'session_id')
            if next_snapshot.parent_id:
                _validate_doc_id(next_snapshot.parent_id, 'parent_id')

            sid = next_snapshot.snapshot_id
            session_id = session_id_of(next_snapshot)
            if not session_id:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message="FirestoreSessionStore requires 'sessionId' on the snapshot.",
                )
            assert sid is not None

            _pointer_ref = self._pointer_ref(session_id, prefix)
            pointer_snap = await _pointer_ref.get(transaction=transaction)
            pointer: _PointerDoc | None = None
            if pointer_snap.exists:
                try:
                    pointer = _PointerDoc.model_validate(pointer_snap.to_dict())
                except Exception:
                    # Corrupt pointer on the write path: treat as missing so this
                    # transaction rewrites it wholesale (self-heal), keeping the
                    # session writable instead of wedging every save.
                    logger.warning("Rewriting invalid session pointer document for '%s'", session_id)
            new_state = _state_to_dict(next_snapshot.state)

            meta: _SnapshotWriteMeta
            if existing_recon is not None:
                existing_doc, existing_state = existing_recon
                # Lineage is fixed at create: re-parenting an existing doc (or a
                # losing concurrent create that retries as an upsert) would
                # rewrite history under descendants that still apply its diff.
                old_parent = existing_doc.parent_id or ''
                new_parent = next_snapshot.parent_id or ''
                if old_parent != new_parent:
                    raise GenkitError(
                        status='FAILED_PRECONDITION',
                        message=(
                            f"Snapshot '{sid}' parent_id is immutable "
                            f"('{old_parent}' -> '{new_parent}'); create a "
                            'new snapshot to branch.'
                        ),
                    )
                # State is tip-only: an interior rewrite would leave children
                # applying a patch against a base that no longer matches.
                # Metadata-only updates (heartbeat, status, …) stay allowed.
                # A missing pointer proves nothing about interiority, so
                # deletion self-heal can still rewrite.
                if existing_state != new_state and pointer is not None and pointer.current_snapshot_id != sid:
                    raise GenkitError(
                        status='FAILED_PRECONDITION',
                        message=(
                            f"Snapshot '{sid}' is not session '{session_id}'s current "
                            "snapshot; rewriting an interior snapshot's state would "
                            "corrupt descendants' diffs; rewrite the session tip or "
                            'create a new snapshot to branch.'
                        ),
                    )
                if existing_doc.kind == 'checkpoint':
                    meta = self._write_checkpoint(
                        transaction,
                        sid,
                        new_state,
                        old_shard_count=existing_doc.checkpoint_shard_count,
                        prefix=prefix,
                    )
                else:
                    parent_id = existing_doc.parent_id
                    parent_state = None
                    if parent_id:
                        parent_recon = await self._reconstruct(transaction, parent_id, prefix=prefix)
                        parent_state = parent_recon[1] if parent_recon else None
                        if parent_recon and parent_recon[0].session_id != next_snapshot.session_id:
                            raise GenkitError(
                                status='FAILED_PRECONDITION',
                                message=(
                                    f"Snapshot '{sid}' has parent '{parent_id}' owned by session "
                                    f"'{parent_recon[0].session_id}', not '{next_snapshot.session_id}'."
                                ),
                            )
                    candidate_patch = _patch_to_json(diff_json(from_value=parent_state, to_value=new_state))
                    # Oversized patches can't live in one doc field; promote to checkpoint.
                    patch_too_large = _byte_length(candidate_patch) > self.shard_size
                    if patch_too_large:
                        meta = self._write_checkpoint(transaction, sid, new_state, prefix=prefix)
                    else:
                        meta = _SnapshotWriteMeta(
                            kind='diff',
                            checkpoint_id=existing_doc.checkpoint_id,
                            checkpoint_shard_count=existing_doc.checkpoint_shard_count,
                            segment_path=existing_doc.segment_path,
                            state_patch=candidate_patch,
                        )
            else:
                parent_meta = None
                if next_snapshot.parent_id:
                    parent_meta = await self._load_parent_chain_meta(
                        transaction,
                        next_snapshot.parent_id,
                        pointer,
                        prefix=prefix,
                    )
                # New root, missing parent chain, or segment full → full checkpoint.
                if (
                    not next_snapshot.parent_id
                    or parent_meta is None
                    or len(parent_meta.segment_path) + 1 >= self.checkpoint_interval
                ):
                    meta = self._write_checkpoint(transaction, sid, new_state, prefix=prefix)
                else:
                    parent_recon = await self._reconstruct_from(
                        transaction,
                        checkpoint_id=parent_meta.checkpoint_id,
                        shard_count=parent_meta.checkpoint_shard_count,
                        segment_path=parent_meta.segment_path,
                        target_id=next_snapshot.parent_id,
                        prefix=prefix,
                    )
                    parent_state = parent_recon[1] if parent_recon else None
                    if parent_recon and parent_recon[0].session_id != next_snapshot.session_id:
                        raise GenkitError(
                            status='FAILED_PRECONDITION',
                            message=(
                                f"Snapshot '{sid}' has parent '{next_snapshot.parent_id}' owned by "
                                f"session '{parent_recon[0].session_id}', not '{next_snapshot.session_id}'."
                            ),
                        )
                    candidate_patch = _patch_to_json(diff_json(from_value=parent_state, to_value=new_state))
                    # Oversized patches can't live in one doc field; promote to checkpoint.
                    patch_too_large = _byte_length(candidate_patch) > self.shard_size
                    if patch_too_large:
                        meta = self._write_checkpoint(transaction, sid, new_state, prefix=prefix)
                    else:
                        meta = _SnapshotWriteMeta(
                            kind='diff',
                            checkpoint_id=parent_meta.checkpoint_id,
                            checkpoint_shard_count=parent_meta.checkpoint_shard_count,
                            segment_path=[*parent_meta.segment_path, sid],
                            state_patch=candidate_patch,
                        )

            kind = meta.kind
            checkpoint_id = meta.checkpoint_id
            checkpoint_shard_count = meta.checkpoint_shard_count
            segment_path = meta.segment_path
            state_patch = meta.state_patch

            doc_model = _SnapshotDoc(
                snapshot_id=sid,
                session_id=session_id,
                parent_id=next_snapshot.parent_id,
                created_at=next_snapshot.created_at,
                updated_at=next_snapshot.updated_at or next_snapshot.created_at,
                status=next_snapshot.status,
                heartbeat_at=next_snapshot.heartbeat_at,
                finish_reason=next_snapshot.finish_reason,
                error=next_snapshot.error,
                kind=kind,
                checkpoint_id=checkpoint_id,
                checkpoint_shard_count=checkpoint_shard_count,
                segment_path=segment_path,
                state_patch=_encode_state_patch(state_patch),
            )
            transaction.set(
                snap_ref,
                doc_model.model_dump(by_alias=True, exclude_none=True, mode='json'),
            )
            await self._update_pointer_in_transaction(
                transaction,
                session_id,
                sid,
                pointer=pointer,
                is_new=existing_recon is None,
                checkpoint_id=checkpoint_id,
                checkpoint_shard_count=checkpoint_shard_count,
                segment_path=segment_path,
                prefix=prefix,
            )
            return next_snapshot

        return await _translate_txn_errors(rmw(transaction))

    async def on_snapshot_status_change(
        self, snapshot_id: str, *, context: dict[str, Any] | None = None
    ) -> SnapshotStatusStream:
        """Subscribe to status changes for a snapshot.

        Returns an async stream that yields each new status from a dedicated
        Firestore watch for this call. If the document does not exist yet, the
        stream stays open and the watch waits for it — missing is not treated
        as end-of-stream. Likewise, deleting a watched snapshot does not end
        its subscription; a stream ends with a terminal status or
        :meth:`close`. Iteration therefore ends in one of two shapes: the last
        yielded status is terminal (the run resolved), or no terminal was
        yielded (this store was closed locally; the run may still be live
        elsewhere).

        Exiting an ``async for`` with ``break`` does not end the subscription;
        close the stream (``async with`` or ``aclose()``) or it stays open
        until a terminal status or ``close()``. Omitting ``context`` leaves the
        prefix function with ``None`` (typically the default ``global`` prefix);
        callers that want ambient action context must pass it in.
        """
        _validate_doc_id(snapshot_id, 'snapshot_id')
        if self._closed:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message='FirestoreSessionStore: store is closed.',
            )
        # Tenant prefix is captured once so two tenants watching the same
        # snapshot id get independent listeners.
        prefix = self._prefix(context)
        async with self._lock:
            if self._closed:
                q: asyncio.Queue[SnapshotStatus | None] = asyncio.Queue()
                q.put_nowait(None)
                return iterate_statuses(q)

            sub = _StatusSubscription(queue=asyncio.Queue())
            state = self._status_loop_state()
            state.subscriptions.add(sub)
            try:
                sub.watch = self._attach_status_watch(snapshot_id, prefix=prefix, sub=sub)
            except Exception:
                state.subscriptions.discard(sub)
                raise

            if self._closed:
                # close() raced the watch attach; end this stream the way close
                # ends every other local stream.
                await self._teardown_subscription(sub)
                with contextlib.suppress(Exception):
                    sub.queue.put_nowait(None)
                return iterate_statuses(sub.queue)

            async def on_close() -> None:
                async with self._lock:
                    await self._teardown_subscription(sub)

            return iterate_statuses(sub.queue, on_close=on_close)

    async def _read_snapshot(
        self,
        snapshot_id: str,
        *,
        prefix: str,
    ) -> SessionSnapshot | None:
        """Read a single snapshot by id, reconstructing state from its checkpoint chain."""
        _validate_doc_id(snapshot_id, 'snapshot_id')
        transaction = self.client.transaction(read_only=True, max_attempts=self.transaction_max_attempts)

        @firestore.async_transactional
        async def read_in_transaction(transaction: AsyncTransaction) -> SessionSnapshot | None:
            reconstructed = await self._reconstruct(transaction, snapshot_id, prefix=prefix)
            return self._to_snapshot(reconstructed) if reconstructed else None

        return await _translate_txn_errors(read_in_transaction(transaction))

    async def _update_pointer_in_transaction(
        self,
        transaction: AsyncTransaction,
        session_id: str,
        snapshot_id: str,
        *,
        pointer: _PointerDoc | None,
        is_new: bool,
        checkpoint_id: str,
        checkpoint_shard_count: int,
        segment_path: list[str],
        prefix: str,
    ) -> None:
        """Advance or refresh the session pointer inside an open transaction.

        The pointer is advanced for a new leaf, refreshed when the current
        leaf is rewritten, and recreated when missing or invalid (TTL expiry,
        corruption) — last writer wins. The document is replaced wholesale so
        a reader never observes a partial update. Writes to other snapshots
        leave the pointer untouched.

        Callers must pass the pointer already loaded earlier in the same
        transaction — Firestore rejects reads after any buffered writes.
        """
        if not (is_new or pointer is None or pointer.current_snapshot_id == snapshot_id):
            return
        payload = _PointerDoc(
            current_snapshot_id=snapshot_id,
            checkpoint_id=checkpoint_id,
            checkpoint_shard_count=checkpoint_shard_count,
            segment_path=segment_path,
        ).model_dump(by_alias=True, exclude_none=False, mode='python')
        payload['updatedAt'] = datetime.now(timezone.utc).isoformat()
        transaction.set(self._pointer_ref(session_id, prefix), payload)

    async def _load_parent_chain_meta(
        self,
        transaction: AsyncTransaction,
        parent_id: str,
        pointer: _PointerDoc | None,
        *,
        prefix: str,
    ) -> _ParentChainMeta | None:
        """Resolve parent checkpoint/segment metadata without materializing state."""
        if pointer and pointer.current_snapshot_id == parent_id:
            if pointer.checkpoint_id and pointer.checkpoint_shard_count is not None:
                return _ParentChainMeta(
                    checkpoint_id=pointer.checkpoint_id,
                    checkpoint_shard_count=pointer.checkpoint_shard_count,
                    segment_path=pointer.segment_path,
                )
        snap = await self._snapshot_ref(parent_id, prefix).get(transaction=transaction)
        if not snap.exists:
            logger.warning("Parent snapshot document '%s' does not exist", parent_id)
            return None
        try:
            doc = _SnapshotDoc.model_validate(snap.to_dict())
        except Exception:
            logger.warning("Parent snapshot document '%s' contains invalid metadata", parent_id)
            return None
        return _ParentChainMeta(
            checkpoint_id=doc.checkpoint_id,
            checkpoint_shard_count=doc.checkpoint_shard_count,
            segment_path=doc.segment_path,
        )

    async def _reconstruct(
        self,
        transaction: AsyncTransaction,
        snapshot_id: str,
        *,
        prefix: str,
    ) -> tuple[_SnapshotDoc, dict[str, Any]] | None:
        snap = await self._snapshot_ref(snapshot_id, prefix).get(transaction=transaction)
        if not snap.exists:
            return None
        try:
            doc = _SnapshotDoc.model_validate(snap.to_dict())
        except Exception as e:
            # Fail loud: returning None would look like "missing" and let
            # save_snapshot overwrite the bad leaf as a brand-new checkpoint.
            raise GenkitError(
                status='DATA_LOSS',
                message=f"FirestoreSessionStore: invalid snapshot document '{snap.reference.path}'.",
            ) from e
        return await self._reconstruct_from(
            transaction,
            checkpoint_id=doc.checkpoint_id,
            shard_count=doc.checkpoint_shard_count,
            segment_path=doc.segment_path,
            target_id=snapshot_id,
            prefix=prefix,
        )

    async def _reconstruct_from(
        self,
        transaction: AsyncTransaction,
        *,
        checkpoint_id: str,
        shard_count: int,
        segment_path: list[str],
        target_id: str,
        prefix: str,
    ) -> tuple[_SnapshotDoc, dict[str, Any]] | None:
        if shard_count < 1:
            raise GenkitError(
                status='DATA_LOSS',
                message=(
                    f"FirestoreSessionStore: checkpoint '{checkpoint_id}' has invalid "
                    f'checkpointShardCount {shard_count} (must be >= 1).'
                ),
            )
        target_is_checkpoint = len(segment_path) == 0
        _snapshots_col = self._snapshots_col(prefix)
        _shards_col = self._shards_col(prefix)
        checkpoint_ref = _snapshots_col.document(checkpoint_id)
        shard_refs = [_shards_col.document(f'{checkpoint_id}_{i}') for i in range(shard_count)]
        seg_refs = [_snapshots_col.document(sid) for sid in segment_path]

        refs: list[AsyncDocumentReference] = []
        if target_is_checkpoint:
            refs.append(checkpoint_ref)
        refs.extend(shard_refs)
        refs.extend(seg_refs)

        if not refs:
            return None

        by_path: dict[str, DocumentSnapshot] = {}
        # AsyncTransaction.get_all awaits an async generator (library bug), so
        # batch-read through the client with the open transaction instead.
        snaps = [snap async for snap in self.client.get_all(refs, transaction=transaction)]
        for snap in snaps:
            by_path[snap.reference.path] = snap

        shard_snaps = [by_path[ref.path] for ref in shard_refs]
        state = self._stitch(shard_snaps)

        if target_is_checkpoint:
            checkpoint_snap = by_path.get(checkpoint_ref.path)
            if checkpoint_snap is None or not checkpoint_snap.exists:
                logger.warning(
                    "Checkpoint snapshot document '%s' does not exist for target '%s'",
                    checkpoint_ref.path,
                    target_id,
                )
                return None
            try:
                checkpoint_doc = _SnapshotDoc.model_validate(checkpoint_snap.to_dict())
            except Exception as e:
                raise GenkitError(
                    status='DATA_LOSS',
                    message=f"FirestoreSessionStore: invalid checkpoint document '{checkpoint_ref.path}'.",
                ) from e
            if checkpoint_doc.snapshot_id != target_id:
                # Pointer and snapshot metadata are written in one transaction,
                # so a doc disagreeing with its own address is corruption.
                raise GenkitError(
                    status='DATA_LOSS',
                    message=(
                        f"FirestoreSessionStore: checkpoint document '{checkpoint_ref.path}' "
                        f"snapshotId mismatch (got '{checkpoint_doc.snapshot_id}', expected '{target_id}')."
                    ),
                )
            if not isinstance(state, dict):
                raise GenkitError(
                    status='DATA_LOSS',
                    message=f"FirestoreSessionStore: checkpoint '{checkpoint_id}' shards decode to a non-object state.",
                )
            return checkpoint_doc, state

        target_doc: _SnapshotDoc | None = None
        for ref in seg_refs:
            seg_snap = by_path.get(ref.path)
            if seg_snap is None or not seg_snap.exists:
                if ref is seg_refs[-1]:
                    # The final segment is the target itself (segmentPath ends
                    # with the target's own id); a wholly deleted target is
                    # not-found (e.g. TTL), matching by-id lookup semantics.
                    return None
                # An *interior* segment referenced by the target's own
                # segmentPath: its absence is a broken chain, the same failure
                # class as a missing shard.
                raise GenkitError(
                    status='DATA_LOSS',
                    message=f"FirestoreSessionStore: missing segment snapshot document '{ref.path}'.",
                )
            try:
                seg_doc = _SnapshotDoc.model_validate(seg_snap.to_dict())
            except Exception as e:
                raise GenkitError(
                    status='DATA_LOSS',
                    message=f"FirestoreSessionStore: invalid segment document '{ref.path}'.",
                ) from e
            try:
                state = apply_json_patch(
                    doc=state,
                    patch=_patch_from_json(_decode_state_patch(seg_doc.state_patch, snapshot_id=seg_doc.snapshot_id)),
                )
            except GenkitError:
                raise
            except Exception as e:
                raise GenkitError(
                    status='DATA_LOSS',
                    message=(
                        f"FirestoreSessionStore: snapshot '{seg_doc.snapshot_id}' has an unreadable "
                        f"'statePatch' field (expected a JSON string; found "
                        f'{type(seg_doc.state_patch).__name__}). '
                        'Documents written by pre-release versions are not readable; '
                        "delete the session's documents and re-create."
                    ),
                ) from e
            target_doc = seg_doc

        if target_doc is None or target_doc.snapshot_id != target_id:
            raise GenkitError(
                status='DATA_LOSS',
                message=(
                    f"FirestoreSessionStore: segment chain for '{target_id}' ends at "
                    f"'{target_doc.snapshot_id if target_doc else None}' instead of the target."
                ),
            )
        if not isinstance(state, dict):
            raise GenkitError(
                status='DATA_LOSS',
                message=f"FirestoreSessionStore: reconstructed state for '{target_id}' is not an object.",
            )
        return target_doc, state

    def _write_shards(
        self,
        transaction: AsyncTransaction,
        checkpoint_id: str,
        state: dict[str, Any],
        *,
        old_shard_count: int = 0,
        prefix: str,
    ) -> int:
        _shards_col = self._shards_col(prefix)
        # ensure_ascii=False so multilingual checkpoints stay compact UTF-8;
        # escaping would inflate size and change shard boundaries. Formats are
        # independent per runtime — don't share one collection across them.
        buf = _json_dumps_utf8(state)
        count = max(1, (len(buf) + self.shard_size - 1) // self.shard_size)
        for i in range(count):
            chunk = buf[i * self.shard_size : (i + 1) * self.shard_size]
            shard_doc = _ShardDoc(chunk=chunk)
            transaction.set(_shards_col.document(f'{checkpoint_id}_{i}'), shard_doc.model_dump(mode='python'))
        for i in range(count, old_shard_count):
            transaction.delete(_shards_col.document(f'{checkpoint_id}_{i}'))
        return count

    def _write_checkpoint(
        self,
        transaction: AsyncTransaction,
        snapshot_id: str,
        state: dict[str, Any],
        *,
        old_shard_count: int = 0,
        prefix: str,
    ) -> _SnapshotWriteMeta:
        shard_count = self._write_shards(
            transaction,
            snapshot_id,
            state,
            old_shard_count=old_shard_count,
            prefix=prefix,
        )
        return _SnapshotWriteMeta(
            kind='checkpoint',
            checkpoint_id=snapshot_id,
            checkpoint_shard_count=shard_count,
            segment_path=[],
            state_patch=None,
        )

    def _stitch(self, shard_snaps: list[DocumentSnapshot]) -> dict[str, Any] | None:
        if not shard_snaps:
            return {}
        buffers: list[bytes] = []
        for snap in shard_snaps:
            if not snap.exists:
                raise GenkitError(
                    status='DATA_LOSS',
                    message=f"FirestoreSessionStore: missing checkpoint shard '{snap.id}'.",
                )
            try:
                shard = _ShardDoc.model_validate(snap.to_dict())
            except Exception as e:
                raise GenkitError(
                    status='DATA_LOSS',
                    message=f"FirestoreSessionStore: invalid checkpoint shard '{snap.id}'.",
                ) from e
            buffers.append(shard.chunk)
        try:
            decoded = json.loads(b''.join(buffers).decode('utf-8'))
        except Exception as e:
            shard_ids = ', '.join(snap.id for snap in shard_snaps)
            raise GenkitError(
                status='DATA_LOSS',
                message=f"FirestoreSessionStore: corrupt shard/snapshot document for '{shard_ids}': {e}",
            ) from e
        return decoded

    def _to_snapshot(self, reconstructed: tuple[_SnapshotDoc, dict[str, Any]] | None) -> SessionSnapshot | None:
        if reconstructed is None:
            return None
        doc, state_raw = reconstructed
        return doc.to_session_snapshot(state_raw)

    def _ensure_sync_client(self) -> firestore.Client:
        """Return this loop's sync client used for realtime watches.

        Uses ``_to_sync_copy()`` on the loop's async client when the
        constructor did not supply a sync client for this loop.
        """
        if self._closed:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message='FirestoreSessionStore: store is closed.',
            )
        state = self._status_loop_state()
        if state.sync_client is not None:
            return state.sync_client
        loop = asyncio.get_running_loop()
        if self._provided_sync_client is not None and self._provided_sync_loop is None:
            state.sync_client = self._provided_sync_client
            state.owns_sync_client = False
            self._provided_sync_loop = loop
            return state.sync_client
        async_client = self._ensure_async_client()
        if hasattr(async_client, '_to_sync_copy'):
            state.sync_client = async_client._to_sync_copy()
            state.owns_sync_client = True
            return state.sync_client
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                'Realtime status watches require a synchronous Firestore client. '
                'Unable to derive sync client from client. '
                "Please pass 'sync_client' to FirestoreSessionStore."
            ),
        )

    def _attach_status_watch(
        self,
        snapshot_id: str,
        *,
        prefix: str,
        sub: _StatusSubscription,
    ) -> Any:  # noqa: ANN401
        """Start a dedicated Firestore listener for one status subscription.

        Realtime watches require the sync client: the async client's
        ``on_snapshot`` is a non-functional stub, so its presence must not
        be used for dispatch. Caller holds ``_lock``.
        """
        sync_client = self._ensure_sync_client()
        ref = sync_client.collection(self.collection).document(prefix).collection('snapshots').document(snapshot_id)
        loop = asyncio.get_running_loop()

        def on_snapshot(doc_snapshots: list[DocumentSnapshot], changes: Any, read_time: Any) -> None:  # noqa: ANN401
            if not doc_snapshots:
                return
            status = _status_from_doc(doc_snapshots[0])
            if status is None:
                # Missing / deleted: keep waiting; not end-of-stream.
                return

            def deliver() -> None:
                if sub.closed:
                    return
                if sub.last_status == status:
                    return
                sub.last_status = status
                with contextlib.suppress(Exception):
                    sub.queue.put_nowait(status)
                if status not in TERMINAL_STATUSES:
                    return
                with contextlib.suppress(Exception):
                    sub.queue.put_nowait(None)
                sub.closed = True

                async def cleanup() -> None:
                    async with self._lock:
                        await self._teardown_subscription(sub)

                cleanup_coro = cleanup()
                try:
                    asyncio.create_task(cleanup_coro)
                except RuntimeError:
                    cleanup_coro.close()
                    # Loop can't schedule (often shutting down). Consumer already
                    # has status+EOS — still drop the watch so it doesn't linger.
                    state = self._status_loop_state()
                    state.subscriptions.discard(sub)
                    watch = sub.watch
                    sub.watch = None
                    if watch is not None:
                        self._unsubscribe_watch(watch)

            loop.call_soon_threadsafe(deliver)

        return ref.on_snapshot(on_snapshot)

    async def _teardown_subscription(self, sub: _StatusSubscription) -> None:
        """Stop one subscription's watch and drop it from the loop bag. Holds ``_lock``."""
        state = self._status_loop_state()
        state.subscriptions.discard(sub)
        watch = sub.watch
        sub.watch = None
        sub.closed = True
        if watch is not None:
            await asyncio.to_thread(self._unsubscribe_watch, watch)

    def close(self) -> None:
        """Stop active watches, end local status streams, and release the store's resources.

        Safe to call more than once. Does not close a ``client`` / ``sync_client``
        the caller passed into the constructor, and never closes async clients.
        Sync clients this store created per loop are closed here.

        Synchronous and fast: watches are unsubscribed on the calling thread,
        and every stream from :meth:`on_snapshot_status_change` in THIS
        process ends so consumers exit their ``async for`` loops. A stream
        that ends without having yielded a terminal status means the store
        shut down locally — the run itself may still be live, and subscribers
        in other processes are unaffected. Ending a run is a different act:
        write a terminal status, which every subscriber everywhere observes.

        Stream endings are delivered through each subscriber's event loop
        (subscriptions and clients are loop-local). Loop bags are kept so an
        in-flight read that suspended across ``close`` can still finish against
        that loop's async client.
        """
        self._closed = True
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        with self._status_by_loop_mu:
            loop_bags = list(self._status_by_loop.items())
        for loop, state in loop_bags:
            subs = list(state.subscriptions)
            # Clear FIRST so in-flight watch callbacks see closed subs, THEN
            # stop watches, THEN wake consumers.
            state.subscriptions.clear()
            watches: list[Any] = []
            queues: list[asyncio.Queue[SnapshotStatus | None]] = []
            for sub in subs:
                sub.closed = True
                if sub.watch is not None:
                    watches.append(sub.watch)
                    sub.watch = None
                queues.append(sub.queue)
            for watch in watches:
                self._unsubscribe_watch(watch)
            for q in queues:
                if running is loop:
                    with contextlib.suppress(Exception):
                        q.put_nowait(None)
                elif not loop.is_closed():
                    # Foreign thread / other loop: queues are loop-bound.
                    with contextlib.suppress(RuntimeError):
                        loop.call_soon_threadsafe(q.put_nowait, None)
            # Drop owned sync watches clients; keep async clients for in-flight I/O.
            self._release_loop_clients(state, async_clients=False)

    @staticmethod
    def _watch_listener_thread(watch: Any) -> threading.Thread | None:  # noqa: ANN401
        """Return the gRPC watch consumer thread, if the SDK exposes it."""
        # google.cloud.firestore_v1.watch.Watch holds BackgroundConsumer._thread.
        consumer = getattr(watch, '_consumer', None)
        if consumer is None:
            return None
        thread = getattr(consumer, '_thread', None)
        return thread if isinstance(thread, threading.Thread) else None

    @staticmethod
    def _unsubscribe_watch(watch: Any) -> None:  # noqa: ANN401
        """Unsubscribe a watch, off-thread when called from its own listener."""

        def stop() -> None:
            with contextlib.suppress(Exception):
                watch.unsubscribe()

        background = FirestoreSessionStore._watch_listener_thread(watch)
        if background is not None and background is threading.current_thread():
            # Joining the listener thread from itself is impossible; hand
            # teardown to a short-lived helper so Watch.close can finish.
            threading.Thread(target=stop, daemon=True, name='fs-watch-teardown').start()
            return
        try:
            watch.unsubscribe()
        except RuntimeError as e:
            if 'cannot join current thread' not in str(e):
                return
            threading.Thread(target=stop, daemon=True, name='fs-watch-teardown').start()
        except Exception:
            return
