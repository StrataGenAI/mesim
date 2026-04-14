"""
mesh/store_forward.py — MESIM DTN store-and-forward

Provides two classes:

ForwardQueue (sync)
    SQLite-backed queue that persists encrypted message payloads for offline
    peers.  Each entry has a 24-hour TTL.  Survives node restart.

StoreForward (async)
    Wraps ForwardQueue and a MeshTransport.  Registers on_peer_connected so
    that queued messages are auto-flushed when a peer reconnects.  Also runs
    a background loop that flushes any pending messages to peers that are
    currently reachable (in case a connect callback was missed).  Exposes
    send_or_queue() — sends immediately if a live session exists, otherwise
    enqueues for later delivery.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.crypto import SessionKey, decrypt_message, encrypt_message

if TYPE_CHECKING:
    from mesh.transport import MeshTransport
    from rich.console import Console

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TTL_SECONDS        = 86_400   # 24 hours
MAX_QUEUE_PER_PEER = 256      # cap per peer to prevent unbounded growth
_FLUSH_INTERVAL    = 30       # seconds between periodic flush sweeps


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueuedMessage:
    """A single entry returned by ForwardQueue.get_pending()."""

    id: int
    peer_id: str
    payload: bytes        # plaintext payload (decrypted in memory)
    queued_at: float
    expires_at: float
    attempt: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_db(path: str) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode. Abstraction point for future SQLCipher."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _encrypt_payload(key: SessionKey, payload: bytes) -> bytes:
    """Returns nonce (12B) || ciphertext_with_tag."""
    nonce, ct = encrypt_message(payload, key)
    return nonce + ct


def _decrypt_payload(key: SessionKey, blob: bytes) -> bytes:
    """Inverse of _encrypt_payload. Raises InvalidTag on failure."""
    nonce, ct = blob[:12], blob[12:]
    return decrypt_message(nonce, ct, key)


def normalize_peer_id(peer_id: str) -> str:
    """Normalize peer_id to canonical UUID4 form (36-char with hyphens).

    Handles the old wire format (32-char hex, no hyphens) that was produced by
    device_id_from_bytes() before the transport canonical-form bug fix.  Any
    peer_id that is already UUID4 form (36 chars) is returned unchanged.
    Unknown formats are passed through without modification.
    """
    if len(peer_id) == 36:
        return peer_id  # already canonical
    if len(peer_id) == 32 and all(c in "0123456789abcdefABCDEF" for c in peer_id):
        p = peer_id.lower()
        return f"{p[0:8]}-{p[8:12]}-{p[12:16]}-{p[16:20]}-{p[20:32]}"
    return peer_id  # unknown format — pass through unchanged


# ---------------------------------------------------------------------------
# ForwardQueue
# ---------------------------------------------------------------------------


class ForwardQueue:
    """
    Persistent, encrypted FIFO queue for store-and-forward messages.

    Each row's payload is encrypted with the provided SessionKey so the
    SQLite file is safe at rest.  All peer_id arguments are normalized to
    canonical UUID4 form before any DB operation so entries are always
    findable regardless of which format the caller uses.

    Usage::

        key = SessionKey(raw=os.urandom(32))
        with ForwardQueue("fwd.db", key) as q:
            qid = q.enqueue("peer_abc", b"raw payload")
            msgs = q.get_pending("peer_abc")
            q.mark_sent(qid)
    """

    def __init__(self, db_path: str, store_key: SessionKey) -> None:
        self._db_path = db_path
        self._key = store_key
        self._conn: sqlite3.Connection | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def open(self) -> None:
        if self._conn is not None:
            return
        self._conn = _open_db(self._db_path)
        self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> ForwardQueue:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Public API ─────────────────────────────────────────────────────────

    def enqueue(self, peer_id: str, payload: bytes) -> int:
        """
        Encrypt and add payload to the queue for peer_id.

        peer_id is normalized to UUID4 form before storage.
        Returns the queue id.
        Raises OverflowError if MAX_QUEUE_PER_PEER is reached for this peer.
        """
        peer_id = normalize_peer_id(peer_id)
        self._assert_open()
        if self.queue_size(peer_id) >= MAX_QUEUE_PER_PEER:
            raise OverflowError(
                f"Forward queue for peer {peer_id!r} is full "
                f"(max {MAX_QUEUE_PER_PEER})"
            )
        now = time.time()
        blob = _encrypt_payload(self._key, payload)
        cur = self._conn.execute(
            "INSERT INTO forward_queue (peer_id, payload, queued_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (peer_id, blob, now, now + TTL_SECONDS),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_pending(self, peer_id: str) -> list[QueuedMessage]:
        """
        Return all non-expired queued messages for peer_id, oldest first.

        peer_id is normalized to UUID4 form before the DB query.
        """
        peer_id = normalize_peer_id(peer_id)
        self._assert_open()
        now = time.time()
        cur = self._conn.execute(
            "SELECT id, peer_id, payload, queued_at, expires_at, attempt"
            " FROM forward_queue"
            " WHERE peer_id=? AND expires_at > ?"
            " ORDER BY queued_at ASC, id ASC",
            (peer_id, now),
        )
        results = []
        for row_id, p_id, blob, queued_at, expires_at, attempt in cur.fetchall():
            payload = _decrypt_payload(self._key, bytes(blob))
            results.append(QueuedMessage(
                id=row_id,
                peer_id=p_id,
                payload=payload,
                queued_at=queued_at,
                expires_at=expires_at,
                attempt=attempt,
            ))
        return results

    def mark_sent(self, queue_id: int) -> None:
        """Remove an entry from the queue. No-op for unknown ids."""
        self._assert_open()
        self._conn.execute("DELETE FROM forward_queue WHERE id=?", (queue_id,))
        self._conn.commit()

    def increment_attempt(self, queue_id: int) -> None:
        """Increment the attempt counter for a queued message."""
        self._assert_open()
        self._conn.execute(
            "UPDATE forward_queue SET attempt=attempt+1 WHERE id=?", (queue_id,)
        )
        self._conn.commit()

    def purge_expired(self) -> int:
        """Delete all entries past their TTL. Returns the number of rows deleted."""
        self._assert_open()
        now = time.time()
        cur = self._conn.execute(
            "DELETE FROM forward_queue WHERE expires_at <= ?", (now,)
        )
        self._conn.commit()
        return cur.rowcount

    def queue_size(self, peer_id: str) -> int:
        """Count of non-expired entries for peer_id. peer_id is normalized."""
        peer_id = normalize_peer_id(peer_id)
        self._assert_open()
        now = time.time()
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM forward_queue WHERE peer_id=? AND expires_at > ?",
            (peer_id, now),
        )
        return cur.fetchone()[0]

    def total_size(self) -> int:
        """Count of all non-expired entries across all peers."""
        self._assert_open()
        now = time.time()
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM forward_queue WHERE expires_at > ?", (now,)
        )
        return cur.fetchone()[0]

    # ── Private helpers ────────────────────────────────────────────────────

    def _assert_open(self) -> None:
        if self._conn is None:
            raise RuntimeError("ForwardQueue is not open — call open() first")

    def _ensure_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS forward_queue (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                peer_id     TEXT    NOT NULL,
                payload     BLOB    NOT NULL,
                queued_at   REAL    NOT NULL,
                expires_at  REAL    NOT NULL,
                attempt     INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_fq_peer_exp
                ON forward_queue (peer_id, expires_at);
        """)
        self._conn.commit()


# ---------------------------------------------------------------------------
# StoreForward
# ---------------------------------------------------------------------------


class StoreForward:
    """
    Async store-and-forward coordinator.

    Wraps a ForwardQueue and a MeshTransport.  Registers an
    on_peer_connected callback so queued messages are delivered
    automatically when a peer comes online.  Also runs a background
    flush loop every _FLUSH_INTERVAL seconds to deliver to any peers
    that are reachable but whose connect callback was missed.

    Usage::

        sf = StoreForward(queue, transport)
        await sf.start()
        delivered = await sf.send_or_queue(peer_id, payload)
        await sf.stop()
    """

    def __init__(
        self,
        queue: ForwardQueue,
        transport: MeshTransport,
        console: Console | None = None,
    ) -> None:
        self._queue = queue
        self._transport = transport
        self._console = console
        self._flush_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Register on_peer_connected callback and start the periodic flush loop."""
        self._transport.on_peer_connected(self._on_peer_connected)
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Cancel the background flush loop."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

    async def send_or_queue(self, peer_id: str, payload: bytes) -> bool:
        """
        Send payload to peer_id immediately if a live session exists,
        otherwise persist it in the ForwardQueue.

        peer_id is normalized to UUID4 form before the session lookup.
        Returns True if delivered immediately, False if queued.
        """
        peer_id = normalize_peer_id(peer_id)
        sessions = self._transport.get_sessions()
        if peer_id in sessions:
            await self._transport.send(peer_id, payload)
            return True
        else:
            self._queue.enqueue(peer_id, payload)
            return False

    async def flush_peer(self, peer_id: str) -> int:
        """
        Deliver all pending queue entries for peer_id.

        peer_id is normalized to UUID4 form.  Prints a console line when
        there are messages to flush.  Returns the number of messages sent.
        Called automatically by _on_peer_connected and _flush_loop; can
        also be called manually.
        """
        peer_id = normalize_peer_id(peer_id)
        pending = self._queue.get_pending(peer_id)
        if pending and self._console is not None:
            sessions = self._transport.get_sessions()
            callsign = (
                sessions[peer_id].peer_bundle.callsign
                if peer_id in sessions
                else peer_id
            )
            self._console.print(
                f"[DTN] flushing {len(pending)} messages to {callsign}",
                markup=False,
            )
        sent = 0
        for msg in pending:
            try:
                await self._transport.send(peer_id, msg.payload)
                self._queue.mark_sent(msg.id)
                sent += 1
            except Exception:
                self._queue.increment_attempt(msg.id)
        return sent

    # ── Private helpers ────────────────────────────────────────────────────

    async def _flush_loop(self) -> None:
        """
        Background task: every _FLUSH_INTERVAL seconds, flush pending messages
        to any peer that currently has an active session.
        """
        while True:
            await asyncio.sleep(_FLUSH_INTERVAL)
            sessions = self._transport.get_sessions()
            for peer_id in list(sessions):
                if self._queue.queue_size(peer_id) > 0:
                    await self.flush_peer(peer_id)

    def _on_peer_connected(self, peer_id: str) -> None:
        """Synchronous callback registered with transport; schedules flush as a task."""
        asyncio.create_task(self.flush_peer(peer_id))
