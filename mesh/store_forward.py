"""
mesh/store_forward.py — MESIM DTN store-and-forward

Provides two classes:

ForwardQueue (sync)
    SQLite-backed queue that persists encrypted message payloads for offline
    peers.  Each entry has a 24-hour TTL.  Survives node restart.

StoreForward (async)
    Wraps ForwardQueue and a MeshTransport.  Registers on_peer_connected so
    that queued messages are auto-flushed when a peer reconnects.  Exposes
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TTL_SECONDS        = 86_400   # 24 hours
MAX_QUEUE_PER_PEER = 256      # cap per peer to prevent unbounded growth


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


# ---------------------------------------------------------------------------
# ForwardQueue
# ---------------------------------------------------------------------------


class ForwardQueue:
    """
    Persistent, encrypted FIFO queue for store-and-forward messages.

    Each row's payload is encrypted with the provided SessionKey so the
    SQLite file is safe at rest.

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

        Returns the queue id.
        Raises OverflowError if MAX_QUEUE_PER_PEER is reached for this peer.
        """
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
        """
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
        """Count of non-expired entries for peer_id."""
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
    automatically when a peer comes online.

    Usage::

        sf = StoreForward(queue, transport)
        await sf.start()
        delivered = await sf.send_or_queue(peer_id, payload)
        await sf.stop()
    """

    def __init__(self, queue: ForwardQueue, transport: MeshTransport) -> None:
        self._queue = queue
        self._transport = transport

    async def start(self) -> None:
        """Register on_peer_connected callback with the transport."""
        self._transport.on_peer_connected(self._on_peer_connected)

    async def stop(self) -> None:
        pass  # no background tasks to cancel

    async def send_or_queue(self, peer_id: str, payload: bytes) -> bool:
        """
        Send payload to peer_id immediately if a live session exists,
        otherwise persist it in the ForwardQueue.

        Returns True if delivered immediately, False if queued.
        """
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

        Returns the number of messages sent.
        Called automatically by _on_peer_connected; can also be called manually.
        """
        pending = self._queue.get_pending(peer_id)
        sent = 0
        for msg in pending:
            try:
                await self._transport.send(peer_id, msg.payload)
                self._queue.mark_sent(msg.id)
                sent += 1
            except Exception:
                self._queue.increment_attempt(msg.id)
        return sent

    def _on_peer_connected(self, peer_id: str) -> None:
        """Synchronous callback registered with transport; schedules flush as a task."""
        asyncio.create_task(self.flush_peer(peer_id))
