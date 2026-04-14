"""
tests/test_store_forward.py — Unit tests for mesh/store_forward.py

Groups:
  1. ForwardQueue lifecycle
  2. enqueue
  3. get_pending
  4. mark_sent
  5. increment_attempt
  6. purge_expired
  7. Encryption at rest
  8. Persistence
  9. StoreForward async integration
"""

import asyncio
import os
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.crypto import SessionKey
from mesh.store_forward import (
    ForwardQueue,
    MAX_QUEUE_PER_PEER,
    QueuedMessage,
    StoreForward,
    TTL_SECONDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store_key():
    return SessionKey(raw=os.urandom(32))


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "fwd.db")


@pytest.fixture
def queue(tmp_db, store_key):
    q = ForwardQueue(tmp_db, store_key)
    q.open()
    yield q
    q.close()


def _make_transport(sessions: dict | None = None):
    """Create a minimal MeshTransport mock."""
    t = MagicMock()
    t.get_sessions.return_value = sessions or {}
    t.send = AsyncMock(return_value=1)
    t._connected_cbs = []

    def on_peer_connected(cb):
        t._connected_cbs.append(cb)

    t.on_peer_connected.side_effect = on_peer_connected
    return t


# ---------------------------------------------------------------------------
# Group 1 — ForwardQueue lifecycle
# ---------------------------------------------------------------------------


def test_queue_open_creates_db_file(tmp_db, store_key):
    q = ForwardQueue(tmp_db, store_key)
    q.open()
    q.close()
    assert os.path.exists(tmp_db)


def test_queue_open_creates_table(tmp_db, store_key):
    q = ForwardQueue(tmp_db, store_key)
    q.open()
    q.close()
    conn = sqlite3.connect(tmp_db)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    conn.close()
    assert "forward_queue" in tables


def test_queue_context_manager(tmp_db, store_key):
    with ForwardQueue(tmp_db, store_key) as q:
        assert q._conn is not None
    assert q._conn is None


def test_queue_close_without_open_is_safe(tmp_db, store_key):
    q = ForwardQueue(tmp_db, store_key)
    q.close()  # must not raise


# ---------------------------------------------------------------------------
# Group 2 — enqueue
# ---------------------------------------------------------------------------


def test_enqueue_returns_int_id(queue):
    qid = queue.enqueue("peer1", b"hello")
    assert isinstance(qid, int)
    assert qid > 0


def test_enqueue_increments_id(queue):
    id1 = queue.enqueue("peer1", b"first")
    id2 = queue.enqueue("peer1", b"second")
    assert id2 > id1


def test_enqueue_sets_ttl(queue):
    before = time.time()
    queue.enqueue("peer1", b"msg")
    after = time.time()
    msgs = queue.get_pending("peer1")
    assert before + TTL_SECONDS <= msgs[0].expires_at <= after + TTL_SECONDS


def test_enqueue_enforces_max_queue_per_peer(queue):
    for _ in range(MAX_QUEUE_PER_PEER):
        queue.enqueue("peer1", b"x")
    with pytest.raises(OverflowError):
        queue.enqueue("peer1", b"overflow")


def test_enqueue_different_peers_independent(queue):
    queue.enqueue("alice", b"a")
    queue.enqueue("bob",   b"b")
    # Neither peer hits the limit
    assert queue.queue_size("alice") == 1
    assert queue.queue_size("bob")   == 1


# ---------------------------------------------------------------------------
# Group 3 — get_pending
# ---------------------------------------------------------------------------


def test_get_pending_empty(queue):
    assert queue.get_pending("nobody") == []


def test_get_pending_returns_queued_message_objects(queue):
    queue.enqueue("peer1", b"hi")
    pending = queue.get_pending("peer1")
    assert len(pending) == 1
    assert isinstance(pending[0], QueuedMessage)


def test_get_pending_payload_roundtrip(queue):
    queue.enqueue("peer1", b"secret payload")
    msg = queue.get_pending("peer1")[0]
    assert msg.payload == b"secret payload"


def test_get_pending_ordered_by_queued_at(queue):
    queue.enqueue("peer1", b"first")
    queue.enqueue("peer1", b"second")
    pending = queue.get_pending("peer1")
    assert pending[0].queued_at <= pending[1].queued_at


def test_get_pending_excludes_expired(queue):
    # Manually insert an expired row
    expired_at = time.time() - 1
    queue._conn.execute(
        "INSERT INTO forward_queue (peer_id, payload, queued_at, expires_at)"
        " VALUES (?, ?, ?, ?)",
        ("peer1", b"\x00" * 28, time.time() - TTL_SECONDS - 10, expired_at),
    )
    queue._conn.commit()
    # Should not appear in get_pending
    assert queue.get_pending("peer1") == []


def test_get_pending_peer_isolation(queue):
    queue.enqueue("alice", b"for alice")
    queue.enqueue("bob",   b"for bob")
    assert len(queue.get_pending("alice")) == 1
    assert queue.get_pending("alice")[0].payload == b"for alice"


# ---------------------------------------------------------------------------
# Group 4 — mark_sent
# ---------------------------------------------------------------------------


def test_mark_sent_removes_entry(queue):
    qid = queue.enqueue("peer1", b"remove me")
    queue.mark_sent(qid)
    assert queue.get_pending("peer1") == []


def test_mark_sent_idempotent(queue):
    qid = queue.enqueue("peer1", b"twice")
    queue.mark_sent(qid)
    queue.mark_sent(qid)  # must not raise
    assert queue.queue_size("peer1") == 0


def test_mark_sent_unknown_id_is_noop(queue):
    queue.mark_sent(999999)  # must not raise


# ---------------------------------------------------------------------------
# Group 5 — increment_attempt
# ---------------------------------------------------------------------------


def test_increment_attempt_increments_counter(queue):
    qid = queue.enqueue("peer1", b"retry me")
    msg = queue.get_pending("peer1")[0]
    assert msg.attempt == 0

    queue.increment_attempt(qid)
    msg = queue.get_pending("peer1")[0]
    assert msg.attempt == 1


def test_increment_attempt_multiple_times(queue):
    qid = queue.enqueue("peer1", b"many retries")
    queue.increment_attempt(qid)
    queue.increment_attempt(qid)
    queue.increment_attempt(qid)
    msg = queue.get_pending("peer1")[0]
    assert msg.attempt == 3


# ---------------------------------------------------------------------------
# Group 6 — purge_expired
# ---------------------------------------------------------------------------


def test_purge_expired_removes_expired(queue):
    # Insert an expired entry directly
    queue._conn.execute(
        "INSERT INTO forward_queue (peer_id, payload, queued_at, expires_at)"
        " VALUES (?, ?, ?, ?)",
        ("peer1", b"\x00" * 28, time.time() - TTL_SECONDS - 10, time.time() - 1),
    )
    queue._conn.commit()
    deleted = queue.purge_expired()
    assert deleted == 1


def test_purge_expired_keeps_fresh(queue):
    queue.enqueue("peer1", b"fresh")
    deleted = queue.purge_expired()
    assert deleted == 0
    assert queue.queue_size("peer1") == 1


def test_purge_expired_returns_count(queue):
    for _ in range(3):
        queue._conn.execute(
            "INSERT INTO forward_queue (peer_id, payload, queued_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            ("peer1", b"\x00" * 28, time.time() - TTL_SECONDS - 10, time.time() - 1),
        )
    queue._conn.commit()
    assert queue.purge_expired() == 3


# ---------------------------------------------------------------------------
# Group 7 — Encryption at rest
# ---------------------------------------------------------------------------


def test_payload_not_in_db_file(tmp_db, store_key):
    secret = b"CLASSIFIED PAYLOAD DATA"
    with ForwardQueue(tmp_db, store_key) as q:
        q.enqueue("peer1", secret)

    raw = open(tmp_db, "rb").read()
    assert secret not in raw


def test_total_size(queue):
    assert queue.total_size() == 0
    queue.enqueue("alice", b"a")
    queue.enqueue("bob",   b"b")
    assert queue.total_size() == 2


# ---------------------------------------------------------------------------
# Group 8 — Persistence
# ---------------------------------------------------------------------------


def test_persistence_survives_close_reopen(tmp_db, store_key):
    with ForwardQueue(tmp_db, store_key) as q:
        q.enqueue("peer1", b"persist me")

    with ForwardQueue(tmp_db, store_key) as q:
        pending = q.get_pending("peer1")
    assert len(pending) == 1
    assert pending[0].payload == b"persist me"


def test_persistence_mark_sent_survives_reopen(tmp_db, store_key):
    with ForwardQueue(tmp_db, store_key) as q:
        qid = q.enqueue("peer1", b"mark me")
        q.mark_sent(qid)

    with ForwardQueue(tmp_db, store_key) as q:
        assert q.get_pending("peer1") == []


# ---------------------------------------------------------------------------
# Group 9 — StoreForward async integration
# ---------------------------------------------------------------------------


async def test_store_forward_send_immediately_when_session_active(tmp_db, store_key):
    from core.crypto import SessionKey as SK
    from core.identity import create_identity, Rank
    from mesh.transport import MeshTransport, SessionInfo, NonceCounter
    from core.identity import get_public_bundle
    import itertools

    # Mock session
    identity_b = create_identity("BOB", Rank.NCO)
    bundle_b = get_public_bundle(identity_b)
    session = MagicMock()
    session.peer_addr = ("127.0.0.1", 9000)

    transport = _make_transport(sessions={"peer1": session})
    q = ForwardQueue(tmp_db, store_key)
    q.open()

    sf = StoreForward(q, transport)
    await sf.start()

    delivered = await sf.send_or_queue("peer1", b"immediate payload")
    assert delivered is True
    transport.send.assert_called_once_with("peer1", b"immediate payload")

    await sf.stop()
    q.close()


async def test_store_forward_queues_when_no_session(tmp_db, store_key):
    transport = _make_transport(sessions={})
    q = ForwardQueue(tmp_db, store_key)
    q.open()

    sf = StoreForward(q, transport)
    await sf.start()

    delivered = await sf.send_or_queue("offline_peer", b"deferred payload")
    assert delivered is False
    assert q.queue_size("offline_peer") == 1

    await sf.stop()
    q.close()


async def test_store_forward_flush_peer_delivers_all(tmp_db, store_key):
    transport = _make_transport(sessions={})
    q = ForwardQueue(tmp_db, store_key)
    q.open()
    q.enqueue("peer1", b"msg1")
    q.enqueue("peer1", b"msg2")

    sf = StoreForward(q, transport)
    await sf.start()

    # Simulate peer becoming reachable
    session = MagicMock()
    transport.get_sessions.return_value = {"peer1": session}

    count = await sf.flush_peer("peer1")
    assert count == 2
    assert q.queue_size("peer1") == 0

    await sf.stop()
    q.close()


async def test_store_forward_on_peer_connected_triggers_flush(tmp_db, store_key):
    transport = _make_transport(sessions={})
    q = ForwardQueue(tmp_db, store_key)
    q.open()
    q.enqueue("peer1", b"queued msg")

    sf = StoreForward(q, transport)
    await sf.start()

    # Simulate transport calling the on_peer_connected callback
    session = MagicMock()
    transport.get_sessions.return_value = {"peer1": session}

    # The StoreForward should have registered a callback
    assert len(transport._connected_cbs) == 1
    cb = transport._connected_cbs[0]
    result = cb("peer1")
    if asyncio.iscoroutine(result):
        await result

    await asyncio.sleep(0.05)  # let tasks settle
    assert q.queue_size("peer1") == 0

    await sf.stop()
    q.close()
